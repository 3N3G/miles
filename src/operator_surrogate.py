"""Small, visit-independent surrogates for one frozen pointwise sublayer."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class SparseOperator(nn.Module):
    """TopK affine-ReLU-affine model, with training-only normalization buffers."""

    def __init__(self, dimension, width, topk):
        super().__init__()
        if not 1 <= topk <= width:
            raise ValueError("topk must be in 1..width")
        self.dimension, self.width, self.topk = dimension, width, topk
        self.encoder = nn.Linear(dimension, width)
        self.decoder = nn.Linear(width, dimension)
        self.register_buffer("x_mean", torch.zeros(dimension))
        self.register_buffer("y_mean", torch.zeros(dimension))
        self.register_buffer("x_scale", torch.ones(()))
        self.register_buffer("y_scale", torch.ones(()))
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.copy_(F.normalize(self.decoder.weight, dim=0))

    @torch.no_grad()
    def set_statistics(self, x, y):
        self.x_mean.copy_(x.mean(0))
        self.y_mean.copy_(y.mean(0))
        self.x_scale.copy_((x - self.x_mean).square().mean().sqrt().clamp_min(1e-6))
        self.y_scale.copy_((y - self.y_mean).square().mean().sqrt().clamp_min(1e-6))

    def encode(self, x):
        positive = F.relu(self.encoder((x - self.x_mean) / self.x_scale))
        if self.topk == self.width:
            return positive
        values, indices = positive.topk(self.topk, dim=-1)
        return torch.zeros_like(positive).scatter(-1, indices, values)

    def forward(self, x):
        return self.decoder(self.encode(x)) * self.y_scale + self.y_mean

    def feature_contribution(self, x, feature):
        """Contribution in the native operator's output units, not normalized units."""
        return self.encode(x)[..., feature, None] * self.decoder.weight[:, feature] * self.y_scale

    def checkpoint(self):
        return {"dimension": self.dimension, "width": self.width, "topk": self.topk,
                "state": {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}}

    @classmethod
    def from_checkpoint(cls, checkpoint, device):
        model = cls(checkpoint["dimension"], checkpoint["width"], checkpoint["topk"])
        model.load_state_dict(checkpoint["state"])
        return model.to(device).eval().requires_grad_(False)


@torch.no_grad()
def predict(model, x, batch_size=256):
    return torch.cat([model(batch).float() for batch in x.split(batch_size)])


def fit_sparse(x, y, xv, yv, *, seed, width, topk, epochs, batch_size=256, learning_rate=0.001, report=None):
    """Each row is seen once per epoch. Select checkpoints using validation only."""
    torch.manual_seed(seed)
    model = SparseOperator(x.shape[-1], width, topk).to(x.device)
    model.set_statistics(x, y)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device=x.device).manual_seed(seed)
    best_error, best, history = math.inf, None, []
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(x), generator=generator, device=x.device)
        loss_sum = 0.0
        for index in permutation.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(x[index]) - y[index]) / model.y_scale).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite surrogate loss")
            loss.backward()
            optimizer.step()
            model.normalize_decoder()
            loss_sum += loss.item() * len(index)
        model.eval()
        with torch.no_grad():
            error = ((predict(model, xv) - yv) / model.y_scale).square().mean().item()
        row = {"epoch": epoch + 1, "train_normalized_mse": loss_sum / len(x), "validation_scaled_mse": error}
        history.append(row)
        if error < best_error:
            best_error, best = error, model.checkpoint()
        if report:
            report(row)
    if best is None:
        raise RuntimeError("No finite checkpoint")
    metadata = {"history": history, "training_rows": len(x), "validation_rows": len(xv),
                "epochs": epochs, "updates": epochs * math.ceil(len(x) / batch_size),
                "parameters": sum(p.numel() for p in model.parameters()),
                "seed": seed, "checkpoint_selection": "validation_scaled_mse only"}
    del optimizer, model
    return best, metadata


@torch.no_grad()
def fit_ridge(x, y, regularizer):
    mx, my = x.mean(0), y.mean(0)
    scale = (x - mx).square().mean().sqrt().clamp_min(1e-6)
    xn = (x - mx) / scale
    gram = xn.T @ xn / len(x)
    gram += regularizer * torch.eye(x.shape[-1], device=x.device)
    matrix = torch.linalg.solve(gram, xn.T @ (y - my) / len(x))
    return {"x_mean": mx, "x_scale": scale, "y_mean": my, "matrix": matrix}


def predict_ridge(x, fit):
    return (x - fit["x_mean"]) / fit["x_scale"] @ fit["matrix"] + fit["y_mean"]


@torch.no_grad()
def document_metrics(prediction, target, *, positions, training_mean):
    if len(target) % positions:
        raise ValueError("Incomplete paragraph")
    # One denominator per evaluated pass, shared by all competing methods.
    variance = (target - target.mean(0)).square().mean().clamp_min(1e-12)
    squared = (prediction - target).square().mean(-1).reshape(-1, positions).mean(-1) / variance
    cosine = F.cosine_similarity(prediction - training_mean, target - training_mean, dim=-1)
    return {"normalized_mse": squared.mean().item(), "per_paragraph_normalized_mse": squared.cpu().tolist(),
            "centered_cosine": cosine.mean().item(),
            "per_paragraph_centered_cosine": cosine.reshape(-1, positions).mean(-1).cpu().tolist()}


class VisitEdit:
    """Forward hook with explicit visit selection and safe native-output edits."""

    def __init__(self, *, replacement=None, pass_index=None, feature=None, random_seed=None):
        self.replacement, self.pass_index = replacement, pass_index
        self.feature, self.random_seed = feature, random_seed
        self.calls = 0
        self.edit_norms = []
        self.actual_edit_norms = []

    def __call__(self, module, args, output):
        visit = self.calls
        self.calls += 1
        if self.pass_index is not None and visit != self.pass_index:
            return None
        x = args[0].float()
        if self.feature is None:
            if self.replacement is None:
                return output
            model = self.replacement[visit] if isinstance(self.replacement, list) else self.replacement
            result = model(x) if callable(model) else model.expand_as(x)
            return result.to(output.dtype)
        contribution = self.replacement.feature_contribution(x[:, -1], self.feature)
        if self.random_seed is not None:
            generator = torch.Generator(device=x.device).manual_seed(self.random_seed)
            direction = torch.randn(contribution.shape[-1], generator=generator, device=x.device)
            direction = F.normalize(direction, dim=0)
            contribution = contribution.norm(dim=-1, keepdim=True) * direction
        result = output.clone()
        result[:, -1] = (output[:, -1].float() - contribution).to(output.dtype)
        self.edit_norms.append(contribution.norm(dim=-1).cpu().tolist())
        self.actual_edit_norms.append((result[:, -1].float() - output[:, -1].float()).norm(dim=-1).cpu().tolist())
        return result
