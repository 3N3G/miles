"""Small geometric estimators used by the transport experiment."""

import torch


def centered(x):
    mean = x.mean(0)
    residual = x - mean
    scale = residual.square().mean().sqrt().clamp_min(1e-8)
    return mean, scale, residual / scale


def fit_rotation(x, y, regularizer=0.1):
    mx, sx, xn = centered(x)
    my, sy, yn = centered(y)
    covariance = xn.T @ yn / len(x)
    covariance += regularizer * torch.eye(x.shape[-1], device=x.device)
    u, singular, vh = torch.linalg.svd(covariance, full_matrices=False)
    q = u @ vh
    # Enforce an actual rotation, rather than an orthogonal reflection.
    if torch.linalg.slogdet(q)[0] < 0:
        q = q - 2 * torch.outer(u[:, -1], vh[-1])
    return {"mx": mx, "sx": sx, "my": my, "sy": sy, "matrix": q}


def fit_ridge(x, y, ridge):
    mx, sx, xn = centered(x)
    my, sy, yn = centered(y)
    gram = xn.T @ xn / len(x)
    gram += ridge * torch.eye(x.shape[-1], device=x.device)
    cross = xn.T @ yn / len(x)
    matrix = torch.linalg.solve(gram, cross)
    return {"mx": mx, "sx": sx, "my": my, "sy": sy, "matrix": matrix}


def transport(x, fit):
    return (x - fit["mx"]) / fit["sx"] @ fit["matrix"] * fit["sy"] + fit["my"]


def plane_extrapolation(previous, current, steps):
    magnitude = current.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    u = previous / previous.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    v = current / magnitude
    cosine = (u * v).sum(-1, keepdim=True).clamp(-1, 1)
    angle = torch.acos(cosine)
    tangent = (cosine * v - u) / torch.sin(angle).clamp_min(1e-8)
    result = magnitude * (torch.cos(steps * angle) * v + torch.sin(steps * angle) * tangent)
    return torch.where(angle < 1e-5, current, result)


def prediction_error(prediction, target):
    denominator = (target - target.mean(0)).square().mean().clamp_min(1e-8)
    return ((prediction - target).square().mean() / denominator).item()


def algebra_checks(device):
    generator = torch.Generator(device=device).manual_seed(1729)
    x = torch.randn(512, 16, device=device, generator=generator)
    q, _ = torch.linalg.qr(torch.randn(16, 16, device=device, generator=generator))
    if torch.linalg.slogdet(q)[0] < 0:
        q[:, -1] *= -1
    shift = torch.randn(16, device=device, generator=generator)
    y = 2 * (x @ q) + shift
    fit = fit_rotation(x, y, regularizer=0)
    error = (transport(x, fit) - y).abs().max().item()
    assert error < 2e-4, error
    a = torch.tensor([[1.0, 0.0]], device=device)
    angle = torch.tensor(0.2, device=device)
    b = torch.stack([angle.cos(), angle.sin()]).view(1, 2)
    expected = torch.stack([(3 * angle).cos(), (3 * angle).sin()]).view(1, 2)
    extrapolation_error = (plane_extrapolation(a, b, 2) - expected).abs().max().item()
    assert extrapolation_error < 1e-5, extrapolation_error
    assert (plane_extrapolation(a, b, 0) - b).abs().max() < 1e-6
    power_error = ((x @ torch.linalg.matrix_power(q, 3)) - (((x @ q) @ q) @ q)).abs().max().item()
    assert power_error < 1e-5, power_error
    permutation = torch.randperm(len(x), device=device, generator=generator)
    shuffled = fit_ridge(x, y[permutation], ridge=0.1)
    xv = torch.randn(128, 16, device=device, generator=generator)
    yv = 2 * (xv @ q) + shift
    shuffled_error = prediction_error(transport(xv, shuffled), yv)
    assert shuffled_error > 0.5, shuffled_error
    return {"known_rotation_max_error": error, "known_plane_extrapolation_error": extrapolation_error, "rotation_power_error": power_error, "shuffled_normalized_error": shuffled_error}
