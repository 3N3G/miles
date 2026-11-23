"""CPU controls for the operator pilot. No pretrained model or GPU is needed."""

import json
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from operator_surrogate import SparseOperator, VisitEdit, document_metrics, fit_ridge, fit_sparse, predict, predict_ridge


class OperatorControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_sparsity_and_native_unit_decomposition(self):
        torch.manual_seed(0)
        model = SparseOperator(8, 16, 3)
        x, y = torch.randn(20, 8) * 3 + 2, torch.randn(20, 8) * 5 - 4
        model.set_statistics(x, y)
        self.assertLessEqual(int((model.encode(x) > 0).sum(-1).max()), 3)
        contributions = sum(model.feature_contribution(x, j) for j in range(16))
        torch.testing.assert_close(contributions + model.y_mean + model.decoder.bias * model.y_scale, model(x))
        restored = SparseOperator.from_checkpoint(model.checkpoint(), "cpu")
        torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)

    def test_only_chosen_visit_and_last_token_change(self):
        model = SparseOperator(8, 16, 3)
        x, native = torch.randn(2, 5, 8), torch.randn(2, 5, 8)
        hook = VisitEdit(replacement=model, pass_index=1, feature=0)
        self.assertIsNone(hook(None, (x,), native))
        edited = hook(None, (x,), native)
        torch.testing.assert_close(edited[:, :-1], native[:, :-1], rtol=0, atol=0)
        torch.testing.assert_close(edited[:, -1], native[:, -1] - model.feature_contribution(x[:, -1], 0))
        self.assertIsNone(hook(None, (x,), native))
        self.assertEqual(hook.calls, 3)
        self.assertEqual(len(hook.edit_norms), 1)

    def test_random_control_matches_feature_edit_norm(self):
        model = SparseOperator(8, 16, 3)
        x, native = torch.randn(4, 5, 8), torch.randn(4, 5, 8)
        specific = VisitEdit(replacement=model, feature=3)
        random = VisitEdit(replacement=model, feature=3, random_seed=73)
        specific(None, (x,), native)
        random(None, (x,), native)
        torch.testing.assert_close(torch.tensor(specific.edit_norms), torch.tensor(random.edit_norms))

    def test_exact_identity_and_pass_specific_replacement(self):
        x, native = torch.randn(1, 5, 8), torch.randn(1, 5, 8)
        identity = VisitEdit()
        self.assertIs(identity(None, (x,), native), native)
        hook = VisitEdit(replacement=[torch.full((8,), float(i)) for i in range(3)])
        for i in range(3):
            torch.testing.assert_close(hook(None, (x,), native), torch.full_like(native, float(i)))

    def test_ridge_known_affine_and_document_boundaries(self):
        torch.manual_seed(1)
        x, matrix = torch.randn(256, 8), torch.randn(8, 8)
        y = x @ matrix + 3
        fit = fit_ridge(x, y, 1e-6)
        torch.testing.assert_close(predict_ridge(x, fit), y, rtol=1e-4, atol=5e-5)
        rows = document_metrics(y, y, positions=16, training_mean=y.mean(0))
        self.assertEqual(rows["normalized_mse"], 0)
        self.assertEqual(len(rows["per_paragraph_normalized_mse"]), 16)
        with self.assertRaises(ValueError):
            document_metrics(y, y, positions=15, training_mean=y.mean(0))

    def test_planted_operator_learns_and_shuffle_fails(self):
        torch.manual_seed(17)
        x = torch.randn(512, 8)
        # A planted sparse nonlinear operator with a simple, known basis.
        y = torch.relu(x)
        kwargs = dict(seed=0, width=32, topk=8, epochs=100, batch_size=64, learning_rate=0.01)
        paired, _ = fit_sparse(x[:384], y[:384], x[384:448], y[384:448], **kwargs)
        permutation = torch.randperm(384)
        shuffled, _ = fit_sparse(x[:384], y[:384][permutation], x[384:448], y[384:448], **kwargs)
        good = SparseOperator.from_checkpoint(paired, "cpu")
        bad = SparseOperator.from_checkpoint(shuffled, "cpu")
        target = y[448:]
        variance = (target - target.mean(0)).square().mean()
        good_error = ((predict(good, x[448:]) - target).square().mean() / variance).item()
        bad_error = ((predict(bad, x[448:]) - target).square().mean() / variance).item()
        self.assertLess(good_error, 0.12)
        self.assertGreater(bad_error, 0.7)
        print(json.dumps({"control": "planted_operator", "paired_test_nmse": good_error, "shuffled_test_nmse": bad_error}))

    def test_bootstrap_keeps_articles_as_sampling_units(self):
        from analyze_operator_pilot import bootstrap_interval, summarize
        differences = np.linspace(-0.1, 0.3, 32)
        rows = []
        for seed in range(3):
            for p in range(1, 4):
                for name, values in [(f"shared_seed{seed}", 0.5+differences), (f"pass{p}_seed{seed}", np.full(32, 0.5))]:
                    rows.append({"model": name, "evaluation_pass": p, "normalized_mse": float(values.mean()),
                                 "per_paragraph_normalized_mse": values.tolist()})
        summary = summarize(rows)
        np.testing.assert_allclose(summary["per_article_differences"], differences)
        np.testing.assert_allclose(summary["paired_95_percent_ci"], bootstrap_interval(differences))
        self.assertEqual(len(summary["per_article_differences"]), 32)

    def test_real_capture_hook_logic_on_cpu_model(self):
        from operator_pilot import capture

        class Router(torch.nn.Module):
            def forward(self, x):
                values, indices = x[:, :16].topk(8, dim=-1)
                return x[:, :16], values.softmax(-1), indices

        class Operator(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = Router()

            def forward(self, x):
                self.gate(x.reshape(-1, x.shape[-1]))
                return x.tanh()

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(256, 2304)
                layer = torch.nn.Module()
                layer.mlp = Operator()
                self.layers = torch.nn.ModuleList([layer])
                self.model = SimpleNamespace(language_model=SimpleNamespace(layers=self.layers))

            def forward(self, input_ids, use_cache=False):
                x = self.embedding(input_ids)
                for p in range(3):
                    x = self.layers[0].mlp(x + p/10)
                return SimpleNamespace(logits=x[:, :, :16])

        model = Model().eval().requires_grad_(False)
        tokenizer = lambda text, add_special_tokens: SimpleNamespace(input_ids=list(range(int(text), int(text)+64)))
        records = [{"text": str(i*64), "token_ids": list(range(i*64, (i+1)*64)),
                    "id": str(i), "split": "fit"} for i in range(2)]
        with tempfile.TemporaryDirectory() as tmp:
            x, y = capture(model, tokenizer, records, 0, Path(tmp))
            self.assertEqual(x.shape, (2, 48, 3, 2304))
            np.testing.assert_allclose(y, np.tanh(x), atol=2e-7)
            route = np.load(Path(tmp)/"routing_001.npz")["indices"]
            self.assertEqual(route.shape, (48, 3, 8))
            controls = json.loads((Path(tmp)/"native_controls.json").read_text())
            self.assertEqual(controls["operator_replay_max_errors"], [0, 0, 0])

    def test_fitting_cannot_see_final_values(self):
        from operator_pilot import fit_all, evaluate_final
        rng = np.random.default_rng(17)
        x = rng.normal(size=(8, 48, 3, 8)).astype(np.float32)
        y = np.maximum(x, 0)
        records = [{"split": "fit" if i < 4 else "validation" if i < 6 else "final"} for i in range(8)]
        config = {"device": "cpu", "seeds": [0], "width": 16, "topk": 4,
                  "epochs": 2, "batch_size": 64, "learning_rate": 0.001,
                  "ridge_regularizers": [0.01, 0.1]}
        poisoned_x, poisoned_y = x.copy(), y.copy()
        poisoned_x[6:] = 9999
        poisoned_y[6:] = -9999
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            a, b = Path(tmp)/"a", Path(tmp)/"b"
            a.mkdir(); b.mkdir()
            fit_all(x, y, records, config, a)
            fit_all(poisoned_x, poisoned_y, records, config, b)
            for path in a.glob("*_seed0.pt"):
                left = torch.load(path, weights_only=True)["state"]
                right = torch.load(b/path.name, weights_only=True)["state"]
                for key in left:
                    torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
            self.assertEqual(json.loads((a/"frozen_fit.json").read_text()), json.loads((b/"frozen_fit.json").read_text()))
            evaluated = evaluate_final(x, y, records, config, a)
            self.assertEqual(len(evaluated), 24)


if __name__ == "__main__":
    unittest.main()
