import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from runpod_ai_lab.backprop import forward_backward
from runpod_ai_lab.train_mlp import build_model, make_data


class CoreTests(unittest.TestCase):
    def test_manual_gradients_with_finite_difference(self):
        torch.manual_seed(1)
        x, y = torch.randn(7, 2, dtype=torch.float64), torch.randn(7, 1, dtype=torch.float64)
        params = [torch.randn(2, 3, dtype=x.dtype), torch.randn(3, dtype=x.dtype),
                  torch.randn(3, 1, dtype=x.dtype), torch.randn(1, dtype=x.dtype)]
        _, grads = forward_backward(x, y, params)
        eps = 1e-6
        for p, grad in zip(params, grads):
            for i in range(p.numel()):
                original = p.flatten()[i].item()
                p.flatten()[i] = original + eps
                plus, _ = forward_backward(x, y, params)
                p.flatten()[i] = original - eps
                minus, _ = forward_backward(x, y, params)
                p.flatten()[i] = original
                self.assertAlmostEqual((plus - minus).item() / (2 * eps), grad.flatten()[i].item(), places=6)

    def test_checkpoint_prediction_roundtrip(self):
        model = build_model(8, 2).eval()
        x, _ = make_data(8, 1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            torch.save(model.state_dict(), path)
            restored = build_model(8, 2).eval()
            restored.load_state_dict(torch.load(path, weights_only=True))
            torch.testing.assert_close(model(x), restored(x))

    def test_independent_cli_and_learning(self):
        cases = [("environment", []), ("tensors", ["--size", "32", "--repeats", "2"]),
                 ("backprop", []), ("train_mlp", ["--epochs", "20"])]
        with tempfile.TemporaryDirectory() as folder:
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            for name, options in cases:
                result = subprocess.run([sys.executable, "-m", "runpod_ai_lab." + name, "--device", "cpu",
                                         "--output", folder, *options], capture_output=True,
                                        text=True, timeout=120, env=env)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            reports = {v["experiment"]: v for p in Path(folder).glob("*/metrics.json")
                       for v in [json.loads(p.read_text())]}
            backprop = reports["backprop"]["metrics"]
            self.assertLess(backprop["final_loss"], backprop["initial_loss"] * 0.2)
            self.assertGreater(reports["train_mlp"]["metrics"]["history"][-1]["val_accuracy"], 0.9)
            self.assertTrue(reports["tensors"]["metrics"]["finite"])


if __name__ == "__main__":
    unittest.main()
