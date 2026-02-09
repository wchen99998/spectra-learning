import unittest

import torch

from linear_probe import run_linear_probe


class LinearProbeTests(unittest.TestCase):
    def test_smoke_1024_bit_metrics_are_finite(self):
        g = torch.Generator().manual_seed(0)
        x_train = torch.randn(128, 16, generator=g)
        y_train = (torch.rand(128, 1024, generator=g) > 0.92).to(torch.float32)
        y_train[:, 0] = 1.0
        x_test = torch.randn(64, 16, generator=g)
        y_test = (torch.rand(64, 1024, generator=g) > 0.92).to(torch.float32)
        y_test[:, 0] = 1.0

        metrics = run_linear_probe([(x_train, y_train)], [(x_test, y_test)], fit_bias=True)

        self.assertTrue(torch.isfinite(metrics["accuracy"]).item())
        self.assertTrue(torch.isfinite(metrics["tanimoto"]).item())
        self.assertTrue(torch.isfinite(metrics["pred_positive_rate"]).item())
        self.assertTrue(torch.isfinite(metrics["target_positive_rate"]).item())

    def test_zero_logit_ties_predict_negative(self):
        x_train = torch.zeros(32, 8)
        y_train = torch.zeros(32, 1024)
        x_test = torch.zeros(16, 8)
        y_test = torch.zeros(16, 1024)

        metrics = run_linear_probe([(x_train, y_train)], [(x_test, y_test)], fit_bias=False)

        self.assertEqual(float(metrics["pred_positive_rate"]), 0.0)
        self.assertEqual(float(metrics["target_positive_rate"]), 0.0)
        self.assertEqual(float(metrics["accuracy"]), 1.0)
        self.assertEqual(float(metrics["tanimoto"]), 1.0)

    def test_bias_term_improves_thresholded_accuracy(self):
        g = torch.Generator().manual_seed(123)
        x_train = torch.rand(4096, 1, generator=g)
        x_test = torch.rand(4096, 1, generator=g)
        y_train = (x_train[:, :1] > 0.6).to(torch.float32)
        y_test = (x_test[:, :1] > 0.6).to(torch.float32)

        no_bias = run_linear_probe([(x_train, y_train)], [(x_test, y_test)], fit_bias=False)
        with_bias = run_linear_probe([(x_train, y_train)], [(x_test, y_test)], fit_bias=True)

        self.assertGreater(float(with_bias["accuracy"]), float(no_bias["accuracy"]))


if __name__ == "__main__":
    unittest.main()
