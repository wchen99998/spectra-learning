import unittest

import torch
import torch.nn.functional as F

from train import _FingerprintMetricAccumulator


class MsgEvalMetricAccumulatorTests(unittest.TestCase):
    def _compute_reference_metrics(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        probs = torch.sigmoid(logits)
        pred_bits = probs > 0.5
        target_bits = targets > 0.5

        intersection = (pred_bits & target_bits).sum(dim=1).float()
        union = (pred_bits | target_bits).sum(dim=1).float()
        tanimoto = torch.where(union > 0, intersection / union, torch.ones_like(union)).mean()
        cosine_similarity = F.cosine_similarity(probs, targets, dim=1).mean()
        bit_accuracy = (pred_bits == target_bits).float().mean()

        tp = (pred_bits & target_bits).float().sum(dim=0)
        fp = (pred_bits & ~target_bits).float().sum(dim=0)
        fn = (~pred_bits & target_bits).float().sum(dim=0)

        precision_per_bit = tp / (tp + fp).clamp(min=1e-8)
        recall_per_bit = tp / (tp + fn).clamp(min=1e-8)
        f1_per_bit = 2 * precision_per_bit * recall_per_bit / (precision_per_bit + recall_per_bit).clamp(min=1e-8)

        has_pred = (tp + fp) > 0
        has_target = (tp + fn) > 0
        precision_per_bit = torch.where(has_pred, precision_per_bit, torch.zeros_like(precision_per_bit))
        recall_per_bit = torch.where(has_target, recall_per_bit, torch.zeros_like(recall_per_bit))
        f1_per_bit = torch.where(has_pred | has_target, f1_per_bit, torch.zeros_like(f1_per_bit))

        return {
            "tanimoto": tanimoto,
            "cosine_similarity": cosine_similarity,
            "bit_accuracy": bit_accuracy,
            "precision": precision_per_bit.mean(),
            "recall": recall_per_bit.mean(),
            "f1": f1_per_bit.mean(),
            "pred_positive_rate": pred_bits.float().mean(),
            "target_positive_rate": target_bits.float().mean(),
        }

    def test_accumulator_matches_reference(self):
        g = torch.Generator().manual_seed(7)
        logits = torch.randn(96, 1024, generator=g)
        targets = (torch.rand(96, 1024, generator=g) > 0.91).float()

        accumulator = _FingerprintMetricAccumulator(fingerprint_bits=1024)
        for start in range(0, logits.shape[0], 17):
            end = min(start + 17, logits.shape[0])
            accumulator.update(logits[start:end], targets[start:end])

        metrics = accumulator.compute(device="cpu")
        expected = self._compute_reference_metrics(logits, targets)

        for key, expected_value in expected.items():
            self.assertTrue(
                torch.allclose(metrics[key], expected_value, atol=1e-6),
                msg=f"Mismatch for {key}: {metrics[key]} vs {expected_value}",
            )

    def test_tanimoto_is_one_for_all_zero_pairs(self):
        logits = torch.full((8, 32), -20.0)
        targets = torch.zeros((8, 32))

        accumulator = _FingerprintMetricAccumulator(fingerprint_bits=32)
        accumulator.update(logits, targets)
        metrics = accumulator.compute(device="cpu")

        self.assertEqual(float(metrics["tanimoto"]), 1.0)
        self.assertEqual(float(metrics["pred_positive_rate"]), 0.0)
        self.assertEqual(float(metrics["target_positive_rate"]), 0.0)


if __name__ == "__main__":
    unittest.main()
