import math
import unittest

import tensorflow as tf
import torch
from torch.nn import functional as F

from input_pipeline import (
    _NUM_PEAKS_OUTPUT,
    _NUM_SPECIAL_TOKENS,
    _PEAK_MZ_MAX,
    _SPECIAL_TOKENS,
    _build_single_spectrum_input,
    _compact_sort_peaks,
    _convert_to_neutral_loss,
    detokenize_spectrum,
)
from models.bert_torch import BERTTorch


class InputPipelineAutoregressiveTests(unittest.TestCase):
    def test_peak_ordering_supports_mz_and_intensity(self):
        example = {
            "mz": tf.constant([50.0, 10.0, 30.0, 0.0], dtype=tf.float32),
            "intensity": tf.constant([0.2, 0.5, 0.9, 0.0], dtype=tf.float32),
        }

        by_mz = _compact_sort_peaks("mz")(dict(example))
        by_intensity = _compact_sort_peaks("intensity")(dict(example))

        self.assertEqual(by_mz["mz"].shape[0], _NUM_PEAKS_OUTPUT)
        self.assertEqual(by_intensity["mz"].shape[0], _NUM_PEAKS_OUTPUT)
        self.assertTrue(tf.reduce_all(tf.equal(by_mz["mz"][:3], [10.0, 30.0, 50.0])).numpy())
        self.assertTrue(
            tf.reduce_all(tf.equal(by_intensity["mz"][:3], [30.0, 10.0, 50.0])).numpy()
        )

    def test_neutral_loss_is_sorted_after_conversion(self):
        original = {
            "mz": tf.constant([10.0, 70.0, 20.0, 0.0], dtype=tf.float32),
            "intensity": tf.constant([0.2, 0.9, 0.5, 0.0], dtype=tf.float32),
            "precursor_mz": tf.constant(100.0, dtype=tf.float32),
        }

        converted_then_sorted = _compact_sort_peaks("mz")(
            _convert_to_neutral_loss()(dict(original))
        )
        sorted_then_converted = _convert_to_neutral_loss()(
            _compact_sort_peaks("mz")(dict(original))
        )

        self.assertTrue(
            tf.reduce_all(tf.equal(converted_then_sorted["mz"][:3], [30.0, 80.0, 90.0])).numpy()
        )
        self.assertTrue(
            tf.reduce_all(tf.equal(sorted_then_converted["mz"][:3], [90.0, 80.0, 30.0])).numpy()
        )

    def test_single_spectrum_input_appends_sep_and_pads(self):
        build = _build_single_spectrum_input(max_len=7)
        example = {
            "mz": tf.constant([10, 11, 12], dtype=tf.int32),
            "intensity": tf.constant([20, 21, 22], dtype=tf.int32),
        }
        output = build(example)
        token_ids = output["token_ids"].numpy().tolist()
        segment_ids = output["segment_ids"].numpy().tolist()

        self.assertEqual(token_ids, [1, 10, 20, 11, 21, 2, 0])
        self.assertEqual(segment_ids, [0, 0, 0, 0, 0, 0, 0])

    def test_detokenize_stops_at_sep(self):
        intensity_offset = _NUM_SPECIAL_TOKENS + int(_PEAK_MZ_MAX) + 1
        tokens = [
            _SPECIAL_TOKENS["[CLS]"],
            _NUM_SPECIAL_TOKENS + 40,
            intensity_offset + 5,
            _SPECIAL_TOKENS["[SEP]"],
            _NUM_SPECIAL_TOKENS + 80,
            intensity_offset + 10,
            _SPECIAL_TOKENS["[PAD]"],
        ]
        decoded = detokenize_spectrum(tokens)

        self.assertEqual(decoded["mz"].shape[0], 1)
        self.assertEqual(float(decoded["mz"][0]), 40.0)


class BERTAutoregressiveTests(unittest.TestCase):
    def _build_model(self) -> BERTTorch:
        return BERTTorch(
            vocab_size=64,
            max_length=8,
            precursor_bins=16,
            precursor_offset=4,
            model_dim=16,
            num_layers=1,
            num_heads=4,
            num_kv_heads=4,
            num_segments=1,
            pad_token_id=0,
            cls_token_id=1,
            sep_token_id=2,
            cache_rope_frequencies=False,
        )

    def test_next_token_metrics_ignore_pad_targets(self):
        model = self._build_model()
        token_ids = torch.tensor([[1, 3, 4, 2, 0]], dtype=torch.long)
        logits = torch.zeros((1, 5, 10), dtype=torch.float32)
        logits[0, 3, 0] = -100.0
        logits[0, 3, 1] = 100.0

        token_loss, _ = model._next_token_metrics(logits, token_ids)

        expected = torch.tensor(math.log(10.0), dtype=torch.float32)
        included = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, 10),
            token_ids[:, 1:].reshape(-1),
        )
        self.assertTrue(torch.allclose(token_loss, expected, atol=1e-6))
        self.assertGreater(float(included - token_loss), 1.0)

    def test_forward_outputs_are_finite(self):
        model = self._build_model()
        batch = {
            "token_ids": torch.tensor(
                [
                    [1, 10, 11, 2, 0, 0, 0, 0],
                    [1, 20, 21, 22, 2, 0, 0, 0],
                ],
                dtype=torch.long,
            ),
            "segment_ids": torch.zeros((2, 8), dtype=torch.long),
            "precursor_mz": torch.tensor([5, 6], dtype=torch.long),
        }

        metrics = model(batch, train=True)

        self.assertTrue(torch.isfinite(metrics["loss"]).item())
        self.assertTrue(torch.isfinite(metrics["token_loss"]).item())
        self.assertTrue(torch.isfinite(metrics["token_accuracy"]).item())

    def test_encode_pools_sep_hidden_state(self):
        model = self._build_model()
        token_ids = torch.tensor(
            [
                [1, 10, 11, 2, 0, 0, 0, 0],
                [1, 20, 21, 22, 2, 0, 0, 0],
            ],
            dtype=torch.long,
        )
        segment_ids = torch.zeros((2, 8), dtype=torch.long)
        batch = {
            "token_ids": token_ids,
            "segment_ids": segment_ids,
            "precursor_mz": torch.tensor([5, 6], dtype=torch.long),
        }

        pooled = model.encode(batch, train=False)

        encoded = model.encoder(
            model._embed_inputs(token_ids, segment_ids),
            train=False,
            attention_mask=None,
        )
        sep_idx = (token_ids == model.sep_token_id).to(torch.int64).argmax(dim=1)
        expected = encoded[torch.arange(token_ids.shape[0]), sep_idx, :]

        self.assertTrue(torch.allclose(pooled, expected))


if __name__ == "__main__":
    unittest.main()
