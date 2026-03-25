import unittest

import numpy as np
import tensorflow as tf

from input_pipeline import _augment_block_jepa_batch_tf


class BlockJEPAInputAugmentationTests(unittest.TestCase):
    def _make_batch(self) -> dict[str, tf.Tensor]:
        peak_mz = tf.constant(
            [
                [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.00, 0.00],
                [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.00],
            ],
            dtype=tf.float32,
        )
        peak_intensity = tf.constant(
            [
                [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.00, 0.00],
                [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.00],
            ],
            dtype=tf.float32,
        )
        peak_valid_mask = tf.constant(
            [
                [True, True, True, True, True, True, False, False],
                [True, True, True, True, True, True, True, False],
            ],
            dtype=tf.bool,
        )
        precursor_mz = tf.constant([0.5, 0.6], dtype=tf.float32)
        return {
            "peak_mz": peak_mz,
            "peak_intensity": peak_intensity,
            "peak_valid_mask": peak_valid_mask,
            "precursor_mz": precursor_mz,
        }

    def test_block_jepa_output_shape(self):
        tf.random.set_seed(1234)
        batch = self._make_batch()
        aug = _augment_block_jepa_batch_tf(
            num_target_blocks=3,
            context_fraction=0.25,
            target_fraction=0.20,
            block_min_len=1,
            mz_jitter_std=0.0,
            intensity_jitter_std=0.0,
        )
        out = aug(batch)

        self.assertEqual(out["peak_mz"].shape, (2, 8))
        self.assertEqual(out["peak_intensity"].shape, (2, 8))
        self.assertEqual(out["peak_valid_mask"].shape, (2, 8))
        self.assertEqual(out["context_mask"].shape, (2, 8))
        self.assertEqual(out["target_masks"].shape, (2, 3, 8))

    def test_masks_are_subsets_of_valid(self):
        tf.random.set_seed(7)
        batch = self._make_batch()
        aug = _augment_block_jepa_batch_tf(
            num_target_blocks=2,
            context_fraction=0.25,
            target_fraction=0.25,
            block_min_len=1,
            mz_jitter_std=0.0,
            intensity_jitter_std=0.0,
        )
        out = aug(batch)

        context_mask = out["context_mask"].numpy()
        target_masks = out["target_masks"].numpy()
        valid_mask = out["peak_valid_mask"].numpy()

        np.testing.assert_array_equal(
            context_mask <= valid_mask, np.ones_like(valid_mask, dtype=bool)
        )
        np.testing.assert_array_equal(
            target_masks <= valid_mask[:, None, :],
            np.ones_like(target_masks, dtype=bool),
        )

    def test_context_and_target_blocks_are_contiguous_and_disjoint_from_context(self):
        tf.random.set_seed(42)
        batch = self._make_batch()
        aug = _augment_block_jepa_batch_tf(
            num_target_blocks=3,
            context_fraction=0.25,
            target_fraction=0.15,
            block_min_len=1,
            mz_jitter_std=0.0,
            intensity_jitter_std=0.0,
        )
        out = aug(batch)

        context_mask = out["context_mask"].numpy()
        target_masks = out["target_masks"].numpy()
        valid_mask = out["peak_valid_mask"].numpy()

        for row_idx in range(valid_mask.shape[0]):
            context_positions = np.flatnonzero(context_mask[row_idx])
            if context_positions.size > 1:
                np.testing.assert_array_equal(
                    np.diff(context_positions),
                    np.ones(context_positions.size - 1, dtype=int),
                )
            for target_idx in range(target_masks.shape[1]):
                block = target_masks[row_idx, target_idx]
                self.assertFalse(np.any(block & context_mask[row_idx]))
                block_positions = np.flatnonzero(block)
                if block_positions.size > 1:
                    np.testing.assert_array_equal(
                        np.diff(block_positions),
                        np.ones(block_positions.size - 1, dtype=int),
                    )
            self.assertTrue(
                np.all(
                    (context_mask[row_idx] | target_masks[row_idx].any(axis=0))
                    <= valid_mask[row_idx]
                )
            )

    def test_target_blocks_can_overlap(self):
        tf.random.set_seed(7)
        batch = self._make_batch()
        aug = _augment_block_jepa_batch_tf(
            num_target_blocks=3,
            context_fraction=0.25,
            target_fraction=0.50,
            block_min_len=1,
            mz_jitter_std=0.0,
            intensity_jitter_std=0.0,
        )
        out = aug(batch)

        target_masks = out["target_masks"].numpy()
        overlap_count = (
            target_masks.sum(axis=1) > 1
        ).sum()
        self.assertGreater(overlap_count, 0)

    def test_zero_jitter_preserves_peak_values(self):
        tf.random.set_seed(202)
        batch = self._make_batch()
        aug = _augment_block_jepa_batch_tf(
            num_target_blocks=2,
            context_fraction=0.25,
            target_fraction=0.25,
            block_min_len=1,
            mz_jitter_std=0.0,
            intensity_jitter_std=0.0,
        )
        out = aug(batch)

        np.testing.assert_allclose(
            out["peak_mz"].numpy(), batch["peak_mz"].numpy(), rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            out["peak_intensity"].numpy(),
            batch["peak_intensity"].numpy(),
            rtol=0.0,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
