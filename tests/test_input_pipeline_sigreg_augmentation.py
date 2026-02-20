import unittest

import tensorflow as tf

from input_pipeline import _augment_sigreg_batch_tf


class SigregInputAugmentationTests(unittest.TestCase):
    def _make_batch(self) -> dict[str, tf.Tensor]:
        peak_mz = tf.constant(
            [
                [0.10, 0.20, 0.30, 0.40, 0.00, 0.00],
                [0.15, 0.25, 0.35, 0.45, 0.55, 0.00],
            ],
            dtype=tf.float32,
        )
        peak_intensity = tf.constant(
            [
                [0.90, 0.80, 0.70, 0.60, 0.00, 0.00],
                [0.95, 0.85, 0.75, 0.65, 0.55, 0.00],
            ],
            dtype=tf.float32,
        )
        peak_valid_mask = tf.constant(
            [
                [True, True, True, True, False, False],
                [True, True, True, True, True, False],
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

    def test_both_views_are_augmented_and_compacted(self):
        tf.random.set_seed(1234)
        batch = self._make_batch()
        aug = _augment_sigreg_batch_tf(
            contiguous_mask_fraction_min=0.30,
            contiguous_mask_fraction_max=0.30,
            random_drop_fraction_min=0.10,
            random_drop_fraction_max=0.10,
            mz_jitter_std=0.0,
            intensity_jitter_std=0.0,
        )
        out = aug(batch)

        fused_valid = out["fused_valid_mask"]
        b = tf.shape(batch["peak_mz"])[0]
        original_valid_count = tf.reduce_sum(tf.cast(batch["peak_valid_mask"], tf.int32), axis=1)
        view1_valid_count = tf.reduce_sum(tf.cast(fused_valid[:b], tf.int32), axis=1)
        view2_valid_count = tf.reduce_sum(tf.cast(fused_valid[b:], tf.int32), axis=1)

        self.assertTrue(bool(tf.reduce_all(view1_valid_count < original_valid_count).numpy()))
        self.assertTrue(bool(tf.reduce_all(view2_valid_count < original_valid_count).numpy()))

        for row in fused_valid.numpy():
            valid_count = int(row.sum())
            self.assertTrue(row[:valid_count].all())
            self.assertFalse(row[valid_count:].any())

    def test_packed_masked_count_matches_dropped_valid_peaks(self):
        tf.random.set_seed(5678)
        batch = self._make_batch()
        aug = _augment_sigreg_batch_tf(
            contiguous_mask_fraction_min=0.40,
            contiguous_mask_fraction_max=0.40,
            random_drop_fraction_min=0.10,
            random_drop_fraction_max=0.10,
            mz_jitter_std=0.0,
            intensity_jitter_std=0.0,
        )
        out = aug(batch)

        b = tf.shape(batch["peak_mz"])[0]
        original_valid_count = tf.reduce_sum(tf.cast(batch["peak_valid_mask"], tf.int32), axis=1)

        fused_valid = out["fused_valid_mask"]
        fused_masked = out["fused_masked_positions"]
        view1_valid_count = tf.reduce_sum(tf.cast(fused_valid[:b], tf.int32), axis=1)
        view2_valid_count = tf.reduce_sum(tf.cast(fused_valid[b:], tf.int32), axis=1)
        view1_masked_count = tf.reduce_sum(tf.cast(fused_masked[:b], tf.int32), axis=1)
        view2_masked_count = tf.reduce_sum(tf.cast(fused_masked[b:], tf.int32), axis=1)

        self.assertTrue(bool(tf.reduce_all(view1_masked_count == (original_valid_count - view1_valid_count)).numpy()))
        self.assertTrue(bool(tf.reduce_all(view2_masked_count == (original_valid_count - view2_valid_count)).numpy()))


if __name__ == "__main__":
    unittest.main()
