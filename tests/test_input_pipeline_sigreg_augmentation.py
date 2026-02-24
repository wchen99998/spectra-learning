import unittest

import numpy as np
import tensorflow as tf

from input_pipeline import _augment_multicrop_batch_tf


class MulticropInputAugmentationTests(unittest.TestCase):
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

    def test_multicrop_output_shape(self):
        tf.random.set_seed(1234)
        batch = self._make_batch()
        num_global = 2
        num_local = 3
        num_views = num_global + num_local
        aug = _augment_multicrop_batch_tf(
            num_global_views=num_global,
            num_local_views=num_local,
            global_keep_fraction=0.80,
            local_keep_fraction=0.25,
            mz_jitter_std=0.0,
            intensity_jitter_std=0.0,
        )
        out = aug(batch)

        B = 2
        N = 6
        self.assertEqual(out["fused_mz"].shape, (num_views * B, N))
        self.assertEqual(out["fused_intensity"].shape, (num_views * B, N))
        self.assertEqual(out["fused_valid_mask"].shape, (num_views * B, N))
        self.assertEqual(out["fused_precursor_mz"].shape, (num_views * B,))

    def test_multicrop_no_masked_positions_key(self):
        batch = self._make_batch()
        aug = _augment_multicrop_batch_tf(
            num_global_views=2,
            num_local_views=2,
            global_keep_fraction=0.80,
            local_keep_fraction=0.25,
            mz_jitter_std=0.0,
            intensity_jitter_std=0.0,
        )
        out = aug(batch)
        self.assertNotIn("fused_masked_positions", out)
        self.assertNotIn("view1_masked_fraction", out)

    def test_multicrop_with_masked_tokens_outputs_mask_keys(self):
        tf.random.set_seed(7)
        batch = self._make_batch()
        aug = _augment_multicrop_batch_tf(
            num_global_views=2,
            num_local_views=2,
            global_keep_fraction=0.8,
            local_keep_fraction=0.25,
            mz_jitter_std=0.0,
            intensity_jitter_std=0.0,
            keep_masked_tokens=True,
        )
        out = aug(batch)

        self.assertIn("fused_masked_positions", out)
        self.assertIn("view1_masked_fraction", out)
        self.assertEqual(out["fused_masked_positions"].shape, out["fused_valid_mask"].shape)
        self.assertGreaterEqual(float(tf.reduce_sum(tf.cast(out["fused_masked_positions"], tf.float32))), 1.0)

        fused_masked = out["fused_masked_positions"].numpy()
        fused_intensity = out["fused_intensity"].numpy()
        peak_intensity = batch["peak_intensity"].numpy()
        batch_size = peak_intensity.shape[0]
        for row_idx, row_masked in enumerate(fused_masked):
            base_idx = row_idx % batch_size
            if row_masked.any():
                np.testing.assert_allclose(
                    fused_intensity[row_idx][row_masked],
                    peak_intensity[base_idx][row_masked],
                    rtol=0.0,
                    atol=0.0,
                )

    def test_global_views_keep_more_than_local(self):
        tf.random.set_seed(42)
        batch = self._make_batch()
        num_global = 2
        num_local = 3
        aug = _augment_multicrop_batch_tf(
            num_global_views=num_global,
            num_local_views=num_local,
            global_keep_fraction=0.80,
            local_keep_fraction=0.25,
            mz_jitter_std=0.0,
            intensity_jitter_std=0.0,
        )
        out = aug(batch)

        B = 2
        fused_valid = out["fused_valid_mask"]
        global_valid = fused_valid[:num_global * B]
        local_valid = fused_valid[num_global * B:]

        global_avg = tf.reduce_mean(tf.reduce_sum(tf.cast(global_valid, tf.float32), axis=1))
        local_avg = tf.reduce_mean(tf.reduce_sum(tf.cast(local_valid, tf.float32), axis=1))

        self.assertGreater(float(global_avg.numpy()), float(local_avg.numpy()))

    def test_dropped_peaks_preserve_original_slots(self):
        tf.random.set_seed(202)
        batch = self._make_batch()
        aug = _augment_multicrop_batch_tf(
            num_global_views=4,
            num_local_views=4,
            global_keep_fraction=0.50,
            local_keep_fraction=0.50,
            mz_jitter_std=0.0,
            intensity_jitter_std=0.0,
        )
        out = aug(batch)

        fused_mz = out["fused_mz"].numpy()
        fused_valid = out["fused_valid_mask"].numpy()
        peak_mz = batch["peak_mz"].numpy()
        peak_valid = batch["peak_valid_mask"].numpy()
        batch_size = peak_mz.shape[0]

        for row_idx, row_valid in enumerate(fused_valid):
            base_idx = row_idx % batch_size
            self.assertFalse(row_valid[~peak_valid[base_idx]].any())
            expected_mz = np.where(row_valid, peak_mz[base_idx], 0.0)
            np.testing.assert_allclose(fused_mz[row_idx], expected_mz, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
