import unittest

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

    def test_compacted_valid_peaks_are_at_front(self):
        tf.random.set_seed(99)
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

        for row in out["fused_valid_mask"].numpy():
            valid_count = int(row.sum())
            self.assertTrue(row[:valid_count].all())
            if valid_count < len(row):
                self.assertFalse(row[valid_count:].any())


if __name__ == "__main__":
    unittest.main()
