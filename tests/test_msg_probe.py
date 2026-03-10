import unittest

import numpy as np
import tensorflow as tf
import torch

from input_pipeline import _prepend_precursor_token_tf
from utils.msg_probe import (
    FG_SMARTS,
    MsgLinearProbe,
    MsgProbeSplitTargets,
    _build_task_spec,
    _collect_split_targets,
    _probe_step,
    iter_massspec_probe,
    probe_steps_per_epoch,
)


class _DummyDataset:
    def __init__(self, batches):
        self._batches = batches

    def as_numpy_iterator(self):
        return iter(self._batches)


class _DummyDataModule:
    def __init__(self, batches, info, batch_size):
        self._dataset = _DummyDataset(batches)
        self.info = info
        self.batch_size = batch_size
        self.calls = []

    def build_dataset(
        self,
        split: str,
        seed: int,
        *,
        peak_ordering: str | None = None,
        shuffle: bool = False,
        drop_remainder: bool = True,
    ):
        self.calls.append(
            {
                "split": split,
                "seed": seed,
                "peak_ordering": peak_ordering,
                "shuffle": shuffle,
                "drop_remainder": drop_remainder,
            }
        )
        return self._dataset


class MsgLinearProbeTests(unittest.TestCase):
    def test_output_shapes_match_task_heads(self):
        probe = MsgLinearProbe(
            input_dim=32,
            task_names=("mol_weight", "hydroxyl", "amine"),
        )
        pooled = torch.randn(7, 32)
        logits = probe(pooled)

        self.assertEqual(logits["mol_weight"].shape, (7, 1))
        self.assertEqual(logits["hydroxyl"].shape, (7, 1))
        self.assertEqual(logits["amine"].shape, (7, 1))

    def test_finite_outputs(self):
        probe = MsgLinearProbe(
            input_dim=16,
            task_names=("mol_weight", "hydroxyl"),
        )
        pooled = torch.randn(3, 16)
        logits = probe(pooled)

        self.assertTrue(torch.isfinite(logits["mol_weight"]).all().item())
        self.assertTrue(torch.isfinite(logits["hydroxyl"]).all().item())


class MsgProbeStepTests(unittest.TestCase):
    def test_probe_step_filters_invalid_targets_and_losses_are_finite(self):
        task_spec = _build_task_spec(
            train_targets=MsgProbeSplitTargets(
                regression={
                    "mol_weight": np.asarray([10.0, 14.0, 16.0], dtype=np.float32),
                    "logp": np.asarray([1.0, 2.0, 2.5], dtype=np.float32),
                    "num_heavy_atoms": np.asarray([2.0, 4.0, 5.0], dtype=np.float32),
                    "num_rings": np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
                },
                classification={
                    name: np.asarray([0, 1, 0], dtype=np.int32) for name in FG_SMARTS
                },
            ),
            test_targets=MsgProbeSplitTargets(
                regression={
                    "mol_weight": np.asarray([10.0, 14.0, 16.0], dtype=np.float32),
                    "logp": np.asarray([1.0, 2.0, 2.5], dtype=np.float32),
                    "num_heavy_atoms": np.asarray([2.0, 4.0, 5.0], dtype=np.float32),
                    "num_rings": np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
                },
                classification={
                    name: np.asarray([0, 1, 0], dtype=np.int32) for name in FG_SMARTS
                },
            ),
        )
        input_dim = 8
        probe = MsgLinearProbe(
            input_dim=input_dim,
            task_names=task_spec.regression_tasks + task_spec.classification_tasks,
        )
        batch = {
            "probe_valid_mol": torch.tensor([True, False, True], dtype=torch.bool),
            "probe_mol_weight": torch.tensor([10.0, 12.0, 16.0], dtype=torch.float32),
            "probe_logp": torch.tensor([1.0, 1.5, 2.5], dtype=torch.float32),
            "probe_num_heavy_atoms": torch.tensor([2.0, 3.0, 5.0], dtype=torch.float32),
            "probe_num_rings": torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32),
        }
        for name in FG_SMARTS:
            batch[f"probe_fg_{name}"] = torch.tensor([0, 1, 1], dtype=torch.int32)

        def dummy_extractor(b):
            n = b["probe_valid_mol"].shape[0]
            return torch.randn(n, input_dim)

        result = _probe_step(
            probe,
            batch,
            task_spec=task_spec,
            device=torch.device("cpu"),
            feature_extractor=dummy_extractor,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["batch_size"], 2)
        self.assertTrue(torch.isfinite(result["loss_total"]).item())


class MsgProbeTaskSpecTests(unittest.TestCase):
    def test_fg_tasks_follow_prevalence_filter(self):
        train_fg = {name: np.zeros(4, dtype=np.int32) for name in FG_SMARTS}
        test_fg = {name: np.zeros(4, dtype=np.int32) for name in FG_SMARTS}
        train_fg["hydroxyl"] = np.asarray([0, 1, 0, 1], dtype=np.int32)
        test_fg["hydroxyl"] = np.asarray([1, 0, 1, 0], dtype=np.int32)
        train_fg["amine"] = np.asarray([0, 0, 0, 0], dtype=np.int32)
        test_fg["amine"] = np.asarray([0, 1, 0, 1], dtype=np.int32)

        task_spec = _build_task_spec(
            train_targets=MsgProbeSplitTargets(
                regression={
                    "mol_weight": np.linspace(10.0, 13.0, 4, dtype=np.float32),
                    "logp": np.linspace(1.0, 2.5, 4, dtype=np.float32),
                    "num_heavy_atoms": np.linspace(2.0, 5.0, 4, dtype=np.float32),
                    "num_rings": np.linspace(0.0, 1.5, 4, dtype=np.float32),
                },
                classification=train_fg,
            ),
            test_targets=MsgProbeSplitTargets(
                regression={
                    "mol_weight": np.linspace(14.0, 17.0, 4, dtype=np.float32),
                    "logp": np.linspace(3.0, 4.5, 4, dtype=np.float32),
                    "num_heavy_atoms": np.linspace(6.0, 9.0, 4, dtype=np.float32),
                    "num_rings": np.linspace(2.0, 3.5, 4, dtype=np.float32),
                },
                classification=test_fg,
            ),
        )

        self.assertEqual(
            task_spec.regression_tasks,
            ("mol_weight", "logp", "num_heavy_atoms", "num_rings"),
        )
        self.assertEqual(task_spec.classification_tasks, ("hydroxyl",))


class MsgProbeCollectionTests(unittest.TestCase):
    def test_collect_split_targets_uses_batch_targets_directly(self):
        dm = _DummyDataModule(
            batches=[
                {
                    "peak_mz": np.zeros((2, 60), dtype=np.float32),
                    "probe_valid_mol": np.asarray([1, 0], dtype=np.int32),
                    "probe_mol_weight": np.asarray([10.0, 20.0], dtype=np.float32),
                    "probe_logp": np.asarray([1.0, 2.0], dtype=np.float32),
                    "probe_num_heavy_atoms": np.asarray([2.0, 3.0], dtype=np.float32),
                    "probe_num_rings": np.asarray([0.0, 1.0], dtype=np.float32),
                    **{
                        f"probe_fg_{name}": np.asarray([0, 1], dtype=np.int32)
                        for name in FG_SMARTS
                    },
                },
                {
                    "peak_mz": np.zeros((2, 60), dtype=np.float32),
                    "probe_valid_mol": np.asarray([1, 1], dtype=np.int32),
                    "probe_mol_weight": np.asarray([30.0, 40.0], dtype=np.float32),
                    "probe_logp": np.asarray([3.0, 4.0], dtype=np.float32),
                    "probe_num_heavy_atoms": np.asarray([4.0, 5.0], dtype=np.float32),
                    "probe_num_rings": np.asarray([2.0, 3.0], dtype=np.float32),
                    **{
                        f"probe_fg_{name}": np.asarray([1, 0], dtype=np.int32)
                        for name in FG_SMARTS
                    },
                },
            ],
            info={
                "massspec_train_size": 0,
                "massspec_val_size": 0,
                "massspec_test_size": 4,
            },
            batch_size=2,
        )

        targets = _collect_split_targets(
            probe_data=dm,
            split="massspec_test",
            peak_ordering="intensity",
            seed=0,
        )

        self.assertTrue(
            np.array_equal(
                targets.regression["mol_weight"],
                np.asarray([10.0, 30.0, 40.0], dtype=np.float32),
            )
        )
        self.assertTrue(
            np.array_equal(
                targets.classification["hydroxyl"],
                np.asarray([0, 1, 0], dtype=np.int32),
            )
        )


class ProbeIterationTests(unittest.TestCase):
    def test_train_probe_uses_shuffle_and_includes_remainder(self):
        batches = [
            {"peak_mz": np.zeros((4, 60), dtype=np.float32)},
            {"peak_mz": np.ones((4, 60), dtype=np.float32)},
        ]
        dm = _DummyDataModule(
            batches=batches,
            info={
                "massspec_train_size": 6,
                "massspec_val_size": 0,
                "massspec_test_size": 0,
            },
            batch_size=4,
        )
        result = list(
            iter_massspec_probe(
                dm,
                "massspec_train",
                seed=123,
                peak_ordering="mz",
                drop_remainder=False,
            )
        )
        self.assertEqual(dm.calls[0]["shuffle"], True)
        self.assertEqual(dm.calls[0]["drop_remainder"], False)
        self.assertEqual(result[0]["peak_mz"].shape[0], 4)
        self.assertEqual(result[1]["peak_mz"].shape[0], 2)

    def test_eval_probe_does_not_shuffle(self):
        batches = [{"peak_mz": np.zeros((2, 60), dtype=np.float32)}]
        dm = _DummyDataModule(
            batches=batches,
            info={
                "massspec_train_size": 0,
                "massspec_val_size": 0,
                "massspec_test_size": 2,
            },
            batch_size=4,
        )
        _ = list(
            iter_massspec_probe(
                dm,
                "massspec_test",
                seed=321,
                peak_ordering="intensity",
                drop_remainder=False,
            )
        )
        self.assertEqual(dm.calls[0]["shuffle"], False)

    def test_probe_iteration_respects_max_samples(self):
        batches = [
            {"peak_mz": np.zeros((4, 60), dtype=np.float32)},
            {"peak_mz": np.ones((4, 60), dtype=np.float32)},
        ]
        dm = _DummyDataModule(
            batches=batches,
            info={
                "massspec_train_size": 8,
                "massspec_val_size": 0,
                "massspec_test_size": 0,
            },
            batch_size=4,
        )
        result = list(
            iter_massspec_probe(
                dm,
                "massspec_train",
                seed=123,
                peak_ordering="mz",
                drop_remainder=False,
                max_samples=5,
            )
        )
        self.assertEqual(result[0]["peak_mz"].shape[0], 4)
        self.assertEqual(result[1]["peak_mz"].shape[0], 1)


class ProbeStepCountTests(unittest.TestCase):
    def test_probe_steps_per_epoch_matches_drop_remainder_policy(self):
        dm = _DummyDataModule(
            batches=[],
            info={
                "massspec_train_size": 10,
                "massspec_val_size": 0,
                "massspec_test_size": 0,
            },
            batch_size=4,
        )
        self.assertEqual(
            probe_steps_per_epoch(dm, split="massspec_train", drop_remainder=False),
            3,
        )
        self.assertEqual(
            probe_steps_per_epoch(dm, split="massspec_train", drop_remainder=True),
            2,
        )
        self.assertEqual(
            probe_steps_per_epoch(
                dm,
                split="massspec_train",
                drop_remainder=False,
                max_samples=5,
            ),
            2,
        )


class ProbePrecursorTokenTfTests(unittest.TestCase):
    def test_prepend_shapes_and_values(self):
        B, N = 3, 5
        batch = {
            "peak_mz": tf.constant(np.random.rand(B, N).astype(np.float32)),
            "peak_intensity": tf.constant(np.random.rand(B, N).astype(np.float32)),
            "peak_valid_mask": tf.constant(np.ones((B, N), dtype=bool)),
            "precursor_mz": tf.constant([0.1, 0.2, 0.3], dtype=tf.float32),
            "fingerprint": tf.constant(np.zeros((B, 4), dtype=np.int32)),
            "probe_valid_mol": tf.constant([True, False, True]),
        }

        out = _prepend_precursor_token_tf(batch)

        # Shapes are [B, N+1]
        self.assertEqual(out["peak_mz"].shape, (B, N + 1))
        self.assertEqual(out["peak_intensity"].shape, (B, N + 1))
        self.assertEqual(out["peak_valid_mask"].shape, (B, N + 1))

        # precursor_mz key is removed
        self.assertNotIn("precursor_mz", out)

        # Position 0 has sentinel intensity=-1 and valid=True
        np.testing.assert_array_equal(
            out["peak_intensity"][:, 0].numpy(), [-1.0, -1.0, -1.0]
        )
        np.testing.assert_array_equal(
            out["peak_valid_mask"][:, 0].numpy(), [True, True, True]
        )

        # Position 0 mz equals original precursor_mz
        np.testing.assert_allclose(out["peak_mz"][:, 0].numpy(), [0.1, 0.2, 0.3])

        # Original peaks shifted to positions 1..N
        np.testing.assert_array_equal(
            out["peak_mz"][:, 1:].numpy(),
            batch["peak_mz"].numpy(),
        )

        # Passthrough keys preserved
        self.assertIn("fingerprint", out)
        self.assertIn("probe_valid_mol", out)
        np.testing.assert_array_equal(
            out["fingerprint"].numpy(), batch["fingerprint"].numpy()
        )


if __name__ == "__main__":
    unittest.main()
