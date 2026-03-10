import unittest

import numpy as np
import torch

from utils.msg_probe import (
    FG_SMARTS,
    MsgAttentiveProbe,
    MsgProbeSplitTargets,
    _build_task_spec,
    _collect_split_targets,
    _probe_step,
    iter_massspec_probe,
    probe_steps_per_epoch,
    should_run_msg_probe,
)


class _DummyEncoder(torch.nn.Module):
    def forward(
        self,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        *,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack((peak_mz, peak_intensity), dim=-1)


class _DummyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _DummyEncoder()
        self.use_precursor_token = False


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
        self.calls.append({
            "split": split,
            "seed": seed,
            "peak_ordering": peak_ordering,
            "shuffle": shuffle,
            "drop_remainder": drop_remainder,
        })
        return self._dataset


class MsgAttentiveProbeTests(unittest.TestCase):
    def test_output_shapes_match_task_heads(self):
        probe = MsgAttentiveProbe(
            input_dim=32,
            hidden_dim=64,
            num_attention_heads=4,
            task_names=("mol_weight", "hydroxyl", "amine"),
        )
        features = torch.randn(7, 60, 32)
        feature_mask = torch.ones(7, 60, dtype=torch.bool)
        logits = probe(features, feature_mask)

        self.assertEqual(logits["mol_weight"].shape, (7, 1))
        self.assertEqual(logits["hydroxyl"].shape, (7, 1))
        self.assertEqual(logits["amine"].shape, (7, 1))

    def test_all_padded_rows_produce_finite_outputs(self):
        probe = MsgAttentiveProbe(
            input_dim=16,
            hidden_dim=32,
            num_attention_heads=4,
            task_names=("mol_weight", "hydroxyl"),
        )
        features = torch.randn(3, 5, 16)
        feature_mask = torch.tensor(
            [
                [False, False, False, False, False],
                [True, True, False, False, False],
                [False, False, False, False, False],
            ],
            dtype=torch.bool,
        )
        logits = probe(features, feature_mask)

        self.assertTrue(torch.isfinite(logits["mol_weight"]).all().item())
        self.assertTrue(torch.isfinite(logits["hydroxyl"]).all().item())

    def test_custom_attention_block_count_is_used(self):
        probe = MsgAttentiveProbe(
            input_dim=16,
            hidden_dim=32,
            num_attention_heads=4,
            num_attention_blocks=5,
            mlp_ratio=2,
            task_names=("mol_weight",),
        )
        self.assertEqual(len(probe.blocks), 5)
        self.assertEqual(probe.blocks[0].mlp[0].out_features, 32)


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
                    name: np.asarray([0, 1, 0], dtype=np.int32)
                    for name in FG_SMARTS
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
                    name: np.asarray([0, 1, 0], dtype=np.int32)
                    for name in FG_SMARTS
                },
            ),
        )
        probe = MsgAttentiveProbe(
            input_dim=2,
            hidden_dim=8,
            num_attention_heads=1,
            task_names=task_spec.regression_tasks + task_spec.classification_tasks,
        )
        batch = {
            "peak_mz": torch.tensor(
                [[0.1, 0.2, 0.3], [0.5, 0.6, 0.7], [0.2, 0.4, 0.6]],
                dtype=torch.float32,
            ),
            "peak_intensity": torch.tensor(
                [[1.0, 0.9, 0.8], [0.7, 0.6, 0.5], [0.4, 0.3, 0.2]],
                dtype=torch.float32,
            ),
            "peak_valid_mask": torch.ones(3, 3, dtype=torch.bool),
            "probe_valid_mol": torch.tensor([True, False, True], dtype=torch.bool),
            "probe_mol_weight": torch.tensor([10.0, 12.0, 16.0], dtype=torch.float32),
            "probe_logp": torch.tensor([1.0, 1.5, 2.5], dtype=torch.float32),
            "probe_num_heavy_atoms": torch.tensor([2.0, 3.0, 5.0], dtype=torch.float32),
            "probe_num_rings": torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32),
        }
        for name in FG_SMARTS:
            batch[f"probe_fg_{name}"] = torch.tensor([0, 1, 1], dtype=torch.int32)

        result = _probe_step(
            probe,
            _DummyBackbone(),
            batch,
            task_spec=task_spec,
            feature_source="encoder",
            device=torch.device("cpu"),
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

        self.assertEqual(task_spec.regression_tasks, ("mol_weight", "logp", "num_heavy_atoms", "num_rings"))
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

        self.assertTrue(np.array_equal(targets.regression["mol_weight"], np.asarray([10.0, 30.0, 40.0], dtype=np.float32)))
        self.assertTrue(np.array_equal(targets.classification["hydroxyl"], np.asarray([0, 1, 0], dtype=np.int32)))


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


class MsgProbeIntervalTests(unittest.TestCase):
    def test_interval_trigger_logic(self):
        self.assertFalse(should_run_msg_probe(global_step=10, every_n_steps=0))
        self.assertFalse(should_run_msg_probe(global_step=9, every_n_steps=5))
        self.assertTrue(should_run_msg_probe(global_step=10, every_n_steps=5))


if __name__ == "__main__":
    unittest.main()
