import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from utils.msg_probe import (
    FG_SMARTS,
    MsgAttentiveProbe,
    MsgProbeTargets,
    _build_task_spec,
    _probe_step,
    iter_massspec_probe,
    load_or_build_msg_probe_targets,
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


def _write_fake_msg_tsv(path: Path) -> None:
    header = [
        "mzs",
        "intensities",
        "precursor_mz",
        "fold",
        "smiles",
        "adduct",
        "instrument_type",
        "collision_energy",
    ]
    rows = [
        ["100,200", "1,0.5", "250.1", "train", "CCO", "[M+H]+", "Orbitrap", "10"],
        ["110,210", "1,0.2", "270.2", "train", "CCN", "[M+H]+", "Orbitrap", "20"],
        ["120,220", "1,0.8", "290.3", "test", "CCO", "[M+Na]+", "QTOF", "30"],
    ]
    with path.open("w") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(row) + "\n")


class MsgProbeCacheTests(unittest.TestCase):
    def test_cache_builds_and_reuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tsv_path = tmp_path / "MassSpecGym.tsv"
            cache_dir = tmp_path / "cache"
            _write_fake_msg_tsv(tsv_path)

            first = load_or_build_msg_probe_targets(
                tsv_path=tsv_path,
                cache_dir=cache_dir,
            )
            cache_files = list(cache_dir.glob("MassSpecGym_probe_targets_v*.npz"))
            self.assertEqual(len(cache_files), 1)

            second = load_or_build_msg_probe_targets(
                tsv_path=tsv_path,
                cache_dir=cache_dir,
            )
            self.assertEqual(first.index_by_smiles, second.index_by_smiles)
            self.assertTrue(np.array_equal(first.valid_mol_mask, second.valid_mol_mask))
            for key in first.mol_props:
                self.assertTrue(np.array_equal(first.mol_props[key], second.mol_props[key]))
            for name in FG_SMARTS:
                self.assertTrue(np.array_equal(first.fg_counts[name], second.fg_counts[name]))


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

    def test_probe_step_filters_invalid_targets_and_losses_are_finite(self):
        probe_targets = MsgProbeTargets(
            smiles=np.asarray(["s0", "s1", "s2", "s3"], dtype=np.str_),
            mol_props={
                "mol_weight": np.asarray([10.0, 12.0, 14.0, 16.0], dtype=np.float32),
                "logp": np.asarray([1.0, 1.5, 2.0, 2.5], dtype=np.float32),
                "num_heavy_atoms": np.asarray([2.0, 3.0, 4.0, 5.0], dtype=np.float32),
                "num_rings": np.asarray([0.0, 1.0, 1.0, 2.0], dtype=np.float32),
            },
            fg_counts={
                name: np.asarray([0, 1, 0, 1], dtype=np.int16)
                for name in FG_SMARTS
            },
            valid_mol_mask=np.asarray([True, False, True, True]),
            index_by_smiles={f"s{i}": i for i in range(4)},
        )
        task_spec = _build_task_spec(
            targets=probe_targets,
            train_idx=np.asarray([0, 2, 3], dtype=np.int64),
            test_idx=np.asarray([0, 2, 3], dtype=np.int64),
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
            "smiles": np.asarray(["s0", "s1", "s3"], dtype=np.str_),
        }
        result = _probe_step(
            probe,
            _DummyBackbone(),
            batch,
            targets=probe_targets,
            task_spec=task_spec,
            feature_source="encoder",
            device=torch.device("cpu"),
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["batch_size"], 2)
        self.assertTrue(torch.isfinite(result["loss_total"]).item())


class MsgProbeTaskSpecTests(unittest.TestCase):
    def test_fg_tasks_follow_prevalence_filter(self):
        fg_counts = {name: np.zeros(8, dtype=np.int16) for name in FG_SMARTS}
        fg_counts["hydroxyl"] = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int16)
        fg_counts["amine"] = np.asarray([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int16)
        targets = MsgProbeTargets(
            smiles=np.asarray([f"s{i}" for i in range(8)], dtype=np.str_),
            mol_props={
                "mol_weight": np.linspace(10.0, 17.0, 8, dtype=np.float32),
                "logp": np.linspace(1.0, 4.5, 8, dtype=np.float32),
                "num_heavy_atoms": np.linspace(2.0, 9.0, 8, dtype=np.float32),
                "num_rings": np.linspace(0.0, 3.5, 8, dtype=np.float32),
            },
            fg_counts=fg_counts,
            valid_mol_mask=np.ones(8, dtype=bool),
            index_by_smiles={f"s{i}": i for i in range(8)},
        )

        task_spec = _build_task_spec(
            targets=targets,
            train_idx=np.asarray([0, 1, 2, 3], dtype=np.int64),
            test_idx=np.asarray([4, 5, 6, 7], dtype=np.int64),
        )

        self.assertEqual(task_spec.regression_tasks, ("mol_weight", "logp", "num_heavy_atoms", "num_rings"))
        self.assertEqual(task_spec.classification_tasks, ("hydroxyl",))


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


class MsgProbeIntervalTests(unittest.TestCase):
    def test_interval_trigger_logic(self):
        self.assertFalse(should_run_msg_probe(global_step=10, every_n_steps=0))
        self.assertFalse(should_run_msg_probe(global_step=9, every_n_steps=5))
        self.assertTrue(should_run_msg_probe(global_step=10, every_n_steps=5))


if __name__ == "__main__":
    unittest.main()
