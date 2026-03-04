import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.msg_linear_probe import (
    FG_SMARTS,
    MsgProbeTargets,
    _fit_msg_linear_probe_metrics,
    load_or_build_msg_probe_targets,
    should_run_msg_linear_probe,
)


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


class MsgLinearProbeCacheTests(unittest.TestCase):
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


class MsgLinearProbeMetricTests(unittest.TestCase):
    def test_metrics_contract_for_regression_and_fg(self):
        rng = np.random.default_rng(7)
        n_train = 80
        n_test = 40
        dim = 12
        n_total = n_train + n_test

        x_train = rng.normal(size=(n_train, dim)).astype(np.float32)
        x_test = rng.normal(size=(n_test, dim)).astype(np.float32)
        x_all = np.concatenate([x_train, x_test], axis=0)

        mol_weight = x_all[:, 0] * 2.0 + x_all[:, 1] * 0.5 + 5.0
        logp = x_all[:, 2] * 1.5 - x_all[:, 3] * 0.2 + 1.0
        num_heavy_atoms = (x_all[:, 4] * 3.0 + 20.0).astype(np.float32)
        num_rings = (x_all[:, 5] * 2.0 + 4.0).astype(np.float32)

        fg_counts = {}
        for i, name in enumerate(FG_SMARTS):
            base = (x_all[:, i % dim] > 0).astype(np.int16)
            fg_counts[name] = base

        smiles = np.asarray([f"s{i}" for i in range(n_total)], dtype=np.str_)
        targets = MsgProbeTargets(
            smiles=smiles,
            mol_props={
                "mol_weight": mol_weight.astype(np.float32),
                "logp": logp.astype(np.float32),
                "num_heavy_atoms": num_heavy_atoms.astype(np.float32),
                "num_rings": num_rings.astype(np.float32),
            },
            fg_counts=fg_counts,
            valid_mol_mask=np.ones(n_total, dtype=bool),
            index_by_smiles={s: i for i, s in enumerate(smiles.tolist())},
        )

        metrics = _fit_msg_linear_probe_metrics(
            train_embeddings=x_train,
            test_embeddings=x_test,
            train_idx=np.arange(n_train, dtype=np.int64),
            test_idx=np.arange(n_train, n_total, dtype=np.int64),
            targets=targets,
        )

        for key in ("mol_weight", "logp", "num_heavy_atoms", "num_rings"):
            self.assertIn(f"msg_linear_probe/train/r2_{key}", metrics)
            self.assertIn(f"msg_linear_probe/test/r2_{key}", metrics)

        self.assertIn("msg_linear_probe/train/auc_fg_mean", metrics)
        self.assertIn("msg_linear_probe/test/auc_fg_mean", metrics)
        self.assertGreater(metrics["msg_linear_probe/num_fg_tasks"], 0.0)


class MsgLinearProbeIntervalTests(unittest.TestCase):
    def test_interval_trigger_logic(self):
        self.assertFalse(should_run_msg_linear_probe(global_step=10, every_n_steps=0))
        self.assertFalse(should_run_msg_linear_probe(global_step=9, every_n_steps=5))
        self.assertTrue(should_run_msg_linear_probe(global_step=10, every_n_steps=5))


if __name__ == "__main__":
    unittest.main()

