import unittest

import numpy as np
import torch
from ml_collections import config_dict

from input_pipeline import _prepend_precursor_token_torch
from utils.spectra_preprocessing import PRECURSOR_TOKEN_INTENSITY
from utils.msg_probe import (
    MsgCovariancePool,
    MsgLinearProbe,
    MsgMeanPool,
    MsgPmaPool,
    MsgProbeSplitTargets,
    MsgSequenceProbe,
    _build_task_spec,
    _collect_split_targets,
    _new_epoch_state,
    _probe_step,
    _probe_task_names,
    _probe_task_output_dims,
    _score_epoch_state,
    _update_epoch_state,
    build_msg_probe_inputs,
    msg_probe_variants_from_config,
    msg_probe_metric_higher_is_better,
    iter_massspec_probe,
    probe_steps_per_epoch,
    resolve_msg_probe_sample_limits,
    resolve_msg_probe_select_metric,
)


def _maccs(rows: list[list[int]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.int32)


class _DummyDataset:
    def __init__(self, batches):
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)

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
    def test_build_msg_probe_inputs_returns_masked_mean_only(self):
        peak_embeddings = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]],
                [[2.0, 0.0], [4.0, 2.0], [6.0, 4.0]],
            ]
        )
        valid_mask = torch.tensor(
            [
                [True, True, False],
                [True, False, False],
            ]
        )

        probe_inputs = build_msg_probe_inputs(
            peak_embeddings,
            valid_mask,
        )

        expected = torch.tensor(
            [
                [2.0, 3.0],
                [2.0, 0.0],
            ]
        )
        self.assertTrue(torch.allclose(probe_inputs, expected))

    def test_output_shapes_match_task_heads(self):
        probe = MsgLinearProbe(
            input_dim=64,
            task_names=("mol_weight", "maccs"),
            task_output_dims={"maccs": 4},
        )
        probe_inputs = torch.randn(7, 64)
        logits = probe(probe_inputs)

        self.assertEqual(logits["mol_weight"].shape, (7, 1))
        self.assertEqual(logits["maccs"].shape, (7, 4))

    def test_output_shapes_match_multiclass_head(self):
        probe = MsgLinearProbe(
            input_dim=64,
            task_names=("mol_weight", "num_rings"),
            task_output_dims={"num_rings": 4},
        )
        probe_inputs = torch.randn(7, 64)
        logits = probe(probe_inputs)

        self.assertEqual(logits["mol_weight"].shape, (7, 1))
        self.assertEqual(logits["num_rings"].shape, (7, 4))

    def test_finite_outputs(self):
        probe = MsgLinearProbe(
            input_dim=32,
            task_names=("mol_weight", "maccs"),
            task_output_dims={"maccs": 4},
        )
        probe_inputs = torch.randn(3, 32)
        logits = probe(probe_inputs)

        self.assertTrue(torch.isfinite(logits["mol_weight"]).all().item())
        self.assertTrue(torch.isfinite(logits["maccs"]).all().item())

    def test_probe_heads_are_linear(self):
        probe = MsgLinearProbe(
            input_dim=32,
            task_names=("mol_weight", "num_rings", "maccs"),
            task_output_dims={"num_rings": 4, "maccs": 4},
        )

        self.assertIsInstance(probe.heads["mol_weight"], torch.nn.Linear)
        self.assertIsInstance(probe.heads["num_rings"], torch.nn.Linear)
        self.assertIsInstance(probe.heads["maccs"], torch.nn.Linear)


class MsgSequenceProbeTests(unittest.TestCase):
    def test_msg_probe_variants_from_config_defaults(self):
        self.assertEqual(
            msg_probe_variants_from_config({}),
            ("mean", "covariance", "pma"),
        )

    def test_covariance_pool_matches_masked_second_moment(self):
        pool = MsgCovariancePool(input_dim=2, compressed_dim=2)
        with torch.no_grad():
            pool.left_proj.weight.copy_(torch.eye(2))
            pool.right_proj.weight.copy_(torch.eye(2))

        peak_embeddings = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]],
                [[2.0, 1.0], [9.0, 9.0], [8.0, 8.0]],
            ]
        )
        valid_mask = torch.tensor(
            [
                [True, True, False],
                [True, False, False],
            ]
        )

        pooled = pool(peak_embeddings, valid_mask)

        expected = torch.tensor(
            [
                [5.0, 7.0, 7.0, 10.0],
                [4.0, 2.0, 2.0, 1.0],
            ]
        )
        self.assertTrue(torch.allclose(pooled, expected))

    def test_pma_pool_returns_fixed_size_vectors(self):
        pool = MsgPmaPool(input_dim=4, num_seeds=3, num_heads=2)
        peak_embeddings = torch.randn(2, 5, 4)
        valid_mask = torch.tensor(
            [
                [True, True, False, False, False],
                [True, True, True, True, False],
            ]
        )

        pooled = pool(peak_embeddings, valid_mask)

        self.assertEqual(pooled.shape, (2, 4))
        self.assertTrue(torch.isfinite(pooled).all().item())

    def test_sequence_probe_output_shapes_match_task_heads_for_all_variants(self):
        peak_embeddings = torch.randn(3, 6, 4)
        valid_mask = torch.tensor(
            [
                [True, True, True, False, False, False],
                [True, True, True, True, False, False],
                [True, True, False, False, False, False],
            ]
        )
        variants = (
            MsgSequenceProbe(
                pooler=MsgMeanPool(),
                pooled_dim=4,
                hidden_dim=8,
                task_names=("mol_weight", "maccs"),
                task_output_dims={"maccs": 4},
            ),
            MsgSequenceProbe(
                pooler=MsgCovariancePool(input_dim=4, compressed_dim=3),
                pooled_dim=9,
                hidden_dim=8,
                task_names=("mol_weight", "maccs"),
                task_output_dims={"maccs": 4},
            ),
            MsgSequenceProbe(
                pooler=MsgPmaPool(input_dim=4, num_seeds=2, num_heads=2),
                pooled_dim=4,
                hidden_dim=8,
                task_names=("mol_weight", "maccs"),
                task_output_dims={"maccs": 4},
            ),
        )

        for probe in variants:
            logits = probe(peak_embeddings, valid_mask)
            self.assertEqual(logits["mol_weight"].shape, (3, 1))
            self.assertEqual(logits["maccs"].shape, (3, 4))


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
                maccs=_maccs(
                    [
                        [0, 1, 0, 1],
                        [1, 0, 1, 0],
                        [0, 1, 1, 0],
                    ]
                ),
            ),
            test_targets=MsgProbeSplitTargets(
                regression={
                    "mol_weight": np.asarray([10.0, 14.0, 16.0], dtype=np.float32),
                    "logp": np.asarray([1.0, 2.0, 2.5], dtype=np.float32),
                    "num_heavy_atoms": np.asarray([2.0, 4.0, 5.0], dtype=np.float32),
                    "num_rings": np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
                },
                maccs=_maccs(
                    [
                        [0, 1, 0, 1],
                        [1, 0, 1, 0],
                        [0, 1, 1, 0],
                    ]
                ),
            ),
        )
        probe_input_dim = 16
        probe = MsgLinearProbe(
            input_dim=probe_input_dim,
            task_names=_probe_task_names(task_spec),
            task_output_dims=_probe_task_output_dims(task_spec),
        )
        batch = {
            "probe_valid_mol": torch.tensor([True, False, True], dtype=torch.bool),
            "probe_mol_weight": torch.tensor([10.0, 12.0, 16.0], dtype=torch.float32),
            "probe_logp": torch.tensor([1.0, 1.5, 2.5], dtype=torch.float32),
            "probe_num_heavy_atoms": torch.tensor([2.0, 3.0, 5.0], dtype=torch.float32),
            "probe_num_rings": torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32),
            "probe_maccs": torch.tensor(
                [
                    [0, 1, 0, 1],
                    [1, 0, 1, 0],
                    [0, 1, 1, 0],
                ],
                dtype=torch.int32,
            ),
        }

        def dummy_extractor(b):
            n = b["probe_valid_mol"].shape[0]
            return torch.randn(n, probe_input_dim)

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
        self.assertEqual(result["predictions"]["num_rings"].shape, (2,))
        self.assertEqual(result["predictions"]["maccs"].shape, (2, 4))


class MsgProbeTaskSpecTests(unittest.TestCase):
    def test_task_spec_includes_num_rings_and_maccs_bits(self):
        task_spec = _build_task_spec(
            train_targets=MsgProbeSplitTargets(
                regression={
                    "mol_weight": np.linspace(10.0, 13.0, 4, dtype=np.float32),
                    "logp": np.linspace(1.0, 2.5, 4, dtype=np.float32),
                    "num_heavy_atoms": np.linspace(2.0, 5.0, 4, dtype=np.float32),
                    "num_rings": np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
                },
                maccs=_maccs(
                    [
                        [0, 1, 0, 1],
                        [1, 0, 1, 0],
                        [0, 1, 1, 0],
                        [1, 1, 0, 0],
                    ]
                ),
            ),
            test_targets=MsgProbeSplitTargets(
                regression={
                    "mol_weight": np.linspace(14.0, 17.0, 4, dtype=np.float32),
                    "logp": np.linspace(3.0, 4.5, 4, dtype=np.float32),
                    "num_heavy_atoms": np.linspace(6.0, 9.0, 4, dtype=np.float32),
                    "num_rings": np.asarray([2.0, 3.0, 4.0, 5.0], dtype=np.float32),
                },
                maccs=_maccs(
                    [
                        [1, 0, 0, 1],
                        [0, 1, 1, 0],
                        [1, 0, 1, 0],
                        [0, 1, 0, 1],
                    ]
                ),
            ),
        )

        self.assertEqual(
            task_spec.regression_tasks,
            ("mol_weight", "logp", "num_heavy_atoms"),
        )
        self.assertEqual(task_spec.num_rings_classes, (0, 1, 2, 3))
        self.assertEqual(task_spec.maccs_bits, 4)
        self.assertEqual(
            _probe_task_names(task_spec),
            ("mol_weight", "logp", "num_heavy_atoms", "num_rings", "maccs"),
        )


class MsgProbeMetricTests(unittest.TestCase):
    def test_score_epoch_state_reports_num_rings_and_maccs_metrics(self):
        task_spec = _build_task_spec(
            train_targets=MsgProbeSplitTargets(
                regression={
                    "mol_weight": np.asarray([10.0, 20.0, 30.0], dtype=np.float32),
                    "logp": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
                    "num_heavy_atoms": np.asarray([2.0, 4.0, 6.0], dtype=np.float32),
                    "num_rings": np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
                },
                maccs=_maccs(
                    [
                        [0, 1, 0, 1],
                        [1, 0, 1, 0],
                        [0, 1, 1, 0],
                    ]
                ),
            ),
            test_targets=MsgProbeSplitTargets(
                regression={
                    "mol_weight": np.asarray([12.0, 18.0, 29.0], dtype=np.float32),
                    "logp": np.asarray([1.5, 2.5, 3.5], dtype=np.float32),
                    "num_heavy_atoms": np.asarray([3.0, 5.0, 7.0], dtype=np.float32),
                    "num_rings": np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
                },
                maccs=_maccs(
                    [
                        [1, 0, 0, 1],
                        [0, 1, 1, 0],
                        [1, 0, 1, 0],
                    ]
                ),
            ),
        )
        epoch_state = _new_epoch_state(task_spec)
        result = {
            "batch_size": 3,
            "predictions": {
                "mol_weight": torch.tensor([10.0, 19.0, 29.0]),
                "logp": torch.tensor([1.0, 2.0, 4.0]),
                "num_heavy_atoms": torch.tensor([2.0, 5.0, 6.0]),
                "num_rings": torch.tensor([0.0, 2.0, 2.0]),
                "maccs": torch.tensor(
                    [
                        [0.1, 0.9, 0.2, 0.8],
                        [0.8, 0.2, 0.9, 0.1],
                        [0.2, 0.8, 0.7, 0.3],
                    ]
                ),
            },
            "targets": {
                "mol_weight": torch.tensor([10.0, 20.0, 30.0]),
                "logp": torch.tensor([1.0, 2.0, 3.0]),
                "num_heavy_atoms": torch.tensor([2.0, 4.0, 6.0]),
                "num_rings": torch.tensor([0.0, 1.0, 2.0]),
                "maccs": torch.tensor(
                    [
                        [0.0, 1.0, 0.0, 1.0],
                        [1.0, 0.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0],
                    ]
                ),
            },
        }
        _update_epoch_state(epoch_state, result, task_spec)
        metrics = _score_epoch_state(
            prefix="msg_probe/test",
            epoch_state=epoch_state,
            task_spec=task_spec,
        )

        self.assertNotIn("msg_probe/test/r2_num_rings", metrics)
        self.assertEqual(metrics["msg_probe/test/acc_num_rings_exact"], 2 / 3)
        self.assertEqual(metrics["msg_probe/test/acc_num_rings_within_1"], 1.0)
        self.assertAlmostEqual(metrics["msg_probe/test/mae_num_rings"], 1 / 3)
        self.assertEqual(
            metrics["msg_probe/test/r2_mean"],
            metrics["msg_probe/test/r2_mean_wo_num_rings"],
        )
        self.assertEqual(metrics["msg_probe/test/num_maccs_auc_bits"], 4.0)
        self.assertGreater(metrics["msg_probe/test/auc_maccs_mean"], 0.9)
        self.assertEqual(metrics["msg_probe/test/num_maccs_recall_bits"], 4.0)
        self.assertGreater(metrics["msg_probe/test/recall_maccs_mean"], 0.9)

    def test_select_metric_uses_tune_metric_fallback(self):
        cfg = {
            "msg_probe_tune_metric": "msg_probe/test/mae_num_rings",
        }
        self.assertEqual(
            resolve_msg_probe_select_metric(cfg),
            "msg_probe/test/mae_num_rings",
        )
        self.assertFalse(
            msg_probe_metric_higher_is_better("msg_probe/test/mae_num_rings")
        )
        self.assertTrue(
            msg_probe_metric_higher_is_better("msg_probe/test/auc_maccs_mean")
        )


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
                    "probe_maccs": _maccs([[0, 1, 0, 1], [1, 0, 1, 0]]),
                },
                {
                    "peak_mz": np.zeros((2, 60), dtype=np.float32),
                    "probe_valid_mol": np.asarray([1, 1], dtype=np.int32),
                    "probe_mol_weight": np.asarray([30.0, 40.0], dtype=np.float32),
                    "probe_logp": np.asarray([3.0, 4.0], dtype=np.float32),
                    "probe_num_heavy_atoms": np.asarray([4.0, 5.0], dtype=np.float32),
                    "probe_num_rings": np.asarray([2.0, 3.0], dtype=np.float32),
                    "probe_maccs": _maccs([[1, 1, 0, 0], [0, 1, 1, 0]]),
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
                targets.maccs,
                _maccs(
                    [
                        [0, 1, 0, 1],
                        [1, 1, 0, 0],
                        [0, 1, 1, 0],
                    ]
                ),
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

    def test_eval_probe_can_randomly_sample(self):
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
                max_samples=1,
                sample_randomly=True,
            )
        )
        self.assertEqual(dm.calls[0]["shuffle"], True)

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


class ProbeConfigTests(unittest.TestCase):
    def test_nist_full_probe_defaults_use_random_subsets(self):
        cfg = config_dict.ConfigDict()
        cfg.probe_dataset = "nist-full"

        train_samples, test_samples, randomize_test_subset = (
            resolve_msg_probe_sample_limits(cfg)
        )

        self.assertEqual(train_samples, 4000)
        self.assertEqual(test_samples, 1000)
        self.assertTrue(randomize_test_subset)


class ProbePrecursorTokenTests(unittest.TestCase):
    def test_prepend_shapes_and_values(self):
        B, N = 3, 5
        batch = {
            "peak_mz": torch.tensor(np.random.rand(B, N).astype(np.float32)),
            "peak_intensity": torch.tensor(np.random.rand(B, N).astype(np.float32)),
            "peak_valid_mask": torch.tensor(np.ones((B, N), dtype=bool)),
            "precursor_mz": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
            "fingerprint": torch.tensor(np.zeros((B, 4), dtype=np.int32)),
            "probe_valid_mol": torch.tensor([True, False, True]),
            "probe_maccs": torch.tensor(np.zeros((B, 4), dtype=np.int32)),
        }

        out = _prepend_precursor_token_torch(batch)

        self.assertEqual(out["peak_mz"].shape, (B, N + 1))
        self.assertEqual(out["peak_intensity"].shape, (B, N + 1))
        self.assertEqual(out["peak_valid_mask"].shape, (B, N + 1))
        self.assertNotIn("precursor_mz", out)
        np.testing.assert_allclose(
            out["peak_intensity"][:, 0].numpy(),
            [PRECURSOR_TOKEN_INTENSITY] * B,
        )
        np.testing.assert_array_equal(
            out["peak_valid_mask"][:, 0].numpy(), [True, True, True]
        )
        np.testing.assert_allclose(out["peak_mz"][:, 0].numpy(), [0.1, 0.2, 0.3])
        np.testing.assert_array_equal(
            out["peak_mz"][:, 1:].numpy(),
            batch["peak_mz"].numpy(),
        )
        self.assertIn("fingerprint", out)
        self.assertIn("probe_valid_mol", out)
        self.assertIn("probe_maccs", out)
        np.testing.assert_array_equal(
            out["fingerprint"].numpy(), batch["fingerprint"].numpy()
        )
        np.testing.assert_array_equal(
            out["probe_maccs"].numpy(), batch["probe_maccs"].numpy()
        )


if __name__ == "__main__":
    unittest.main()
