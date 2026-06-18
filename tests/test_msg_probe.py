import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest import mock

import numpy as np
import torch
from ml_collections import config_dict
from sklearn.metrics import average_precision_score, roc_auc_score

from spectra_learning.models.pooling import CovariancePool
from spectra_learning.probes.massspec.msg_modules import (
    FrozenPooler,
    MsgCovariancePool,
    MsgLinearProbe,
    MsgMeanPool,
    MsgPmaPool,
    MsgSequenceProbe,
    MsgSinglePairClsPool,
    MsgSinglePairClsProbe,
    MsgSinglePairCovariancePool,
    MsgSinglePairCovarianceProbe,
    MsgSinglePairLinearProbe,
    MsgSinglePairPmaPool,
    _probe_task_names,
    _probe_task_output_dims,
    _uses_pair_features,
    build_msg_sequence_probe as _build_msg_sequence_probe,
)
from spectra_learning.data.loading import local_batch_size
from spectra_learning.probes.massspec.msg_settings import (
    MsgProbeSplitTargets,
    MsgProbeTaskSpec,
    build_msg_probe_inputs,
    msg_probe_variants_from_config,
    resolve_msg_probe_pairwise_alignment_num_pairs,
    resolve_msg_probe_num_repeats,
    resolve_msg_probe_fingerprint,
)
from spectra_learning.probes.massspec.msg_probe import (
    _compute_pairwise_similarity_alignment,
    _build_task_spec,
    _collect_split_targets,
    _new_epoch_state,
    _online_probe_covariance_pooler,
    _probe_step,
    _resolve_probe_warmup_steps,
    _compute_pairwise_similarity_alignment_for_indices,
    _plot_pairwise_similarity_alignment,
    _run_msg_probe_once,
    _score_epoch_state,
    _update_epoch_state,
    msg_probe_metric_higher_is_better,
    iter_massspec_probe,
    probe_steps_per_epoch,
    resolve_msg_probe_select_metric,
    run_msg_probe,
)
from spectra_learning.probes.massspec.pr_curves import PrecisionRecallCurve
from spectra_learning.data.massspec_targets import (
    MORGAN_PROBE_FINGERPRINT_BITS,
    MORGAN_PROBE_FINGERPRINT_RADIUS,
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
        max_samples: int | None = None,
        distributed_world_size: int = 1,
        distributed_rank: int = 0,
        pad_distributed: bool = False,
    ):
        self.calls.append(
            {
                "split": split,
                "seed": seed,
                "peak_ordering": peak_ordering,
                "shuffle": shuffle,
                "drop_remainder": drop_remainder,
                "max_samples": max_samples,
                "distributed_world_size": distributed_world_size,
                "distributed_rank": distributed_rank,
                "pad_distributed": pad_distributed,
            }
        )
        return self._dataset


class _SplitDummyDataModule:
    def __init__(self, batches_by_split, info, batch_size):
        self._batches_by_split = batches_by_split
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
        max_samples: int | None = None,
        distributed_world_size: int = 1,
        distributed_rank: int = 0,
        pad_distributed: bool = False,
    ):
        self.calls.append(
            {
                "split": split,
                "seed": seed,
                "peak_ordering": peak_ordering,
                "shuffle": shuffle,
                "drop_remainder": drop_remainder,
                "max_samples": max_samples,
                "distributed_world_size": distributed_world_size,
                "distributed_rank": distributed_rank,
                "pad_distributed": pad_distributed,
            }
        )
        return _DummyDataset(self._batches_by_split[split])


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
            task_names=("maccs",),
            task_output_dims={"maccs": 7},
        )
        probe_inputs = torch.randn(7, 64)
        logits = probe(probe_inputs)

        self.assertEqual(logits["maccs"].shape, (7, 7))

    def test_output_shapes_match_regression_only_heads(self):
        probe = MsgLinearProbe(
            input_dim=64,
            task_names=("mol_weight", "logp"),
        )
        probe_inputs = torch.randn(7, 64)
        logits = probe(probe_inputs)

        self.assertEqual(logits["mol_weight"].shape, (7, 1))
        self.assertEqual(logits["logp"].shape, (7, 1))

    def test_finite_outputs(self):
        probe = MsgLinearProbe(
            input_dim=32,
            task_names=("maccs",),
            task_output_dims={"maccs": 7},
        )
        probe_inputs = torch.randn(3, 32)
        logits = probe(probe_inputs)

        self.assertTrue(torch.isfinite(logits["maccs"]).all().item())

    def test_probe_head_is_linear(self):
        probe = MsgLinearProbe(
            input_dim=32,
            task_names=("maccs",),
            task_output_dims={"maccs": 7},
        )

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

    def test_covariance_pool_ignores_cls_token(self):
        pool = MsgCovariancePool(input_dim=2, compressed_dim=2)
        with torch.no_grad():
            pool.left_proj.weight.copy_(torch.eye(2))
            pool.right_proj.weight.copy_(torch.eye(2))

        peak_embeddings = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]]])
        valid_mask = torch.tensor([[True, True]])

        pooled = pool(peak_embeddings, valid_mask)

        torch.testing.assert_close(pooled, torch.tensor([[5.0, 7.0, 7.0, 10.0]]))

    def test_single_pair_covariance_pool_uses_off_diagonal_pair_second_moment(self):
        pool = MsgSinglePairCovariancePool(
            single_dim=2,
            pair_dim=2,
            compressed_dim=2,
        )
        with torch.no_grad():
            pool.single_pool.left_proj.weight.copy_(torch.eye(2))
            pool.single_pool.right_proj.weight.copy_(torch.eye(2))
            pool.pair_left_proj.weight.copy_(torch.eye(2))
            pool.pair_right_proj.weight.copy_(torch.eye(2))
            pool.output_proj.weight.zero_()
            pool.output_proj.bias.zero_()
            pool.output_proj.weight[:, 4:].copy_(torch.eye(4))

        peak_embeddings = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [9.0, 9.0], [99.0, 99.0]]])
        pair_embeddings = torch.zeros(1, 4, 4, 2)
        pair_embeddings[0, 0, 0] = torch.tensor([100.0, 100.0])
        pair_embeddings[0, 1, 1] = torch.tensor([100.0, 100.0])
        pair_embeddings[0, 0, 1] = torch.tensor([1.0, 2.0])
        pair_embeddings[0, 1, 0] = torch.tensor([3.0, 4.0])
        pair_embeddings[0, 0, 2] = torch.tensor([100.0, 100.0])
        pair_embeddings[0, 3, 3] = torch.tensor([99.0, 99.0])
        valid_mask = torch.tensor([[True, True, False]])

        pooled = pool(peak_embeddings, valid_mask, pair_embeddings)

        expected = torch.nn.functional.layer_norm(
            torch.tensor([[5.0, 7.0, 7.0, 10.0]]),
            (4,),
        )
        torch.testing.assert_close(pooled, expected)

    def test_single_pair_covariance_pool_uses_latent_pairs_when_pair_grid_is_compact(self):
        pool = MsgSinglePairCovariancePool(
            single_dim=2,
            pair_dim=2,
            compressed_dim=2,
        )
        with torch.no_grad():
            pool.single_pool.left_proj.weight.copy_(torch.eye(2))
            pool.single_pool.right_proj.weight.copy_(torch.eye(2))
            pool.pair_left_proj.weight.copy_(torch.eye(2))
            pool.pair_right_proj.weight.copy_(torch.eye(2))
            pool.output_proj.weight.zero_()
            pool.output_proj.bias.zero_()
            pool.output_proj.weight[:, 4:].copy_(torch.eye(4))

        peak_embeddings = torch.randn(1, 4, 2)
        latent_pair_embeddings = torch.zeros(1, 2, 2, 2)
        latent_pair_embeddings[0, 0, 0] = torch.tensor([100.0, 100.0])
        latent_pair_embeddings[0, 1, 1] = torch.tensor([100.0, 100.0])
        latent_pair_embeddings[0, 0, 1] = torch.tensor([1.0, 2.0])
        latent_pair_embeddings[0, 1, 0] = torch.tensor([3.0, 4.0])
        valid_mask = torch.tensor([[True, True, True]])

        pooled = pool(peak_embeddings, valid_mask, latent_pair_embeddings)

        expected = torch.nn.functional.layer_norm(
            torch.tensor([[5.0, 7.0, 7.0, 10.0]]),
            (4,),
        )
        torch.testing.assert_close(pooled, expected)

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

    def test_single_pair_pma_pool_flattens_single_and_pair_tokens(self):
        pool = MsgSinglePairPmaPool(
            single_dim=4,
            pair_dim=6,
            latent_dim=4,
            num_tokens=3,
            num_heads=2,
            num_blocks=2,
            hidden_dim=8,
            norm_eps=1e-5,
        )
        peak_embeddings = torch.randn(2, 5, 4)
        pair_embeddings = torch.randn(2, 5, 5, 6)
        valid_mask = torch.tensor(
            [
                [True, True, False, False, False],
                [True, True, True, True, False],
            ]
        )

        pooled = pool(peak_embeddings, valid_mask, pair_embeddings)

        self.assertEqual(pooled.shape, (2, 2 * 3 * 4))
        self.assertTrue(torch.isfinite(pooled).all().item())

    def test_single_pair_pma_pool_accepts_compact_latent_pair_tokens(self):
        pool = MsgSinglePairPmaPool(
            single_dim=4,
            pair_dim=6,
            latent_dim=4,
            num_tokens=3,
            num_heads=2,
            num_blocks=2,
            hidden_dim=8,
            norm_eps=1e-5,
        )
        peak_embeddings = torch.randn(2, 5, 4)
        latent_pair_embeddings = torch.randn(2, 2, 2, 6)
        valid_mask = torch.ones(2, 5, dtype=torch.bool)

        pooled = pool(peak_embeddings, valid_mask, latent_pair_embeddings)

        self.assertEqual(pooled.shape, (2, 2 * 3 * 4))
        self.assertTrue(torch.isfinite(pooled).all().item())


    def test_cls_pool_concatenates_single_cls_and_pair_cls(self):
        pool = MsgSinglePairClsPool()
        peak_embeddings = torch.randn(2, 4, 3)
        pair_embeddings = torch.randn(2, 4, 4, 5)
        valid_mask = torch.ones(2, 3, dtype=torch.bool)

        pooled = pool(peak_embeddings, valid_mask, pair_embeddings)

        expected = torch.cat(
            [
                peak_embeddings[:, 3],
                pair_embeddings[:, 3, 3],
            ],
            dim=-1,
        )
        torch.testing.assert_close(pooled, expected)

    def test_cls_probe_uses_pair_features_and_mlp_heads(self):
        config = config_dict.ConfigDict()
        config.model_dim = 4
        config.pairmixer_pair_dim = 6
        config.msg_probe_mlp_hidden_dim = 8
        task_spec = MsgProbeTaskSpec(
            regression_tasks=("mol_weight",),
            maccs_bits=4,
            regression_means={"mol_weight": 0.0},
            regression_stds={"mol_weight": 1.0},
            fingerprint_task="maccs",
        )
        probe = _build_msg_sequence_probe(
            "cls",
            config=config,
            task_spec=task_spec,
        )
        peak_embeddings = torch.randn(3, 6, 4)
        pair_embeddings = torch.randn(3, 6, 6, 6)
        valid_mask = torch.ones(3, 5, dtype=torch.bool)

        logits = probe(peak_embeddings, valid_mask, pair_embeddings)

        self.assertTrue(_uses_pair_features("cls"))
        self.assertIsInstance(probe, MsgSinglePairClsProbe)
        head = cast(torch.nn.Sequential, probe.heads.heads["maccs"])
        first_head = cast(torch.nn.Linear, head[0])
        self.assertEqual(first_head.in_features, 10)
        self.assertEqual(logits["maccs"].shape, (3, 5))

    def test_single_pair_pma_probe_uses_linear_heads(self):
        config = config_dict.ConfigDict()
        config.model_dim = 4
        config.pairmixer_pair_dim = 6
        config.encoder_num_heads = 2
        config.msg_probe_mlp_hidden_dim = 8
        task_spec = MsgProbeTaskSpec(
            regression_tasks=("mol_weight",),
            maccs_bits=4,
            regression_means={"mol_weight": 0.0},
            regression_stds={"mol_weight": 1.0},
            fingerprint_task="maccs",
        )
        probe = _build_msg_sequence_probe(
            "single_pair_pma_2",
            config=config,
            task_spec=task_spec,
        )
        peak_embeddings = torch.randn(3, 5, 4)
        pair_embeddings = torch.randn(3, 5, 5, 6)
        valid_mask = torch.ones(3, 5, dtype=torch.bool)

        logits = probe(peak_embeddings, valid_mask, pair_embeddings)

        self.assertIsInstance(probe, MsgSinglePairLinearProbe)
        self.assertEqual(len(probe.pooler.single_encoder.blocks), 2)
        self.assertIsInstance(probe.heads.heads["maccs"], torch.nn.Linear)
        self.assertEqual(logits["maccs"].shape, (3, 5))

    def test_single_pair_covariance_probe_uses_mlp_heads(self):
        config = config_dict.ConfigDict()
        config.model_dim = 4
        config.pairmixer_pair_dim = 6
        config.covariance_pooling_dim = 3
        config.msg_probe_mlp_hidden_dim = 8
        task_spec = MsgProbeTaskSpec(
            regression_tasks=("mol_weight",),
            maccs_bits=4,
            regression_means={"mol_weight": 0.0},
            regression_stds={"mol_weight": 1.0},
            fingerprint_task="maccs",
        )
        probe = _build_msg_sequence_probe(
            "single_pair_covariance",
            config=config,
            task_spec=task_spec,
        )
        peak_embeddings = torch.randn(3, 5, 4)
        pair_embeddings = torch.randn(3, 5, 5, 6)
        valid_mask = torch.ones(3, 5, dtype=torch.bool)

        logits = probe(peak_embeddings, valid_mask, pair_embeddings)

        self.assertIsInstance(probe, MsgSinglePairCovarianceProbe)
        self.assertEqual(probe.pooler.output_dim, 3 * 3)
        head = cast(torch.nn.Sequential, probe.heads.heads["maccs"])
        first_head = cast(torch.nn.Linear, head[0])
        self.assertEqual(first_head.in_features, 3 * 3)
        self.assertEqual(logits["maccs"].shape, (3, 5))

    def test_single_pair_covariance_probe_reuses_trained_pooler(self):
        config = config_dict.ConfigDict()
        config.model_dim = 4
        config.pairmixer_pair_dim = 6
        config.covariance_pooling_dim = 7
        config.msg_probe_mlp_hidden_dim = 8
        task_spec = MsgProbeTaskSpec(
            regression_tasks=("mol_weight",),
            maccs_bits=4,
            regression_means={"mol_weight": 0.0},
            regression_stds={"mol_weight": 1.0},
            fingerprint_task="maccs",
        )
        trained_pooler = MsgSinglePairCovariancePool(
            single_dim=4,
            pair_dim=6,
            compressed_dim=3,
        )

        probe = _build_msg_sequence_probe(
            "single_pair_covariance",
            config=config,
            task_spec=task_spec,
            covariance_pooler=_online_probe_covariance_pooler(
                "single_pair_covariance",
                trained_pooler,
            ),
        )

        self.assertIsInstance(probe.pooler, FrozenPooler)
        self.assertIs(probe.pooler.pooler, trained_pooler)
        head = cast(torch.nn.Sequential, probe.heads.heads["maccs"])
        first_head = cast(torch.nn.Linear, head[0])
        self.assertEqual(first_head.in_features, 3 * 3)
        probe_param_ids = {id(param) for param in probe.parameters()}
        trained_param_ids = {id(param) for param in trained_pooler.parameters()}
        self.assertFalse(probe_param_ids & trained_param_ids)

    def test_single_pair_covariance_probe_can_train_supplied_pooler(self):
        config = config_dict.ConfigDict()
        config.model_dim = 4
        config.pairmixer_pair_dim = 6
        config.covariance_pooling_dim = 7
        config.msg_probe_mlp_hidden_dim = 8
        config.msg_probe_freeze_supplied_pooler = False
        task_spec = MsgProbeTaskSpec(
            regression_tasks=("mol_weight",),
            maccs_bits=4,
            regression_means={"mol_weight": 0.0},
            regression_stds={"mol_weight": 1.0},
            fingerprint_task="maccs",
        )
        trained_pooler = MsgSinglePairCovariancePool(
            single_dim=4,
            pair_dim=6,
            compressed_dim=3,
        )

        probe = _build_msg_sequence_probe(
            "single_pair_covariance",
            config=config,
            task_spec=task_spec,
            covariance_pooler=trained_pooler,
        )

        self.assertIs(probe.pooler, trained_pooler)
        probe_param_ids = {id(param) for param in probe.parameters()}
        trained_param_ids = {id(param) for param in trained_pooler.parameters()}
        self.assertTrue(probe_param_ids & trained_param_ids)

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
                task_names=("maccs",),
                task_output_dims={"maccs": 7},
            ),
            MsgSequenceProbe(
                pooler=MsgCovariancePool(input_dim=4, compressed_dim=3),
                pooled_dim=9,
                hidden_dim=8,
                task_names=("maccs",),
                task_output_dims={"maccs": 7},
            ),
            MsgSequenceProbe(
                pooler=MsgPmaPool(input_dim=4, num_seeds=2, num_heads=2),
                pooled_dim=4,
                hidden_dim=8,
                task_names=("maccs",),
                task_output_dims={"maccs": 7},
            ),
        )

        for probe in variants:
            logits = probe(peak_embeddings, valid_mask)
            self.assertEqual(logits["maccs"].shape, (3, 7))

    def test_sequence_probe_supports_deeper_mlp_heads(self):
        probe = MsgSequenceProbe(
            pooler=MsgMeanPool(),
            pooled_dim=4,
            hidden_dim=8,
            num_layers=4,
            task_names=("maccs",),
            task_output_dims={"maccs": 7},
        )

        head = cast(torch.nn.Sequential, probe.heads.heads["maccs"])
        linear_layers = [
            layer for layer in head if isinstance(layer, torch.nn.Linear)
        ]

        self.assertEqual(len(linear_layers), 4)
        self.assertEqual(linear_layers[0].in_features, 4)
        self.assertEqual(linear_layers[1].in_features, 8)
        self.assertEqual(linear_layers[-1].out_features, 7)

    def test_covariance_probe_uses_learned_pooler_when_provided(self):
        config = config_dict.ConfigDict()
        config.model_dim = 4
        config.msg_probe_mlp_hidden_dim = 8
        config.covariance_pooling_dim = 3
        task_spec = MsgProbeTaskSpec(
            regression_tasks=("mol_weight",),
            maccs_bits=0,
            regression_means={"mol_weight": 0.0},
            regression_stds={"mol_weight": 1.0},
            fingerprint_task="maccs",
        )
        learned_pooler = CovariancePool(input_dim=4, compressed_dim=2)

        probe = _build_msg_sequence_probe(
            "covariance",
            config=config,
            task_spec=task_spec,
            covariance_pooler=learned_pooler,
        )

        self.assertIsInstance(probe.pooler, FrozenPooler)
        self.assertIs(probe.pooler.pooler, learned_pooler)
        head = cast(torch.nn.Sequential, probe.heads.heads["mol_weight"])
        first_head = cast(torch.nn.Linear, head[0])
        self.assertEqual(first_head.in_features, 2 * 2)
        probe_param_ids = {id(param) for param in probe.parameters()}
        learned_param_ids = {id(param) for param in learned_pooler.parameters()}
        self.assertFalse(probe_param_ids & learned_param_ids)

    def test_covariance_probe_trains_own_pooler_when_main_pooler_is_frozen(self):
        config = config_dict.ConfigDict()
        config.model_dim = 4
        config.covariance_pooling_dim = 2
        config.msg_probe_mlp_hidden_dim = 8
        task_spec = MsgProbeTaskSpec(
            regression_tasks=("mol_weight",),
            maccs_bits=0,
            regression_means={"mol_weight": 0.0},
            regression_stds={"mol_weight": 1.0},
            fingerprint_task="maccs",
        )
        probe = _build_msg_sequence_probe(
            "covariance",
            config=config,
            task_spec=task_spec,
            covariance_pooler=_online_probe_covariance_pooler("covariance"),
        )
        optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3)

        self.assertIsInstance(probe.pooler, MsgCovariancePool)
        self.assertEqual(probe.pooler.left_proj.out_features, 2)
        self.assertTrue(all(param.requires_grad for param in probe.pooler.parameters()))
        probe_param_ids = {id(param) for param in probe.parameters()}
        probe_pooler_param_ids = {id(param) for param in probe.pooler.parameters()}
        optimizer_param_ids = {
            id(param)
            for group in optimizer.param_groups
            for param in group["params"]
        }

        self.assertTrue(probe_pooler_param_ids <= optimizer_param_ids)
        self.assertTrue(probe_pooler_param_ids <= probe_param_ids)

    def test_covariance_probe_reuses_frozen_pooler_when_main_pooler_is_trainable(self):
        main_pooler = CovariancePool(input_dim=4, compressed_dim=2)

        pooler = _online_probe_covariance_pooler("covariance", main_pooler)

        self.assertIs(pooler, main_pooler)

    def test_online_covariance_probe_does_not_override_trainable_pooler(self):
        config = config_dict.ConfigDict()
        config.model_dim = 4
        config.covariance_pooling_dim = 7
        config.msg_probe_mlp_hidden_dim = 8
        task_spec = MsgProbeTaskSpec(
            regression_tasks=("mol_weight",),
            maccs_bits=0,
            regression_means={"mol_weight": 0.0},
            regression_stds={"mol_weight": 1.0},
            fingerprint_task="maccs",
        )
        main_pooler = CovariancePool(input_dim=4, compressed_dim=2)
        original_state = {
            name: param.detach().clone()
            for name, param in main_pooler.state_dict().items()
        }

        probe = _build_msg_sequence_probe(
            "covariance",
            config=config,
            task_spec=task_spec,
            covariance_pooler=_online_probe_covariance_pooler(
                "covariance",
                main_pooler,
            ),
        )
        optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3)
        peak_embeddings = torch.randn(3, 5, 4)
        valid_mask = torch.ones(3, 5, dtype=torch.bool)
        logits = probe(peak_embeddings, valid_mask)["mol_weight"]
        loss = logits.square().mean()
        loss.backward()
        optimizer.step()
        probe.load_state_dict(probe.state_dict())

        self.assertIsInstance(probe.pooler, FrozenPooler)
        self.assertIs(probe.pooler.pooler, main_pooler)
        head = cast(torch.nn.Sequential, probe.heads.heads["mol_weight"])
        first_head = cast(torch.nn.Linear, head[0])
        self.assertEqual(first_head.in_features, 2 * 2)
        self.assertEqual(main_pooler.left_proj.out_features, 2)
        probe_param_ids = {id(param) for param in probe.parameters()}
        main_pooler_param_ids = {id(param) for param in main_pooler.parameters()}
        self.assertFalse(probe_param_ids & main_pooler_param_ids)
        self.assertFalse(main_pooler.state_dict().keys() <= probe.state_dict().keys())
        for name, param in main_pooler.state_dict().items():
            torch.testing.assert_close(param, original_state[name])


class PairwiseAlignmentTests(unittest.TestCase):
    def test_pairwise_alignment_uses_cosine_and_tanimoto(self):
        embeddings = np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        morgan = np.asarray(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=np.int32,
        )

        alignment = _compute_pairwise_similarity_alignment(
            embeddings=embeddings,
            morgan_bits=morgan,
            num_pairs=200,
            seed=3,
        )

        self.assertEqual(alignment.tanimoto.shape, (200,))
        self.assertEqual(alignment.cosine.shape, (200,))
        self.assertGreater(alignment.pearson, 0.99)
        self.assertTrue(set(np.unique(alignment.tanimoto)).issubset({0.0, 1.0}))

    def test_pairwise_alignment_can_use_prepared_pair_indices(self):
        alignment = _compute_pairwise_similarity_alignment_for_indices(
            embeddings=np.asarray(
                [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]],
                dtype=np.float32,
            ),
            tanimoto=np.asarray([0.9, 0.1, 0.8, 0.2], dtype=np.float32),
            left_idx=np.asarray([0, 0, 2, 2], dtype=np.int64),
            right_idx=np.asarray([1, 2, 3, 1], dtype=np.int64),
        )

        self.assertEqual(alignment.tanimoto.shape, (4,))
        self.assertEqual(alignment.cosine.shape, (4,))
        self.assertGreater(alignment.pearson, 0.9)

    def test_pairwise_alignment_plot_writes_png_and_pdf(self):
        alignment = _compute_pairwise_similarity_alignment(
            embeddings=np.asarray(
                [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]],
                dtype=np.float32,
            ),
            morgan_bits=np.asarray(
                [[1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1]],
                dtype=np.int32,
            ),
            num_pairs=64,
            seed=5,
        )

        with TemporaryDirectory() as tmpdir:
            png_path, pdf_path = _plot_pairwise_similarity_alignment(
                alignment,
                Path(tmpdir) / "alignment",
            )

            self.assertTrue(png_path.exists())
            self.assertTrue(pdf_path.exists())
            self.assertGreater(png_path.stat().st_size, 0)
            self.assertGreater(pdf_path.stat().st_size, 0)


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
            "probe_fluorine": torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32),
            "probe_sulfur": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
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
        self.assertNotIn("mol_weight", result["predictions"])
        self.assertNotIn("num_rings", result["predictions"])
        self.assertEqual(result["predictions"]["fluorine"].shape, (2,))
        self.assertEqual(result["predictions"]["sulfur"].shape, (2,))
        self.assertEqual(result["predictions"]["maccs"].shape, (2, 4))


class MsgProbeTaskSpecTests(unittest.TestCase):
    def test_task_spec_uses_one_fingerprint_head_for_maccs_bits(self):
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

        self.assertEqual(task_spec.regression_tasks, ())
        self.assertEqual(task_spec.binary_tasks, ("fluorine", "sulfur"))
        self.assertEqual(task_spec.maccs_bits, 4)
        self.assertEqual(_probe_task_names(task_spec), ("fluorine", "sulfur", "maccs"))
        self.assertEqual(_probe_task_output_dims(task_spec), {"maccs": 4})


class MsgProbeMetricTests(unittest.TestCase):
    def test_score_epoch_state_reports_maccs_metrics(self):
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
                "fluorine": torch.tensor([0.1, 0.9, 0.8]),
                "sulfur": torch.tensor([0.8, 0.1, 0.2]),
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
                "fluorine": torch.tensor([0.0, 1.0, 1.0]),
                "sulfur": torch.tensor([1.0, 0.0, 0.0]),
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
        self.assertNotIn("msg_probe/test/acc_num_rings_exact", metrics)
        self.assertNotIn("msg_probe/test/mae_num_rings", metrics)
        self.assertNotIn("msg_probe/test/r2_mean_wo_num_rings", metrics)
        self.assertNotIn("msg_probe/test/r2_mean", metrics)
        self.assertNotIn("msg_probe/test/mae_mean", metrics)
        self.assertEqual(metrics["msg_probe/test/num_maccs_auc_bits"], 4.0)
        self.assertGreater(metrics["msg_probe/test/auc_maccs_mean"], 0.9)
        self.assertEqual(
            metrics["msg_probe/test/num_maccs_average_precision_bits"], 4.0
        )
        self.assertGreater(
            metrics["msg_probe/test/average_precision_maccs_mean"], 0.9
        )
        self.assertEqual(metrics["msg_probe/test/num_maccs_recall_bits"], 4.0)
        self.assertGreater(metrics["msg_probe/test/recall_maccs_mean"], 0.9)
        self.assertEqual(metrics["msg_probe/test/num_maccs_precision_bits"], 4.0)
        self.assertGreater(metrics["msg_probe/test/precision_maccs_mean"], 0.9)
        self.assertAlmostEqual(metrics["msg_probe/test/tanimoto_maccs_mean"], 1.0)
        self.assertGreater(metrics["msg_probe/test/cosine_maccs_mean"], 0.95)
        self.assertGreater(metrics["msg_probe/test/auc_fluorine"], 0.9)
        self.assertGreater(metrics["msg_probe/test/auc_sulfur"], 0.9)

    def test_score_epoch_state_matches_per_bit_fingerprint_metrics(self):
        task_spec = MsgProbeTaskSpec(
            regression_tasks=(),
            maccs_bits=5,
            regression_means={},
            regression_stds={},
            fingerprint_task="maccs",
        )
        target = np.asarray(
            [
                [0, 0, 1, 0, 1],
                [1, 0, 1, 0, 0],
                [0, 0, 1, 1, 1],
                [1, 0, 1, 1, 0],
                [0, 0, 1, 0, 0],
                [1, 0, 1, 1, 1],
            ],
            dtype=np.float32,
        )
        pred = np.asarray(
            [
                [0.1, 0.2, 0.9, 0.2, 0.8],
                [0.9, 0.1, 0.7, 0.4, 0.3],
                [0.2, 0.4, 0.8, 0.7, 0.6],
                [0.8, 0.3, 0.6, 0.9, 0.2],
                [0.4, 0.2, 0.95, 0.1, 0.4],
                [0.7, 0.1, 0.85, 0.8, 0.7],
            ],
            dtype=np.float32,
        )
        epoch_state = _new_epoch_state(task_spec)
        _update_epoch_state(
            epoch_state,
            {
                "batch_size": int(target.shape[0]),
                "predictions": {"maccs": torch.from_numpy(pred)},
                "targets": {"maccs": torch.from_numpy(target)},
            },
            task_spec,
        )

        metrics = _score_epoch_state(
            prefix="msg_probe/test",
            epoch_state=epoch_state,
            task_spec=task_spec,
        )

        auc_values = []
        average_precision_values = []
        recall_values = []
        precision_values = []
        tanimoto_values = []
        cosine_values = []
        for bit_idx in range(target.shape[1]):
            bit_target = target[:, bit_idx]
            bit_pred = pred[:, bit_idx] >= 0.5
            if np.unique(bit_target).size < 2:
                if np.count_nonzero(bit_target) > 0:
                    recall_values.append(float(bit_pred[bit_target == 1].mean()))
                    precision_values.append(
                        float(bit_target[bit_pred].mean())
                        if np.count_nonzero(bit_pred) > 0
                        else 0.0
                    )
                continue
            auc_values.append(float(roc_auc_score(bit_target, pred[:, bit_idx])))
            average_precision_values.append(
                float(average_precision_score(bit_target, pred[:, bit_idx]))
            )
            recall_values.append(float(bit_pred[bit_target == 1].mean()))
            precision_values.append(
                float(bit_target[bit_pred].mean())
                if np.count_nonzero(bit_pred) > 0
                else 0.0
            )
        bit_pred = pred >= 0.5
        for sample_idx in range(target.shape[0]):
            intersection = np.count_nonzero(bit_pred[sample_idx] & (target[sample_idx] > 0))
            union = np.count_nonzero(bit_pred[sample_idx] | (target[sample_idx] > 0))
            tanimoto_values.append(intersection / max(union, 1))
            denominator = float(
                np.linalg.norm(pred[sample_idx]) * np.linalg.norm(target[sample_idx])
            )
            cosine_values.append(
                float(np.dot(pred[sample_idx], target[sample_idx]) / max(denominator, 1e-12))
            )

        self.assertEqual(metrics["msg_probe/test/num_maccs_auc_bits"], 3.0)
        self.assertEqual(metrics["msg_probe/test/num_maccs_recall_bits"], 4.0)
        self.assertAlmostEqual(
            metrics["msg_probe/test/auc_maccs_mean"], float(np.mean(auc_values))
        )
        self.assertAlmostEqual(
            metrics["msg_probe/test/average_precision_maccs_mean"],
            float(np.mean(average_precision_values)),
        )
        self.assertAlmostEqual(
            metrics["msg_probe/test/recall_maccs_mean"], float(np.mean(recall_values))
        )
        self.assertAlmostEqual(
            metrics["msg_probe/test/precision_maccs_mean"], float(np.mean(precision_values))
        )
        self.assertAlmostEqual(
            metrics["msg_probe/test/tanimoto_maccs_mean"], float(np.mean(tanimoto_values))
        )
        self.assertAlmostEqual(
            metrics["msg_probe/test/cosine_maccs_mean"], float(np.mean(cosine_values))
        )

    def test_score_epoch_state_names_morgan_similarity_metrics(self):
        task_spec = MsgProbeTaskSpec(
            regression_tasks=(),
            maccs_bits=3,
            regression_means={},
            regression_stds={},
            fingerprint_task="morgan",
        )
        epoch_state = _new_epoch_state(task_spec)
        _update_epoch_state(
            epoch_state,
            {
                "batch_size": 2,
                "predictions": {
                    "morgan": torch.tensor(
                        [
                            [0.9, 0.1, 0.8],
                            [0.2, 0.7, 0.6],
                        ],
                        dtype=torch.float32,
                    )
                },
                "targets": {
                    "morgan": torch.tensor(
                        [
                            [1.0, 0.0, 1.0],
                            [0.0, 1.0, 1.0],
                        ],
                        dtype=torch.float32,
                    )
                },
            },
            task_spec,
        )

        metrics = _score_epoch_state(
            prefix="msg_probe/test",
            epoch_state=epoch_state,
            task_spec=task_spec,
        )

        self.assertIn("msg_probe/test/auc_morgan_mean", metrics)
        self.assertIn("msg_probe/test/average_precision_morgan_mean", metrics)
        self.assertEqual(metrics["msg_probe/test/tanimoto_morgan_mean"], 1.0)
        self.assertGreater(metrics["msg_probe/test/cosine_morgan_mean"], 0.95)
        self.assertNotIn("msg_probe/test/auc_maccs_mean", metrics)

    def test_select_metric_rejects_tune_metric(self):
        cfg = {
            "msg_probe_tune_metric": "msg_probe/test/mae_mol_weight",
        }
        with self.assertRaisesRegex(ValueError, "msg_probe_tune_metric"):
            resolve_msg_probe_select_metric(cfg)
        self.assertFalse(
            msg_probe_metric_higher_is_better("msg_probe/test/mae_mol_weight")
        )
        self.assertTrue(
            msg_probe_metric_higher_is_better("msg_probe/test/auc_maccs_mean")
        )

    def test_select_metric_rejects_probe_dataset(self):
        with self.assertRaisesRegex(ValueError, "probe_dataset"):
            resolve_msg_probe_select_metric({"probe_dataset": "nist-murcko"})

    def test_fingerprint_rejects_fingerprint_type_alias(self):
        cfg = {"msg_probe_fingerprint_type": "morgan"}

        with self.assertRaisesRegex(ValueError, "msg_probe_fingerprint_type"):
            resolve_msg_probe_fingerprint(cfg)


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
                    "probe_fluorine": np.asarray([0.0, 1.0], dtype=np.float32),
                    "probe_sulfur": np.asarray([1.0, 0.0], dtype=np.float32),
                    "probe_maccs": _maccs([[0, 1, 0, 1], [1, 0, 1, 0]]),
                },
                {
                    "peak_mz": np.zeros((2, 60), dtype=np.float32),
                    "probe_valid_mol": np.asarray([1, 1], dtype=np.int32),
                    "probe_mol_weight": np.asarray([30.0, 40.0], dtype=np.float32),
                    "probe_logp": np.asarray([3.0, 4.0], dtype=np.float32),
                    "probe_num_heavy_atoms": np.asarray([4.0, 5.0], dtype=np.float32),
                    "probe_num_rings": np.asarray([2.0, 3.0], dtype=np.float32),
                    "probe_fluorine": np.asarray([1.0, 0.0], dtype=np.float32),
                    "probe_sulfur": np.asarray([0.0, 1.0], dtype=np.float32),
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

        self.assertEqual(targets.regression, {})
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
        self.assertTrue(
            np.array_equal(
                targets.binary["fluorine"],
                np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            )
        )
        self.assertTrue(
            np.array_equal(
                targets.binary["sulfur"],
                np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
            )
        )

    def test_collect_split_targets_can_select_morgan_targets(self):
        dm = _DummyDataModule(
            batches=[
                {
                    "peak_mz": np.zeros((2, 60), dtype=np.float32),
                    "probe_valid_mol": np.asarray([1, 1], dtype=np.int32),
                    "probe_mol_weight": np.asarray([10.0, 20.0], dtype=np.float32),
                    "probe_logp": np.asarray([1.0, 2.0], dtype=np.float32),
                    "probe_num_heavy_atoms": np.asarray([2.0, 3.0], dtype=np.float32),
                    "probe_num_rings": np.asarray([0.0, 1.0], dtype=np.float32),
                    "probe_fluorine": np.asarray([0.0, 1.0], dtype=np.float32),
                    "probe_sulfur": np.asarray([1.0, 0.0], dtype=np.float32),
                    "probe_maccs": _maccs([[0, 1], [1, 0]]),
                    "probe_morgan": _maccs([[1, 1, 0], [0, 1, 1]]),
                },
            ],
            info={
                "massspec_train_size": 2,
                "massspec_val_size": 0,
                "massspec_test_size": 0,
            },
            batch_size=2,
        )

        targets = _collect_split_targets(
            probe_data=dm,
            split="massspec_train",
            peak_ordering="intensity",
            seed=0,
            fingerprint_task="morgan",
        )

        np.testing.assert_array_equal(
            targets.maccs,
            _maccs([[1, 1, 0], [0, 1, 1]]),
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

    def test_probe_steps_per_epoch_treats_batch_size_as_global_under_ddp(self):
        dm = _DummyDataModule(
            batches=[],
            info={
                "massspec_train_size": 16,
                "massspec_val_size": 0,
                "massspec_test_size": 0,
            },
            batch_size=8,
        )

        self.assertEqual(local_batch_size(dm.batch_size, 4), 2)
        self.assertEqual(
            probe_steps_per_epoch(
                dm,
                split="massspec_train",
                drop_remainder=False,
                distributed_world_size=4,
            ),
            2,
        )


class ProbeConfigTests(unittest.TestCase):
    def test_nist_murcko_probe_repeat_defaults_to_one(self):
        cfg = config_dict.ConfigDict()

        self.assertEqual(resolve_msg_probe_num_repeats(cfg), 1)

    def test_nist_murcko_probe_repeat_override_is_used(self):
        cfg = config_dict.ConfigDict()
        cfg.nist_murcko_probe_num_repeats = 3

        self.assertEqual(resolve_msg_probe_num_repeats(cfg), 3)

    def test_msg_probe_warmup_epochs_override_step_count(self):
        cfg = config_dict.ConfigDict()
        cfg.msg_probe_warmup_epochs = 0.5
        cfg.msg_probe_warmup_steps = 0

        self.assertEqual(_resolve_probe_warmup_steps(cfg, steps_per_epoch=118), 59)

    def test_msg_probe_warmup_steps_still_work(self):
        cfg = config_dict.ConfigDict()
        cfg.msg_probe_warmup_steps = 17

        self.assertEqual(_resolve_probe_warmup_steps(cfg, steps_per_epoch=118), 17)

    def test_msg_probe_fingerprint_defaults_to_maccs(self):
        cfg = config_dict.ConfigDict()

        self.assertEqual(resolve_msg_probe_fingerprint(cfg), "maccs")
        self.assertEqual(resolve_msg_probe_select_metric(cfg), "msg_probe/test/auc_fluorine")

    def test_msg_probe_fingerprint_can_select_morgan_metric(self):
        cfg = config_dict.ConfigDict()
        cfg.msg_probe_fingerprint = "morgan"

        self.assertEqual(resolve_msg_probe_fingerprint(cfg), "morgan")
        self.assertEqual(resolve_msg_probe_select_metric(cfg), "msg_probe/test/auc_fluorine")

    def test_pairwise_alignment_defaults_to_disabled(self):
        cfg = config_dict.ConfigDict()

        self.assertEqual(resolve_msg_probe_pairwise_alignment_num_pairs(cfg), 0)

    def test_pairwise_alignment_num_pairs_is_opt_in(self):
        cfg = config_dict.ConfigDict()
        cfg.msg_probe_pairwise_alignment_num_pairs = 20_000

        self.assertEqual(resolve_msg_probe_pairwise_alignment_num_pairs(cfg), 20_000)

    def test_morgan_probe_config_targets_4096_radius2_and_alignment(self):
        from configs.wandb_pa645zxs_morgan import get_config

        cfg = get_config()

        self.assertEqual(resolve_msg_probe_fingerprint(cfg), "morgan")
        self.assertEqual(
            resolve_msg_probe_select_metric(cfg),
            "msg_probe/test/auc_fluorine",
        )
        self.assertEqual(MORGAN_PROBE_FINGERPRINT_BITS, 4096)
        self.assertEqual(MORGAN_PROBE_FINGERPRINT_RADIUS, 2)
        self.assertEqual(resolve_msg_probe_pairwise_alignment_num_pairs(cfg), 20_000)


class MsgProbeRunTests(unittest.TestCase):
    @staticmethod
    def _probe_batch(scale: float) -> dict[str, np.ndarray]:
        peak_mz = np.asarray(
            [
                [0.1, 0.2, 0.3],
                [0.2, 0.4, 0.6],
                [0.3, 0.6, 0.9],
                [0.4, 0.8, 1.2],
            ],
            dtype=np.float32,
        )
        peak_intensity = peak_mz * scale
        return {
            "peak_mz": peak_mz,
            "peak_intensity": peak_intensity,
            "peak_valid_mask": np.ones_like(peak_mz, dtype=bool),
            "probe_valid_mol": np.ones(4, dtype=bool),
            "probe_mol_weight": np.asarray([10.0, 12.0, 14.0, 16.0], dtype=np.float32),
            "probe_logp": np.asarray([1.0, 1.5, 2.0, 2.5], dtype=np.float32),
            "probe_num_heavy_atoms": np.asarray([2.0, 3.0, 4.0, 5.0], dtype=np.float32),
            "probe_num_rings": np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
            "probe_fluorine": np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
            "probe_sulfur": np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
            "probe_maccs": _maccs(
                [
                    [0, 1, 0, 1],
                    [1, 0, 1, 0],
                    [0, 1, 1, 0],
                    [1, 0, 0, 1],
                ]
            ),
        }

    def test_validation_selected_msg_probe_evaluates_test_once_after_selection(self):
        cfg = config_dict.ConfigDict()
        cfg.seed = 11
        cfg.model_dim = 4
        cfg.msg_probe_variants = ("mean",)
        cfg.msg_probe_mlp_hidden_dim = 8
        cfg.msg_probe_num_epochs = 2
        cfg.msg_probe_learning_rate = 1e-3
        cfg.msg_probe_weight_decay = 0.0
        cfg.msg_probe_warmup_steps = 0
        cfg.msg_probe_early_stopping = True
        cfg.msg_probe_early_stopping_patience = 10
        cfg.msg_probe_pairwise_alignment_num_pairs = 0

        probe_data = _SplitDummyDataModule(
            batches_by_split={
                "massspec_train": [self._probe_batch(1.0)],
                "massspec_val": [self._probe_batch(1.5)],
                "massspec_test": [self._probe_batch(2.0)],
                "massspec_mcebio_test": [self._probe_batch(2.5)],
            },
            info={
                "massspec_train_size": 4,
                "massspec_val_size": 4,
                "massspec_test_size": 4,
                "massspec_mcebio_test_size": 4,
                "probe_morgan_bits": 0,
            },
            batch_size=4,
        )

        class DummyEncoder(torch.nn.Module):
            def forward(
                self,
                peak_mz,
                peak_intensity,
                *,
                valid_mask,
                precursor_mz=None,
            ):
                values = peak_mz + peak_intensity
                return values.unsqueeze(-1).repeat(1, 1, cfg.model_dim)

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = DummyEncoder()

        curve: list[dict[str, float]] = []
        with mock.patch(
            "spectra_learning.probes.massspec.msg_probe.MassSpecProbeData.from_config",
            return_value=probe_data,
        ):
            metrics = _run_msg_probe_once(
                config=cfg,
                model=DummyModel(),
                device=torch.device("cpu"),
                on_epoch_end=curve.append,
            )

        test_calls = [
            call for call in probe_data.calls if call["split"] == "massspec_test"
        ]
        mcebio_test_calls = [
            call
            for call in probe_data.calls
            if call["split"] == "massspec_mcebio_test"
        ]
        self.assertEqual(len(test_calls), 1)
        self.assertEqual(len(mcebio_test_calls), 1)
        self.assertIn("msg_probe/mean/test/auc_fluorine", metrics)
        self.assertIn("msg_probe/mean/test/auc_sulfur", metrics)
        self.assertIsInstance(
            metrics["msg_probe/mean/test/pr_curve_fluorine"],
            PrecisionRecallCurve,
        )
        self.assertIsInstance(
            metrics["msg_probe/mean/test/pr_curve_sulfur"],
            PrecisionRecallCurve,
        )
        self.assertIn("msg_probe/mean/test/auc_maccs_mean", metrics)
        self.assertIn("msg_probe/mean/mcebio_sulfur_test/auc_sulfur", metrics)
        self.assertIsInstance(
            metrics["msg_probe/mean/mcebio_sulfur_test/pr_curve_sulfur"],
            PrecisionRecallCurve,
        )
        self.assertNotIn("msg_probe/mean/mcebio_sulfur_test/auc_fluorine", metrics)
        self.assertNotIn("msg_probe/mean/mcebio_sulfur_test/pr_curve_fluorine", metrics)
        self.assertIn("msg_probe/mean/val/auc_maccs_mean", metrics)
        self.assertIn("msg_probe/mean/val/auc_maccs_mean", curve[0])
        self.assertNotIn("msg_probe/mean/test/auc_maccs_mean", curve[0])
        self.assertNotIn("msg_probe/mean/test/pr_curve_fluorine", curve[0])


class RepeatedProbeTests(unittest.TestCase):
    @staticmethod
    def _msg_probe_metrics(*, auc: float, epoch: float) -> dict[str, float]:
        return {
            "msg_probe/mean/test/auc_maccs_mean": auc,
            "msg_probe/mean/epoch": epoch,
        }

    def test_run_msg_probe_averages_best_metrics_and_epoch_curves(self):
        cfg = config_dict.ConfigDict()
        cfg.seed = 7
        cfg.nist_murcko_probe_num_repeats = 2

        repeat_payloads = (
            (
                self._msg_probe_metrics(auc=0.6, epoch=2.0),
                [
                    self._msg_probe_metrics(auc=0.4, epoch=1.0),
                    self._msg_probe_metrics(auc=0.6, epoch=2.0),
                ],
            ),
            (
                self._msg_probe_metrics(auc=0.8, epoch=4.0),
                [
                    self._msg_probe_metrics(auc=0.5, epoch=1.0),
                    self._msg_probe_metrics(auc=0.9, epoch=2.0),
                ],
            ),
        )

        def fake_run_once(
            *,
            config,
            model,
            device,
            covariance_pooler,
            on_epoch_end,
            repeat_index,
            plot_dir,
            plot_step,
            distributed,
        ):
            metrics, curve = repeat_payloads[repeat_index]
            for epoch_metrics in curve:
                if on_epoch_end is not None:
                    on_epoch_end(dict(epoch_metrics))
            return dict(metrics)

        curve: list[dict[str, float]] = []
        with mock.patch(
            "spectra_learning.probes.massspec.msg_probe._run_msg_probe_once",
            side_effect=fake_run_once,
        ):
            metrics = run_msg_probe(
                config=cfg,
                model=mock.sentinel.model,
                device=torch.device("cpu"),
                on_epoch_end=curve.append,
            )

        self.assertAlmostEqual(metrics["msg_probe/mean/test/auc_maccs_mean"], 0.7)
        self.assertAlmostEqual(metrics["msg_probe/mean/epoch"], 3.0)
        self.assertAlmostEqual(metrics["msg_probe/repeats"], 2.0)
        self.assertNotIn("msg_probe/test/auc_maccs_mean", metrics)
        self.assertNotIn("msg_probe_epoch", metrics)
        self.assertEqual(len(curve), 2)
        self.assertAlmostEqual(curve[0]["msg_probe/mean/test/auc_maccs_mean"], 0.45)
        self.assertAlmostEqual(curve[0]["msg_probe/mean/epoch"], 1.0)
        self.assertAlmostEqual(curve[1]["msg_probe/mean/test/auc_maccs_mean"], 0.75)
        self.assertAlmostEqual(curve[1]["msg_probe/mean/epoch"], 2.0)


if __name__ == "__main__":
    unittest.main()
