import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import torch.nn.functional as F
from ml_collections import config_dict

import train
from spectra_learning.data import murcko as murcko_data
from spectra_learning.data.massspec_probe import massspec_source_cache_dir
from spectra_learning.data.massspec_targets import MACCS_FINGERPRINT_BITS
from spectra_learning.training import contrastive as contrastive_training
from spectra_learning.training.contrastive import (
    ContrastiveBatchCollator,
    ContrastiveOnlineBatchCollator,
    NistMurckoContrastivePairs,
    _load_contrastive_split,
    train_contrastive,
)
from spectra_learning.training.checkpointing import covariance_pooler_checkpoint_path


def test_train_routes_contrastive_task(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_train(config, workdir):
        calls.append((config, workdir))
        return {"run/training_task": "contrastive"}

    monkeypatch.setattr(contrastive_training, "train_contrastive", fake_train)
    config = {"training_task": "contrastive"}

    assert train._train(config, tmp_path)["run/training_task"] == "contrastive"
    assert calls == [(config, tmp_path)]


def test_train_rejects_jax_contrastive_backend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not support device_backend='jax'"):
        train._train(
            {
                "training_task": "contrastive",
                "device_backend": "jax",
            },
            tmp_path,
        )


@pytest.mark.parametrize(
    "removed_key",
    (
        "contrastive_model_learning_rate",
        "contrastive_pooler_learning_rate",
        "contrastive_online_probe_learning_rate",
        "contrastive_init_full_checkpoint_path",
    ),
)
def test_contrastive_training_rejects_removed_alternate_paths(
    removed_key: str,
    tmp_path: Path,
) -> None:
    config = config_dict.ConfigDict({removed_key: 1e-3})

    with pytest.raises(ValueError, match=f"{removed_key} has been removed"):
        train_contrastive(config, tmp_path)


def _write_split(root: Path, split: str, rows: list[tuple[str, float]]) -> None:
    n = len(rows)
    payload = {
        "spectrum_index": pa.array(list(range(n)), type=pa.int64()),
        "fold": pa.array([split] * n, type=pa.string()),
        "precursor_mz": pa.array([250.0 + idx for idx in range(n)], type=pa.float32()),
        "num_peaks": pa.array([3] * n, type=pa.int32()),
        "spectrum_mz": pa.array(
            [[100.0 + idx, 125.0 + idx, 150.0 + idx] for idx in range(n)],
            type=pa.list_(pa.float32()),
        ),
        "spectrum_intensity": pa.array([[1.0, 0.5, 0.25] for _ in range(n)], type=pa.list_(pa.float32())),
        "smiles": pa.array([smiles for smiles, _ in rows], type=pa.string()),
        "canonical_smiles": pa.array([smiles for smiles, _ in rows], type=pa.string()),
        "adduct": pa.array(["[M+H]+"] * n, type=pa.string()),
        "instrument_type": pa.array(["Q-TOF"] * n, type=pa.string()),
        "collision_energy": pa.array([ce for _, ce in rows], type=pa.float32()),
        "collision_energy_present": pa.array([1] * n, type=pa.int32()),
        "has_fluorine": pa.array([False] * n, type=pa.bool_()),
        "has_sulfur": pa.array([False] * n, type=pa.bool_()),
        "maccs_166": pa.FixedSizeListArray.from_arrays(
            pa.array(np.zeros(n * 166, dtype=np.int8), type=pa.int8()),
            166,
        ),
        "murcko_hist_key": pa.array(["{}"] * n, type=pa.string()),
        "murcko_hist_json": pa.array(["{}"] * n, type=pa.string()),
        "metadata_json": pa.array(["{}"] * n, type=pa.string()),
    }
    pq.write_table(pa.table(payload), root / f"{split}.parquet")


def _write_artifact(root: Path) -> None:
    root.mkdir(parents=True)
    rows_by_split = {
        "train": [("CCO", 10.0), ("CCO", 20.0), ("CCN", 15.0), ("CCN", 35.0)],
        "val": [("CCC", 10.0), ("CCC", 30.0), ("CCF", 20.0), ("CCF", 40.0)],
        "test": [("CCCl", 10.0), ("CCCl", 20.0)],
    }
    for split, rows in rows_by_split.items():
        _write_split(root, split, rows)
    metadata = {
        "metadata_version": murcko_data.NIST_MURCKO_METADATA_VERSION,
        "artifact_format": murcko_data.NIST_MURCKO_ARTIFACT_FORMAT,
        "storage_format": "parquet",
        "min_precursor_mz": 1.0,
        "max_precursor_mz": 1000.0,
        "num_peaks_input": 128,
        "adduct_vocab": {"[M+H]+": 0},
        "instrument_type_vocab": {"Q-TOF": 0},
        "dreams_dim": 0,
        "probe_maccs_bits": 166,
        "probe_morgan_bits": 4096,
        "probe_morgan_radius": 2,
        "pairwise_alignment_available": False,
        "pairwise_alignment_num_pairs": 0,
        "pairwise_alignment_num_endpoints": 0,
    }
    for split, rows in rows_by_split.items():
        metadata[f"{split}_files"] = [f"{split}.parquet"]
        metadata[f"{split}_lengths"] = [len(rows)]
        metadata[f"{split}_size"] = len(rows)
        metadata[f"{split}_positive"] = 0
    (root / "metadata.json").write_text(json.dumps(metadata))


def _config(artifact_dir: Path) -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()
    cfg.artifact_dir = str(artifact_dir)
    cfg.nist_murcko_probe_hf_subdir = murcko_data.NIST_MURCKO_PREPARED_SUBDIR
    cfg.nist_murcko_probe_repo_id = "unused/local"
    cfg.nist_murcko_probe_revision = "main"
    cfg.seed = 7
    cfg.training_mode = "contrastive"
    cfg.batch_size = 4
    cfg.contrastive_batch_size = 4
    cfg.contrastive_pairs_per_epoch = 2
    cfg.contrastive_val_pairs_per_epoch = 2
    cfg.contrastive_val_every_n_steps = 0
    cfg.training_max_steps = 1
    cfg.num_epochs = 1
    cfg.num_peaks = 4
    cfg.max_precursor_mz = 1000.0
    cfg.min_peak_intensity = 0.0
    cfg.peak_drop_min_intensity = 0.0
    cfg.precursor_peak_exclusion_window_da = 0.0
    cfg.peak_ordering = "mz"
    cfg.model_dim = 16
    cfg.encoder_num_layers = 1
    cfg.encoder_num_heads = 4
    cfg.encoder_use_position_embedding = False
    cfg.encoder_apply_final_norm = True
    cfg.encoder_mz_embedding = "fourier"
    cfg.encoder_fourier_num_freqs = 1
    cfg.encoder_fourier_mlp_hidden_dim = 16
    cfg.encoder_fourier_mlp_num_layers = 2
    cfg.feature_mlp_hidden_dim = 16
    cfg.pairmixer_pair_dim = 16
    cfg.pairmixer_pair_feature_hidden_dim = 16
    cfg.attention_mlp_multiple = 2.0
    cfg.masked_latent_predictor_num_layers = 1
    cfg.masked_latent_predictor_num_heads = 4
    cfg.jepa_num_target_blocks = 1
    cfg.predictor_dim = 16
    cfg.contrastive_covariance_dim = 4
    cfg.contrastive_online_probe_hidden_dim = 16
    cfg.contrastive_temperature = 0.2
    cfg.contrastive_loss_weight = 1.0
    cfg.online_probe_loss_weight = 1.0
    cfg.learning_rate = 1e-3
    cfg.min_learning_rate = 1e-4
    cfg.warmup_steps = 0
    cfg.weight_decay = 0.0
    cfg.b2 = 0.99
    cfg.optimizer = "adam"
    cfg.optimizer_fused = False
    cfg.grad_clip_norm = 1.0
    cfg.autocast_dtype = "none"
    cfg.compile_mode = "none"
    cfg.dataloader_num_workers = 0
    cfg.dataloader_pin_memory = False
    cfg.log_every_n_steps = 0
    cfg.checkpoint_every_steps = 1
    cfg.enable_wandb = False
    return cfg


def test_contrastive_pairs_use_same_compound_different_collision_energy(tmp_path: Path):
    root = tmp_path / "artifacts" / "nist_murcko_probe"
    _write_artifact(root)
    split = _load_contrastive_split([str(root / "train.parquet")], max_samples=None)
    dataset = NistMurckoContrastivePairs(split, pairs_per_epoch=16, seed=1)

    for idx in range(len(dataset)):
        pair = dataset[idx]
        assert split.smiles[pair["left_idx"]] == split.smiles[pair["right_idx"]]
        assert (
            split.collision_energy[pair["left_idx"]]
            != split.collision_energy[pair["right_idx"]]
        )


def test_dreams_triplets_use_mass_matched_different_compound_negative():
    split = contrastive_training.ContrastiveSplit(
        spectra=np.zeros((6, 2, 3), dtype=np.float32),
        precursor_mz=np.asarray(
            [100.00, 100.20, 100.03, 100.40, 200.00, 200.20],
            dtype=np.float32,
        ),
        smiles=np.asarray(["CCO", "CCO", "CCN", "CCN", "CCC", "CCC"]),
        collision_energy=np.asarray([10.0, 20.0, 10.0, 20.0, 10.0, 20.0]),
        collision_energy_present=np.ones(6, dtype=np.int32),
        probe_maccs=np.zeros((6, MACCS_FINGERPRINT_BITS), dtype=np.int8),
    )
    dataset = NistMurckoContrastivePairs(
        split,
        pairs_per_epoch=16,
        seed=3,
        negative_mass_tolerance_da=0.05,
    )

    for idx in range(len(dataset)):
        triplet = dataset[idx]
        assert split.smiles[triplet["left_idx"]] == split.smiles[triplet["right_idx"]]
        assert (
            split.collision_energy[triplet["left_idx"]]
            != split.collision_energy[triplet["right_idx"]]
        )
        assert (
            split.smiles[triplet["left_idx"]]
            != split.smiles[triplet["negative_idx"]]
        )
        assert (
            abs(
                split.precursor_mz[triplet["left_idx"]]
                - split.precursor_mz[triplet["negative_idx"]]
            )
            <= 0.05
        )


def test_contrastive_collators_normalize_collision_energy() -> None:
    split = contrastive_training.ContrastiveSplit(
        spectra=np.asarray(
            [
                [[100.0, 120.0, 0.0], [1.0, 0.5, 0.0]],
                [[110.0, 130.0, 0.0], [0.8, 0.4, 0.0]],
                [[140.0, 160.0, 0.0], [0.7, 0.3, 0.0]],
            ],
            dtype=np.float32,
        ),
        precursor_mz=np.asarray([500.0, 520.0, 540.0], dtype=np.float32),
        smiles=np.asarray(["CCO", "CCO", "CCN"]),
        collision_energy=np.asarray([20.0, 150.0, -5.0], dtype=np.float32),
        collision_energy_present=np.ones(3, dtype=np.int32),
        probe_maccs=np.zeros((3, MACCS_FINGERPRINT_BITS), dtype=np.int8),
    )
    collator_kwargs = {
        "num_peaks": 2,
        "max_precursor_mz": 1000.0,
        "min_peak_intensity": 0.0,
        "peak_drop_min_intensity": 0.0,
        "peak_ordering": "mz",
        "precursor_peak_exclusion_window_da": 0.0,
    }

    pair_batch = ContrastiveBatchCollator(split, **collator_kwargs)(
        [{"left_idx": 0, "right_idx": 1, "compound_id": 0}]
    )
    online_batch = ContrastiveOnlineBatchCollator(split, **collator_kwargs)([0, 2])

    torch.testing.assert_close(
        pair_batch["collision_energy"],
        torch.tensor([0.2, 1.0], dtype=torch.float32),
    )
    torch.testing.assert_close(
        pair_batch["charge"],
        torch.tensor([1.0, 1.0], dtype=torch.float32),
    )
    torch.testing.assert_close(
        online_batch["collision_energy"],
        torch.tensor([0.2, 0.0], dtype=torch.float32),
    )
    torch.testing.assert_close(
        online_batch["charge"],
        torch.tensor([1.0, 1.0], dtype=torch.float32),
    )


def test_contrastive_smoke_training_writes_frozen_pooler_checkpoint(tmp_path: Path):
    artifact_dir = tmp_path / "artifacts"
    source_dir = massspec_source_cache_dir(
        artifact_dir,
        "unused/local",
        "main",
    )
    _write_artifact(source_dir / murcko_data.NIST_MURCKO_PREPARED_SUBDIR)
    workdir = tmp_path / "work"

    results = train_contrastive(_config(artifact_dir), workdir)

    assert results["run/final_global_step"] == 1.0
    saved_config = json.loads((workdir / "config.json").read_text())
    assert saved_config["training_mode"] == "contrastive"
    last_path = workdir / "checkpoints" / "last.pt"
    pooler_path = covariance_pooler_checkpoint_path(last_path)
    assert last_path.exists()
    assert pooler_path.exists()
    ckpt = torch.load(last_path, map_location="cpu", weights_only=True)
    assert ckpt["training_mode"] == "contrastive"
    assert ckpt["covariance_pooler_checkpoint"] == pooler_path.name
    assert "contrastive_projector" not in ckpt


def test_contrastive_loss_uses_normalized_pooler_output(monkeypatch):
    pooled = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ],
        dtype=torch.float32,
    )
    captured = {}

    def fake_info_nce_loss(
        features: torch.Tensor,
        *,
        positive_index: torch.Tensor,
        compound_id: torch.Tensor,
        temperature: float,
        **kwargs,
    ):
        captured["features"] = features.detach().clone()
        return features.sum() * 0.0 + features.new_tensor(1.0), features.sum() * 0.0

    monkeypatch.setattr(
        contrastive_training,
        "info_nce_loss",
        fake_info_nce_loss,
    )

    class FakeEncoder(torch.nn.Module):
        def forward_with_pair(
            self,
            peak_mz,
            peak_intensity,
            *,
            valid_mask,
            precursor_mz,
            spectrum_metadata=None,
        ):
            batch_size, num_peaks = peak_mz.shape
            peak_embeddings = peak_mz.new_zeros(batch_size, num_peaks, 3)
            pair_embeddings = peak_mz.new_zeros(batch_size, num_peaks, num_peaks, 2)
            return peak_embeddings, pair_embeddings

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = FakeEncoder()

    class FixedPooler(torch.nn.Module):
        def forward(
            self,
            peak_embeddings,
            valid_mask,
            pair_embeddings,
        ):
            return pooled.to(device=peak_embeddings.device)

    module = contrastive_training.ContrastiveTrainingModule(
        model=FakeModel(),
        pooler=FixedPooler(),
        online_probe=contrastive_training.OnlineProbeHead(
            input_dim=pooled.shape[1],
            hidden_dim=8,
            output_dim=MACCS_FINGERPRINT_BITS,
        ),
        teacher_model=None,
        temperature=0.1,
        loss_type="info_nce",
        triplet_margin=0.2,
        triplet_negative_max_maccs_tanimoto=None,
        triplet_hard_fraction=None,
        info_nce_negatives=None,
        fingerprint_target_temperature=0.1,
        contrastive_loss_weight=1.0,
        online_probe_loss_weight=1.0,
        encoder_anchor_loss_weight=0.0,
    )
    batch = {
        "peak_mz": torch.zeros(4, 2),
        "peak_intensity": torch.zeros(4, 2),
        "peak_valid_mask": torch.ones(4, 2, dtype=torch.bool),
        "precursor_mz": torch.zeros(4),
        "positive_index": torch.tensor([1, 0, 3, 2]),
        "compound_id": torch.tensor([0, 0, 1, 1]),
        "probe_maccs": torch.zeros(4, MACCS_FINGERPRINT_BITS),
    }

    module(batch)

    torch.testing.assert_close(captured["features"], F.normalize(pooled, dim=-1))
    assert not hasattr(module, "projector")


def test_encoder_anchor_loss_compares_student_to_teacher_pooler_output():
    class ConstantEncoder(torch.nn.Module):
        def __init__(self, value: float) -> None:
            super().__init__()
            self.value = value

        def forward_with_pair(
            self,
            peak_mz,
            peak_intensity,
            *,
            valid_mask,
            precursor_mz,
            spectrum_metadata=None,
        ):
            batch_size, num_peaks = peak_mz.shape
            peak_embeddings = peak_mz.new_full((batch_size, num_peaks, 1), self.value)
            pair_embeddings = peak_mz.new_zeros(batch_size, num_peaks, num_peaks, 1)
            return peak_embeddings, pair_embeddings

    class FakeModel(torch.nn.Module):
        def __init__(self, value: float) -> None:
            super().__init__()
            self.encoder = ConstantEncoder(value)

    class FirstPeakPooler(torch.nn.Module):
        def forward(
            self,
            peak_embeddings,
            valid_mask,
            pair_embeddings,
        ):
            return peak_embeddings[:, 0, :].repeat(1, 4)

    module = contrastive_training.ContrastiveTrainingModule(
        model=FakeModel(1.0),
        pooler=FirstPeakPooler(),
        online_probe=contrastive_training.OnlineProbeHead(
            input_dim=4,
            hidden_dim=8,
            output_dim=MACCS_FINGERPRINT_BITS,
        ),
        teacher_model=FakeModel(-1.0),
        temperature=0.1,
        loss_type="dreams_triplet",
        triplet_margin=0.2,
        triplet_negative_max_maccs_tanimoto=None,
        triplet_hard_fraction=None,
        info_nce_negatives=None,
        fingerprint_target_temperature=0.1,
        contrastive_loss_weight=0.0,
        online_probe_loss_weight=0.0,
        encoder_anchor_loss_weight=0.5,
    )
    batch = {
        "peak_mz": torch.zeros(4, 2),
        "peak_intensity": torch.zeros(4, 2),
        "peak_valid_mask": torch.ones(4, 2, dtype=torch.bool),
        "precursor_mz": torch.zeros(4),
        "positive_index": torch.tensor([1, 0, 3, 2]),
        "compound_id": torch.tensor([0, 0, 1, 1]),
        "probe_maccs": torch.zeros(4, MACCS_FINGERPRINT_BITS),
    }

    result = module(batch)

    torch.testing.assert_close(result["encoder_anchor_loss"], torch.tensor(2.0))
    torch.testing.assert_close(result["loss"], torch.tensor(1.0))


def test_dreams_triplet_loss_uses_cosine_margin_against_negatives():
    features = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.8, 0.6],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )

    loss, accuracy = contrastive_training.dreams_triplet_loss(
        features,
        positive_index=torch.tensor([1, 0, 3, 2]),
        compound_id=torch.tensor([0, 0, 1, 1]),
        margin=0.2,
    )

    cosine = features @ features.T
    positive = cosine.gather(1, torch.tensor([[1], [0], [3], [2]]))
    negatives = torch.tensor(
        [
            [False, False, True, True],
            [False, False, True, True],
            [True, True, False, False],
            [True, True, False, False],
        ]
    )
    expected = torch.clamp_min(0.2 - positive + cosine, 0)[negatives].mean()
    expected_accuracy = (positive > cosine)[negatives].float().mean()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(accuracy, expected_accuracy)


def test_info_nce_loss_can_limit_negatives_to_positive_only():
    features = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.8, 0.6],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )

    loss, accuracy = contrastive_training.info_nce_loss(
        features,
        positive_index=torch.tensor([1, 0, 3, 2]),
        compound_id=torch.tensor([0, 0, 1, 1]),
        temperature=0.1,
        max_negatives=0,
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))
    torch.testing.assert_close(accuracy, torch.tensor(1.0))


def test_info_nce_loss_can_ignore_maccs_similar_negatives():
    features = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.8, 0.6],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    maccs = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    loss, accuracy = contrastive_training.info_nce_loss(
        features,
        positive_index=torch.tensor([1, 0, 3, 2]),
        compound_id=torch.tensor([0, 0, 1, 1]),
        temperature=0.1,
        anchor_maccs=maccs,
        candidate_maccs=maccs,
        negative_max_maccs_tanimoto=0.5,
    )

    logits = features @ features.T / 0.1
    allowed = torch.tensor(
        [
            [False, True, False, True],
            [True, False, False, True],
            [False, False, False, True],
            [True, True, True, False],
        ]
    )
    expected_logits = logits.masked_fill(~allowed, -torch.finfo(logits.dtype).max)
    expected_loss = F.cross_entropy(expected_logits, torch.tensor([1, 0, 3, 2]))
    expected_accuracy = (
        expected_logits.argmax(dim=1) == torch.tensor([1, 0, 3, 2])
    ).float().mean()
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(accuracy, expected_accuracy)


def test_dreams_triplet_loss_can_ignore_maccs_similar_negatives():
    features = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.8, 0.6],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    maccs = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    loss, accuracy = contrastive_training.dreams_triplet_loss(
        features,
        positive_index=torch.tensor([1, 0, 3, 2]),
        compound_id=torch.tensor([0, 0, 1, 1]),
        margin=0.2,
        anchor_maccs=maccs,
        candidate_maccs=maccs,
        negative_max_maccs_tanimoto=0.5,
    )

    cosine = features @ features.T
    positive = cosine.gather(1, torch.tensor([[1], [0], [3], [2]]))
    negatives = torch.tensor(
        [
            [False, False, False, True],
            [False, False, False, True],
            [False, False, False, False],
            [True, True, False, False],
        ]
    )
    expected = torch.clamp_min(0.2 - positive + cosine, 0)[negatives].mean()
    expected_accuracy = (positive > cosine)[negatives].float().mean()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(accuracy, expected_accuracy)


def test_dreams_triplet_loss_can_focus_hard_negatives():
    features = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.8, 0.6],
                [0.0, 1.0],
                [0.7, 0.7],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )

    full_loss, _ = contrastive_training.dreams_triplet_loss(
        features,
        positive_index=torch.tensor([1, 0, 3, 2]),
        compound_id=torch.tensor([0, 0, 1, 1]),
        margin=0.2,
    )
    hard_loss, _ = contrastive_training.dreams_triplet_loss(
        features,
        positive_index=torch.tensor([1, 0, 3, 2]),
        compound_id=torch.tensor([0, 0, 1, 1]),
        margin=0.2,
        hard_fraction=0.125,
    )

    cosine = features @ features.T
    positive = cosine.gather(1, torch.tensor([[1], [0], [3], [2]]))
    negatives = torch.tensor(
        [
            [False, False, True, True],
            [False, False, True, True],
            [True, True, False, False],
            [True, True, False, False],
        ]
    )
    triplet_losses = torch.clamp_min(0.2 - positive + cosine, 0)[negatives]
    torch.testing.assert_close(hard_loss, triplet_losses.max())
    assert hard_loss > full_loss


def test_explicit_dreams_triplet_loss_uses_supplied_negative():
    features = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.8, 0.6],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.7, 0.7],
                [-1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )

    loss, accuracy = contrastive_training.explicit_dreams_triplet_loss(
        features,
        anchor_index=torch.tensor([0, 3]),
        positive_index=torch.tensor([1, 4]),
        negative_index=torch.tensor([2, 5]),
        margin=0.1,
    )

    cosine = features @ features.T
    expected = torch.clamp_min(
        0.1
        - cosine[torch.tensor([0, 3]), torch.tensor([1, 4])]
        + cosine[torch.tensor([0, 3]), torch.tensor([2, 5])],
        0,
    ).mean()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(accuracy, torch.tensor(1.0))


def test_explicit_dreams_triplet_loss_can_focus_hard_negatives():
    features = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.8, 0.6],
                [1.0, 0.0],
                [0.7, 0.7],
                [0.95, 0.05],
                [-1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )

    full_loss, _ = contrastive_training.explicit_dreams_triplet_loss(
        features,
        anchor_index=torch.tensor([0, 2]),
        positive_index=torch.tensor([1, 3]),
        negative_index=torch.tensor([4, 5]),
        margin=0.2,
    )
    hard_loss, _ = contrastive_training.explicit_dreams_triplet_loss(
        features,
        anchor_index=torch.tensor([0, 2]),
        positive_index=torch.tensor([1, 3]),
        negative_index=torch.tensor([4, 5]),
        margin=0.2,
        hard_fraction=0.5,
    )

    cosine = features @ features.T
    triplet_losses = torch.clamp_min(
        0.2
        - cosine[torch.tensor([0, 2]), torch.tensor([1, 3])]
        + cosine[torch.tensor([0, 2]), torch.tensor([4, 5])],
        0,
    )
    torch.testing.assert_close(hard_loss, triplet_losses.max())
    assert hard_loss > full_loss


def test_explicit_dreams_triplet_loss_can_ignore_maccs_similar_negative():
    features = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.8, 0.6],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.7, 0.7],
                [-1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    maccs = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    loss, accuracy = contrastive_training.explicit_dreams_triplet_loss(
        features,
        anchor_index=torch.tensor([0, 3]),
        positive_index=torch.tensor([1, 4]),
        negative_index=torch.tensor([2, 5]),
        margin=0.1,
        candidate_maccs=maccs,
        negative_max_maccs_tanimoto=0.5,
    )

    cosine = features @ features.T
    expected = torch.clamp_min(0.1 - cosine[3, 4] + cosine[3, 5], 0)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(accuracy, torch.tensor(1.0))


def test_maccs_soft_contrastive_targets_fingerprint_neighbors():
    features = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    maccs = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    loss, accuracy = contrastive_training.maccs_soft_contrastive_loss(
        features,
        candidate_features=features,
        anchor_index=torch.arange(features.shape[0]),
        anchor_maccs=maccs,
        candidate_maccs=maccs,
        temperature=0.2,
        target_temperature=0.05,
    )

    assert torch.isfinite(loss)
    torch.testing.assert_close(accuracy, torch.tensor(1.0))
    torch.testing.assert_close(
        contrastive_training.maccs_tanimoto(maccs[:1], maccs)[0],
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
    )


def test_maccs_similarity_loss_matches_fingerprint_tanimoto():
    features = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.7, 0.7],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    maccs = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    loss, accuracy = contrastive_training.maccs_similarity_loss(
        features,
        candidate_features=features,
        anchor_index=torch.arange(features.shape[0]),
        anchor_maccs=maccs,
        candidate_maccs=maccs,
    )

    cosine = features @ features.T
    target = contrastive_training.maccs_tanimoto(maccs, maccs)
    mask = ~torch.eye(features.shape[0], dtype=torch.bool)
    torch.testing.assert_close(loss, F.mse_loss(cosine[mask], target[mask]))
    assert torch.isfinite(accuracy)


def test_maccs_centroid_auc_loss_prefers_bit_separated_features():
    maccs = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    good_features = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.0, 1.0],
                [0.2, 0.8],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    bad_features = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )

    good_loss, good_accuracy = contrastive_training.maccs_centroid_auc_loss(
        good_features,
        candidate_features=good_features,
        anchor_maccs=maccs,
        candidate_maccs=maccs,
        temperature=0.2,
    )
    bad_loss, bad_accuracy = contrastive_training.maccs_centroid_auc_loss(
        bad_features,
        candidate_features=bad_features,
        anchor_maccs=maccs,
        candidate_maccs=maccs,
        temperature=0.2,
    )

    assert good_loss < bad_loss
    assert good_accuracy > bad_accuracy


def test_pairwise_maccs_auc_loss_prefers_ranked_positives():
    target = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    good_logits = torch.tensor(
        [
            [2.0, -1.0],
            [1.5, -0.5],
            [-0.5, 1.5],
            [-1.0, 2.0],
        ]
    )
    bad_logits = -good_logits

    good_loss = contrastive_training.pairwise_maccs_auc_loss(good_logits, target)
    bad_loss = contrastive_training.pairwise_maccs_auc_loss(bad_logits, target)

    assert good_loss < bad_loss


def test_pairwise_maccs_auc_loss_can_focus_hard_pairs():
    target = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
    logits = torch.tensor([[2.0], [-1.0], [1.5], [-2.0]])

    full_loss = contrastive_training.pairwise_maccs_auc_loss(logits, target)
    hard_loss = contrastive_training.pairwise_maccs_auc_loss(
        logits,
        target,
        hard_fraction=0.25,
    )

    pair_losses = F.softplus(
        -(
            logits[target[:, 0] > 0.5, 0][:, None]
            - logits[target[:, 0] <= 0.5, 0][None, :]
        )
    ).flatten()
    torch.testing.assert_close(hard_loss, pair_losses.max())
    assert hard_loss > full_loss


def test_online_probe_auc_bce_weight_controls_auc_mix():
    logits = torch.zeros(4, MACCS_FINGERPRINT_BITS)
    logits[:, 0] = torch.tensor([2.0, 1.0, -1.0, -2.0])
    batch = {"probe_maccs": torch.zeros(4, MACCS_FINGERPRINT_BITS)}
    batch["probe_maccs"][:, 0] = torch.tensor([1.0, 1.0, 0.0, 0.0])

    loss, _, _ = contrastive_training.online_probe_loss(
        logits,
        batch,
        maccs_loss_type="auc_bce",
        auc_loss_weight=1.0,
    )
    expected = contrastive_training.pairwise_maccs_auc_loss(
        logits,
        batch["probe_maccs"],
    )
    torch.testing.assert_close(loss, expected)


def test_online_probe_loss_uses_only_maccs_targets():
    logits = torch.zeros(2, MACCS_FINGERPRINT_BITS)
    batch = {
        "probe_maccs": torch.zeros(2, MACCS_FINGERPRINT_BITS),
    }

    loss, maccs_bce, _ = contrastive_training.online_probe_loss(
        logits,
        batch,
        maccs_loss_weight=1.0,
    )
    torch.testing.assert_close(loss, maccs_bce)
