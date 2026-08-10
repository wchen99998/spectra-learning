from __future__ import annotations

import argparse
import itertools

import jax
import numpy as np
import torch
from ml_collections import config_dict

from spectra_learning.data.gems.collate import GemsBatchCollator
from spectra_learning.models.model_jax import PeakSetJEPAJax
from spectra_learning.training.checkpointing_jax import (
    EmergencyCheckpointMonitor,
    build_jax_checkpoint_manager,
    jax_training_checkpoint_metadata,
)
from spectra_learning.training.logging import MetricLogger
from spectra_learning.training.pretrain_jax import (
    _run_jax_training_loop,
    initialize_jax_model_from_torch_seed,
)


_TINY_MODEL_KWARGS = {
    "training_mode": "mae",
    "model_dim": 4,
    "encoder_num_layers": 1,
    "encoder_num_heads": 1,
    "attention_mlp_multiple": 1.0,
    "feature_mlp_hidden_dim": 4,
    "encoder_fourier_num_freqs": 1,
    "pairmixer_fourier_num_freqs": 1,
    "pairmixer_pair_dim": 4,
    "pairmixer_pair_feature_hidden_dim": 4,
    "masked_latent_predictor_num_layers": 1,
    "masked_latent_predictor_num_heads": 1,
    "num_peaks": 3,
    "jepa_num_target_blocks": 1,
    "distogram_loss_weight": 0.0,
    "predictor_dropout": 0.0,
    "target_projector_dim": -1,
    "jepa_mae_mz_bin_size": 100.0,
    "jepa_mae_intensity_bin_size": 0.5,
}


class _TinyDataModule:
    def __init__(self, batch: dict[str, np.ndarray], train_steps: int) -> None:
        self.batch = batch
        self.train_steps = train_steps
        self.global_batch_size = int(batch["peak_mz"].shape[0])
        self.batch_size = self.global_batch_size
        self.gradient_accumulation_steps = 1

    def train_loader_for_epoch(self, epoch: int, start_batch: int = 0):
        del epoch
        return itertools.repeat(self.batch, self.train_steps - start_batch)

    def set_mask_fractions(
        self,
        context_fraction: float,
        target_fraction: float,
    ) -> None:
        del context_fraction, target_fraction


class _SmokeLogger(MetricLogger):
    def log_metrics(self, metrics, step=None) -> None:
        if "train/loss" in metrics:
            print(f"SMOKE_STEP step={step}", flush=True)


def _config() -> config_dict.ConfigDict:
    return config_dict.ConfigDict(
        {
            **_TINY_MODEL_KWARGS,
            "seed": 5,
            "num_epochs": 1,
            "learning_rate": 1e-3,
            "jax_mesh_devices": str(jax.device_count()),
            "checkpoint_every_steps": 0,
            "jax_enable_async_checkpointing": False,
            "log_every_n_steps": 100,
            "msg_probe_every_n_steps": -1,
        }
    )


def _batch() -> dict[str, np.ndarray]:
    torch.manual_seed(123)

    def sample(mz, intensity, precursor_mz, collision_energy, charge):
        spectra = np.zeros((2, 128), dtype=np.float32)
        spectra[0, : len(mz)] = np.asarray(mz, dtype=np.float32)
        spectra[1, : len(intensity)] = np.asarray(intensity, dtype=np.float32)
        return {
            "spectra": spectra,
            "precursor_mz_raw": np.asarray(precursor_mz, dtype=np.float32),
            "collision_energy": np.asarray(collision_energy, dtype=np.float32),
            "charge": np.asarray(charge, dtype=np.float32),
        }

    collator = GemsBatchCollator(
        augment=True,
        num_target_blocks=1,
        context_fraction=0.4,
        target_fraction=0.35,
        block_min_len=1,
        num_peaks=3,
        max_precursor_mz=1000.0,
        min_peak_intensity=0.0,
        peak_drop_min_intensity=0.0,
        peak_ordering="mz",
        precursor_peak_exclusion_window_da=0.0,
        mask_strategy="contiguous",
        mask_lengths=(1, 2, 3),
        mask_round_from=2,
        output_format="numpy",
    )
    samples = [
        sample([100.0, 125.0, 150.0], [1.0, 0.8, 0.4], 500.0, 20.0, 1.0),
        sample([220.0, 240.0, 300.0], [0.9, 0.3, 0.2], 620.0, 40.0, 2.0),
    ]
    return collator(samples * 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args()

    config = _config()
    manager = build_jax_checkpoint_manager(
        args.workdir,
        max_to_keep=2,
        enable_async_checkpointing=False,
    )
    restored_step = manager.latest_step()
    print(f"SMOKE_START restored_step={restored_step}", flush=True)

    model = PeakSetJEPAJax(**_TINY_MODEL_KWARGS)
    if restored_step is None:
        initialize_jax_model_from_torch_seed(config, model)
    total_steps = 1_000_000 if restored_step is None else restored_step + 3
    datamodule = _TinyDataModule(_batch(), train_steps=total_steps)
    metadata = jax_training_checkpoint_metadata(
        "emergency_checkpoint_smoke",
        {"model": "tiny_peakset_v1"},
    )

    with EmergencyCheckpointMonitor.for_current_environment() as monitor:
        metrics = _run_jax_training_loop(
            config=config,
            datamodule=datamodule,
            model=model,
            logger=_SmokeLogger(),
            total_steps=total_steps,
            checkpoint_manager=manager,
            resume_step=restored_step,
            checkpoint_metadata=metadata,
            metric_reduction="mean",
            enable_msg_probe=False,
            emergency_checkpoint=monitor,
        )
    manager.close()
    final_step = int(metrics["run/final_global_step"])
    assert restored_step is not None
    assert final_step == restored_step + 3
    print(
        f"SMOKE_RECOVERY_SUCCEEDED restored_step={restored_step} "
        f"final_step={final_step}",
        flush=True,
    )


if __name__ == "__main__":
    main()
