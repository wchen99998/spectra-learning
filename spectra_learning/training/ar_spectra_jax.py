from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ml_collections import config_dict

from spectra_learning.data.ar_spectra import SpectraARGemsDataModule
from spectra_learning.models.ar_spectra_jax import build_spectra_ar_model_jax_from_config
from spectra_learning.training.pretrain_jax import (
    JaxTrainingTask,
    jax_config_checkpoint_contract,
    train_and_evaluate_jax_task,
)


def train_and_evaluate_ar_spectra_jax(
    config: config_dict.ConfigDict,
    workdir: str | Path,
) -> dict[str, object]:
    task = JaxTrainingTask(
        name="ar_spectra",
        build_datamodule=_build_ar_jax_datamodule,
        build_model=lambda task_config, datamodule: (
            build_spectra_ar_model_jax_from_config(
                task_config,
                datamodule.ar_tokenizer,
            )
        ),
        metric_reduction="token_weighted",
        run_metadata=_ar_jax_run_metadata,
        checkpoint_contract=_ar_jax_checkpoint_contract,
        log_start=_log_ar_jax_start,
    )
    return train_and_evaluate_jax_task(config, workdir, task=task)


def _build_ar_jax_datamodule(
    config: config_dict.ConfigDict,
    process_count: int,
    process_index: int,
) -> SpectraARGemsDataModule:
    return SpectraARGemsDataModule(
        config,
        seed=int(config.seed),
        distributed_world_size=process_count,
        distributed_rank=process_index,
        distributed_local_rank=0,
    )


def _ar_jax_run_metadata(datamodule: SpectraARGemsDataModule) -> dict[str, object]:
    tokenizer = datamodule.ar_tokenizer
    return {
        "run/ar_vocab_size": float(tokenizer.vocab_size),
        "run/ar_sequence_length": float(tokenizer.sequence_length),
        "run/ar_prefix_length": float(tokenizer.prefix_length),
        "run/ar_tokens_per_peak": float(tokenizer.tokens_per_peak),
    }


def _ar_jax_checkpoint_contract(
    config: config_dict.ConfigDict,
    datamodule: SpectraARGemsDataModule,
    total_steps: int,
) -> dict[str, Any]:
    tokenizer_config = asdict(datamodule.ar_tokenizer_config)
    tokenizer_config["mz_bin_widths"] = list(tokenizer_config["mz_bin_widths"])
    return {
        **jax_config_checkpoint_contract(config),
        "tokenizer": tokenizer_config,
        "dataset": {
            "repo_id": str(config.gems_hdf5_repo_id),
            "revision": str(config.gems_hdf5_revision),
            "manifest": str(config.gems_hdf5_manifest),
            "spectrum_dataset": str(config.gems_hdf5_spectrum_dataset),
            "precursor_dataset": str(config.gems_hdf5_precursor_dataset),
            "rows_per_block": int(config.gems_hdf5_rows_per_block),
        },
        "preprocessing": {
            "num_peaks": int(datamodule.num_peaks_output),
            "max_precursor_mz": float(datamodule.max_precursor_mz),
            "min_peak_intensity": float(datamodule.min_peak_intensity),
            "peak_drop_min_intensity": float(datamodule.peak_drop_min_intensity),
            "peak_filtering": str(datamodule.peak_filtering),
            "grouped_peak_shoulder_da": float(datamodule.grouped_peak_shoulder_da),
            "grouped_peak_isotope_charges": list(
                datamodule.grouped_peak_isotope_charges
            ),
            "peak_ordering": str(datamodule.peak_ordering),
            "precursor_peak_exclusion_window_da": float(
                datamodule.precursor_peak_exclusion_window_da
            ),
        },
        "model": {
            "model_dim": int(config.ar_model_dim),
            "num_layers": int(config.ar_num_layers),
            "num_heads": int(config.ar_num_heads),
            "mlp_multiple": float(config.ar_mlp_multiple),
            "rope_base": float(config.ar_rope_base),
            "attention_kernel": str(config.ar_attention_kernel),
            "attention_block_size": int(config.ar_attention_block_size),
            "gelu_approximation": str(config.ar_gelu_approximation),
            "compute_dtype": str(config.autocast_dtype),
        },
        "optimizer": {
            "name": str(config.optimizer),
            "learning_rate": float(config.learning_rate),
            "min_learning_rate": (
                None
                if config.min_learning_rate is None
                else float(config.min_learning_rate)
            ),
            "warmup_steps": int(config.warmup_steps),
            "weight_decay": float(config.weight_decay),
            "b1": float(config.b1),
            "b2": float(config.b2),
            "grad_clip_norm": float(config.grad_clip_norm),
        },
        "training": {
            "seed": int(config.seed),
            "num_epochs": float(config.num_epochs),
            "global_batch_size": int(datamodule.global_batch_size),
            "gradient_accumulation_steps": int(
                datamodule.gradient_accumulation_steps
            ),
            "drop_remainder": bool(datamodule.drop_remainder),
            "train_steps_per_epoch": int(datamodule.train_steps),
            "total_steps": int(total_steps),
        },
    }


def _log_ar_jax_start(
    datamodule: SpectraARGemsDataModule,
    _total_steps: int,
) -> None:
    logging.info("AR tokenizer: %s", asdict(datamodule.ar_tokenizer_config))
    logging.info("AR vocab size: %d", datamodule.ar_tokenizer.vocab_size)
    logging.info("AR sequence length: %d", datamodule.ar_tokenizer.sequence_length)
