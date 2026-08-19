from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from flax import nnx
from ml_collections import config_dict

from spectra_learning.data.gems.artifacts import MASSIVE_V2_HDF5_FORMAT
from spectra_learning.data.gems.grouped import GroupedGemsDataModule
from spectra_learning.models.grouped_jepa_jax import GroupedSpectrumJEPAJax
from spectra_learning.models.settings import PeakSetJEPASettings
from spectra_learning.training.checkpointing_jax import restore_jax_encoder_state
from spectra_learning.training.pretrain_jax import (
    JaxTrainingTask,
    jax_config_checkpoint_contract,
    train_and_evaluate_jax_task,
)


def train_grouped_jepa_jax(
    config: config_dict.ConfigDict,
    workdir: str | Path,
) -> dict[str, object]:
    task = JaxTrainingTask(
        name="grouped_jepa",
        build_datamodule=_build_datamodule,
        build_model=_build_model,
        checkpoint_contract=_checkpoint_contract,
        initialize_model=_initialize_model,
        run_metadata=_run_metadata,
        log_start=_log_start,
    )
    return train_and_evaluate_jax_task(config, workdir, task=task)


def _build_datamodule(
    config: config_dict.ConfigDict,
    process_count: int,
    process_index: int,
) -> GroupedGemsDataModule:
    return GroupedGemsDataModule(
        config,
        seed=int(config.seed),
        distributed_world_size=process_count,
        distributed_rank=process_index,
        distributed_local_rank=0,
    )


def _build_model(
    config: config_dict.ConfigDict,
    _datamodule: GroupedGemsDataModule,
) -> GroupedSpectrumJEPAJax:
    settings = PeakSetJEPASettings.from_config(config)
    use_ema_teacher = bool(config.get("group_jepa_use_ema_teacher", True))
    return GroupedSpectrumJEPAJax(
        settings,
        teacher_spectra_per_group=int(config.group_jepa_teacher_spectra_per_group),
        ema_momentum=(
            float(config.group_jepa_ema_momentum) if use_ema_teacher else None
        ),
        invariance_loss_weight=float(
            config.get("group_jepa_invariance_loss_weight", 1.0)
        ),
        visreg_loss_weight=float(config.get("group_jepa_visreg_loss_weight", 0.0)),
        visreg_num_projections=int(
            config.get("group_jepa_visreg_num_projections", 0)
        ),
        visreg_center_weight=float(
            config.get("group_jepa_visreg_center_weight", 1.0)
        ),
        visreg_scale_weight=float(
            config.get("group_jepa_visreg_scale_weight", 1.0)
        ),
        visreg_shape_weight=float(
            config.get("group_jepa_visreg_shape_weight", 1.0)
        ),
        visreg_data_axis_name=(
            "data"
            if bool(config.get("group_jepa_visreg_gather_embeddings", False))
            else None
        ),
        rngs=nnx.Rngs(int(config.seed)),
    )


def _initialize_model(
    config: config_dict.ConfigDict,
    model: GroupedSpectrumJEPAJax,
) -> None:
    checkpoint = config.group_jepa_init_checkpoint_path
    encoder_state = nnx.as_pure(nnx.state(model.encoder, nnx.Param))
    restored = restore_jax_encoder_state(checkpoint, encoder_state)
    nnx.update(model.encoder, restored)
    if model.teacher_encoder is not None:
        nnx.update(model.teacher_encoder, restored)
    logging.info(
        "Initialized grouped JEPA encoder%s from %s",
        " and EMA teacher" if model.teacher_encoder is not None else "",
        checkpoint,
    )


def _checkpoint_contract(
    config: config_dict.ConfigDict,
    datamodule: GroupedGemsDataModule,
    _total_steps: int,
) -> dict[str, Any]:
    contract = jax_config_checkpoint_contract(config)
    assert datamodule.artifact.format == MASSIVE_V2_HDF5_FORMAT
    contract["dataset"] = {
        "format": datamodule.info["gems_hdf5_format"],
        "repo_id": datamodule.info["gems_hdf5_repo_id"],
        "revision": datamodule.info["gems_hdf5_revision"],
        "manifest_sha256": datamodule.info["gems_manifest_sha256"],
        "shard_plan_sha256": datamodule.info["gems_shard_plan_sha256"],
        "grouping": datamodule.info["gems_split"]["assigned_entity_key"],
        "spectra_per_group": datamodule.spectra_per_group,
        "teacher_spectra_per_group": datamodule.teacher_spectra_per_group,
    }
    return contract


def _run_metadata(datamodule: GroupedGemsDataModule) -> dict[str, object]:
    return {
        "run/groups_per_batch": float(datamodule.global_batch_size),
        "run/spectra_per_group": float(datamodule.spectra_per_group),
        "run/effective_spectra_per_step": float(
            datamodule.global_spectra_batch_size
        ),
        "run/train_groups": float(datamodule.info["train_groups"]),
        "run/validation_groups": float(datamodule.info["validation_groups"]),
    }


def _log_start(datamodule: GroupedGemsDataModule, total_steps: int) -> None:
    logging.info(
        "Grouped JEPA: %d train groups, %d validation groups, %d groups/step, "
        "%d spectra/group, %d optimizer steps.",
        datamodule.info["train_groups"],
        datamodule.info["validation_groups"],
        datamodule.global_batch_size,
        datamodule.spectra_per_group,
        total_steps,
    )
