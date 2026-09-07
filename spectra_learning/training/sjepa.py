from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from flax import nnx
from ml_collections import config_dict

from spectra_learning.data.gems.artifacts import MASSIVE_V2_HDF5_FORMAT
from spectra_learning.data.gems.datamodule import GemsDataModule
from spectra_learning.models.settings import PeakSetJEPASettings
from spectra_learning.models.sjepa_jax import SJEPASpectrumJax
from spectra_learning.training.pretrain_jax import (
    JaxTrainingTask,
    jax_config_checkpoint_contract,
    train_and_evaluate_jax_task,
)


def train_sjepa_jax(
    config: config_dict.ConfigDict,
    workdir: str | Path,
) -> dict[str, object]:
    task = JaxTrainingTask(
        name="sjepa",
        build_datamodule=_build_datamodule,
        build_model=_build_model,
        checkpoint_contract=_checkpoint_contract,
        run_metadata=_run_metadata,
        log_start=_log_start,
    )
    return train_and_evaluate_jax_task(config, workdir, task=task)


def _build_datamodule(
    config: config_dict.ConfigDict,
    process_count: int,
    process_index: int,
) -> GemsDataModule:
    return GemsDataModule(
        config,
        seed=int(config.seed),
        distributed_world_size=process_count,
        distributed_rank=process_index,
        distributed_local_rank=0,
    )


def _build_model(
    config: config_dict.ConfigDict,
    _datamodule: GemsDataModule,
) -> SJEPASpectrumJax:
    return SJEPASpectrumJax(
        PeakSetJEPASettings.from_config(config),
        num_global_views=int(config.sjepa_num_global_views),
        num_local_views=int(config.sjepa_num_local_views),
        global_view_fraction=float(config.sjepa_global_view_fraction),
        local_view_fraction=float(config.sjepa_local_view_fraction),
        invariance_loss_weight=float(
            config.get("sjepa_invariance_loss_weight", 1.0)
        ),
        sigreg_loss_weight=float(config.sjepa_sigreg_loss_weight),
        teacher_stop_gradient=bool(
            config.get("sjepa_teacher_stop_gradient", True)
        ),
        sigreg_num_slices=int(config.sjepa_sigreg_num_slices),
        sigreg_data_axis_name=(
            "data" if bool(config.sjepa_sigreg_gather_embeddings) else None
        ),
        sigreg_seed=int(config.seed),
        rngs=nnx.Rngs(int(config.seed)),
    )


def _checkpoint_contract(
    config: config_dict.ConfigDict,
    datamodule: GemsDataModule,
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
    }
    return contract


def _run_metadata(datamodule: GemsDataModule) -> dict[str, object]:
    num_views = datamodule.sjepa_num_global_views + datamodule.sjepa_num_local_views
    return {
        "run/groups_per_optimizer_step": float(datamodule.global_batch_size),
        "run/views_per_spectrum": float(num_views),
        "run/effective_view_batch_size": float(
            datamodule.global_batch_size * num_views
        ),
        "run/global_views_per_spectrum": float(
            datamodule.sjepa_num_global_views
        ),
        "run/local_views_per_spectrum": float(datamodule.sjepa_num_local_views),
    }


def _log_start(datamodule: GemsDataModule, total_steps: int) -> None:
    num_views = datamodule.sjepa_num_global_views + datamodule.sjepa_num_local_views
    logging.info(
        "SJEPA: %d groups/optimizer step, %d views/optimizer step, "
        "%d global view(s) at %.0f%%, %d local view(s) at %.0f%%, "
        "%d optimizer steps.",
        datamodule.global_batch_size,
        datamodule.global_batch_size * num_views,
        datamodule.sjepa_num_global_views,
        100 * datamodule.sjepa_global_view_fraction,
        datamodule.sjepa_num_local_views,
        100 * datamodule.sjepa_local_view_fraction,
        total_steps,
    )
