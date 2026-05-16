import hashlib
import json
import logging
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch
from ml_collections import config_dict

from spectra_learning.models.model import PeakSetSIGReg
from spectra_learning.training.checkpointing import save_probe_checkpoint
from spectra_learning.training.logging import _config_to_wandb_dict


ModalProbeSubmitter = Callable[
    [str, Path, Path, int],
    str | None,
]

log = logging.getLogger(__name__)


def msg_probe_backend(config: config_dict.ConfigDict) -> str:
    return str(
        config.get(
            "msg_probe_backend",
            config.get("msg_probe_run_mode", "inline"),
        )
    ).lower()


def should_run_msg_probe_on_modal(config: config_dict.ConfigDict) -> bool:
    return msg_probe_backend(config) == "modal"


def save_and_submit_modal_msg_probe(
    *,
    config: config_dict.ConfigDict,
    model: PeakSetSIGReg,
    covariance_pooler: torch.nn.Module | None,
    checkpoint_dir: Path,
    workdir: Path,
    global_step: int,
    epoch: int,
    loss: float,
    wandb_run_id: str | None = None,
    submitter: ModalProbeSubmitter | None = None,
) -> dict[str, object]:
    checkpoint_path = checkpoint_dir / f"modal-probe-step-{global_step:08d}.pt"
    save_probe_checkpoint(
        checkpoint_path,
        model,
        global_step,
        epoch,
        loss,
        wandb_run_id,
        covariance_pooler=covariance_pooler,
    )

    config_json = json.dumps(_config_to_wandb_dict(config), sort_keys=True)
    config_json_path = checkpoint_path.with_suffix(".config.json")
    config_json_path.write_text(config_json)

    checkpoint_in_volume = _path_is_modal_volume(checkpoint_path, config)
    if submitter is not None:
        call_id = submitter(config_json, checkpoint_path, workdir, global_step)
    elif checkpoint_in_volume:
        call_id = _spawn_modal_probe_from_volume(
            config_json=config_json,
            checkpoint_path=checkpoint_path,
            workdir=workdir,
            global_step=global_step,
        )
    else:
        call_id = _spawn_modal_probe_submission_process(
            config=config,
            config_json_path=config_json_path,
            checkpoint_path=checkpoint_path,
            workdir=workdir,
            global_step=global_step,
        )

    submission_path = checkpoint_path.with_suffix(".submission.json")
    submission_path.write_text(
        json.dumps(
            {
                "call_id": call_id,
                "checkpoint_in_volume": checkpoint_in_volume,
                "checkpoint_path": str(checkpoint_path),
                "global_step": global_step,
            },
            indent=2,
            sort_keys=True,
        )
    )
    log.info(
        "Submitted Modal MSG probe for global_step=%d checkpoint=%s call_id=%s",
        global_step,
        checkpoint_path,
        call_id,
    )
    return {
        "msg_probe/modal/submitted": 1.0,
        "msg_probe/modal/checkpoint_in_volume": float(checkpoint_in_volume),
        "msg_probe/modal/global_step": float(global_step),
        "msg_probe/modal/call_id": str(call_id),
    }


def _path_is_modal_volume(path: Path, config: config_dict.ConfigDict) -> bool:
    volume_mount = Path(str(config.get("modal_probe_volume_mount_path", "/vol")))
    absolute_path = path.expanduser().absolute()
    return absolute_path == volume_mount or absolute_path.is_relative_to(volume_mount)


def _spawn_modal_probe_from_volume(
    *,
    config_json: str,
    checkpoint_path: Path,
    workdir: Path,
    global_step: int,
) -> str:
    import modal_train

    modal_train.volume.commit()
    handle = cast(Any, modal_train.run_probe_checkpoint).spawn(
        config_json=config_json,
        checkpoint_path=str(checkpoint_path),
        workdir=str(workdir),
        global_step=global_step,
    )
    return str(handle.object_id)


def _spawn_modal_probe_submission_process(
    *,
    config: config_dict.ConfigDict,
    config_json_path: Path,
    checkpoint_path: Path,
    workdir: Path,
    global_step: int,
) -> str:
    project_root = Path(__file__).resolve().parents[2]
    modal_cli = str(
        config.get(
            "modal_probe_cli",
            Path(sys.executable).with_name("modal"),
        )
    )
    modal_entrypoint = str(config.get("modal_probe_entrypoint", "modal_train.py"))
    log_path = checkpoint_path.with_suffix(".submit.log")
    command = [
        modal_cli,
        "run",
        "--detach",
        modal_entrypoint,
        "--submit-probe-config-json-path",
        str(config_json_path),
        "--submit-probe-checkpoint-path",
        str(checkpoint_path),
        "--submit-probe-workdir",
        str(workdir),
        "--submit-probe-global-step",
        str(global_step),
    ]
    stdout = log_path.open("ab")
    process = subprocess.Popen(
        command,
        cwd=project_root,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    stdout.close()
    return f"local-submit-pid-{process.pid}"


def modal_probe_remote_label(path: str | Path, global_step: int) -> str:
    source = str(Path(path).expanduser().resolve())
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:10]
    name = Path(source).name or "run"
    return f"{name}-{digest}/step-{global_step:08d}"
