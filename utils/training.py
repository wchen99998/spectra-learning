import datetime
import importlib.util
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.loggers import CSVLogger
from ml_collections import config_dict

from models.model import PeakSetSIGReg


def load_config(path: str | Path) -> config_dict.ConfigDict:
    path = Path(path)
    spec = importlib.util.spec_from_file_location("experiment_config", path)
    assert spec is not None, f"Could not load module spec from {path}"
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.get_config()


def build_model_from_config(config: config_dict.ConfigDict) -> PeakSetSIGReg:
    return PeakSetSIGReg(
        model_dim=int(config.model_dim),
        encoder_num_layers=int(config.num_layers),
        encoder_num_heads=int(config.num_heads),
        encoder_num_kv_heads=config.get("num_kv_heads", None),
        attention_mlp_multiple=float(config.attention_mlp_multiple),
        feature_mlp_hidden_dim=int(config.get("feature_mlp_hidden_dim", 128)),
        encoder_use_rope=bool(config.get("encoder_use_rope", False)),
        masked_token_loss_weight=float(config.get("masked_token_loss_weight", 0.0)),
        masked_token_loss_type=str(config.get("masked_token_loss_type", "l1")),
        normalize_jepa_targets=bool(config.get("normalize_jepa_targets", False)),
        representation_regularizer=str(
            config.get("representation_regularizer", "sigreg")
        ),
        masked_latent_predictor_num_layers=int(
            config.get("masked_latent_predictor_num_layers", 2)
        ),
        sigreg_num_slices=int(config.get("sigreg_num_slices", 256)),
        sigreg_lambda=float(config.get("sigreg_lambda", 0.1)),
        sigreg_lambda_warmup_steps=int(config.get("sigreg_lambda_warmup_steps", 0)),
        gco_constraints=list(config.get("gco_constraints", ())),
        gco_alpha=float(config.get("gco_alpha", 0.99)),
        gco_eta=float(config.get("gco_eta", 1e-3)),
        gco_log_lambda_init=float(config.get("gco_log_lambda_init", -8.0)),
        gco_log_lambda_min=float(config.get("gco_log_lambda_min", -12.0)),
        gco_log_lambda_max=float(config.get("gco_log_lambda_max", 2.0)),
        jepa_num_target_blocks=int(config.get("jepa_num_target_blocks", 2)),
        use_ema_teacher_target=bool(config.get("use_ema_teacher_target", False)),
        teacher_ema_decay=float(config.get("teacher_ema_decay", 0.996)),
        teacher_ema_decay_start=float(config.get("teacher_ema_decay_start", 0.0)),
        teacher_ema_decay_warmup_steps=int(
            config.get("teacher_ema_decay_warmup_steps", 0)
        ),
        teacher_ema_update_every=int(config.get("teacher_ema_update_every", 1)),
        encoder_qk_norm=bool(config.get("encoder_qk_norm", False)),
        norm_type=str(config.get("norm_type", "rmsnorm")),
        use_precursor_token=bool(config.get("use_precursor_token", False)),
    )


def collect_and_log_param_metrics(model: torch.nn.Module) -> dict[str, float]:
    by_module: dict[str, list[int]] = {}
    total = trainable = 0
    for name, param in model.named_parameters():
        numel = int(param.numel())
        module_name = name.split(".", 1)[0]
        counts = by_module.setdefault(module_name, [0, 0])
        total += numel
        counts[0] += numel
        if param.requires_grad:
            trainable += numel
            counts[1] += numel
    logging.info(
        "Model parameters: total=%s trainable=%s non_trainable=%s",
        f"{total:,}",
        f"{trainable:,}",
        f"{total - trainable:,}",
    )
    metrics: dict[str, float] = {
        "model/params_total": float(total),
        "model/params_trainable": float(trainable),
        "model/params_non_trainable": float(total - trainable),
    }
    for module_name in sorted(by_module):
        mod_total, mod_train = by_module[module_name]
        logging.info(
            "  [%s] total=%s trainable=%s",
            module_name,
            f"{mod_total:,}",
            f"{mod_train:,}",
        )
        metrics[f"model/params_total/{module_name}"] = float(mod_total)
        metrics[f"model/params_trainable/{module_name}"] = float(mod_train)
    return metrics


def _build_wandb_init_kwargs(config: Any | None) -> dict[str, Any]:
    if config is None:
        return {}
    wandb_kwargs = dict(config.get("wandb_kwargs", {}) or {})
    if resume_id := os.environ.get("WANDB_RESUME_ID"):
        wandb_kwargs.setdefault("id", resume_id)
        wandb_kwargs.setdefault("resume", "must")
        wandb_kwargs.pop("name", None)
        return wandb_kwargs
    prefix = config.get("wandb_run_name_prefix")
    if prefix and "name" not in wandb_kwargs:
        counter_path = Path(
            config.get("wandb_run_name_counter_path", ".wandb_run_counter")
        )
        if bool(config.get("wandb_run_name_use_increment", True)):
            idx = (
                int(counter_path.read_text().strip()) if counter_path.exists() else 0
            ) + 1
            counter_path.parent.mkdir(parents=True, exist_ok=True)
            counter_path.write_text(str(idx))
            wandb_kwargs["name"] = f"{str(prefix).strip()}-{idx:04d}"
        else:
            wandb_kwargs["name"] = (
                f"{str(prefix).strip()}-{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M')}"
            )
    return wandb_kwargs


def _to_serialisable_config(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _to_serialisable_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serialisable_config(v) for v in value]
    return str(value)


def _config_to_wandb_dict(config: Any | None) -> dict[str, Any]:
    if config is None:
        return {}
    if callable(getattr(config, "to_dict", None)):
        return dict(_to_serialisable_config(config.to_dict()))
    if isinstance(config, Mapping):
        return dict(_to_serialisable_config(config))
    return dict(_to_serialisable_config(vars(config)))


def build_logger(config: config_dict.ConfigDict, workdir: Path) -> pl.loggers.Logger:
    if config.get("enable_wandb", False):
        from lightning.pytorch.loggers import WandbLogger

        wandb_kwargs = _build_wandb_init_kwargs(config)
        logger = WandbLogger(
            project=config.get("wandb_project", "md4"),
            save_dir=str(workdir),
            log_model=False,
            **wandb_kwargs,
        )
        logger.log_hyperparams(_config_to_wandb_dict(config))
        return logger
    return CSVLogger(save_dir=str(workdir), name="csv_logs")


def load_pretrained_weights(
    model: PeakSetSIGReg,
    checkpoint_path: str,
) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    sd = ckpt.get("state_dict") or ckpt.get("model") or ckpt
    prefixed = {
        k.removeprefix("model."): v for k, v in sd.items() if k.startswith("model.")
    }
    model.load_state_dict(prefixed or sd, strict=True)


def patch_muon_batched_step() -> None:
    """Monkey-patch torch.optim.Muon.step with batched Newton-Schulz via bmm."""
    from torch.optim._muon import _adjust_lr, EPS as _MUON_EPS

    @torch.no_grad()
    def _batched_muon_step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            lr = lr.item() if isinstance(lr, torch.Tensor) else float(lr)
            wd = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            a, b, c = group["ns_coefficients"]
            eps = group["eps"]
            ns_steps = group["ns_steps"]
            adjust_fn = group["adjust_lr_fn"]

            params_with_grad = []
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        p.grad, memory_format=torch.preserve_format
                    )
                params_with_grad.append(p)

            if not params_with_grad:
                continue

            grads = [p.grad for p in params_with_grad]
            bufs = [self.state[p]["momentum_buffer"] for p in params_with_grad]

            torch._foreach_lerp_(bufs, grads, 1 - momentum)
            if nesterov:
                updates = torch._foreach_lerp(grads, bufs, momentum)
            else:
                updates = list(bufs)

            orthos = []
            needs_ts = []
            adj_lrs = []
            for i, p in enumerate(params_with_grad):
                ortho = updates[i].bfloat16()
                needs_t = ortho.size(0) > ortho.size(1)
                if needs_t:
                    ortho = ortho.T
                orthos.append(ortho)
                needs_ts.append(needs_t)
                adj_lrs.append(_adjust_lr(lr, adjust_fn, p.shape))

            norms = torch._foreach_norm(orthos)
            torch._foreach_clamp_min_(norms, eps)
            torch._foreach_div_(orthos, norms)

            entries = [
                (params_with_grad[i], orthos[i], adj_lrs[i], needs_ts[i])
                for i in range(len(params_with_grad))
            ]

            shape_groups: dict[tuple[int, int], list[int]] = {}
            for i, (_, ortho, _, _) in enumerate(entries):
                key = (ortho.size(0), ortho.size(1))
                shape_groups.setdefault(key, []).append(i)

            apply_wd = wd != 0
            wd_factor = 1 - lr * wd

            all_params = []
            all_results = []
            all_neg_lrs = []

            for shape_key, idxs in shape_groups.items():
                if len(idxs) < 2:
                    i = idxs[0]
                    p, ortho, adj_lr, needs_t = entries[i]
                    for _ in range(ns_steps):
                        gram = ortho @ ortho.T
                        gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
                        ortho = torch.addmm(ortho, gram_update, ortho, beta=a)
                    if needs_t:
                        ortho = ortho.T
                    all_params.append(p)
                    all_results.append(ortho)
                    all_neg_lrs.append(-adj_lr)
                else:
                    batch = torch.stack([entries[i][1] for i in idxs])
                    for _ in range(ns_steps):
                        gram = torch.bmm(batch, batch.transpose(1, 2))
                        gram_update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
                        batch = torch.baddbmm(batch, gram_update, batch, beta=a)

                    for j, i in enumerate(idxs):
                        _, _, adj_lr, needs_t = entries[i]
                        result = batch[j]
                        if needs_t:
                            result = result.T
                        all_params.append(entries[i][0])
                        all_results.append(result)
                        all_neg_lrs.append(-adj_lr)

            if apply_wd:
                torch._foreach_mul_(all_params, wd_factor)
            torch._foreach_mul_(all_results, all_neg_lrs)
            torch._foreach_add_(all_params, all_results)

        return loss

    torch.optim.Muon.step = _batched_muon_step
    logging.info("Patched torch.optim.Muon.step with batched Newton-Schulz.")


def latest_ckpt_path(directory: Path) -> str | None:
    checkpoint_dir = directory / "checkpoints"
    root = checkpoint_dir if checkpoint_dir.exists() else directory
    ckpts = sorted(
        [*root.rglob("*.ckpt"), *root.rglob("*.pt")],
        key=lambda p: p.stat().st_mtime,
    )
    return str(ckpts[-1]) if ckpts else None
