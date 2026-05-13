import torch
from ml_collections import config_dict

from spectra_learning.training.schedules import make_cosine_schedule, scaled_min_lr

PREDICTOR_PARAM_PREFIXES = (
    "encoder_to_predictor_proj.",
    "masked_latent_predictor.",
    "predictor_slot_embedding.",
    "predictor_final_norm.",
    "masked_latent_readout.",
    "target_projector.",
    "jepa_mae_mz_head.",
    "jepa_mae_intensity_head.",
)
PREDICTOR_PARAM_NAMES = {
    "predictor_register_tokens",
}

def is_weight_decay_target(name: str, param: torch.nn.Parameter) -> bool:
    return param.ndim >= 2 and name.endswith("weight")


def _model_param_name(name: str) -> str:
    return name.removeprefix("model.")


def is_predictor_parameter(name: str) -> bool:
    name = _model_param_name(name)
    return name in PREDICTOR_PARAM_NAMES or name.startswith(PREDICTOR_PARAM_PREFIXES)


def build_adamw_param_groups(
    decay_params: list[torch.nn.Parameter],
    no_decay_params: list[torch.nn.Parameter],
    weight_decay: float,
) -> list[dict]:
    param_groups = []
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": weight_decay})
    return param_groups


def build_optimizers(
    config: config_dict.ConfigDict,
    model: torch.nn.Module,
    total_steps: int,
    device: torch.device,
) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.LRScheduler]]:
    base_lr = float(config.learning_rate)
    predictor_lr_ratio = float(config.get("predictor_learning_rate_ratio", 1.0))
    optimizer_type = str(config.get("optimizer", "adamw")).lower()
    settings = _optimizer_settings(config, device)
    if optimizer_type == "muon":
        return _build_muon_optimizers(
            config,
            model,
            total_steps,
            predictor_lr_ratio,
            settings,
        )
    if predictor_lr_ratio != 1.0:
        return _build_split_adamw_optimizers(
            config,
            model,
            total_steps,
            predictor_lr_ratio,
            settings,
        )
    return _build_single_adamw_optimizer(config, model, total_steps, settings)


def _optimizer_settings(
    config: config_dict.ConfigDict,
    device: torch.device,
) -> dict:
    is_cuda = device.type == "cuda"
    fused_cfg = config.get("optimizer_fused", None)
    return {
        "base_lr": float(config.learning_rate),
        "warmup_steps": int(config.get("warmup_steps", 0)),
        "min_learning_rate": config.get("min_learning_rate", None),
        "b2": float(config.get("b2", 0.999)),
        "weight_decay": float(config.weight_decay),
        "is_cuda": is_cuda,
        "fused": is_cuda if fused_cfg is None else bool(fused_cfg) and is_cuda,
    }


def _adamw(
    param_groups: list[dict],
    *,
    lr: float,
    b2: float,
    fused: bool,
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=(0.9, b2),
        fused=fused,
    )


def _build_muon_optimizers(
    config: config_dict.ConfigDict,
    model: torch.nn.Module,
    total_steps: int,
    predictor_lr_ratio: float,
    settings: dict,
) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.LRScheduler]]:
    if predictor_lr_ratio != 1.0:
        return _build_split_muon_optimizers(
            config,
            model,
            total_steps,
            predictor_lr_ratio,
            settings,
        )
    return _build_single_muon_optimizer(config, model, total_steps, settings)


def _muon_kwargs(config: config_dict.ConfigDict, settings: dict) -> dict:
    adjust_lr_fn = config.get("muon_adjust_lr_fn", "match_rms_adamw")
    if adjust_lr_fn is not None:
        adjust_lr_fn = str(adjust_lr_fn)
    return dict(
        weight_decay=float(
            config.get("muon_weight_decay", None) or settings["weight_decay"]
        ),
        momentum=float(config.get("muon_momentum", 0.95)),
        nesterov=bool(config.get("muon_nesterov", True)),
        ns_steps=int(config.get("muon_ns_steps", 5)),
        adjust_lr_fn=adjust_lr_fn,
    )


def _split_muon_parameters(
    model: torch.nn.Module,
) -> tuple[list, list, list, list]:
    base_muon_params = []
    predictor_muon_params = []
    base_adamw_no_decay_params = []
    predictor_adamw_no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_predictor = is_predictor_parameter(name)
        if is_weight_decay_target(name, param):
            (predictor_muon_params if is_predictor else base_muon_params).append(param)
        else:
            (
                predictor_adamw_no_decay_params
                if is_predictor
                else base_adamw_no_decay_params
            ).append(param)
    return (
        base_muon_params,
        predictor_muon_params,
        base_adamw_no_decay_params,
        predictor_adamw_no_decay_params,
    )


def _build_split_muon_optimizers(
    config: config_dict.ConfigDict,
    model: torch.nn.Module,
    total_steps: int,
    predictor_lr_ratio: float,
    settings: dict,
) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.LRScheduler]]:
    groups = _split_muon_parameters(model)
    muon_lr = float(config.get("muon_lr", None) or settings["base_lr"])
    adamw_lr = float(config.get("adamw_lr", None) or settings["base_lr"])
    optimizer_specs = [
        (
            "muon",
            "muon",
            groups[0],
            muon_lr,
            settings["min_learning_rate"],
        ),
        (
            "predictor_muon",
            "muon",
            groups[1],
            muon_lr * predictor_lr_ratio,
            scaled_min_lr(settings["min_learning_rate"], predictor_lr_ratio),
        ),
        (
            "adamw",
            "adamw",
            build_adamw_param_groups([], groups[2], settings["weight_decay"]),
            adamw_lr,
            settings["min_learning_rate"],
        ),
        (
            "predictor_adamw",
            "adamw",
            build_adamw_param_groups([], groups[3], settings["weight_decay"]),
            adamw_lr * predictor_lr_ratio,
            scaled_min_lr(settings["min_learning_rate"], predictor_lr_ratio),
        ),
    ]
    return _build_muon_specs(config, total_steps, optimizer_specs, settings)


def _build_muon_specs(
    config: config_dict.ConfigDict,
    total_steps: int,
    optimizer_specs: list[tuple],
    settings: dict,
) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.LRScheduler]]:
    optimizers = []
    schedulers = []
    for label, optimizer_type, params, lr, min_lr in optimizer_specs:
        if not params:
            continue
        if optimizer_type == "muon":
            optimizer = torch.optim.Muon(
                params,
                lr=lr,
                **_muon_kwargs(config, settings),
            )
        else:
            optimizer = _adamw(
                params,
                lr=lr,
                b2=settings["b2"],
                fused=settings["fused"],
            )
        optimizer._spectra_lr_label = label
        optimizers.append(optimizer)
        schedulers.append(
            make_cosine_schedule(
                optimizer,
                total_steps,
                settings["warmup_steps"],
                min_lr,
            )
        )
    return optimizers, schedulers


def _build_single_muon_optimizer(
    config: config_dict.ConfigDict,
    model: torch.nn.Module,
    total_steps: int,
    settings: dict,
) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.LRScheduler]]:
    muon_params: list[torch.nn.Parameter] = []
    adamw_no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_weight_decay_target(name, param):
            muon_params.append(param)
        else:
            adamw_no_decay_params.append(param)
    optimizer_specs = [
        (
            "muon",
            "muon",
            muon_params,
            float(config.get("muon_lr", None) or settings["base_lr"]),
            settings["min_learning_rate"],
        ),
        (
            "adamw",
            "adamw",
            build_adamw_param_groups(
                [],
                adamw_no_decay_params,
                settings["weight_decay"],
            ),
            float(config.get("adamw_lr", None) or settings["base_lr"]),
            settings["min_learning_rate"],
        ),
    ]
    return _build_muon_specs(config, total_steps, optimizer_specs, settings)


def _split_adamw_parameters(
    model: torch.nn.Module,
) -> tuple[list, list, list, list]:
    base_decay_params = []
    base_no_decay_params = []
    predictor_decay_params = []
    predictor_no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_predictor = is_predictor_parameter(name)
        if is_weight_decay_target(name, param):
            (predictor_decay_params if is_predictor else base_decay_params).append(param)
        else:
            (predictor_no_decay_params if is_predictor else base_no_decay_params).append(param)
    return base_decay_params, base_no_decay_params, predictor_decay_params, predictor_no_decay_params


def _build_split_adamw_optimizers(
    config: config_dict.ConfigDict,
    model: torch.nn.Module,
    total_steps: int,
    predictor_lr_ratio: float,
    settings: dict,
) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.LRScheduler]]:
    base_decay, base_no_decay, pred_decay, pred_no_decay = _split_adamw_parameters(model)
    optimizer_specs = [
        (
            build_adamw_param_groups(base_decay, base_no_decay, settings["weight_decay"]),
            settings["base_lr"],
            settings["min_learning_rate"],
        ),
        (
            build_adamw_param_groups(pred_decay, pred_no_decay, settings["weight_decay"]),
            settings["base_lr"] * predictor_lr_ratio,
            scaled_min_lr(settings["min_learning_rate"], predictor_lr_ratio),
        ),
    ]
    optimizers = []
    schedulers = []
    for param_groups, lr, min_lr in optimizer_specs:
        if not param_groups:
            continue
        optimizer = _adamw(
            param_groups,
            lr=lr,
            b2=settings["b2"],
            fused=settings["fused"],
        )
        optimizers.append(optimizer)
        schedulers.append(
            make_cosine_schedule(
                optimizer,
                total_steps,
                settings["warmup_steps"],
                min_lr,
            )
        )
    return optimizers, schedulers


def _build_single_adamw_optimizer(
    config: config_dict.ConfigDict,
    model: torch.nn.Module,
    total_steps: int,
    settings: dict,
) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.LRScheduler]]:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and is_weight_decay_target(name, param):
            decay_params.append(param)
        elif param.requires_grad:
            no_decay_params.append(param)
    optimizer = _adamw(
        [
            {"params": no_decay_params, "weight_decay": 0.0},
            {"params": decay_params, "weight_decay": settings["weight_decay"]},
        ],
        lr=settings["base_lr"],
        b2=settings["b2"],
        fused=settings["fused"],
    )
    scheduler = make_cosine_schedule(
        optimizer,
        total_steps,
        settings["warmup_steps"],
        settings["min_learning_rate"],
    )
    return [optimizer], [scheduler]
