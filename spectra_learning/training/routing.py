from typing import Any


def resolve_training_route(config: Any) -> tuple[str, str]:
    task = str(config.get("training_task", "pretrain")).lower()
    backend = str(config.get("device_backend", "auto")).lower()
    if task not in {
        "pretrain",
        "contrastive",
        "ar_spectra",
        "adversarial_fake_peak",
    }:
        raise ValueError(f"Unknown training_task: {task}")
    if backend not in {"auto", "torch", "jax"}:
        raise ValueError(f"Unknown device_backend: {backend}")
    if task == "contrastive" and backend == "jax":
        raise ValueError("contrastive training does not support device_backend='jax'")
    if task == "adversarial_fake_peak" and backend != "torch":
        raise ValueError(f"{task} training requires device_backend='torch'")
    if task == "ar_spectra" and backend != "jax":
        raise ValueError("ar_spectra training requires device_backend='jax'")
    return task, backend
