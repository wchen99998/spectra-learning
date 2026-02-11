from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path

import ray
from ml_collections import config_dict
from ray import tune
from ray.tune import RunConfig
from ray.air.integrations.wandb import WandbLoggerCallback

from train import train_and_evaluate

_TUNE_DIST_REGISTRY: dict[str, callable] = {
    "loguniform": tune.loguniform,
    "uniform": tune.uniform,
    "choice": tune.choice,
    "randint": tune.randint,
    "quniform": tune.quniform,
    "grid_search": tune.grid_search,
}


def _load_config(path: str | Path) -> config_dict.ConfigDict:
    path = Path(path)
    spec = importlib.util.spec_from_file_location("experiment_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.get_config()


def _build_param_space(cfg: config_dict.ConfigDict) -> dict[str, object]:
    specs = cfg.get("tune_param_space", [])
    assert specs, (
        "cfg.tune_param_space is empty. "
        "Define tunable parameters in the config file via apply_tune_defaults() "
        "or set cfg.tune_param_space directly."
    )
    space: dict[str, object] = {}
    for entry in specs:
        name = entry["param"]
        dist_name = entry["dist"]
        args = entry["args"]
        dist_fn = _TUNE_DIST_REGISTRY[dist_name]
        # choice/grid_search take a single list; others take unpacked args
        if dist_name in ("choice", "grid_search"):
            space[name] = dist_fn(args)
        else:
            space[name] = dist_fn(*args)
    return space


def _trainable(
    trial_config: dict[str, object],
    *,
    base_config_path: str,
    resolved_tfrecord_dir: str,
) -> None:
    cfg = _load_config(base_config_path)
    for key, value in trial_config.items():
        cfg[key] = value

    # Use the pre-resolved absolute path so trials don't re-download data.
    cfg.tfrecord_dir = resolved_tfrecord_dir

    # Ray handles W&B logging through WandbLoggerCallback.
    cfg.enable_wandb = False

    trial_dir = Path(tune.get_context().get_trial_dir())
    final_metrics = train_and_evaluate(cfg, workdir=trial_dir)
    tune_metrics = {
        key.replace("/", "_"): float(value)
        for key, value in final_metrics.items()
    }
    tune.report(tune_metrics)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ray Tune HPO for pretraining + final attentive probe.")
    parser.add_argument("--config", required=True, help="Path to training config python file.")
    parser.add_argument("--workdir", required=True, help="Ray results root directory.")
    parser.add_argument("--num-samples", type=int, default=16, help="Number of trials to sample.")
    parser.add_argument("--cpus-per-trial", type=float, default=8.0, help="CPU resources per trial.")
    parser.add_argument("--gpus-per-trial", type=float, default=1.0, help="GPU resources per trial.")
    parser.add_argument("--max-concurrent-trials", type=int, default=2, help="Max concurrent trials.")
    parser.add_argument(
        "--metric",
        default="final_probe_test_acc_precursor_bin",
        help="Metric to optimize (Ray-reported key).",
    )
    parser.add_argument(
        "--mode",
        choices=("min", "max"),
        default="max",
        help="Optimization direction for metric.",
    )
    parser.add_argument(
        "--wandb-project",
        default="",
        help="W&B project for Ray trial-level logging. Empty disables callback.",
    )
    parser.add_argument(
        "--override-json",
        default="",
        help=(
            "Optional JSON dict to pin specific parameters to fixed values. "
            "Example: '{\"final_probe_num_epochs\": 3}'. "
            "These override the config's tune_param_space distributions."
        ),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    cfg = _load_config(args.config)

    param_space = _build_param_space(cfg)
    if args.override_json:
        overrides = json.loads(args.override_json)
        param_space.update(overrides)

    callbacks = []
    wandb_project = args.wandb_project or str(cfg.get("wandb_project", ""))
    if wandb_project:
        callbacks.append(WandbLoggerCallback(project=wandb_project, log_config=True))

    # Resolve tfrecord_dir to an absolute path so trials reuse existing data.
    resolved_tfrecord_dir = str(
        Path(cfg.get("tfrecord_dir", "data/gems_peaklist_tfrecord")).expanduser().resolve()
    )

    ray.init(ignore_reinit_error=True)
    trainable = tune.with_parameters(
        _trainable,
        base_config_path=str(Path(args.config).expanduser().resolve()),
        resolved_tfrecord_dir=resolved_tfrecord_dir,
    )
    trainable = tune.with_resources(
        trainable,
        resources={
            "cpu": float(args.cpus_per_trial),
            "gpu": float(args.gpus_per_trial),
        },
    )

    tuner = tune.Tuner(
        trainable,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            metric=args.metric,
            mode=args.mode,
            num_samples=int(args.num_samples),
            max_concurrent_trials=int(args.max_concurrent_trials),
        ),
        run_config=RunConfig(
            name="spectra_hpo",
            storage_path=str(workdir),
            callbacks=callbacks,
        ),
    )
    results = tuner.fit()
    best = results.get_best_result(metric=args.metric, mode=args.mode)

    print("Best trial metric:", best.metrics[args.metric])
    print("Best trial config:", best.config)


if __name__ == "__main__":
    main()
