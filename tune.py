import argparse
import itertools
import json
import logging
import math
import random
from pathlib import Path

from train import train_and_evaluate
from utils.training import load_config

_PARAM_ABBREVS: dict[str, str] = {
    "learning_rate": "lr",
    "weight_decay": "wd",
    "sigreg_lambda": "lam",
    "multicrop_local_keep_fraction": "lkf",
}


def _sample_value(dist: str, args: list, rng: random.Random) -> object:
    match dist:
        case "loguniform":
            return math.exp(rng.uniform(math.log(args[0]), math.log(args[1])))
        case "uniform":
            return rng.uniform(args[0], args[1])
        case "choice":
            return rng.choice(args)
        case "randint":
            return rng.randint(args[0], args[1] - 1)
        case "quniform":
            return round(rng.uniform(args[0], args[1]) / args[2]) * args[2]
        case _:
            raise ValueError(f"Unknown distribution: {dist!r}")


def generate_trial_configs(
    param_space: list[dict],
    num_samples: int,
    seed: int,
) -> list[dict[str, object]]:
    grid_params, random_params = [], []
    for entry in param_space:
        if entry["dist"] == "grid":
            grid_params.append((entry["param"], entry["args"]))
        else:
            random_params.append(entry)
    if grid_params:
        names, values = zip(*grid_params)
        grid_combos = [dict(zip(names, combo)) for combo in itertools.product(*values)]
    else:
        grid_combos = [{}]
    if not random_params:
        return grid_combos
    rng = random.Random(seed)
    trials = []
    for grid_combo in grid_combos:
        for _ in range(num_samples):
            trial = dict(grid_combo)
            for entry in random_params:
                trial[entry["param"]] = _sample_value(entry["dist"], entry["args"], rng)
            trials.append(trial)
    return trials


def build_trial_run_name(idx: int, trial_config: dict[str, object]) -> str:
    parts = [
        f"{_PARAM_ABBREVS.get(p, p)}={v:.2g}"
        if isinstance(v, float)
        else f"{_PARAM_ABBREVS.get(p, p)}={v}"
        for p, v in trial_config.items()
    ]
    return f"tune-{idx:03d}-{'_'.join(parts)}"


def run_trials(
    *,
    config_path: str,
    workdir: Path,
    trial_configs: list[dict[str, object]],
    metric: str,
    mode: str,
    wandb_project: str,
    overrides: dict[str, object],
) -> list[dict]:
    results = []
    for idx, trial_params in enumerate(trial_configs):
        trial_dir = workdir / f"trial_{idx:03d}"
        trial_name = build_trial_run_name(idx, trial_params)
        logging.info("=== Trial %d/%d: %s ===", idx + 1, len(trial_configs), trial_name)
        cfg = load_config(config_path)
        cfg.update(overrides)
        cfg.update(trial_params)
        if wandb_project:
            cfg.enable_wandb = True
            cfg.wandb_project = wandb_project
            cfg.wandb_kwargs = {"name": trial_name}
            cfg.wandb_run_name_prefix = ""
        final_metrics = train_and_evaluate(cfg, workdir=trial_dir)
        import wandb

        if wandb.run is not None:
            wandb.finish()
        for key in sorted(final_metrics):
            logging.info(
                "Trial %d/%d %s = %s",
                idx + 1,
                len(trial_configs),
                key,
                final_metrics[key],
            )
        results.append(
            {
                "idx": idx,
                "name": trial_name,
                "params": trial_params,
                "metrics": final_metrics,
                "metric_value": final_metrics.get(metric),
                "workdir": str(trial_dir),
            }
        )
    return results


def print_summary(
    results: list[dict],
    metric: str,
    mode: str,
) -> None:
    valid = [r for r in results if r["metric_value"] is not None]
    if not valid:
        logging.warning("No trials returned the metric %r.", metric)
        return
    ranked = sorted(valid, key=lambda r: r["metric_value"], reverse=mode == "max")
    print(f"\n{'=' * 80}")
    print(f"  Tune results — {len(valid)} trials, metric={metric}, mode={mode}")
    print(f"{'=' * 80}")
    print(f"  {'Rank':<6}{'Trial':<40}{metric:<20}{'Dir'}")
    print(f"  {'-' * 6}{'-' * 40}{'-' * 20}{'-' * 14}")
    for rank, r in enumerate(ranked, 1):
        print(f"  {rank:<6}{r['name']:<40}{r['metric_value']:<20.6f}{r['workdir']}")
    best = ranked[0]
    print(f"\n  Best: {best['name']}  ({metric} = {best['metric_value']:.6f})")
    print(f"  Config: {best['params']}")
    print(f"  Workdir: {best['workdir']}")
    print(f"{'=' * 80}\n")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Lightweight HPO for pretraining + periodic MSG probe."
    )
    parser.add_argument(
        "--config", required=True, help="Path to training config python file."
    )
    parser.add_argument(
        "--workdir", required=True, help="Root directory for trial outputs."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=16,
        help="Random samples per grid combination.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible sampling."
    )
    parser.add_argument(
        "--metric",
        default="msg_probe/test/auc_fg_mean",
        help="Metric key to optimize (from train_and_evaluate).",
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
        help="W&B project for per-trial logging. Empty disables WandB.",
    )
    parser.add_argument(
        "--override-json",
        default="",
        help=(
            "Optional JSON dict of config overrides applied to every trial. "
            'Example: \'{"num_epochs": 1, "limit_train_batches": 0.01}\''
        ),
    )
    args = parser.parse_args()
    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    config_path = str(Path(args.config).expanduser().resolve())
    cfg = load_config(config_path)
    param_space = cfg.get("tune_param_space", [])
    assert param_space, (
        "cfg.tune_param_space is empty. "
        "Define tunable parameters directly in the config file via `tune_param_space` "
        "or set cfg.tune_param_space directly."
    )
    trial_configs = generate_trial_configs(
        list(param_space),
        num_samples=args.num_samples,
        seed=args.seed,
    )
    logging.info("Generated %d trial configurations.", len(trial_configs))
    results = run_trials(
        config_path=config_path,
        workdir=workdir,
        trial_configs=trial_configs,
        metric=args.metric,
        mode=args.mode,
        wandb_project=args.wandb_project or str(cfg.get("wandb_project", "")),
        overrides=json.loads(args.override_json) if args.override_json else {},
    )
    print_summary(results, metric=args.metric, mode=args.mode)


if __name__ == "__main__":
    main()
