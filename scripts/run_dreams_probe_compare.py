"""Run the online probe setup on precomputed DreaMS embeddings and save a Markdown report."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _default_output_path(config_path: Path) -> Path:
    return Path("reports") / f"dreams_probe_{config_path.stem}.md"


def _format_metric_value(value: float) -> str:
    if isinstance(value, float) and math.isfinite(value):
        return f"{value:.4f}"
    return str(value)


def _build_markdown_report(
    *,
    config_path: Path,
    output_path: Path,
    device: torch.device,
    config,
    metrics: dict[str, float],
) -> str:
    from utils.msg_probe import resolve_msg_probe_select_metric

    probe_batch_size = int(config.get("msg_probe_batch_size", config.get("batch_size", 512)))
    select_metric = resolve_msg_probe_select_metric(config).replace(
        "msg_probe/",
        "dreams_probe/",
    )
    summary_keys = [
        "dreams_probe_epoch",
        select_metric,
        "dreams_probe/test/auc_maccs_mean",
        "dreams_probe/test/recall_maccs_mean",
        "dreams_probe/test/r2_mean_wo_num_rings",
        "dreams_probe/test/mae_num_rings",
        "dreams_probe/test/mae_mean",
    ]
    summary_keys = list(dict.fromkeys(summary_keys))
    all_metric_rows = [
        (key, metrics[key]) for key in sorted(metrics) if key.startswith("dreams_probe/")
    ]

    lines = [
        "# DreaMS Probe Report",
        "",
        f"- Generated: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`",
        f"- Config: `{config_path}`",
        f"- Output: `{output_path}`",
        f"- Device: `{device}`",
        f"- Probe dataset: `{config.probe_dataset}`",
        f"- Probe epochs: `{int(config.msg_probe_num_epochs)}`",
        f"- Probe batch size: `{probe_batch_size}`",
        f"- Probe learning rate: `{float(config.msg_probe_learning_rate):g}`",
        f"- Probe weight decay: `{float(config.msg_probe_weight_decay):g}`",
        f"- Probe warmup steps: `{int(config.msg_probe_warmup_steps)}`",
        f"- Selection metric: `{select_metric}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in summary_keys:
        if key in metrics:
            lines.append(f"| `{key}` | `{_format_metric_value(metrics[key])}` |")
    lines.extend(
        [
            "",
            "## All Metrics",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
        ]
    )
    for key, value in all_metric_rows:
        lines.append(f"| `{key}` | `{_format_metric_value(value)}` |")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the online probe setup on precomputed DreaMS embeddings."
    )
    parser.add_argument(
        "--config",
        default="configs/gems_small.py",
        help="Path to the experiment config whose msg_probe_* settings should be reused.",
    )
    parser.add_argument(
        "--output-markdown",
        default=None,
        help="Where to write the Markdown report. Defaults to reports/dreams_probe_<config>.md.",
    )
    return parser.parse_args()


def main() -> None:
    from utils.msg_probe import run_dreams_probe
    from utils.training import load_config

    args = parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = (
        Path(args.output_markdown)
        if args.output_markdown
        else _default_output_path(config_path)
    )

    print(f"Device: {device}")
    print(f"Config: {config_path}")
    print(f"Probe dataset: {cfg.probe_dataset}")
    print(f"Probe epochs: {cfg.msg_probe_num_epochs}")

    metrics = run_dreams_probe(config=cfg, device=device)

    if not metrics:
        print("ERROR: No DreaMS embeddings available")
        sys.exit(1)

    markdown = _build_markdown_report(
        config_path=config_path,
        output_path=output_path,
        device=device,
        config=cfg,
        metrics=metrics,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown)

    print(f"\nSaved Markdown report to {output_path}")
    print("\n=== DreaMS Probe Results ===")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {_format_metric_value(v)}")


if __name__ == "__main__":
    main()
