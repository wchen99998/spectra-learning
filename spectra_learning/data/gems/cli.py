from pathlib import Path
from typing import Any

from spectra_learning.data.gems.visualization import visualize_real_mask_strategies


def _parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="GeMS data tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    visualize_parser = subparsers.add_parser(
        "visualize-masks",
        help="Visualize JEPA mask modes on real GeMS samples.",
    )
    visualize_parser.add_argument("--config", type=Path, default=Path("configs/gems_small.py"))
    visualize_parser.add_argument("--split", choices=("train", "validation"), default="validation")
    visualize_parser.add_argument("--start-index", type=int, default=0)
    visualize_parser.add_argument("--num-samples", type=int, default=3)
    visualize_parser.add_argument("--seed", type=int, default=7)
    visualize_parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/input_pipeline_real_masks.png"),
    )
    visualize_parser.add_argument("--strategies", nargs="+", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "visualize-masks":
        visualize_real_mask_strategies(
            config_path=args.config,
            split=args.split,
            start_index=args.start_index,
            num_samples=args.num_samples,
            seed=args.seed,
            output_path=args.output,
            strategies=None if args.strategies is None else tuple(args.strategies),
        )


if __name__ == "__main__":
    main()
