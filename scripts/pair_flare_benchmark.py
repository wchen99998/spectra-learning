"""Run the reference FLARE loops with full pair statistics or a matched control.

The reference source remains unchanged. The diagonal control has exactly the
same parameter tensors and initialization as the full-statistics readout, but
replaces its incoming/outgoing channels with copies of the diagonal channels.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch


def diagonal_control(base_class: type[torch.nn.Module]) -> type[torch.nn.Module]:
    class DiagonalOnlyEncoder(base_class):
        def forward(self, tokens, mz, intensity, cls):
            diagonal = tokens[..., :tokens.shape[-1] // 3]
            return super().forward(diagonal.repeat(1, 1, 3), mz, intensity, cls)

    return DiagonalOnlyEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-scripts", required=True, type=Path)
    parser.add_argument("--stage", choices=["baseline", "candidate"], required=True)
    parser.add_argument("--readout", choices=["marginals", "diagonal"], required=True)
    args, remaining = parser.parse_known_args()
    out_dir = Path(remaining[remaining.index("--out-dir") + 1])
    if "--init-checkpoint" in remaining:
        init = Path(remaining[remaining.index("--init-checkpoint") + 1])
        previous = json.loads((init.parent / "pair_method.json").read_text())
        if previous["readout"] != args.readout:
            raise ValueError("Cannot continue or evaluate a checkpoint with a different pair readout")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pair_method.json").write_text(json.dumps({
        "readout": args.readout, "input_dim": 4608, "pair_dim": 1536,
        "cls_identical_across_readouts": True, "reference_scripts": str(args.reference_scripts),
    }, indent=2) + "\n")
    sys.path.insert(0, str(args.reference_scripts))
    filename = "09_jepa_flare.py" if args.stage == "baseline" else "10_candidate_flare.py"
    spec = importlib.util.spec_from_file_location("pair_flare_stage", args.reference_scripts / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if args.readout == "diagonal":
        hybrid = module if args.stage == "baseline" else module.hybrid
        hybrid.HybridSpectrumEncoder = diagonal_control(hybrid.HybridSpectrumEncoder)
    ranks = []
    if args.stage == "candidate" and "--eval-only" in remaining:
        rank_function = module.ranks_with_ties

        def record_rank(scores, true_index):
            result = rank_function(scores, true_index)
            ranks.append(result)
            return result

        module.ranks_with_ties = record_rank
    sys.argv = [filename, *remaining]
    module.main()
    if ranks:
        (out_dir / "test_ranks.json").write_text(json.dumps(ranks) + "\n")


if __name__ == "__main__":
    main()
