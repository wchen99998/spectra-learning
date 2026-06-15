from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra_learning.data.gems.datamodule import GemsNativeDataModule
from spectra_learning.probes.massspec.data import MassSpecProbeData
from spectra_learning.training.api import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and validate the prepared GeMS and NIST Murcko datasets."
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="Directory where GeMS and NIST Murcko artifacts will be stored.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/medium_pairmixer_encoder.py"),
        help="Config whose dataset repo IDs and preprocessing settings should be used.",
    )
    parser.add_argument("--skip-gems", action="store_true")
    parser.add_argument("--skip-nist-murcko", action="store_true")
    parser.add_argument(
        "--include-morgan",
        action="store_true",
        help="Also download NIST Murcko Morgan auxiliary files.",
    )
    parser.add_argument(
        "--include-dreams",
        action="store_true",
        help="Also download NIST Murcko DreaMS auxiliary files.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    config = load_config(args.config)
    config.artifact_dir = str(args.artifact_dir.expanduser().resolve())

    if args.include_morgan:
        config.msg_probe_fingerprint = "morgan"
    if args.include_dreams:
        config.nist_murcko_probe_include_dreams_auxiliary = True

    if not args.skip_gems:
        gems = GemsNativeDataModule(config, seed=int(config.seed))
        print(f"GeMS artifact: {gems.gems_dir}")
        print(f"GeMS train shards: {len(gems.gems_train_shards)}")
        print(f"GeMS validation shards: {len(gems.gems_validation_shards)}")

    if not args.skip_nist_murcko:
        nist = MassSpecProbeData.from_config(config)
        nist_dir = Path(nist.train_files[0]).parent
        print(f"NIST Murcko artifact: {nist_dir}")
        print(f"NIST Murcko train files: {len(nist.train_files)}")
        print(f"NIST Murcko val files: {len(nist.val_files)}")
        print(f"NIST Murcko test files: {len(nist.test_files)}")


if __name__ == "__main__":
    main()
