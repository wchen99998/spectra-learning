from __future__ import annotations

import argparse
import logging
from pathlib import Path

from spectra_learning.data.gems.datamodule import GemsNativeDataModule
from spectra_learning.data.massspec_probe import MassSpecProbeData
from spectra_learning.training.api import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and validate prepared data artifacts."
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="Directory where data artifacts will be stored.",
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


def download_data_artifacts(
    *,
    artifact_dir: Path,
    config_path: Path,
    skip_gems: bool = False,
    skip_nist_murcko: bool = False,
    include_morgan: bool = False,
    include_dreams: bool = False,
) -> None:
    config = load_config(config_path)
    config.artifact_dir = str(artifact_dir.expanduser().resolve())

    if include_morgan:
        config.msg_probe_fingerprint = "morgan"
    if include_dreams:
        config.nist_murcko_probe_include_dreams_auxiliary = True

    if not skip_gems:
        gems = GemsNativeDataModule(config, seed=int(config.seed))
        print(f"GeMS artifact: {gems.gems_dir}")
        print(f"GeMS train shards: {len(gems.gems_train_shards)}")
        print(f"GeMS validation shards: {len(gems.gems_validation_shards)}")

    if not skip_nist_murcko:
        nist = MassSpecProbeData.from_config(config)
        nist_dir = Path(nist.train_files[0]).parent
        print(f"NIST Murcko artifact: {nist_dir}")
        print(f"NIST Murcko train files: {len(nist.train_files)}")
        print(f"NIST Murcko val files: {len(nist.val_files)}")
        print(f"NIST Murcko test files: {len(nist.test_files)}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    download_data_artifacts(
        artifact_dir=args.artifact_dir,
        config_path=args.config,
        skip_gems=args.skip_gems,
        skip_nist_murcko=args.skip_nist_murcko,
        include_morgan=args.include_morgan,
        include_dreams=args.include_dreams,
    )


if __name__ == "__main__":
    main()
