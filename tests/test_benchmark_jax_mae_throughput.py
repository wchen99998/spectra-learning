import sys

import pytest

from scripts import benchmark_jax_mae_throughput as benchmark


def _base_argv() -> list[str]:
    return [
        "benchmark_jax_mae_throughput.py",
        "--config",
        "configs/1b_pairmixer_dense_adamw.py",
        "--artifact-dir",
        "data/massive_v1_ms2_100m_stratified_x16",
    ]


def test_benchmark_parses_canonical_checkpoint_mode(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [*_base_argv(), "--activation-checkpoint-mode", "selective"],
    )

    args = benchmark.parse_args()

    assert args.activation_checkpoint_mode == "selective"


@pytest.mark.parametrize(
    "removed_flag",
    (
        "--projection-kernel",
        "--encoder-projection-kernel",
        "--predictor-projection-kernel",
    ),
)
def test_benchmark_rejects_removed_projection_kernel_flags(
    monkeypatch,
    removed_flag: str,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [*_base_argv(), removed_flag, "pallas"],
    )

    with pytest.raises(SystemExit) as error:
        benchmark.parse_args()

    assert error.value.code == 2
