from typing import Any, NamedTuple

import numpy as np
import torch
from ml_collections import config_dict

from spectra_learning.probes.massspec.targets import (
    MACCS_FINGERPRINT_BITS,
    MORGAN_PROBE_FINGERPRINT_BITS,
)


class MsgProbeTaskSpec(NamedTuple):
    regression_tasks: tuple[str, ...]
    maccs_bits: int
    regression_means: dict[str, float]
    regression_stds: dict[str, float]
    fingerprint_task: str = "maccs"


class MsgProbeSplitTargets(NamedTuple):
    regression: dict[str, np.ndarray]
    maccs: np.ndarray


class MsgProbePairwiseAlignment(NamedTuple):
    tanimoto: np.ndarray
    cosine: np.ndarray
    pearson: float


def build_msg_probe_inputs(
    peak_embeddings: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    peak_embeddings = peak_embeddings[:, : valid_mask.shape[1]]
    mask = valid_mask.unsqueeze(-1).to(dtype=peak_embeddings.dtype)
    return (peak_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


MACCS_TASK = "maccs"
MORGAN_TASK = "morgan"
REGRESSION_PROBE_TASKS: tuple[str, ...] = ()
PROBE_FINGERPRINT_BITS = {
    MACCS_TASK: MACCS_FINGERPRINT_BITS,
    MORGAN_TASK: MORGAN_PROBE_FINGERPRINT_BITS,
}
def _config_get(config: Any, key: str, default: Any) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def msg_probe_variants_from_config(
    config: Any,
) -> tuple[str, ...]:
    raw_variants = _config_get(config, "msg_probe_variants", ("mean", "covariance", "pma"))
    if isinstance(raw_variants, str):
        return (raw_variants.lower(),)
    return tuple(str(variant).lower() for variant in raw_variants)


def resolve_msg_probe_fingerprint(
    config: Any,
) -> str:
    return str(
        _config_get(
            config,
            "msg_probe_fingerprint",
            _config_get(config, "msg_probe_fingerprint_type", MACCS_TASK),
        )
    ).lower()


def resolve_msg_probe_sample_limits(
    config: Any,
) -> tuple[int | None, int | None, int | None, bool]:
    raw_sample_size = _config_get(config, "msg_probe_sample_size", None)
    raw_train = _config_get(config, "msg_probe_max_train_samples", None) or raw_sample_size
    raw_val = _config_get(config, "msg_probe_max_val_samples", None) or raw_sample_size
    raw_test = _config_get(config, "msg_probe_max_test_samples", None) or raw_sample_size
    if raw_train is None:
        raw_train = _config_get(
            config,
            "nist_murcko_probe_train_samples",
            4_000,
        )
    if raw_val is None:
        raw_val = _config_get(
            config,
            "nist_murcko_probe_val_samples",
            1_000,
        )
    if raw_test is None:
        raw_test = _config_get(
            config,
            "nist_murcko_probe_test_samples",
            1_000,
        )
    max_train_samples = int(raw_train) if raw_train is not None else None
    max_val_samples = int(raw_val) if raw_val is not None else None
    max_test_samples = int(raw_test) if raw_test is not None else None
    randomize_test_subset = max_test_samples is not None
    return max_train_samples, max_val_samples, max_test_samples, randomize_test_subset


def resolve_msg_probe_num_repeats(
    config: Any,
) -> int:
    raw_repeats = _config_get(config, "msg_probe_num_repeats", None)
    if raw_repeats is None:
        raw_repeats = _config_get(
            config,
            "nist_murcko_probe_num_repeats",
            1,
        )
    return int(raw_repeats) if raw_repeats is not None else 1


def resolve_msg_probe_pairwise_alignment_num_pairs(
    config: Any,
) -> int:
    return int(_config_get(config, "msg_probe_pairwise_alignment_num_pairs", 0))
