from typing import Any, NamedTuple

import numpy as np

from spectra_learning.data.massspec_targets import (
    MACCS_FINGERPRINT_BITS,
    MORGAN_PROBE_FINGERPRINT_BITS,
)
from spectra_learning.config.msg_probe import validate_msg_probe_config


class MsgProbeTaskSpec(NamedTuple):
    regression_tasks: tuple[str, ...]
    maccs_bits: int
    regression_means: dict[str, float]
    regression_stds: dict[str, float]
    fingerprint_task: str = "maccs"
    binary_tasks: tuple[str, ...] = ()
    single_pair_covariance_include_diagonal: bool = False


class MsgProbeSplitTargets(NamedTuple):
    regression: dict[str, np.ndarray]
    maccs: np.ndarray
    binary: dict[str, np.ndarray] = {}


class MsgProbePairwiseAlignment(NamedTuple):
    tanimoto: np.ndarray
    cosine: np.ndarray
    pearson: float


def build_msg_probe_inputs(
    peak_embeddings,
    valid_mask,
):
    peak_embeddings = peak_embeddings[:, : valid_mask.shape[1]]
    mask = valid_mask.unsqueeze(-1).to(dtype=peak_embeddings.dtype)
    return (peak_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


MACCS_TASK = "maccs"
MORGAN_TASK = "morgan"
REGRESSION_PROBE_TASKS: tuple[str, ...] = ()
BINARY_PROBE_TASKS: tuple[str, ...] = ("fluorine", "sulfur")
PROBE_FINGERPRINT_BITS = {
    MACCS_TASK: MACCS_FINGERPRINT_BITS,
    MORGAN_TASK: MORGAN_PROBE_FINGERPRINT_BITS,
}
def msg_probe_variants_from_config(
    config: Any,
) -> tuple[str, ...]:
    raw_variants = config.get("msg_probe_variants", ("mean", "covariance", "pma"))
    if isinstance(raw_variants, str):
        return (raw_variants.lower(),)
    return tuple(str(variant).lower() for variant in raw_variants)


def resolve_msg_probe_fingerprint(
    config: Any,
) -> str:
    validate_msg_probe_config(config)
    return str(config.get("msg_probe_fingerprint", MACCS_TASK)).lower()


def resolve_msg_probe_num_repeats(
    config: Any,
) -> int:
    raw_repeats = config.get("msg_probe_num_repeats", None)
    if raw_repeats is None:
        raw_repeats = config.get(
            "nist_murcko_probe_num_repeats",
            1,
        )
    return int(raw_repeats) if raw_repeats is not None else 1


def resolve_msg_probe_pairwise_alignment_num_pairs(
    config: Any,
) -> int:
    return int(config.get("msg_probe_pairwise_alignment_num_pairs", 0))
