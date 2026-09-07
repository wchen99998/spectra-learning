"""Stable, domain-separated CPU randomness for data order and augmentations."""

from contextlib import contextmanager
import hashlib
import json
from collections.abc import Iterator

import torch

INPUT_RANDOMNESS_VERSION = 2


def data_seed(*parts: int | str) -> int:
    payload = json.dumps(parts, separators=(",", ":"), ensure_ascii=True).encode()
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8, person=b"spectra-input-v2").digest(),
        "little",
    )


@contextmanager
def sample_rng(seed: int) -> Iterator[None]:
    # Only CPU RNG is involved; do not initialize or seed CUDA from a worker.
    state = torch.get_rng_state()
    torch.default_generator.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(state)
