from __future__ import annotations

from typing import Any

import torch

from spectra_learning.data.spectra import (
    ASSUMED_PRECURSOR_CHARGE,
    PRECURSOR_CHARGE_MAX,
    SPECTRUM_METADATA_KEYS,
)


def torch_spectrum_metadata_from_batch(
    batch: dict[str, torch.Tensor],
) -> torch.Tensor | None:
    if SPECTRUM_METADATA_KEYS[0] not in batch:
        return None
    collision_energy = batch["collision_energy"].to(dtype=torch.float32)
    raw_charge = batch.get("charge")
    charge = (
        torch.full_like(collision_energy, ASSUMED_PRECURSOR_CHARGE)
        if raw_charge is None
        else raw_charge.to(dtype=torch.float32)
    )
    charge = charge.clamp(0.0, PRECURSOR_CHARGE_MAX) / PRECURSOR_CHARGE_MAX
    return torch.stack(
        [collision_energy, charge],
        dim=-1,
    )


def jax_spectrum_metadata_from_batch(batch: dict[str, Any]) -> Any | None:
    if SPECTRUM_METADATA_KEYS[0] not in batch:
        return None
    import jax.numpy as jnp

    collision_energy = batch["collision_energy"].astype(jnp.float32)
    raw_charge = batch.get("charge")
    charge = (
        jnp.full_like(collision_energy, ASSUMED_PRECURSOR_CHARGE)
        if raw_charge is None
        else raw_charge.astype(jnp.float32)
    )
    charge = jnp.clip(charge, 0.0, PRECURSOR_CHARGE_MAX) / PRECURSOR_CHARGE_MAX
    return jnp.stack(
        [collision_energy, charge],
        axis=-1,
    )
