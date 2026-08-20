from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch

from spectra_learning.data.spectra import (
    ASSUMED_PRECURSOR_CHARGE,
    PRECURSOR_CHARGE_MAX,
    SPECTRUM_METADATA_KEYS,
)


MASSIVE_V2_ACQUISITION_SCHEMA = "massive_v2_acquisition_v1"
MASSIVE_V2_CONDITION_DIM = 28

POLARITY_UNKNOWN = 0
POLARITY_POSITIVE = 1
POLARITY_NEGATIVE = 2
ACQUISITION_UNKNOWN = 0
ACQUISITION_DDA = 1
ACQUISITION_DIA = 2
INSTRUMENT_FAMILIES = (
    "unknown",
    "orbitrap",
    "qtof",
    "tof",
    "ion_trap",
    "triple_quadrupole",
    "fticr",
)


def polarity_id(value: object) -> int:
    text = str(value).strip().lower()
    if text in {"1", "+", "+1", "positive", "pos", "[m+h]+"}:
        return POLARITY_POSITIVE
    if text in {"0", "negative", "neg", "-", "-1", "[m-h]-"}:
        return POLARITY_NEGATIVE
    if re.search(r"\]\s*\d*\+$", text):
        return POLARITY_POSITIVE
    if re.search(r"\]\s*\d*-$", text):
        return POLARITY_NEGATIVE
    return POLARITY_UNKNOWN


def acquisition_type_id(value: object) -> int:
    text = str(value).strip().lower()
    if text == "dda":
        return ACQUISITION_DDA
    if text == "dia":
        return ACQUISITION_DIA
    return ACQUISITION_UNKNOWN


def instrument_family_id(value: object) -> int:
    text = re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
    if any(
        keyword in text
        for keyword in (
            "orbitrap",
            "q exactive",
            "exploris",
            "fusion lumos",
            "fusion tribrid",
        )
    ):
        return 1
    if any(
        keyword in text
        for keyword in (
            "q tof",
            "qtof",
            "quadrupole time of flight",
            "tripletof",
            "triple tof",
        )
    ):
        return 2
    if any(keyword in text for keyword in ("tof", "time of flight")):
        return 3
    if any(keyword in text for keyword in ("ion trap", "linear trap", "ltq")):
        return 4
    if any(
        keyword in text
        for keyword in ("triple quadrupole", "triple quad", "qqq", "tsq")
    ):
        return 5
    if any(keyword in text for keyword in ("fticr", "ft icr", "fourier transform icr")):
        return 6
    return 0


def _torch_field(
    batch: dict[str, torch.Tensor],
    key: str,
    reference: torch.Tensor,
    default: float = 0.0,
) -> torch.Tensor:
    value = batch.get(key)
    if value is None:
        return torch.full_like(reference, default, dtype=torch.float32)
    return value.to(dtype=torch.float32)


def _torch_one_hot(ids: torch.Tensor, classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(ids.to(torch.int64), classes).to(torch.float32)


def torch_massive_v2_condition_from_batch(
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    reference = batch.get("precursor_mz")
    if reference is None:
        reference = batch["collision_energy"]
    reference = reference.to(dtype=torch.float32)
    precursor = _torch_field(batch, "precursor_mz", reference)
    precursor_present = _torch_field(
        batch,
        "precursor_mz_present",
        reference,
        1.0 if "precursor_mz" in batch else 0.0,
    )
    collision = _torch_field(batch, "collision_energy", reference)
    collision_present = _torch_field(
        batch,
        "collision_energy_present",
        reference,
        1.0 if "collision_energy" in batch else 0.0,
    )
    charge = _torch_field(batch, "charge", reference)
    charge_present = _torch_field(
        batch,
        "charge_present",
        reference,
        1.0 if "charge" in batch else 0.0,
    )
    accuracy = _torch_field(batch, "mass_accuracy", reference)
    accuracy_present = _torch_field(batch, "mass_accuracy_present", reference)
    rt = _torch_field(batch, "retention_time_fraction", reference)
    rt_present = _torch_field(batch, "retention_time_present", reference)
    intensity = _torch_field(batch, "precursor_intensity_zscore", reference)
    intensity_present = _torch_field(
        batch, "precursor_intensity_present", reference
    )
    lower = _torch_field(batch, "isolation_window_lower_offset", reference)
    upper = _torch_field(batch, "isolation_window_upper_offset", reference)
    isolation_present = _torch_field(batch, "isolation_window_present", reference)
    polarity = _torch_field(batch, "polarity_id", reference).long()
    acquisition = _torch_field(batch, "acquisition_type_id", reference).long()
    instrument = _torch_field(batch, "instrument_family_id", reference).long()
    return torch.cat(
        [
            torch.stack(
                [
                    precursor,
                    precursor_present,
                    collision,
                    collision_present,
                    charge / PRECURSOR_CHARGE_MAX,
                    charge_present,
                    accuracy,
                    accuracy_present,
                    rt,
                    rt_present,
                    intensity,
                    intensity_present,
                ],
                dim=-1,
            ),
            _torch_one_hot(polarity, 3),
            _torch_one_hot(acquisition, 3),
            torch.stack([lower, upper, isolation_present], dim=-1),
            _torch_one_hot(instrument, len(INSTRUMENT_FAMILIES)),
        ],
        dim=-1,
    )


def torch_spectrum_metadata_from_batch(
    batch: dict[str, torch.Tensor],
    schema: str | None = None,
) -> torch.Tensor | None:
    if "spectrum_metadata" in batch:
        return batch["spectrum_metadata"].to(dtype=torch.float32)
    if schema == MASSIVE_V2_ACQUISITION_SCHEMA:
        return torch_massive_v2_condition_from_batch(batch)
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
    return torch.stack([collision_energy, charge], dim=-1)


def jax_spectrum_metadata_from_batch(
    batch: dict[str, Any],
    schema: str | None = None,
) -> Any | None:
    if "spectrum_metadata" in batch:
        import jax.numpy as jnp

        return batch["spectrum_metadata"].astype(jnp.float32)
    if schema == MASSIVE_V2_ACQUISITION_SCHEMA:
        import jax
        import jax.numpy as jnp

        reference = batch.get("precursor_mz", batch["collision_energy"]).astype(
            jnp.float32
        )

        def field(key: str, default: float = 0.0):
            value = batch.get(key)
            return (
                jnp.full_like(reference, default, dtype=jnp.float32)
                if value is None
                else value.astype(jnp.float32)
            )

        precursor = field("precursor_mz")
        precursor_present = field(
            "precursor_mz_present", 1.0 if "precursor_mz" in batch else 0.0
        )
        collision = field("collision_energy")
        collision_present = field(
            "collision_energy_present",
            1.0 if "collision_energy" in batch else 0.0,
        )
        charge = field("charge")
        charge_present = field("charge_present", 1.0 if "charge" in batch else 0.0)
        values = jnp.stack(
            [
                precursor,
                precursor_present,
                collision,
                collision_present,
                charge / PRECURSOR_CHARGE_MAX,
                charge_present,
                field("mass_accuracy"),
                field("mass_accuracy_present"),
                field("retention_time_fraction"),
                field("retention_time_present"),
                field("precursor_intensity_zscore"),
                field("precursor_intensity_present"),
            ],
            axis=-1,
        )
        isolation = jnp.stack(
            [
                field("isolation_window_lower_offset"),
                field("isolation_window_upper_offset"),
                field("isolation_window_present"),
            ],
            axis=-1,
        )
        return jnp.concatenate(
            [
                values,
                jax.nn.one_hot(field("polarity_id").astype(jnp.int32), 3),
                jax.nn.one_hot(field("acquisition_type_id").astype(jnp.int32), 3),
                isolation,
                jax.nn.one_hot(
                    field("instrument_family_id").astype(jnp.int32),
                    len(INSTRUMENT_FAMILIES),
                ),
            ],
            axis=-1,
        )
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
    return jnp.stack([collision_energy, charge], axis=-1)


def drop_massive_v2_metadata_numpy(
    metadata: np.ndarray,
    probability: float,
) -> np.ndarray:
    metadata = metadata.copy()
    batch_size = metadata.shape[0]
    dropped = np.random.random((batch_size, 9)) < probability
    for field, (value, present) in enumerate(
        ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11))
    ):
        rows = dropped[:, field]
        metadata[rows, value] = 0.0
        metadata[rows, present] = 0.0
    rows = dropped[:, 6]
    metadata[rows, 12:15] = (1.0, 0.0, 0.0)
    rows = dropped[:, 7]
    metadata[rows, 15:18] = (1.0, 0.0, 0.0)
    rows = dropped[:, 8]
    metadata[rows, 18:21] = 0.0
    instrument_rows = np.random.random(batch_size) < probability
    metadata[instrument_rows, 21:28] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return metadata


def drop_massive_v2_metadata_torch(
    metadata: torch.Tensor,
    probability: float,
) -> torch.Tensor:
    metadata = metadata.clone()
    batch_size = metadata.shape[0]
    dropped = torch.rand(batch_size, 9, device=metadata.device) < probability
    for field, (value, present) in enumerate(
        ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11))
    ):
        rows = dropped[:, field]
        metadata[rows, value] = 0.0
        metadata[rows, present] = 0.0
    metadata[dropped[:, 6], 12:15] = metadata.new_tensor((1.0, 0.0, 0.0))
    metadata[dropped[:, 7], 15:18] = metadata.new_tensor((1.0, 0.0, 0.0))
    metadata[dropped[:, 8], 18:21] = 0.0
    instrument_rows = torch.rand(batch_size, device=metadata.device) < probability
    metadata[instrument_rows, 21:28] = metadata.new_tensor(
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )
    return metadata
