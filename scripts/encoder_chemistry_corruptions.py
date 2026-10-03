"""Fixed input interventions for a native 127-peak chemistry audit.

Every output retains physical slot identities. Encoder inputs contain only model
features; chemical labels and matching provenance are returned separately. These
are computational interventions, not assertions about realizable chemistry.
"""
from __future__ import annotations

import numpy as np

from spectra_learning.data.spectra import PEAK_MZ_MAX, PEAK_MZ_MIN

ENCODER_KEYS = (
    "peak_mz", "peak_intensity", "peak_valid_mask", "precursor_mz",
    "collision_energy", "collision_energy_present", "charge", "charge_present",
    "polarity_id", "instrument_family_id",
)
LOSS_NAMES = ("H2O", "NH3", "CO", "CO2", "HCN", "H2S")
# Register names/doses before model outcomes. Attenuation is applied after native
# normalization with no renormalization; removal only changes the validity mask.
DEFINITIONS = {
    **{f"{mode}_{selection}20": {
        "kind": "fraction", "mode": mode, "selection": selection,
        "fraction": 0.2, "attenuation": 0.1, "rng_family": 1,
    } for mode in ("attenuate", "remove") for selection in ("top", "random")},
    "permute_intensity": {"kind": "permute_intensity", "rng_family": 2},
    "mz_jitter_0p01da": {"kind": "mz_jitter", "amplitude_da": 0.01, "rng_family": 3},
    "mz_shift_plus_0p1da": {"kind": "mz_shift", "shift_da": 0.1, "rng_family": 4},
    "observed_ce_metadata": {"kind": "observed_ce_metadata", "minimum_raw_ce_gap": 10.0,
                             "rng_family": 5},
    **{f"{mode}_loss_{'matched_' if matched else ''}{loss}": {
        "kind": "loss", "mode": mode, "loss": loss, "matched_control": matched,
        "attenuation": 0.1, "mass_caliper_da": 25.0, "intensity_ratio_caliper": 2.0,
        "rng_family": 10 + j,
    } for j, loss in enumerate(LOSS_NAMES) for mode in ("attenuate", "remove")
      for matched in (False, True)},
}


def match_peak_controls(
    mz_da: np.ndarray, intensity: np.ndarray, valid: np.ndarray,
    targets: np.ndarray, rng: np.random.Generator, *, mass_caliper_da: float = 25.0,
    intensity_ratio_caliper: float = 2.0,
) -> tuple[np.ndarray, bool]:
    """Random full bipartite matching without replacement under fixed calipers.

    A partial match is never used: target and control arms must have identical
    counts. Only non-target physical peaks can serve as controls.
    """
    selected = np.flatnonzero(targets & valid)
    candidates = np.flatnonzero(valid & ~targets)
    empty = np.zeros_like(valid)
    if not len(selected) or len(candidates) < len(selected):
        return empty, False
    edges = []
    for target in selected:
        ratio = intensity[candidates] / intensity[target]
        allowed = candidates[(np.abs(mz_da[candidates] - mz_da[target]) <= mass_caliper_da)
                             & (ratio >= 1.0 / intensity_ratio_caliper)
                             & (ratio <= intensity_ratio_caliper)]
        edges.append(rng.permutation(allowed).tolist())
    owner: dict[int, int] = {}

    def augment(target: int, seen: set[int]) -> bool:
        for candidate in edges[target]:
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate not in owner or augment(owner[candidate], seen):
                owner[candidate] = target
                return True
        return False

    for target in sorted(range(len(selected)), key=lambda j: len(edges[j])):
        if not augment(target, set()):
            return empty, False
    result = empty.copy()
    result[list(owner)] = True
    return result, True


def observed_ce_donors(data: dict[str, np.ndarray], rows: np.ndarray,
                       repeats: dict[str, np.ndarray], *, minimum_raw_ce_gap: float = 10.0
                       ) -> tuple[np.ndarray, dict]:
    """Select an observed alternate acquisition that changes native CE metadata.

    Raw CE values above the encoder's cap can differ without changing its input;
    such pairs are excluded. Tie-break by smallest raw gap, then original row.
    """
    donor = np.full(len(rows), -1, np.int64)
    available = np.zeros(len(rows), np.int32)
    saturated = np.zeros(len(rows), np.int32)
    for index, row in enumerate(rows):
        candidates = np.flatnonzero(repeats["primary_index"] == row)
        same = ((repeats["computed_inchikey"][candidates] == data["computed_inchikey"][row])
                & (repeats["canonical_smiles"][candidates] == data["canonical_smiles"][row])
                & (repeats["instrument_type"][candidates] == data["instrument_type"][row])
                & (repeats["adduct"][candidates] == data["adduct"][row])
                & repeats["collision_energy_present"][candidates].astype(bool)
                & bool(data["collision_energy_present"][row]))
        gap = np.abs(repeats["collision_energy_raw"][candidates]
                     - data["collision_energy_raw"][row])
        observed = same & (gap >= minimum_raw_ce_gap)
        changed = repeats["collision_energy"][candidates] != data["collision_energy"][row]
        saturated[index] = np.sum(observed & ~changed)
        eligible = np.flatnonzero(observed & changed)
        available[index] = len(eligible)
        if len(eligible):
            order = np.lexsort((repeats["rows"][candidates[eligible]], gap[eligible]))
            donor[index] = candidates[eligible[order[0]]]
    return donor, {"eligible_repeat_counts": available.tolist(),
                   "excluded_raw_ce_changed_but_native_ce_equal_counts": saturated.tolist()}


def corrupt_inputs(
    data: dict[str, np.ndarray], rows: np.ndarray, *, name: str,
    seed: int = 20260930, repeats: dict[str, np.ndarray] | None = None,
    weak_losses: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    """Return copied model inputs, targeted physical slots, and intervention facts.

    Randomness depends on source-row identity and corruption family, so batching
    and row order cannot change interventions. Unsupported rows remain unchanged.
    Metadata-only interventions target no physical peak slots.
    """
    if name not in DEFINITIONS:
        raise ValueError(f"Unknown fixed corruption: {name}")
    rows = np.asarray(rows)
    if rows.ndim != 1 or rows.dtype.kind not in "iu" or not len(rows):
        raise ValueError("rows must be a nonempty one-dimensional integer array")
    if np.any(rows < 0) or np.any(rows >= len(data["peak_mz"])):
        raise ValueError("rows lie outside the primary panel")
    definition = DEFINITIONS[name]
    inputs = {key: np.array(data[key][rows], copy=True) for key in ENCODER_KEYS}
    valid = inputs["peak_valid_mask"]
    targets = np.zeros_like(valid, dtype=bool)
    supported = np.zeros(len(rows), bool)
    facts: dict = {}
    kind = definition["kind"]
    if kind == "observed_ce_metadata":
        if repeats is None:
            raise ValueError("Observed CE replacement requires measured repeat_inputs")
        donor, facts = observed_ce_donors(data, rows, repeats,
            minimum_raw_ce_gap=definition["minimum_raw_ce_gap"])
        supported = donor >= 0
        for key in ("collision_energy", "collision_energy_present"):
            inputs[key][supported] = repeats[key][donor[supported]]
        repeat_source_rows = np.full(len(rows), -1, np.int64)
        repeat_source_rows[supported] = repeats["rows"][donor[supported]]
        facts.update(repeat_indices=donor.tolist(), repeat_source_rows=repeat_source_rows.tolist(),
            targeted_metadata_fields=["collision_energy", "collision_energy_present"],
            interpretation="Metadata-only hybrid: retain the anchor spectrum and replace CE with a measured same-molecule acquisition value. This does not recreate the alternate spectrum.")
    else:
        if kind == "loss":
            if weak_losses is None:
                raise ValueError("Neutral-loss interventions require aligned weak_losses")
            np.testing.assert_array_equal(weak_losses["rows"][rows], data["rows"][rows],
                                          err_msg="Weak-loss rows differ from primary input rows")
            loss_columns = np.flatnonzero(weak_losses["loss_names"] == definition["loss"])
            if len(loss_columns) != 1:
                raise ValueError("The requested theoretical loss must occur exactly once")
            loss_column = int(loss_columns[0])
            candidate_counts = np.zeros(len(rows), np.int32)
            match_supported = np.zeros(len(rows), bool)
            actual_targets = np.zeros_like(valid)
        for index, row in enumerate(rows):
            slots = np.flatnonzero(valid[index])
            rng = np.random.default_rng(np.random.SeedSequence(
                [seed, int(data["rows"][row]), definition["rng_family"]]))
            if kind == "fraction":
                count = min(len(slots) - 1, max(1, int(np.floor(definition["fraction"] * len(slots)))))
                if count < 1:
                    continue
                if definition["selection"] == "top":
                    selected = slots[np.argsort(-inputs["peak_intensity"][index, slots], kind="stable")[:count]]
                else:
                    selected = rng.choice(slots, count, replace=False)
                targets[index, selected] = True
                supported[index] = True
            elif kind == "permute_intensity":
                if len(slots) < 2:
                    continue
                permutation = rng.permutation(len(slots))
                if np.array_equal(permutation, np.arange(len(slots))):
                    permutation = np.roll(permutation, 1)
                original = inputs["peak_intensity"][index, slots].copy()
                inputs["peak_intensity"][index, slots] = original[permutation]
                targets[index, slots] = original != original[permutation]
                supported[index] = targets[index].any()
            elif kind in ("mz_jitter", "mz_shift"):
                physical = inputs["peak_mz"][index, slots].astype(np.float64) * PEAK_MZ_MAX
                if np.any(np.diff(physical) < 0):
                    raise ValueError("Rank-preserving m/z interventions require native m/z order")
                dose = (rng.uniform(-definition["amplitude_da"], definition["amplitude_da"], len(slots))
                        if kind == "mz_jitter" else definition["shift_da"])
                changed = np.clip(physical + dose, PEAK_MZ_MIN, PEAK_MZ_MAX)
                # Monotone projection preserves source slot/rank order, including
                # duplicate m/z. This can correlate jitter and is explicitly OOD.
                if kind == "mz_jitter":
                    changed = np.maximum.accumulate(changed)
                original = inputs["peak_mz"][index, slots].copy()
                inputs["peak_mz"][index, slots] = changed / PEAK_MZ_MAX
                targets[index, slots] = inputs["peak_mz"][index, slots] != original
                supported[index] = targets[index].any()
            elif kind == "loss":
                flagged = weak_losses["peak_matches"][row, :, loss_column] & valid[index]
                candidate_counts[index] = flagged.sum()
                if not flagged.any() or flagged.sum() >= len(slots):
                    continue
                control, matched = match_peak_controls(
                    inputs["peak_mz"][index] * PEAK_MZ_MAX,
                    inputs["peak_intensity"][index], valid[index], flagged, rng,
                    mass_caliper_da=definition["mass_caliper_da"],
                    intensity_ratio_caliper=definition["intensity_ratio_caliper"])
                if not matched:
                    continue
                actual_targets[index] = flagged
                match_supported[index] = True
                supported[index] = True
                targets[index] = control if definition["matched_control"] else flagged
        if kind in ("fraction", "loss"):
            if definition["mode"] == "attenuate":
                inputs["peak_intensity"][targets] *= definition["attenuation"]
            else:
                inputs["peak_valid_mask"][targets] = False
            facts.update(intervention_after_native_normalization=True,
                         intensity_renormalization=False,
                         removal_semantics="Mask only; original m/z/intensity values and all slot indices retained")
        if kind == "loss":
            facts.update(observed_weak_loss_peak_counts=candidate_counts.tolist(),
                         complete_control_match=match_supported.tolist(),
                         original_loss_target_slots=[np.flatnonzero(mask).tolist() for mask in actual_targets],
                         interpretation="Mass-consistent theoretical loss only; not a measured fragment identity or confirmed neutral-loss mechanism. Both target and matched-control arms require a complete caliper-constrained matching.")
        if kind in ("mz_jitter", "mz_shift"):
            facts.update(artificial_out_of_distribution=True,
                         physical_mass_bounds_da=[PEAK_MZ_MIN, PEAK_MZ_MAX],
                         slot_reordering=False, precursor_mass_changed=False,
                         jitter_projection="Forward cumulative maximum after clipping; retains nondecreasing original m/z rank and bounded dose")
    original_mask = data["peak_valid_mask"][rows]
    if np.any(original_mask.any(1) & ~inputs["peak_valid_mask"].any(1)):
        raise AssertionError("An intervention removed every visible physical peak")
    changed = np.zeros(len(rows), bool)
    for key in ENCODER_KEYS:
        difference = inputs[key] != data[key][rows]
        changed |= difference.reshape(len(rows), -1).any(1)
    metadata = {"name": name, "definition": dict(definition), "seed": seed,
                "source_rows": data["rows"][rows].tolist(),
                "supported_rows": supported.tolist(), "supported_row_count": int(supported.sum()),
                "changed_rows": changed.tolist(), "changed_row_count": int(changed.sum()),
                "target_peak_counts": targets.sum(1).tolist(),
                "input_fields": list(ENCODER_KEYS), "chemical_labels_sent_to_encoder": False,
                "physical_slot_indices_preserved": True, "facts": facts}
    return inputs, targets, metadata
