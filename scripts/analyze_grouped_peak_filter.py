from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectra_learning.data.gems.hdf5 import GemsHdf5ShardDataset
from spectra_learning.data.spectra import (
    DEFAULT_GROUPED_PEAK_ISOTOPE_CHARGES,
    DEFAULT_GROUPED_PEAK_SHOULDER_DA,
    DEFAULT_MIN_PEAK_INTENSITY,
    PEAK_MZ_MAX,
    PEAK_MZ_MIN,
)

C13_NEUTRON_DA = 1.0033548378
DEFAULT_OUTPUT_DIR = Path("artifacts/grouped_peak_filter_analysis")
SIRIUS_MZ_ISO_ERRT = 0.002
SIRIUS_ISOTOPE_RANGES = (
    (
        0.99664664 - SIRIUS_MZ_ISO_ERRT,
        1.00342764 + SIRIUS_MZ_ISO_ERRT,
    ),
    (
        1.99653883209004 - SIRIUS_MZ_ISO_ERRT,
        2.0067426280592295 + SIRIUS_MZ_ISO_ERRT,
    ),
    (
        2.9950584 - SIRIUS_MZ_ISO_ERRT,
        3.00995027 + SIRIUS_MZ_ISO_ERRT,
    ),
    (
        3.99359037 - SIRIUS_MZ_ISO_ERRT,
        4.01300058 + SIRIUS_MZ_ISO_ERRT,
    ),
    (
        4.9937908 - SIRIUS_MZ_ISO_ERRT,
        5.01572941 + SIRIUS_MZ_ISO_ERRT,
    ),
)


def _resolve_manifest(path: Path) -> Path:
    if path.suffix == ".json":
        return path
    return path / "fdataloader_shards.json"


def _sample_indices(total: int, sample_rows: int, seed: int) -> np.ndarray:
    count = min(sample_rows, total)
    return np.sort(np.random.default_rng(seed).choice(total, size=count, replace=False))


def _iter_sampled_spectra(
    manifest_or_dir: Path,
    sample_rows: int,
    seed: int,
):
    manifest_path = _resolve_manifest(manifest_or_dir)
    dataset = GemsHdf5ShardDataset(
        manifest_path,
        spectrum_dataset="spectrum",
        precursor_dataset="precursor_mz",
    )
    sample_idx = _sample_indices(len(dataset), sample_rows, seed)
    for row_id in sample_idx.tolist():
        sample = dataset[int(row_id)]
        yield manifest_path.name, int(row_id), sample["spectra"], float(
            sample["precursor_mz_raw"]
        )


def _find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: list[int], a: int, b: int) -> None:
    root_a = _find(parent, a)
    root_b = _find(parent, b)
    if root_a != root_b:
        parent[root_b] = root_a


def _group_peaks(
    mz: np.ndarray,
    intensity: np.ndarray,
    *,
    shoulder_da: float,
    isotope_mode: str,
    isotope_tol_da: float,
    isotope_spacings: tuple[float, ...],
    isotope_max_relative_intensity: float,
    isotope_charges: tuple[int, ...],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    parent = list(range(len(mz)))
    edges: list[tuple[int, int, str]] = []

    for i, value in enumerate(mz):
        shoulder_end = int(np.searchsorted(mz, value + shoulder_da, side="right"))
        for j in range(i + 1, shoulder_end):
            _union(parent, i, j)
            edges.append((i, j, "shoulder"))

        if isotope_mode == "c13":
            for spacing in isotope_spacings:
                lo = int(
                    np.searchsorted(
                        mz,
                        value + spacing - isotope_tol_da,
                        side="left",
                    )
                )
                hi = int(
                    np.searchsorted(
                        mz,
                        value + spacing + isotope_tol_da,
                        side="right",
                    )
                )
                for j in range(max(i + 1, lo), hi):
                    if intensity[j] <= intensity[i] * isotope_max_relative_intensity:
                        _union(parent, i, j)
                        edges.append((i, j, "isotope"))
        else:
            for charge in isotope_charges:
                if charge > 1 and value / charge < 100.0:
                    continue
                pattern_edges: list[tuple[int, int]] = []
                for lo_delta, hi_delta in SIRIUS_ISOTOPE_RANGES:
                    lo = int(
                        np.searchsorted(
                            mz,
                            value + lo_delta / charge,
                            side="left",
                        )
                    )
                    hi = int(
                        np.searchsorted(
                            mz,
                            value + hi_delta / charge,
                            side="right",
                        )
                    )
                    if hi <= max(i + 1, lo):
                        break
                    window_indices = range(max(i + 1, lo), hi)
                    pattern_edges.extend((i, j) for j in window_indices)
                if pattern_edges:
                    for a, b in pattern_edges:
                        _union(parent, a, b)
                        edges.append((a, b, "isotope"))

    roots = [_find(parent, i) for i in range(len(mz))]
    root_to_group: dict[int, int] = {}
    members_by_group: list[list[int]] = []
    group_id_of_peak = np.empty(len(mz), dtype=np.int32)
    for peak_idx, root in enumerate(roots):
        group_id = root_to_group.get(root)
        if group_id is None:
            group_id = len(members_by_group)
            root_to_group[root] = group_id
            members_by_group.append([])
        members_by_group[group_id].append(peak_idx)
        group_id_of_peak[peak_idx] = group_id

    edge_types = [set() for _ in members_by_group]
    for i, j, edge_type in edges:
        group_id = int(group_id_of_peak[i])
        if group_id == int(group_id_of_peak[j]):
            edge_types[group_id].add(edge_type)

    groups = []
    for group_id, member_list in enumerate(members_by_group):
        members = np.asarray(member_list, dtype=np.int32)
        representative = int(members[np.argmax(intensity[members])])
        types = edge_types[group_id]
        if len(members) == 1:
            group_type = "single"
        elif types == {"shoulder"}:
            group_type = "shoulder"
        elif types == {"isotope"}:
            group_type = "isotope"
        else:
            group_type = "mixed"
        groups.append(
            {
                "members": members,
                "representative": representative,
                "max_intensity": float(intensity[members].max()),
                "sum_intensity": float(intensity[members].sum()),
                "type": group_type,
            }
        )
    return groups, group_id_of_peak


def _quantiles(values: list[float] | list[int]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values)
    return {
        str(q): float(np.percentile(arr, q))
        for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)
    }


def _mean(values: list[float] | list[int]) -> float:
    return float(np.asarray(values).mean()) if values else 0.0


def _mean_ci95(values: list[float] | list[int]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if arr.size == 1:
        return {"mean": mean, "lo": mean, "hi": mean}
    half_width = float(1.96 * arr.std(ddof=1) / np.sqrt(arr.size))
    return {"mean": mean, "lo": mean - half_width, "hi": mean + half_width}


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _peak_rows(mz: np.ndarray, intensity: np.ndarray, indices: np.ndarray) -> list[dict]:
    return [
        {
            "mz": float(mz[idx]),
            "intensity": float(intensity[idx]),
        }
        for idx in indices.tolist()
    ]


def _group_rows(
    mz: np.ndarray,
    intensity: np.ndarray,
    groups: list[dict[str, Any]],
    group_indices: np.ndarray,
) -> list[dict]:
    rows = []
    for group_id in group_indices.tolist():
        group = groups[group_id]
        members = group["members"]
        rows.append(
            {
                "type": group["type"],
                "representative_mz": float(mz[group["representative"]]),
                "representative_intensity": float(intensity[group["representative"]]),
                "max_intensity": group["max_intensity"],
                "sum_intensity": group["sum_intensity"],
                "size": int(len(members)),
                "members": _peak_rows(mz, intensity, members),
            }
        )
    return rows


def analyze(
    *,
    artifact_dir: Path,
    sample_rows: int,
    seed: int,
    num_peaks: int,
    min_peak_intensity: float,
    precursor_peak_exclusion_window_da: float,
    shoulder_da: float,
    isotope_mode: str,
    isotope_tol_da: float,
    isotope_charges: tuple[int, ...],
    isotope_max_relative_intensity: float,
    group_score: str,
    examples: int,
) -> dict[str, Any]:
    isotope_spacings = tuple(C13_NEUTRON_DA / charge for charge in isotope_charges)

    scalar = {
        "rows": 0,
        "current_selected_peaks": 0,
        "grouped_selected_peaks": 0,
        "current_redundant_peaks": 0,
        "overlap_representative_peaks": 0,
        "added_representative_peaks": 0,
        "selected_from_beyond_current_topn": 0,
        "all_groups": 0,
        "selected_groups": 0,
    }
    type_counts = {
        "all": {"single": 0, "shoulder": 0, "isotope": 0, "mixed": 0},
        "selected": {"single": 0, "shoulder": 0, "isotope": 0, "mixed": 0},
    }
    series: dict[str, list[float] | list[int]] = {
        "valid_peaks": [],
        "groups": [],
        "current_unique_groups_in_topn": [],
        "grouped_selected_groups": [],
        "current_redundant_peaks": [],
        "added_representative_peaks": [],
        "changed_spectrum": [],
        "selection_jaccard": [],
        "added_representative_relative_intensity": [],
        "added_representative_intensity_rank": [],
        "all_group_size": [],
        "selected_group_size": [],
        "collapsed_peak_count": [],
    }
    saved_examples: list[dict[str, Any]] = []

    for shard_name, row_id, spectrum, precursor_mz in _iter_sampled_spectra(
        artifact_dir,
        sample_rows,
        seed,
    ):
        mz0 = spectrum[0]
        intensity0 = spectrum[1]
        precursor_upper = precursor_mz - precursor_peak_exclusion_window_da
        keep = (
            (mz0 >= PEAK_MZ_MIN)
            & (mz0 <= PEAK_MZ_MAX)
            & (intensity0 >= min_peak_intensity)
            & (
                (precursor_peak_exclusion_window_da <= 0.0)
                | (mz0 <= precursor_upper)
            )
        )
        if not keep.any():
            continue

        mz = mz0[keep].astype(np.float32, copy=False)
        intensity = intensity0[keep].astype(np.float32, copy=False)
        mz_order = np.argsort(mz, kind="stable")
        mz = mz[mz_order]
        intensity = intensity[mz_order]

        groups, group_id_of_peak = _group_peaks(
            mz,
            intensity,
            shoulder_da=shoulder_da,
            isotope_mode=isotope_mode,
            isotope_tol_da=isotope_tol_da,
            isotope_spacings=isotope_spacings,
            isotope_max_relative_intensity=isotope_max_relative_intensity,
            isotope_charges=isotope_charges,
        )
        current_order = np.argsort(-intensity, kind="stable")
        current_peak_indices = current_order[: min(num_peaks, len(current_order))]

        score_key = "max_intensity" if group_score == "max" else "sum_intensity"
        group_scores = np.asarray([group[score_key] for group in groups])
        grouped_order = np.argsort(-group_scores, kind="stable")
        selected_group_indices = grouped_order[: min(num_peaks, len(grouped_order))]
        representative_indices = np.asarray(
            [groups[group_id]["representative"] for group_id in selected_group_indices],
            dtype=np.int32,
        )

        current_set = set(current_peak_indices.tolist())
        representative_set = set(representative_indices.tolist())
        current_group_set = set(group_id_of_peak[current_peak_indices].tolist())
        added_peaks = representative_set - current_set
        selected_union = current_set | representative_set

        intensity_rank = np.empty(len(intensity), dtype=np.int32)
        intensity_rank[current_order] = np.arange(1, len(intensity) + 1)

        scalar["rows"] += 1
        scalar["current_selected_peaks"] += len(current_peak_indices)
        scalar["grouped_selected_peaks"] += len(representative_indices)
        scalar["current_redundant_peaks"] += len(current_peak_indices) - len(
            current_group_set
        )
        scalar["overlap_representative_peaks"] += len(
            current_set & representative_set
        )
        scalar["added_representative_peaks"] += len(added_peaks)
        scalar["selected_from_beyond_current_topn"] += sum(
            int(intensity_rank[peak_idx] > num_peaks) for peak_idx in added_peaks
        )
        scalar["all_groups"] += len(groups)
        scalar["selected_groups"] += len(selected_group_indices)

        series["valid_peaks"].append(len(mz))
        series["groups"].append(len(groups))
        series["current_unique_groups_in_topn"].append(len(current_group_set))
        series["grouped_selected_groups"].append(len(selected_group_indices))
        series["current_redundant_peaks"].append(
            len(current_peak_indices) - len(current_group_set)
        )
        series["added_representative_peaks"].append(len(added_peaks))
        series["changed_spectrum"].append(float(current_set != representative_set))
        series["selection_jaccard"].append(
            _fraction(len(current_set & representative_set), len(selected_union))
        )
        series["collapsed_peak_count"].append(len(mz) - len(groups))
        for group in groups:
            type_counts["all"][group["type"]] += 1
            series["all_group_size"].append(int(len(group["members"])))
        for group_id in selected_group_indices:
            group = groups[int(group_id)]
            type_counts["selected"][group["type"]] += 1
            series["selected_group_size"].append(int(len(group["members"])))
        for peak_idx in added_peaks:
            series["added_representative_relative_intensity"].append(
                float(intensity[peak_idx] / intensity.max())
            )
            series["added_representative_intensity_rank"].append(
                int(intensity_rank[peak_idx])
            )

        if len(saved_examples) < examples and len(added_peaks) > 0:
            saved_examples.append(
                {
                    "shard": shard_name,
                    "row": row_id,
                    "precursor_mz": precursor_mz,
                    "valid_peaks": int(len(mz)),
                    "groups": int(len(groups)),
                    "current_redundant_peaks": int(
                        len(current_peak_indices) - len(current_group_set)
                    ),
                    "current_top_peaks": _peak_rows(
                        mz,
                        intensity,
                        current_peak_indices[: min(num_peaks, 12)],
                    ),
                    "grouped_top_groups": _group_rows(
                        mz,
                        intensity,
                        groups,
                        selected_group_indices[: min(num_peaks, 12)],
                    ),
                }
            )

    rows = scalar["rows"]
    all_groups = scalar["all_groups"]
    selected_groups = scalar["selected_groups"]
    summary = {
        "artifact_dir": str(artifact_dir),
        "sample_rows_requested": sample_rows,
        "sample_rows_analyzed": rows,
        "seed": seed,
        "num_peaks": num_peaks,
        "min_peak_intensity": min_peak_intensity,
        "precursor_peak_exclusion_window_da": precursor_peak_exclusion_window_da,
        "shoulder_da": shoulder_da,
        "isotope_mode": isotope_mode,
        "isotope_tol_da": isotope_tol_da,
        "isotope_charges": list(isotope_charges),
        "isotope_max_relative_intensity": isotope_max_relative_intensity,
        "group_score": group_score,
        "means": {key: _mean(values) for key, values in series.items()},
        "mean_ci95": {key: _mean_ci95(values) for key, values in series.items()},
        "quantiles": {key: _quantiles(values) for key, values in series.items()},
        "rates": {
            "current_redundant_fraction_of_selected": _fraction(
                scalar["current_redundant_peaks"],
                scalar["current_selected_peaks"],
            ),
            "grouped_representative_overlap_fraction": _fraction(
                scalar["overlap_representative_peaks"],
                scalar["grouped_selected_peaks"],
            ),
            "grouped_added_fraction": _fraction(
                scalar["added_representative_peaks"],
                scalar["grouped_selected_peaks"],
            ),
            "selected_from_beyond_current_topn_fraction": _fraction(
                scalar["selected_from_beyond_current_topn"],
                scalar["grouped_selected_peaks"],
            ),
        },
        "per_spectrum": {
            "current_redundant_peaks": _fraction(
                scalar["current_redundant_peaks"], rows
            ),
            "grouped_added_representative_peaks": _fraction(
                scalar["added_representative_peaks"], rows
            ),
            "selected_from_beyond_current_topn": _fraction(
                scalar["selected_from_beyond_current_topn"], rows
            ),
            "overlap_representative_peaks": _fraction(
                scalar["overlap_representative_peaks"], rows
            ),
        },
        "group_type_fraction": {
            "all": {
                group_type: _fraction(count, all_groups)
                for group_type, count in type_counts["all"].items()
            },
            "selected": {
                group_type: _fraction(count, selected_groups)
                for group_type, count in type_counts["selected"].items()
            },
        },
        "examples": saved_examples,
    }
    return summary


def _default_output_path(args: argparse.Namespace) -> Path:
    dataset = args.artifact_dir.name
    charges = "-".join(str(charge) for charge in args.isotope_charges)
    return (
        DEFAULT_OUTPUT_DIR
        / f"{dataset}_n{args.num_peaks}_rows{args.sample_rows}"
        f"_shoulder{args.shoulder_da:g}_{args.isotope_mode}"
        f"_z{charges}_iso{args.isotope_tol_da:g}_{args.group_score}.json"
    )


def _print_summary(summary: dict[str, Any]) -> None:
    means = summary["means"]
    per_spectrum = summary["per_spectrum"]
    rates = summary["rates"]
    print(f"artifact: {summary['artifact_dir']}")
    print(f"rows analyzed: {summary['sample_rows_analyzed']}")
    print(
        "valid peaks -> groups mean: "
        f"{means['valid_peaks']:.2f} -> {means['groups']:.2f}"
    )
    print(
        "current unique groups in top-n mean: "
        f"{means['current_unique_groups_in_topn']:.2f} / {summary['num_peaks']}"
    )
    print(
        "grouped selected groups mean: "
        f"{means['grouped_selected_groups']:.2f} / {summary['num_peaks']}"
    )
    print(
        "redundant current top-n peaks per spectrum: "
        f"{per_spectrum['current_redundant_peaks']:.2f}"
    )
    print(
        "grouped representatives added per spectrum: "
        f"{per_spectrum['grouped_added_representative_peaks']:.2f}"
    )
    print(
        "selected from beyond current top-n per spectrum: "
        f"{per_spectrum['selected_from_beyond_current_topn']:.2f}"
    )
    print(
        "selected overlap/add/beyond rates: "
        f"{rates['grouped_representative_overlap_fraction']:.3f} / "
        f"{rates['grouped_added_fraction']:.3f} / "
        f"{rates['selected_from_beyond_current_topn_fraction']:.3f}"
    )
    print(
        "changed spectra / selection jaccard mean: "
        f"{means['changed_spectrum']:.3f} / {means['selection_jaccard']:.3f}"
    )
    print("all group type fraction:", summary["group_type_fraction"]["all"])
    print("selected group type fraction:", summary["group_type_fraction"]["selected"])
    print(
        "added representative intensity rank quantiles:",
        summary["quantiles"]["added_representative_intensity_rank"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze grouped peak filtering on HDF5 shards."
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--sample-rows", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--num-peaks", type=int, required=True)
    parser.add_argument(
        "--min-peak-intensity",
        type=float,
        default=DEFAULT_MIN_PEAK_INTENSITY,
    )
    parser.add_argument("--precursor-peak-exclusion-window-da", type=float, default=0.0)
    parser.add_argument(
        "--shoulder-da",
        type=float,
        default=DEFAULT_GROUPED_PEAK_SHOULDER_DA,
    )
    parser.add_argument(
        "--isotope-mode",
        choices=("c13", "sirius-window"),
        default="sirius-window",
    )
    parser.add_argument("--isotope-tol-da", type=float, default=0.02)
    parser.add_argument(
        "--isotope-charges",
        type=int,
        nargs="+",
        default=list(DEFAULT_GROUPED_PEAK_ISOTOPE_CHARGES),
    )
    parser.add_argument("--isotope-max-relative-intensity", type=float, default=0.8)
    parser.add_argument("--group-score", choices=("max", "sum"), default="max")
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    summary = analyze(
        artifact_dir=args.artifact_dir,
        sample_rows=args.sample_rows,
        seed=args.seed,
        num_peaks=args.num_peaks,
        min_peak_intensity=args.min_peak_intensity,
        precursor_peak_exclusion_window_da=args.precursor_peak_exclusion_window_da,
        shoulder_da=args.shoulder_da,
        isotope_mode=args.isotope_mode,
        isotope_tol_da=args.isotope_tol_da,
        isotope_charges=tuple(args.isotope_charges),
        isotope_max_relative_intensity=args.isotope_max_relative_intensity,
        group_score=args.group_score,
        examples=args.examples,
    )
    output_path = args.output_json or _default_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    _print_summary(summary)
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
