from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import torch

from spectra_learning.data.spectra import (
    DEFAULT_GROUPED_PEAK_ISOTOPE_CHARGES,
    DEFAULT_GROUPED_PEAK_SHOULDER_DA,
    DEFAULT_MAX_PRECURSOR_MZ,
    DEFAULT_MIN_PEAK_INTENSITY,
    DEFAULT_PEAK_FILTERING,
)
from spectra_learning.data import septic_shock


def _mwtab_line(sample_id: str, label: str, raw_file_name: str) -> str:
    return (
        "SUBJECT_SAMPLE_FACTORS            \t-\t"
        f"{sample_id}\tSubject_ID:{label} | Sample source:Serum\t"
        f"Disease={label}; RAW_FILE_NAME(Raw file name)={raw_file_name}"
    )


def _encoded_peaks(pairs: list[tuple[float, float]]) -> str:
    values = np.asarray(pairs, dtype=">f4")
    return base64.b64encode(values.tobytes()).decode("ascii")


def _scan_xml(
    num: int,
    pairs: list[tuple[float, float]],
    *,
    ms_level: int = 1,
    precursor_mz: float | None = None,
) -> str:
    precursor = (
        f"<precursorMz>{precursor_mz}</precursorMz>"
        if precursor_mz is not None
        else ""
    )
    peaks = _encoded_peaks(pairs)
    return (
        f'<scan num="{num}" msLevel="{ms_level}" peaksCount="{len(pairs)}">'
        f"{precursor}"
        '<peaks precision="32" byteOrder="network" pairOrder="m/z-int">'
        f"{peaks}</peaks></scan>"
    )


def _write_mzxml(path: Path, scans: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<?xml version=\"1.0\"?><mzXML><msRun>" + "".join(scans) + "</msRun></mzXML>")


def _data(metadata: dict, root: Path) -> septic_shock.SepticShockData:
    return septic_shock.SepticShockData(
        metadata=metadata,
        root=root,
        batch_size=2,
        num_peaks=2,
        max_precursor_mz=DEFAULT_MAX_PRECURSOR_MZ,
        min_peak_intensity=DEFAULT_MIN_PEAK_INTENSITY,
        peak_drop_min_intensity=DEFAULT_MIN_PEAK_INTENSITY,
        peak_filtering=DEFAULT_PEAK_FILTERING,
        grouped_peak_shoulder_da=DEFAULT_GROUPED_PEAK_SHOULDER_DA,
        grouped_peak_isotope_charges=DEFAULT_GROUPED_PEAK_ISOTOPE_CHARGES,
        peak_ordering="mz",
        precursor_peak_exclusion_window_da=0.0,
        ms_level=None,
    )


def test_parse_septic_shock_mwtab_sample_factors() -> None:
    text = "\n".join(
        [
            "#SUBJECT_SAMPLE_FACTORS:\tignored",
            _mwtab_line("1", "Non-septic shock", "1.mzXML"),
            _mwtab_line("4", "Septic shock", "4.mzXML"),
            _mwtab_line("105", "Non-septic shock", "105r.mzXML"),
        ]
    )

    samples = septic_shock.parse_septic_shock_mwtab(text)

    assert [sample["sample_id"] for sample in samples] == ["1", "4", "105"]
    assert [sample["raw_file_name"] for sample in samples] == [
        "1.mzXML",
        "4.mzXML",
        "105r.mzXML",
    ]
    assert [sample["label"] for sample in samples] == [0, 1, 0]
    assert [sample["label_name"] for sample in samples] == [
        "Non-septic shock",
        "Septic shock",
        "Non-septic shock",
    ]


def test_septic_shock_split_counts_match_plan() -> None:
    samples = [
        {
            "sample_index": i,
            "sample_id": str(i),
            "raw_file_name": f"{i}.mzXML",
            "label_name": "Non-septic shock" if i < 92 else "Septic shock",
            "label": 0 if i < 92 else 1,
        }
        for i in range(124)
    ]

    records = septic_shock.build_septic_shock_sample_records(
        samples,
        split_seed=42,
    )
    counts = septic_shock.septic_shock_split_counts(records)

    assert counts == {
        "train": {"size": 68, "positive": 17, "negative": 51},
        "val": {"size": 18, "positive": 5, "negative": 13},
        "test": {"size": 38, "positive": 10, "negative": 28},
    }


def test_read_mzxml_sample_spectra_xml_fallback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(septic_shock, "USE_PYTEOMICS_MZXML", False)
    path = tmp_path / "sample.mzXML"
    _write_mzxml(
        path,
        [
            _scan_xml(1, [(25.0, 2.0), (50.0, 4.0)], ms_level=1),
            _scan_xml(2, [(100.0, 8.0)], ms_level=2, precursor_mz=250.0),
        ],
    )

    spectra, precursor_mz = septic_shock.read_mzxml_sample_spectra(
        path,
        ms_level=2,
    )

    assert spectra.shape == (1, 2, 128)
    assert torch.allclose(torch.from_numpy(spectra[0, 0, :1]), torch.tensor([100.0]))
    assert torch.allclose(torch.from_numpy(spectra[0, 1, :1]), torch.tensor([8.0]))
    assert torch.allclose(torch.from_numpy(precursor_mz), torch.tensor([250.0]))


def test_septic_shock_loader_preserves_sample_grouping(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(septic_shock, "USE_PYTEOMICS_MZXML", False)
    root = tmp_path / "septic"
    _write_mzxml(
        root / "raw" / "mzxml" / "1.mzXML",
        [
            _scan_xml(1, [(25.0, 5.0), (50.0, 10.0), (900.0, 1.0)]),
            _scan_xml(2, [(100.0, 3.0), (125.0, 1.0)]),
        ],
    )
    _write_mzxml(
        root / "raw" / "mzxml" / "4.mzXML",
        [_scan_xml(1, [(30.0, 7.0), (45.0, 14.0)])],
    )
    metadata = {
        "samples": [
            {
                "sample_index": 0,
                "sample_id": "1",
                "raw_file_name": "1.mzXML",
                "label": 0,
                "label_name": "Non-septic shock",
                "split": "train",
                "mzxml_path": "raw/mzxml/1.mzXML",
            },
            {
                "sample_index": 1,
                "sample_id": "4",
                "raw_file_name": "4.mzXML",
                "label": 1,
                "label_name": "Septic shock",
                "split": "train",
                "mzxml_path": "raw/mzxml/4.mzXML",
            },
        ]
    }
    (root / "metadata.json").write_text(json.dumps(metadata))
    data = _data(metadata, root)

    batch = next(
        iter(
            septic_shock.build_septic_shock_loader(
                data,
                "train",
                shuffle=False,
                seed=0,
            )
        )
    )

    assert batch["sample_id"] == ["1", "4"]
    assert torch.equal(batch["label"], torch.tensor([0.0, 1.0]))
    assert torch.equal(batch["spectrum_count"], torch.tensor([2, 1]))
    assert torch.equal(batch["spectrum_sample_index"], torch.tensor([0, 0, 1]))
    assert batch["peak_mz"].shape == (3, 2)
    assert torch.allclose(batch["peak_mz"][0], torch.tensor([0.025, 0.05]))
    assert torch.allclose(batch["peak_intensity"][0], torch.tensor([0.5, 1.0]))
    assert torch.equal(batch["peak_valid_mask"][0], torch.tensor([True, True]))


def test_build_peaklist_artifact_uses_project_loader(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(septic_shock, "USE_PYTEOMICS_MZXML", False)
    cache_dir = tmp_path / "cache"
    samples = []
    for i in range(124):
        label = 0 if i < 92 else 1
        raw_file_name = f"{i}.mzXML"
        _write_mzxml(
            cache_dir / "raw" / "mzxml" / raw_file_name,
            [_scan_xml(1, [(25.0, 5.0), (50.0, 10.0), (900.0, 1.0)])],
        )
        samples.append(
            {
                "sample_index": i,
                "sample_id": str(i),
                "raw_file_name": raw_file_name,
                "label_name": "Non-septic shock" if label == 0 else "Septic shock",
                "label": label,
                "mzxml_path": f"raw/mzxml/{raw_file_name}",
            }
        )
    (cache_dir / "metadata.json").write_text(json.dumps({"samples": samples}))
    artifact_dir = tmp_path / "artifact"

    metadata = septic_shock.build_septic_shock_peaklist_artifact(
        cache_dir=cache_dir,
        output_dir=artifact_dir,
        split_seed=42,
    )

    assert metadata["artifact_format"] == "raw_peaklist_v1"
    assert metadata["num_peaks_input"] == 128
    assert metadata["train_num_scans"] == 68
    assert (artifact_dir / "train" / "shard-00000-of-00001" / "spectra.npy").exists()

    data = septic_shock.build_septic_shock_data(
        cache_dir=artifact_dir,
        batch_size=2,
        num_peaks=2,
    )
    batch = next(
        iter(
            septic_shock.build_septic_shock_loader(
                data,
                "train",
                shuffle=False,
                seed=0,
                max_samples=2,
            )
        )
    )

    assert batch["peak_mz"].shape == (2, 2)
    assert torch.allclose(batch["peak_mz"][0], torch.tensor([0.025, 0.05]))
    assert torch.allclose(batch["peak_intensity"][0], torch.tensor([0.5, 1.0]))
    assert torch.equal(batch["spectrum_count"], torch.tensor([1, 1]))
