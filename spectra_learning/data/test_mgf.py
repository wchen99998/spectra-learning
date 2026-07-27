from __future__ import annotations

import hashlib
from pathlib import Path

import h5py

from spectra_learning.data.mgf import build_nist_probe_hdf5


def test_build_nist_probe_hdf5_records_source_and_precursor_bounds(
    tmp_path: Path,
) -> None:
    mgf_path = tmp_path / "source.mgf"
    mgf_path.write_text(
        """BEGIN IONS
PEPMASS=100.0
SMILES=CCO
PRECURSORTYPE=[M+H]+
10 100
20 50
END IONS
"""
    )
    output_path = tmp_path / "probe.h5"
    source_bytes = mgf_path.read_bytes()

    metadata = build_nist_probe_hdf5(
        mgf_path,
        output_path,
        min_precursor_mz=50.0,
        max_precursor_mz=500.0,
        compression=None,
    )

    expected_source = {
        "path": str(mgf_path.resolve()),
        "bytes": len(source_bytes),
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
    }
    assert metadata["source"] == expected_source
    assert metadata["min_precursor_mz"] == 50.0
    assert metadata["max_precursor_mz"] == 500.0
    with h5py.File(output_path) as file:
        assert file.attrs["source_path"] == expected_source["path"]
        assert file.attrs["source_bytes"] == expected_source["bytes"]
        assert file.attrs["source_sha256"] == expected_source["sha256"]
        assert file.attrs["min_precursor_mz"] == 50.0
        assert file.attrs["max_precursor_mz"] == 500.0
