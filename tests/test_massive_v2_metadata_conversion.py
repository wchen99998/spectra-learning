import json
from pathlib import Path

import h5py
import numpy as np

from spectra_learning.data.gems.enrich_massive_v2_metadata import (
    index_project_runs,
    read_project_metadata,
    write_project,
)
from spectra_learning.data.gems.prepare_massive_v2 import ACQUISITION_DATASETS


def test_project_metadata_normalizes_within_raw_file(tmp_path: Path) -> None:
    path = tmp_path / "MSV000000001_t0.95_l0.80_grouped.hdf5"
    levels = np.asarray([1, 2, 2, 2, 2, 1], dtype=np.int8)
    with h5py.File(path, "w") as file:
        file.create_dataset("MS level", data=levels)
        file.create_dataset("RT", data=np.asarray([1, 2, 4, 10, 20, 30], dtype=np.float32))
        file.create_dataset("precursor_mz", data=np.asarray([100, 200, 300, 400, 500, 600], dtype=np.float32))
        file.create_dataset("collision_energy", data=np.asarray([10, 20, np.nan, 40, 50, 60], dtype=np.float32))
        file.create_dataset("charge", data=np.asarray([1, 2, 0, 3, 4, 5], dtype=np.int8))
        file.create_dataset("massive_id", data=np.asarray([b"MSV000000001"] * 6))
        file.create_dataset("file_id", data=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32))
        file.create_dataset("group_id", data=np.asarray([-1, 1, 1, -1, -1, -1], dtype=np.int32))
        file.create_dataset("global_group_id", data=np.asarray([-1, 1, 1, -1, -1, -1], dtype=np.int64))
        file.create_dataset(
            "unique_spectrum_id",
            data=np.asarray([f"id-{index}".encode() for index in range(6)]),
        )
        file.create_dataset("acquisition_type", data=np.asarray([b"DDA", b"DDA", b"DDA", b"DIA", b"DIA", b"DIA"]))
        file.create_dataset("positive polarity", data=np.asarray([1, 1, 1, 0, 0, 0], dtype=np.int8))
        file.create_dataset("window lo", data=np.ones(6, dtype=np.float32))
        file.create_dataset("window uo", data=np.full(6, 2, dtype=np.float32))
        file.create_dataset("instrument accuracy est.", data=np.full(6, 1e-4, dtype=np.float32))
        file.create_dataset("precursor intensity", data=np.asarray([1, 2, 4, 8, 16, 32], dtype=np.float32))
        metadata = file.create_group("metadata")
        metadata.create_dataset("instrument name", data=np.asarray([b"Q Exactive", b"Q-TOF"]))

    arrays, validation = read_project_metadata(path)

    np.testing.assert_allclose(
        arrays["retention_time_fraction"],
        np.asarray([0.5, 1.0, 1 / 3, 2 / 3]),
    )
    np.testing.assert_array_equal(arrays["instrument_family_id"], [1, 1, 2, 2])
    np.testing.assert_array_equal(arrays["acquisition_type_id"], [1, 1, 2, 2])
    np.testing.assert_array_equal(arrays["polarity_id"], [1, 1, 2, 2])
    np.testing.assert_array_equal(arrays["collision_energy_present"], [True, False, True, True])
    np.testing.assert_array_equal(arrays["charge_present"], [True, False, True, True])
    np.testing.assert_allclose(arrays["mass_accuracy"], 0.4)
    assert validation.shape == (4,)


def test_enrichment_aligns_by_id_and_marks_source_absent_rows_unknown(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    (artifact / "train").mkdir(parents=True)
    path = artifact / "train/shard_00000.hdf5"
    destination_ids = np.asarray([b"legacy", b"a2", b"a1", b"a0", b"a3"])
    with h5py.File(path, "w") as file:
        file.create_dataset("spectrum", shape=(5, 2, 128), dtype=np.float32)
        file.create_dataset("massive_id", data=np.asarray([b"A"] * 5))
        file.create_dataset("unique_spectrum_id", data=destination_ids)
        file.create_dataset("precursor_mz", data=np.full(5, 500, dtype=np.float32))
        file.create_dataset("collision_energy", data=np.full(5, 30, dtype=np.float32))
        file.create_dataset("charge", data=np.full(5, 2, dtype=np.int8))
        file.create_dataset("RT", data=np.asarray([5, 2, 3, 4, 10], dtype=np.float32))
        file.create_dataset("file_id", data=np.zeros(5, dtype=np.int32))
    manifest = {
        "splits": {
            "train": {"shards": [{"path": "train/shard_00000.hdf5"}]},
            "validation": {"shards": []},
        }
    }
    manifest_path = artifact / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    runs = index_project_runs(manifest_path, tmp_path / "runs.json")

    arrays = {
        "massive_id": np.asarray([b"A"] * 4),
        "unique_spectrum_id": np.asarray([b"a0", b"a1", b"a2", b"a3"]),
        "group_id": np.asarray([-1, 2, 1, -1]),
        "global_group_id": np.asarray([-1, 2, 1, -1]),
    }
    boolean = {
        "precursor_mz_present",
        "collision_energy_present",
        "charge_present",
        "isolation_window_present",
        "mass_accuracy_present",
        "retention_time_present",
        "precursor_intensity_present",
    }
    categorical = {"polarity_id", "acquisition_type_id", "instrument_family_id"}
    for index, name in enumerate(ACQUISITION_DATASETS):
        dtype = np.bool_ if name in boolean else np.int8 if name in categorical else np.float32
        arrays[name] = np.full(4, index + 1, dtype=dtype)

    write_project(artifact, runs["A"], arrays, np.zeros(4, dtype=np.bool_))

    with h5py.File(path, "r") as file:
        assert file["instrument_family_id"][0] == 0
        assert file["precursor_mz_present"][0]
        assert file["retention_time_present"][0]
        assert file["retention_time_fraction"][0] == 0.5
        np.testing.assert_array_equal(
            file["mass_accuracy"][[1, 2, 3, 4]],
            arrays["mass_accuracy"][[2, 1, 0, 3]],
        )
