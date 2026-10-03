"""Run the pinned diffusion extension using the original native study protocol.

Run prepare, fit, then download with .venv/bin/python -m scripts.run_encoder_chemistry_diffusion.
Only raw19/final27 tokens are retained; the encoder still executes its full trace.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scripts import modal_encoder_chemistry as native
from scripts import modal_encoder_chemistry_probes as probes
from scripts import modal_encoder_chemistry_sae as sae

NAME = "diffusion_831k"
EXTENSION = native.LOCAL / "diffusion_extension"
DICTIONARIES = ((32, 0), (32, 1), (64, 0))


def prepare() -> None:
    import google.auth
    from google.auth.transport.requests import Request

    if (native.LOCAL / NAME / "extraction.json").exists():
        raise FileExistsError("Pinned extraction already exists; use fit/download without replacing its provenance")
    with native.volume.batch_upload(force=True) as upload:
        for filename in ("config.json", "endpoint_selection.json"):
            upload.put_file(EXTENSION / filename,
                            str((native.ROOT / "diffusion_extension" / filename).relative_to("/results")))
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/devstorage.read_only"])
    credentials.refresh(Request())
    with native.app.run():
        staged = native.stage_latest.remote(credentials.token, NAME)
        native.write_json(EXTENSION / "copy_manifest.json", staged)
        print("Pinned diffusion copy ready:", staged["copied"], "objects", flush=True)
        pilot = native.pilot.remote(NAME, compare_export=False)
        native.write_json(native.LOCAL / NAME / "pilot.json", pilot)
        if not pilot["native_vs_instrumented"]["bitwise_equal"] or pilot["max_sham_relative_l2"] != 0:
            raise ValueError("Diffusion instrumentation must match the native graph exactly")
        extraction = native.extract.remote(NAME, token_depths=(19, 27))
        native.write_json(native.LOCAL / NAME / "extraction.json", extraction)
        print("Diffusion extraction complete; native cache bitwise parity:",
              extraction["extraction_vs_native_pilot"]["bitwise_equal"], flush=True)


def fit() -> None:
    with probes.app.run(), sae.app.run():
        jobs = []
        for kind in ("attention", "mean"):
            jobs.append((f"readout_{kind}", probes.fit.spawn(NAME, depth=27, kind=kind)))
        for k, seed in DICTIONARIES:
            jobs.append((f"sae_k{k}_seed{seed}", sae.fit.spawn(NAME, k=k, seed=seed)))
        native.write_json(EXTENSION / "launch.json", {
            "name": NAME, "jobs": [{"task": task, "call_id": job.object_id} for task, job in jobs],
        })
        for task, job in jobs:
            result = job.get()
            native.write_json(EXTENSION / f"{task}_result.json", result)
            print(task, "complete", flush=True)


def download_one(relative: Path) -> str:
    destination = native.LOCAL / NAME / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    remote = str((native.ROOT / NAME / relative).relative_to("/results"))
    with temporary.open("wb") as stream:
        for chunk in native.volume.read_file(remote):
            stream.write(chunk)
    temporary.replace(destination)
    return str(relative)


def download() -> None:
    files = [Path("pilot.json"), Path("extraction.json"), Path("readouts/panel_rows.npz")]
    for kind in ("attention", "mean"):
        for seed in range(3):
            directory = Path(f"readouts/depth27/{kind}/seed{seed}")
            files.extend(directory / filename for filename in (
                "trainedprobe.json", "params.npz", "train_predictions.npz", "val_predictions.npz", "test_predictions.npz",
            ))
    for k, seed in DICTIONARIES:
        directory = Path(f"sae/depth19/k{k}/seed{seed}")
        files.extend(directory / filename for filename in (
            "trained_sae.json", "params.npz", "molecule_max.npy", "random_dictionary_molecule_max.npy",
            "dense_channel_molecule_max.npy", "feature_statistics.npz", "dictionary_statistics.npz",
        ))
    with ThreadPoolExecutor(max_workers=6) as executor:
        for job in as_completed([executor.submit(download_one, path) for path in files]):
            print("Downloaded", job.result(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "fit", "download"))
    arguments = parser.parse_args()
    {"prepare": prepare, "fit": fit, "download": download}[arguments.phase]()
