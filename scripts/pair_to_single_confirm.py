"""Final-only transfer check on previously unprobed validation molecules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from scripts.pair_to_single_pilot import encode, evaluate, restore_model


def export(args):
    metadata = json.loads((args.directory / "training.json").read_text())
    if metadata["steps"] != 15000 or metadata["schedule_steps"] != 15000:
        raise ValueError("Confirmation is restricted to the prespecified 15000-step endpoint")
    data = dict(np.load(args.inputs))
    confirmation = np.load(args.confirmation)
    indices = confirmation["probe_test"]
    model = restore_model(args.directory)
    features = []
    for start in range(0, len(indices), 64):
        idx = indices[start:start + 64]
        batch = {key: jnp.asarray(data[key][idx]) for key in
                 ("peak_mz", "peak_intensity", "peak_valid_mask", "precursor_mz")}
        features.append(np.asarray(encode(model, batch)))
    # Reuse the exact original training and validation features so fitting and
    # hyperparameter selection remain identical to the primary evaluation.
    stored = np.load(args.directory / "embeddings.npz")
    keep = data["fold"][stored["indices"]] != "test"
    destination = args.directory / "confirmation"
    destination.mkdir(exist_ok=True)
    temporary = destination / "embeddings.tmp.npz"
    np.savez(temporary,
             indices=np.concatenate((stored["indices"][keep], indices)),
             features=np.concatenate((stored["features"][keep], *features)))
    temporary.replace(destination / "embeddings.npz")
    print(json.dumps({"confirmation_export": str(destination), "spectra": len(indices)}), flush=True)


def score(args):
    destination = args.directory / "confirmation"
    for readout, filename in (("all", "evaluation.json"), ("mean", "evaluation_mean.json")):
        evaluate(SimpleNamespace(inputs=args.confirmation, output=str(destination), readout=readout))
        actual = json.loads((destination / filename).read_text())
        original = json.loads((args.directory / filename).read_text())
        assert actual["ridge_alpha"] == original["ridge_alpha"]
        np.testing.assert_allclose(actual["maccs_val_auc"], original["maccs_val_auc"], rtol=0, atol=1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export", "score"))
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    {"export": export, "score": score}[args.mode](args)


if __name__ == "__main__":
    main()
