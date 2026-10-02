"""Test whether direct pair messages carry chemistry to singles.

Export intact/zero/shuffled bridge test embeddings on GPU, then score
them with the unchanged ridge probe trained on saved intact train embeddings.
By default only the final bridge changes, keeping all earlier blocks intact.
The all-block scope also includes indirect effects on subsequent pair updates
and attention bias through the modified single states.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from scripts.pair_to_single_pilot import encode, restore_model


def export(args):
    data = dict(np.load(args.inputs))
    model = restore_model(args.directory)
    if model.encoder.blocks[-1].bridge is None:
        raise ValueError("Content intervention requires a final pair-value bridge")
    indices = data["probe_test"]
    outputs = {"indices": indices}
    moved = 0
    for condition in ("intact", "zero", "shuffle"):
        blocks = model.encoder.blocks[-1:] if args.scope == "final" else model.encoder.blocks
        for block in blocks:
            block.intervention = condition
        batches = []
        for start in range(0, len(indices), 64):
            idx = indices[start:start + 64]
            batch = {k: jnp.asarray(data[k][idx]) for k in
                     ("peak_mz", "peak_intensity", "peak_valid_mask", "precursor_mz")}
            batches.append(np.asarray(encode(model, batch)))
            if condition == "shuffle":
                # Compact masks are valid prefixes, so matching is exactly
                # matching peak counts (every spectrum also contains CLS).
                counts = data["peak_valid_mask"][idx].sum(1)
                moved += int(((counts[:, None] == counts[None, :]).sum(1) > 1).sum())
        outputs[condition] = np.concatenate(batches)
        print(f"Exported {args.scope}-bridge {condition}: {outputs[condition].shape}", flush=True)
    outputs["changed_donor_fraction"] = moved / len(indices)
    np.savez(intervention_path(args, "npz"), **outputs)


def intervention_path(args, suffix):
    stem = "content_interventions" if args.scope == "final" else "all_content_interventions"
    return args.directory / f"{stem}.{suffix}"


def score(args):
    from sklearn.metrics import roc_auc_score

    data = dict(np.load(args.inputs))
    saved = np.load(args.directory / "embeddings.npz")
    counterfactual = np.load(intervention_path(args, "npz"))
    idx = saved["indices"]
    train, val, test = [data["fold"][idx] == fold for fold in ("train", "val", "test")]
    np.testing.assert_array_equal(idx[test], counterfactual["indices"])
    labels = data["fingerprints"][data["molecule"][idx]].astype(np.float64)
    eligible = (labels[train].sum(0) >= 20) & ((1 - labels[train]).sum(0) >= 20)
    eligible &= labels[val].min(0) < labels[val].max(0)
    bits = eligible & (labels[test].min(0) < labels[test].max(0))
    result = {"scope": f"{args.scope} direct-value bridge intervention; fixed intact-trained chemical probe",
              "changed_donor_fraction": float(counterfactual["changed_donor_fraction"])}
    for readout, width, filename in (("mean_cls", 256, "evaluation.json"),
                                     ("peak_mean", 128, "evaluation_mean.json")):
        reference = json.loads((args.directory / filename).read_text())
        features = saved["features"][:, :width].astype(np.float64)
        mean, scale = features[train].mean(0), np.maximum(features[train].std(0), 1e-5)

        def design(z):
            x = (z.astype(np.float64) - mean) / scale
            return np.concatenate((x, np.ones((len(x), 1))), axis=1)

        x = design(features[train])
        regularizer = np.eye(width + 1) * reference["ridge_alpha"]
        regularizer[-1, -1] = 0
        weights = np.linalg.solve(x.T @ x / train.sum() + regularizer,
                                  x.T @ labels[train] / train.sum())

        def auc(z):
            predictions = design(z) @ weights
            return float(roc_auc_score(labels[test][:, bits], predictions[:, bits], average="macro"))

        original_auc = auc(features[test])
        np.testing.assert_allclose(original_auc, reference["maccs_test_auc"], atol=1e-12, rtol=0)
        result[readout] = {"saved_intact_auc": original_auc,
                           "restored_max_abs_feature_difference": float(np.max(np.abs(
                               counterfactual["intact"][:, :width] - features[test]))),
                           **{f"{condition}_auc": auc(counterfactual[condition][:, :width])
                              for condition in ("intact", "zero", "shuffle")}}
    intervention_path(args, "json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export", "score"))
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--scope", choices=("final", "all"), default="final")
    args = parser.parse_args()
    {"export": export, "score": score}[args.mode](args)


if __name__ == "__main__":
    main()
