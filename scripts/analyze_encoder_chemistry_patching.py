"""Molecule-paired activation rescue curves, with absolute effects and sham gates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit


def paired_summary(values: np.ndarray, weights: np.ndarray) -> dict:
    return {"mean": float(values.mean()), "ci95": np.quantile(weights @ values, [.025, .975]).tolist()}


def analyze(root: Path, resamples: int = 2000) -> dict:
    data = np.load(root / "panel/inputs.npz")
    labels = np.load(root / "panel/labels.npz")
    truth = np.concatenate([labels["maccs"], labels["binary"]], axis=1).astype(float)
    training = truth[data["fold"] == "train"]
    out = {"scope": "Frozen chemical token readouts. Selected physical slots received top20% intensity attenuation; random rescue restores the same number of uncorrupted slots. No molecule or fragment causal claim.",
           "bootstrap": {"unit": "molecule", "resamples": resamples, "seed": 20260930, "multiplicity": "Descriptive unadjusted intervals across correlated depths; no selected-depth confirmation claim."}, "arms": {}}
    for arm in ("parent_1190k", "periodic_1209k"):
        path = root / arm / "causal/patching_test_256/predictions.npz"
        saved = np.load(path)
        y = truth[saved["rows"]]
        assert np.all(data["fold"][saved["rows"]] == "test")
        assert len(np.unique(data["molecule"][saved["rows"]])) == len(y)
        names = saved["variants"].astype(str).tolist()
        assert names[:2] == ["clean", "corrupt"]
        logits = saved["logits"].mean(2).astype(float)
        # Average seed scores for the loss; logit drift is a separate quantity.
        scores = np.clip(expit(saved["logits"].astype(float)).mean(2), 1e-7, 1-1e-7)
        supported = (y.sum(0) >= 5) & ((1-y).sum(0) >= 5) & (training.sum(0) >= 5) & ((1-training).sum(0) >= 5)
        bce = -y[None]*np.log(scores) - (1-y[None])*np.log1p(-scores)
        balance = np.where(y > 0, len(y)/(2*np.maximum(y.sum(0),1)), len(y)/(2*np.maximum((1-y).sum(0),1)))
        loss = (bce[:,:,supported]*balance[:,supported]).mean(-1)
        logit_error = ((logits[:,:,supported]-logits[0,:,supported].T)**2).mean(-1)
        rng = np.random.default_rng(20260930)
        weights = rng.multinomial(len(y), np.full(len(y),1/len(y)), size=resamples) / len(y)
        record = {"source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "molecules": len(y), "supported_targets": int(supported.sum()),
                  "clean_balanced_bce": float(loss[0].mean()), "corruption_balanced_bce_damage": paired_summary(loss[1]-loss[0],weights),
                  "corruption_logit_mse": float(logit_error[1].mean()), "depths": []}
        for depth in range(26):
            s,r,a = [names.index(f"restore_{kind}_depth{depth}") for kind in ("selected","random","all")]
            if not np.array_equal(saved["logits"][a], saved["logits"][0]):
                raise ValueError("Full-state restoration must exactly rescue the clean output")
            record["depths"].append({"depth": depth,
                "selected_bce_recovery": paired_summary(loss[1]-loss[s],weights),
                "random_bce_recovery": paired_summary(loss[1]-loss[r],weights),
                "selected_minus_random_bce_recovery": paired_summary(loss[r]-loss[s],weights),
                "selected_logit_mse": paired_summary(logit_error[s],weights),
                "random_logit_mse": paired_summary(logit_error[r],weights),
                "selected_minus_random_mse_reduction": paired_summary(logit_error[r]-logit_error[s],weights),
                "full_restore_max_abs_error": float(np.max(np.abs(saved["logits"][a]-saved["logits"][0])))})
        out["arms"][arm] = record
    return out


def plot(summary: dict, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(11,7),layout="constrained")
    for col,(arm,record) in enumerate(summary["arms"].items()):
        for kind,color in (("selected","#237c95"),("random","#b67b41")):
            for row,key in ((0,f"{kind}_bce_recovery"),(1,f"{kind}_logit_mse")):
                vals=[r[key] for r in record["depths"]]
                axes[row,col].plot(range(26),[v["mean"] for v in vals],color=color,label=kind.title()+" slots")
                axes[row,col].fill_between(range(26),[v["ci95"][0] for v in vals],[v["ci95"][1] for v in vals],color=color,alpha=.14)
        axes[0,col].axhline(record["corruption_balanced_bce_damage"]["mean"],ls=":",color="#666666",label="Full-state rescue")
        axes[0,col].set_title(arm.replace("_"," "))
        axes[0,col].set_ylabel("Balanced BCE recovery (higher = better)")
        axes[1,col].set_ylabel("Squared logit distance to clean (lower = better)")
        axes[1,col].set_xlabel("Clean raw state restored after block (0 = embedding)")
        for ax in axes[:,col]:
            ax.axhline(0,color="#777777",lw=.6)
            ax.spines[["top","right"]].set_visible(False)
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle("Activation rescue after attenuating the top20% of peaks · 256 test molecules")
    for suffix in ("png","pdf"):
        fig.savefig(output / f"activation_rescue.{suffix}",dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("artifacts/encoder_chemistry_20260930"))
    args=parser.parse_args()
    output=args.root/"patching_analysis"
    output.mkdir(exist_ok=True)
    summary=analyze(args.root)
    (output/"results.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    plot(summary,output)
    for arm,r in summary["arms"].items():
        print(arm,r["corruption_balanced_bce_damage"],[(d["depth"],d["selected_minus_random_bce_recovery"]) for d in r["depths"] if d["depth"] in (0,5,10,19,25)])


if __name__ == "__main__":
    main()
