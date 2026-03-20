"""Sweep PMA num_seeds for MLP probe and plot results."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import torch

logging.basicConfig(level=logging.INFO, stream=sys.stdout)


def main() -> None:
    from configs.gems_a_masked_latent_index_small import get_config
    from models.model import PeakSetSIGReg
    from utils.msg_probe import run_msg_probe

    cfg = get_config()
    cfg.msg_probe_type = "mlp"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PeakSetSIGReg(
        model_dim=cfg.model_dim,
        encoder_num_layers=cfg.num_layers,
        encoder_num_heads=cfg.num_heads,
        encoder_num_kv_heads=cfg.num_kv_heads,
        attention_mlp_multiple=cfg.attention_mlp_multiple,
        feature_mlp_hidden_dim=cfg.feature_mlp_hidden_dim,
        encoder_use_rope=cfg.encoder_use_rope,
        encoder_qk_norm=cfg.encoder_qk_norm,
        representation_regularizer=cfg.representation_regularizer,
        sigreg_num_slices=cfg.sigreg_num_slices,
        sigreg_lambda=cfg.sigreg_lambda,
        norm_type=cfg.norm_type,
        use_precursor_token=cfg.use_precursor_token,
        masked_token_loss_weight=cfg.masked_token_loss_weight,
        masked_token_loss_type=cfg.masked_token_loss_type,
        masked_latent_predictor_num_layers=cfg.masked_latent_predictor_num_layers,
    )

    project_root = Path(__file__).resolve().parent.parent
    ckpt_dir = project_root / "experiments/TEST_FAST_CORRECT_3_noemawarmup/trial_000/checkpoints"
    ckpt_path = str(sorted(ckpt_dir.glob("step-*.pt"))[-1])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = {
        k: v for k, v in ckpt["model"].items()
        if not k.startswith("teacher_encoder") and k != "gco_constraint_targets"
    }
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    seed_counts = [8, 16, 32, 64, 128, 256]
    results: dict[int, dict[str, float]] = {}

    for seeds in seed_counts:
        print(f"\n{'='*60}")
        print(f"Running MLP probe with PMA seeds={seeds}")
        print(f"{'='*60}")
        cfg.msg_probe_pma_num_seeds = seeds
        metrics = run_msg_probe(config=cfg, model=model, device=device)
        results[seeds] = {
            "r2_mean": metrics["msg_probe/test/r2_mean"],
            "auc_mean": metrics["msg_probe/test/auc_fg_mean"],
            "r2_mol_weight": metrics["msg_probe/test/r2_mol_weight"],
            "r2_num_heavy_atoms": metrics["msg_probe/test/r2_num_heavy_atoms"],
            "r2_num_rings": metrics["msg_probe/test/r2_num_rings"],
            "r2_logp": metrics["msg_probe/test/r2_logp"],
        }
        print(f"  -> R2 mean={results[seeds]['r2_mean']:.4f}, AUC mean={results[seeds]['auc_mean']:.4f}")

    # Print summary table
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Seeds':>6}  {'R2 mean':>8}  {'AUC mean':>9}  {'R2 mw':>7}  {'R2 ha':>7}  {'R2 rings':>8}  {'R2 logp':>8}")
    for seeds in seed_counts:
        r = results[seeds]
        print(f"{seeds:>6}  {r['r2_mean']:>8.4f}  {r['auc_mean']:>9.4f}  {r['r2_mol_weight']:>7.4f}  {r['r2_num_heavy_atoms']:>7.4f}  {r['r2_num_rings']:>8.4f}  {r['r2_logp']:>8.4f}")

    # Save raw results
    with open(project_root / "scripts/pma_seed_sweep_results.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    r2_means = [results[s]["r2_mean"] for s in seed_counts]
    auc_means = [results[s]["auc_mean"] for s in seed_counts]

    ax1.plot(seed_counts, r2_means, "o-", linewidth=2, markersize=8, color="#2563eb")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(seed_counts)
    ax1.set_xticklabels([str(s) for s in seed_counts])
    ax1.set_xlabel("PMA num_seeds")
    ax1.set_ylabel("Test R2 (mean)")
    ax1.set_title("MLP Probe: R2 vs PMA Seeds")
    ax1.grid(True, alpha=0.3)

    ax2.plot(seed_counts, auc_means, "o-", linewidth=2, markersize=8, color="#dc2626")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(seed_counts)
    ax2.set_xticklabels([str(s) for s in seed_counts])
    ax2.set_xlabel("PMA num_seeds")
    ax2.set_ylabel("Test AUC (mean)")
    ax2.set_title("MLP Probe: AUC vs PMA Seeds")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Impact of PMA Seeds on MLP Probe Performance\n(step 2.4M checkpoint)", fontsize=13)
    fig.tight_layout()
    out_path = project_root / "scripts/pma_seed_sweep.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
