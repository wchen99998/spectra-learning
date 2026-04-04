"""Benchmark: float32 matmul precision "high" (TF32) vs "medium" (bf16 internal).

Teacher now has explicit @autocast in model.py. This measures whether
set_float32_matmul_precision affects overall step throughput for the
remaining fp32 matmuls (embedding stats, loss, etc.).

Uses Muon optimizer + reduce-overhead compile to match gems_small.py.
"""

import time
import torch
import sys
sys.path.insert(0, ".")

from ml_collections import config_dict
from utils.training import build_model_from_config


def get_bench_config() -> config_dict.ConfigDict:
    """Config matching gems_small.py."""
    cfg = config_dict.ConfigDict()
    cfg.num_peaks = 64
    cfg.model_dim = 768
    cfg.encoder_num_layers = 12
    cfg.encoder_num_heads = 12
    cfg.encoder_num_kv_heads = 12
    cfg.encoder_num_register_tokens = 4
    cfg.encoder_apply_final_norm = False
    cfg.encoder_qk_norm = False
    cfg.encoder_fourier_strategy = "lin_float_int"
    cfg.encoder_fourier_x_min = 1e-4
    cfg.encoder_fourier_x_max = 1000.0
    cfg.encoder_fourier_funcs = "sin"
    cfg.encoder_fourier_num_freqs = 512
    cfg.encoder_fourier_sigma = 10.0
    cfg.encoder_fourier_trainable = True
    cfg.attention_mlp_multiple = 4.0
    cfg.feature_mlp_hidden_dim = 1024
    cfg.jepa_num_target_blocks = 2
    cfg.jepa_context_fraction = 0.35
    cfg.jepa_target_fraction = 0.2
    cfg.jepa_block_min_len = 1
    cfg.norm_type = "layernorm"
    cfg.predictor_num_register_tokens = 4
    cfg.predictor_apply_final_norm = False
    cfg.masked_latent_predictor_num_layers = 10
    cfg.masked_latent_predictor_num_heads = 16
    cfg.temporal_predictor_num_layers = 0
    cfg.predictor_dim = 384
    cfg.predictor_dropout = 0.1
    cfg.masked_token_loss_weight = 1.0
    cfg.masked_token_loss_type = "l2"
    cfg.jepa_target_normalization = "none"
    cfg.jepa_target_layers = [1, 4, 8, 12]
    cfg.use_ema_teacher_target = True
    cfg.teacher_ema_decay = 0.995
    cfg.teacher_ema_decay_start = 0.99
    cfg.teacher_ema_decay_warmup_steps = 500_000
    cfg.teacher_ema_update_every = 1
    cfg.use_precursor_token = False
    cfg.augmentation_mz_jitter_std = 0.0002
    cfg.augmentation_intensity_jitter_std = 0.001
    cfg.optimizer = "muon"
    cfg.learning_rate = 5e-4
    cfg.weight_decay = 0.05
    cfg.b2 = 0.95
    cfg.warmup_steps = 0
    cfg.min_learning_rate = 3e-5
    cfg.optimizer_capturable = True
    cfg.optimizer_fused = True
    cfg.muon_lr = None
    cfg.adamw_lr = None
    cfg.muon_momentum = 0.95
    cfg.muon_nesterov = True
    cfg.muon_weight_decay = None
    return cfg


def make_fake_batch(batch_size: int, num_peaks: int, device: torch.device):
    peak_mz = torch.rand(batch_size, num_peaks, device=device) * 1000.0
    peak_mz, _ = peak_mz.sort(dim=1)
    peak_intensity = torch.rand(batch_size, num_peaks, device=device)
    peak_valid_mask = torch.ones(batch_size, num_peaks, device=device, dtype=torch.bool)
    peak_valid_mask &= torch.rand(batch_size, num_peaks, device=device) > 0.1
    context_mask = peak_valid_mask & (
        torch.rand(batch_size, num_peaks, device=device) < 0.35
    )
    context_mask[:, 0] = peak_valid_mask[:, 0]
    K = 2
    target_masks = (
        torch.rand(batch_size, K, num_peaks, device=device) < 0.2
    ) & peak_valid_mask.unsqueeze(1)
    return {
        "peak_mz": peak_mz,
        "peak_intensity": peak_intensity,
        "peak_valid_mask": peak_valid_mask,
        "context_mask": context_mask,
        "target_masks": target_masks,
        "precursor_mz": torch.rand(batch_size, device=device) * 500.0,
    }


def bench_steps(
    model,
    step_fn,
    batch,
    autocast_dtype,
    warmup: int = 20,
    measure: int = 100,
) -> float:
    model.train()
    for _ in range(warmup):
        torch.compiler.cudagraph_mark_step_begin()
        with torch.autocast("cuda", dtype=autocast_dtype):
            metrics = model.forward_augmented(batch)
        step_fn(metrics)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(measure):
        torch.compiler.cudagraph_mark_step_begin()
        with torch.autocast("cuda", dtype=autocast_dtype):
            metrics = model.forward_augmented(batch)
        step_fn(metrics)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return measure / elapsed


def build_and_bench(
    precision: str,
    cfg: config_dict.ConfigDict,
    device: torch.device,
    batch_size: int,
    warmup: int,
    steps: int,
) -> float:
    """Fresh model + optimizer for a given precision setting. Returns steps/s."""
    from train import _build_optimizers

    torch.compiler.reset()
    torch.set_float32_matmul_precision(precision)

    model = build_model_from_config(cfg).to(device)
    model.forward_augmented = torch.compile(
        model.forward_augmented, mode="reduce-overhead", fullgraph=False,
    )
    total_steps = 100_000
    optimizers, schedulers = _build_optimizers(cfg, model, total_steps, device)
    batch = make_fake_batch(batch_size, cfg.num_peaks, device)

    def _step(metrics):
        metrics["loss"].backward()
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)
        for sched in schedulers:
            sched.step()
        model.update_teacher()

    sps = bench_steps(model, _step, batch, torch.bfloat16, warmup=warmup, measure=steps)

    del model, optimizers, schedulers, batch
    torch.cuda.empty_cache()
    return sps


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda")
    cfg = get_bench_config()

    model_tmp = build_model_from_config(cfg)
    param_count = sum(p.numel() for p in model_tmp.parameters())
    teacher_count = sum(
        p.numel() for p in model_tmp.teacher_encoder.parameters()
    ) if model_tmp.teacher_encoder is not None else 0
    del model_tmp
    print(f"Model params: {param_count:,}  Teacher params: {teacher_count:,}")
    print(f"Batch size: {args.batch_size}  Num peaks: {cfg.num_peaks}")
    print(f"Optimizer: Muon (GNS)  Compile: reduce-overhead")
    print(f"Teacher: explicit @autocast(bf16)")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    results: list[tuple[str, float]] = []
    for precision in ["high", "medium"]:
        print("=" * 60)
        print(f"float32_matmul_precision = \"{precision}\"")
        print("=" * 60)
        sps = build_and_bench(
            precision, cfg, device, args.batch_size, args.warmup, args.steps,
        )
        print(f"  {sps:.2f} steps/s")
        print()
        results.append((precision, sps))

    # Summary
    high_sps = results[0][1]
    med_sps = results[1][1]
    delta = (med_sps - high_sps) / high_sps * 100
    print("=" * 60)
    print(f"  high (TF32):    {high_sps:.2f} steps/s")
    print(f"  medium (bf16):  {med_sps:.2f} steps/s")
    print(f"  Delta:          {delta:+.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
