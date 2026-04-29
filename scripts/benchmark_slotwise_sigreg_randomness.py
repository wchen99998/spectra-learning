"""Benchmark compiled SlotwiseSIGReg under different randomness strategies."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

sys.path.insert(0, ".")

from spectra_learning.models.losses import SlotwiseSIGReg

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def sample_directions(
    dim: int,
    num_slices: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    directions = torch.randn(dim, num_slices, device=device, dtype=dtype)
    return directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)


def sample_direction_pool(
    pool_size: int,
    dim: int,
    num_slices: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    pool = torch.randn(pool_size, dim, num_slices, device=device, dtype=dtype)
    return pool / pool.norm(dim=1, keepdim=True).clamp_min(1e-12)


def make_inputs(
    batch_size: int,
    num_views: int,
    num_slots: int,
    dim: int,
    invalid_fraction: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    proj = torch.randn(
        batch_size,
        num_views,
        num_slots,
        dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    valid_mask = torch.rand(batch_size, num_views, num_slots, device=device) > invalid_fraction
    valid_mask[..., 0] = True
    return proj, valid_mask


class InlineRandomness(nn.Module):
    def __init__(self, sigreg: SlotwiseSIGReg):
        super().__init__()
        self.sigreg = sigreg

    def forward(self, proj: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return self.sigreg(proj, valid_mask=valid_mask)


class InputRandomness(nn.Module):
    def __init__(self, sigreg: SlotwiseSIGReg):
        super().__init__()
        self.sigreg = sigreg

    def forward(
        self,
        proj: torch.Tensor,
        valid_mask: torch.Tensor,
        directions: torch.Tensor,
    ) -> torch.Tensor:
        return self.sigreg(
            proj,
            valid_mask=valid_mask,
            directions=directions,
        )


class BufferedRandomness(nn.Module):
    def __init__(self, sigreg: SlotwiseSIGReg, directions: torch.Tensor):
        super().__init__()
        self.sigreg = sigreg
        self.register_buffer("directions", directions)

    def forward(self, proj: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return self.sigreg(
            proj,
            valid_mask=valid_mask,
            directions=self.directions,
        )


@dataclass
class Variant:
    name: str
    description: str
    module: nn.Module
    before_step: Callable[[int], None]
    args_for_step: Callable[[int], tuple[torch.Tensor, ...]]


@dataclass
class BenchResult:
    name: str
    description: str
    steps_per_second: float
    slot_tokens_per_second: float
    peak_memory_gb: float
    step_rates: tuple[float, ...]


def build_variants(
    proj: torch.Tensor,
    valid_mask: torch.Tensor,
    num_slices: int,
    pool_size: int,
) -> list[Variant]:
    device = proj.device
    dtype = proj.dtype
    dim = proj.shape[-1]
    direction_pool = sample_direction_pool(pool_size, dim, num_slices, device, dtype)
    initial_directions = direction_pool[0].clone()

    inline_module = InlineRandomness(SlotwiseSIGReg(num_slices=num_slices).to(device))
    input_randn_module = InputRandomness(SlotwiseSIGReg(num_slices=num_slices).to(device))
    input_precomputed_module = InputRandomness(
        SlotwiseSIGReg(num_slices=num_slices).to(device)
    )
    buffer_refresh_randn_module = BufferedRandomness(
        SlotwiseSIGReg(num_slices=num_slices).to(device),
        initial_directions.clone(),
    )
    buffer_refresh_precomputed_module = BufferedRandomness(
        SlotwiseSIGReg(num_slices=num_slices).to(device),
        initial_directions.clone(),
    )
    buffer_fixed_module = BufferedRandomness(
        SlotwiseSIGReg(num_slices=num_slices).to(device),
        initial_directions.clone(),
    )

    return [
        Variant(
            name="inline_randn",
            description="sample directions inside the compiled forward",
            module=inline_module,
            before_step=lambda _: None,
            args_for_step=lambda _step, proj=proj, valid_mask=valid_mask: (proj, valid_mask),
        ),
        Variant(
            name="input_randn",
            description="sample directions outside compile and pass them as an input",
            module=input_randn_module,
            before_step=lambda _: None,
            args_for_step=(
                lambda _step,
                proj=proj,
                valid_mask=valid_mask,
                dim=dim,
                num_slices=num_slices,
                device=device,
                dtype=dtype: (
                    proj,
                    valid_mask,
                    sample_directions(dim, num_slices, device, dtype),
                )
            ),
        ),
        Variant(
            name="input_precomputed_pool",
            description="pass a precomputed directions tensor as an input",
            module=input_precomputed_module,
            before_step=lambda _: None,
            args_for_step=(
                lambda step,
                proj=proj,
                valid_mask=valid_mask,
                direction_pool=direction_pool,
                pool_size=pool_size: (
                    proj,
                    valid_mask,
                    direction_pool[step % pool_size],
                )
            ),
        ),
        Variant(
            name="buffer_refresh_randn",
            description="refresh a stable directions buffer from fresh randn each step",
            module=buffer_refresh_randn_module,
            before_step=(
                lambda _step,
                module=buffer_refresh_randn_module,
                dim=dim,
                num_slices=num_slices,
                device=device,
                dtype=dtype: module.directions.copy_(
                    sample_directions(dim, num_slices, device, dtype)
                )
            ),
            args_for_step=lambda _step, proj=proj, valid_mask=valid_mask: (proj, valid_mask),
        ),
        Variant(
            name="buffer_refresh_precomputed",
            description="refresh a stable directions buffer from a precomputed pool",
            module=buffer_refresh_precomputed_module,
            before_step=(
                lambda step,
                module=buffer_refresh_precomputed_module,
                direction_pool=direction_pool,
                pool_size=pool_size: module.directions.copy_(
                    direction_pool[step % pool_size]
                )
            ),
            args_for_step=lambda _step, proj=proj, valid_mask=valid_mask: (proj, valid_mask),
        ),
        Variant(
            name="buffer_fixed",
            description="reuse one fixed directions buffer for every step",
            module=buffer_fixed_module,
            before_step=lambda _: None,
            args_for_step=lambda _step, proj=proj, valid_mask=valid_mask: (proj, valid_mask),
        ),
    ]


def benchmark_variant(
    variant: Variant,
    proj: torch.Tensor,
    warmup: int,
    steps: int,
    repeats: int,
    compile_mode: str,
    fullgraph: bool,
) -> BenchResult:
    torch.compiler.reset()
    compiled_module = torch.compile(
        variant.module,
        mode=compile_mode,
        fullgraph=fullgraph,
    )

    for step in range(warmup):
        torch.compiler.cudagraph_mark_step_begin()
        proj.grad = None
        variant.before_step(step)
        loss = compiled_module(*variant.args_for_step(step))
        loss.backward()

    torch.cuda.synchronize()
    slot_tokens = proj.shape[0] * proj.shape[1] * proj.shape[2]
    step_rates: list[float] = []
    peak_memories: list[float] = []
    for repeat in range(repeats):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for step in range(steps):
            global_step = warmup + repeat * steps + step
            torch.compiler.cudagraph_mark_step_begin()
            proj.grad = None
            variant.before_step(global_step)
            loss = compiled_module(*variant.args_for_step(global_step))
            loss.backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        step_rates.append(steps / elapsed)
        peak_memories.append(torch.cuda.max_memory_allocated() / (1024 ** 3))

    median_steps_per_second = statistics.median(step_rates)
    result = BenchResult(
        name=variant.name,
        description=variant.description,
        steps_per_second=median_steps_per_second,
        slot_tokens_per_second=slot_tokens * median_steps_per_second,
        peak_memory_gb=max(peak_memories),
        step_rates=tuple(step_rates),
    )

    del compiled_module
    torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--num-slots", type=int, default=64)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--num-slices", type=int, default=256)
    parser.add_argument("--invalid-fraction", type=float, default=0.1)
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="float32")
    parser.add_argument(
        "--float32-matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
    )
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pool-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fullgraph", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision(args.float32_matmul_precision)

    device = torch.device("cuda")
    dtype = _DTYPES[args.dtype]
    proj, valid_mask = make_inputs(
        batch_size=args.batch_size,
        num_views=args.num_views,
        num_slots=args.num_slots,
        dim=args.dim,
        invalid_fraction=args.invalid_fraction,
        device=device,
        dtype=dtype,
    )
    variants = build_variants(
        proj=proj,
        valid_mask=valid_mask,
        num_slices=args.num_slices,
        pool_size=args.pool_size,
    )

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        "Compile:",
        f"mode={args.compile_mode}",
        f"fullgraph={args.fullgraph}",
    )
    print(
        "Shape:",
        f"batch={args.batch_size}",
        f"views={args.num_views}",
        f"slots={args.num_slots}",
        f"dim={args.dim}",
        f"slices={args.num_slices}",
    )
    print(
        "Runtime:",
        f"dtype={args.dtype}",
        f"matmul={args.float32_matmul_precision}",
        f"warmup={args.warmup}",
        f"steps={args.steps}",
        f"repeats={args.repeats}",
        f"pool={args.pool_size}",
    )
    print()

    results: list[BenchResult] = []
    for variant in variants:
        print("=" * 80)
        print(f"{variant.name}: {variant.description}")
        result = benchmark_variant(
            variant=variant,
            proj=proj,
            warmup=args.warmup,
            steps=args.steps,
            repeats=args.repeats,
            compile_mode=args.compile_mode,
            fullgraph=args.fullgraph,
        )
        print(
            f"  {result.steps_per_second:.2f} steps/s"
            f"  {result.slot_tokens_per_second / 1e6:.2f}M slot-tokens/s"
            f"  {result.peak_memory_gb:.2f} GB peak"
        )
        print(
            " ",
            "runs=",
            ", ".join(f"{rate:.2f}" for rate in result.step_rates),
        )
        print()
        results.append(result)

    baseline = next(result for result in results if result.name == "inline_randn")
    print("=" * 80)
    print(
        f"{'variant':<28}"
        f"{'steps/s':>12}"
        f"{'relative':>12}"
        f"{'slot-tok/s':>16}"
        f"{'peak GB':>12}"
    )
    print("-" * 80)
    for result in results:
        print(
            f"{result.name:<28}"
            f"{result.steps_per_second:>12.2f}"
            f"{result.steps_per_second / baseline.steps_per_second:>12.3f}"
            f"{result.slot_tokens_per_second / 1e6:>16.2f}"
            f"{result.peak_memory_gb:>12.2f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
