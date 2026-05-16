import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ml_collections import config_dict

from spectra_learning.data.gems.collate import GemsBatchCollator
from spectra_learning.data.gems.datamodule import GemsNativeDataModule
from spectra_learning.data.gems.masking import (
    JEPA_MASK_STRATEGIES,
    _normalize_mask_strategy_name,
)


def _mask_block_ranges(mask: torch.Tensor) -> list[tuple[int, int]]:
    positions = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if positions.numel() == 0:
        return []
    ranges: list[tuple[int, int]] = []
    start = int(positions[0].item())
    prev = start
    for value in positions[1:].tolist():
        current = int(value)
        if current != prev + 1:
            ranges.append((start, prev))
            start = current
        prev = current
    ranges.append((start, prev))
    return ranges


def _mask_block_ranges_in_active_order(
    mask: torch.Tensor,
    active_positions: torch.Tensor,
) -> list[tuple[int, int]]:
    active_indices = torch.nonzero(active_positions, as_tuple=False).squeeze(-1)
    if active_indices.numel() == 0:
        return []
    return _mask_block_ranges(mask[active_indices])


def _format_block_ranges(ranges: list[tuple[int, int]]) -> str:
    if not ranges:
        return "[]"
    return "[" + ", ".join(f"({start}, {end})" for start, end in ranges) + "]"


def _make_visualization_collator_kwargs(datamodule: GemsNativeDataModule) -> dict[str, Any]:
    return {
        "num_target_blocks": datamodule.jepa_num_target_blocks,
        "context_fraction": datamodule.jepa_context_fraction,
        "target_fraction": datamodule.jepa_target_fraction,
        "block_min_len": datamodule.jepa_block_min_len,
        "mask_strategy": datamodule.jepa_mask_strategy,
        "mask_lengths": datamodule.jepa_mask_lengths,
        "mask_round_from": datamodule.jepa_mask_round_from,
        "intensity_aware_mask_config": datamodule.jepa_intensity_aware_mask_config,
        "allow_target_overlap": datamodule.jepa_allow_target_overlap,
        "use_precursor_token": datamodule.use_precursor_token,
        "num_peaks": datamodule.num_peaks_output,
        "max_precursor_mz": datamodule.max_precursor_mz,
        "min_peak_intensity": datamodule.min_peak_intensity,
        "peak_drop_min_intensity": datamodule.peak_drop_min_intensity,
        "peak_ordering": datamodule.peak_ordering,
        "precursor_peak_exclusion_window_da": datamodule.precursor_peak_exclusion_window_da,
    }


def _resolve_visualization_strategies(
    config_mask_strategy: str,
    strategies: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    if strategies is not None:
        resolved = [_normalize_mask_strategy_name(strategy) for strategy in strategies]
        return tuple(dict.fromkeys(resolved))
    default_strategies = list(JEPA_MASK_STRATEGIES)
    config_strategy = _normalize_mask_strategy_name(config_mask_strategy)
    if config_strategy not in default_strategies:
        default_strategies.append(config_strategy)
    return tuple(default_strategies)


def _load_real_mask_visualization_batches(
    *,
    config_path: str | Path,
    split: str,
    start_index: int,
    num_samples: int,
    seed: int,
    strategies: tuple[str, ...] | None = None,
) -> tuple[
    config_dict.ConfigDict,
    dict[str, torch.Tensor],
    dict[str, dict[str, torch.Tensor]],
    list[int],
]:
    from spectra_learning.training.api import load_config

    config = load_config(Path(config_path).expanduser().resolve())
    datamodule = GemsNativeDataModule(config, seed=seed)
    dataset = datamodule._get_dataset(split)
    sample_indices = [start_index + offset for offset in range(num_samples)]
    samples = [dataset[index] for index in sample_indices]
    collator_kwargs = _make_visualization_collator_kwargs(datamodule)
    resolved_strategies = _resolve_visualization_strategies(
        datamodule.jepa_mask_strategy,
        strategies,
    )
    raw_batch = GemsBatchCollator(augment=False, **collator_kwargs)(samples)
    strategy_batches: dict[str, dict[str, torch.Tensor]] = {}
    for strategy in resolved_strategies:
        torch.manual_seed(seed)
        strategy_batches[strategy] = GemsBatchCollator(
            augment=True,
            **(collator_kwargs | {"mask_strategy": strategy}),
        )(samples)
    return config, raw_batch, strategy_batches, sample_indices


def _mask_rows_for_plot(
    peak_valid_mask: torch.Tensor,
    context_mask: torch.Tensor,
    target_masks: torch.Tensor,
) -> tuple[np.ndarray, list[str]]:
    rows = [
        peak_valid_mask,
        context_mask,
        *[target_masks[target_idx] for target_idx in range(target_masks.shape[0])],
    ]
    labels = [
        f"valid ({int(peak_valid_mask.sum().item())})",
        f"context ({int(context_mask.sum().item())})",
        *[
            f"target {target_idx} ({int(target_masks[target_idx].sum().item())})"
            for target_idx in range(target_masks.shape[0])
        ],
    ]
    matrix = torch.stack(rows, dim=0).to(torch.float32).cpu().numpy()
    return matrix, labels


def _set_slot_ticks(ax: Any, *, num_slots: int, use_precursor_token: bool) -> None:
    step = max(math.ceil(float(num_slots) / 8.0), 1)
    ticks = list(range(0, num_slots, step))
    if ticks[-1] != num_slots - 1:
        ticks.append(num_slots - 1)
    labels = [str(tick) for tick in ticks]
    if use_precursor_token and ticks:
        labels[0] = "P"
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


def _plot_mask_strategy_panel(
    *,
    ax_slots: Any,
    ax_masks: Any,
    peak_intensity: torch.Tensor,
    peak_valid_mask: torch.Tensor,
    context_mask: torch.Tensor,
    target_masks: torch.Tensor,
    title: str,
    use_precursor_token: bool,
) -> None:
    x = np.arange(peak_intensity.shape[0])
    valid = peak_valid_mask.cpu().numpy().astype(bool)
    context = context_mask.cpu().numpy().astype(bool)
    any_target = target_masks.any(dim=0).cpu().numpy().astype(bool)
    free_valid = valid & (~context) & (~any_target)
    padded = ~valid
    heights = peak_intensity.cpu().numpy()
    _plot_slot_bars(ax_slots, x, heights, free_valid, context, any_target, padded)
    ax_slots.set_xlim(-0.5, len(x) - 0.5)
    ax_slots.set_title(title, fontsize=11, fontweight="bold")
    ax_slots.set_ylabel("Intensity")
    ax_slots.grid(axis="y", alpha=0.2)
    _mark_precursor_slot(ax_slots, use_precursor_token)
    ax_slots.legend(fontsize=7, loc="upper right")
    _set_slot_ticks(
        ax_slots,
        num_slots=peak_intensity.shape[0],
        use_precursor_token=use_precursor_token,
    )
    _plot_mask_rows(
        ax_masks,
        peak_valid_mask=peak_valid_mask,
        context_mask=context_mask,
        target_masks=target_masks,
        use_precursor_token=use_precursor_token,
    )


def _plot_slot_bars(
    ax: Any,
    x: np.ndarray,
    heights: np.ndarray,
    free_valid: np.ndarray,
    context: np.ndarray,
    any_target: np.ndarray,
    padded: np.ndarray,
) -> None:
    if np.any(free_valid):
        ax.bar(x[free_valid], heights[free_valid], width=0.82, color="#cbd5e1", label="Valid unused")
    if np.any(context):
        ax.bar(x[context], heights[context], width=0.82, color="#2563eb", label="Context")
    if np.any(any_target):
        ax.bar(x[any_target], heights[any_target], width=0.82, color="#f97316", label="Target")
    if np.any(padded):
        ax.scatter(
            x[padded],
            np.zeros(int(padded.sum())),
            color="#94a3b8",
            marker="x",
            s=18,
            linewidths=1.0,
            label="Padding",
            zorder=5,
        )


def _mark_precursor_slot(ax: Any, use_precursor_token: bool) -> None:
    if use_precursor_token:
        ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.35)
        ax.text(
            0.01,
            0.95,
            "slot P = precursor",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
        )


def _plot_mask_rows(
    ax: Any,
    *,
    peak_valid_mask: torch.Tensor,
    context_mask: torch.Tensor,
    target_masks: torch.Tensor,
    use_precursor_token: bool,
) -> None:
    mask_matrix, row_labels = _mask_rows_for_plot(
        peak_valid_mask=peak_valid_mask,
        context_mask=context_mask,
        target_masks=target_masks,
    )
    ax.imshow(mask_matrix, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("Model input slot (P = precursor)" if use_precursor_token else "Peak slot")
    _set_slot_ticks(
        ax,
        num_slots=peak_valid_mask.shape[0],
        use_precursor_token=use_precursor_token,
    )


def _print_mask_strategy_summary(
    *,
    strategy: str,
    batch: dict[str, torch.Tensor],
    sample_index: int,
    dataset_index: int,
    use_precursor_token: bool,
) -> None:
    full_valid = batch["peak_valid_mask"][sample_index]
    full_context = batch["context_mask"][sample_index]
    full_targets = batch["target_masks"][sample_index]
    peak_valid = full_valid[1:] if use_precursor_token else full_valid
    peak_context = full_context[1:] if use_precursor_token else full_context
    peak_targets = full_targets[:, 1:] if use_precursor_token else full_targets
    valid_target_positions = peak_valid & (~peak_context)
    print(
        f"{strategy} | dataset_index={dataset_index} | "
        f"valid={int(full_valid.sum().item())} | "
        f"context={int(full_context.sum().item())} | "
        f"target_counts={[int(mask.sum().item()) for mask in full_targets]}"
    )
    if use_precursor_token:
        print("  model slot P is the precursor token; active-order blocks ignore it")
    print(
        "  context: "
        f"model-slot={_format_block_ranges(_mask_block_ranges(full_context))} | "
        f"active-order={_format_block_ranges(_mask_block_ranges_in_active_order(peak_context, peak_valid))}"
    )
    for target_idx in range(full_targets.shape[0]):
        print(
            f"  target {target_idx}: "
            f"model-slot={_format_block_ranges(_mask_block_ranges(full_targets[target_idx]))} | "
            f"active-order={_format_block_ranges(_mask_block_ranges_in_active_order(peak_targets[target_idx], valid_target_positions))}"
        )


def visualize_real_mask_strategies(
    *,
    config_path: str | Path,
    split: str,
    start_index: int,
    num_samples: int,
    seed: int,
    output_path: str | Path,
    strategies: tuple[str, ...] | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config, raw_batch, strategy_batches, sample_indices = _load_real_mask_visualization_batches(
        config_path=config_path,
        split=split,
        start_index=start_index,
        num_samples=num_samples,
        seed=seed,
        strategies=strategies,
    )
    resolved_strategies = tuple(strategy_batches.keys())
    use_precursor_token = bool(config.get("use_precursor_token", False))
    fig, axes = plt.subplots(
        num_samples * 2,
        len(resolved_strategies),
        figsize=(5.8 * len(resolved_strategies), 4.2 * num_samples),
        height_ratios=[ratio for _ in range(num_samples) for ratio in (3.0, 1.2)],
        squeeze=False,
    )
    fig.suptitle(
        "Real-data JEPA masking on "
        f"{split} split | samples {sample_indices[0]}-{sample_indices[-1]} | "
        f"seed={seed} | strategies={','.join(resolved_strategies)}",
        fontsize=14,
        fontweight="bold",
    )
    _draw_strategy_grid(
        axes=axes,
        raw_batch=raw_batch,
        strategy_batches=strategy_batches,
        sample_indices=sample_indices,
        num_samples=num_samples,
        use_precursor_token=use_precursor_token,
    )
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved real-data mask visualization to {output_path}")
    return output_path


def _draw_strategy_grid(
    *,
    axes: Any,
    raw_batch: dict[str, torch.Tensor],
    strategy_batches: dict[str, dict[str, torch.Tensor]],
    sample_indices: list[int],
    num_samples: int,
    use_precursor_token: bool,
) -> None:
    for row_offset, dataset_index in enumerate(sample_indices):
        peak_intensity = raw_batch["peak_intensity"][row_offset]
        peak_valid_mask = raw_batch["peak_valid_mask"][row_offset]
        for col_idx, (strategy, batch) in enumerate(strategy_batches.items()):
            context_mask = batch["context_mask"][row_offset]
            target_masks = batch["target_masks"][row_offset]
            _plot_mask_strategy_panel(
                ax_slots=axes[row_offset * 2, col_idx],
                ax_masks=axes[row_offset * 2 + 1, col_idx],
                peak_intensity=peak_intensity,
                peak_valid_mask=peak_valid_mask,
                context_mask=context_mask,
                target_masks=target_masks,
                title=_strategy_panel_title(strategy, dataset_index, context_mask, target_masks),
                use_precursor_token=use_precursor_token,
            )
            if row_offset == num_samples - 1:
                axes[row_offset * 2, col_idx].set_xlabel(
                    "Model input slot (P = precursor)" if use_precursor_token else "Peak slot"
                )
            _print_mask_strategy_summary(
                strategy=strategy,
                batch=batch,
                sample_index=row_offset,
                dataset_index=dataset_index,
                use_precursor_token=use_precursor_token,
            )


def _strategy_panel_title(
    strategy: str,
    dataset_index: int,
    context_mask: torch.Tensor,
    target_masks: torch.Tensor,
) -> str:
    return (
        f"{strategy.title()} | dataset[{dataset_index}] | "
        f"context={int(context_mask.sum().item())} | "
        f"targets={[int(mask.sum().item()) for mask in target_masks]}"
    )
