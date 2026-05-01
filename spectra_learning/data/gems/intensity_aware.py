from __future__ import annotations

import torch

INTENSITY_AWARE_MASK_STRATEGY = "intensity_aware"

AWARE_MIXED_MASK_CONFIG = {
    "tau": 0.5,
    "alpha": 0.75,
    "beta_context": 0.85,
    "beta_target": 0.85,
    "eps_context": 0.04,
    "eps_target": 0.08,
    "anchor_keep": 0.60,
    "min_eff_context": 4.0,
    "min_eff_target": 1.8,
    "tail_target_mix": 0.10,
    "local_gap_da": 1.0,
    "local_gap_probability": 0.50,
    "min_unused_mass": 0.09,
}


def _effective_count_from_stats(
    mass: torch.Tensor,
    weighted_log_mass: torch.Tensor,
) -> float:
    if float(mass.item()) <= 0.0:
        return 0.0
    entropy = torch.log(mass) - weighted_log_mass / mass
    return float(torch.exp(entropy).item())


def _weighted_sample_until(
    *,
    indices: torch.Tensor,
    p_row: torch.Tensor,
    weights: torch.Tensor,
    mass_target: float,
    min_eff: float,
    max_count: int | None = None,
) -> torch.Tensor:
    mask = torch.zeros_like(p_row, dtype=torch.bool)
    if indices.numel() == 0:
        return mask
    count_cap = (
        int(indices.numel())
        if max_count is None
        else min(int(max_count), int(indices.numel()))
    )
    order = indices[
        torch.multinomial(weights / weights.sum(), int(indices.numel()), replacement=False)
    ]
    mass = p_row.new_zeros(())
    weighted_log_mass = p_row.new_zeros(())
    for idx in order[:count_cap]:
        mask[idx] = True
        value = p_row[idx]
        mass = mass + value
        weighted_log_mass = weighted_log_mass + value * torch.log(value)
        if float(mass.item()) >= float(mass_target) and _effective_count_from_stats(
            mass,
            weighted_log_mass,
        ) >= float(min_eff):
            break
    return mask


def _weighted_fill_context_until(
    *,
    base_mask: torch.Tensor,
    indices: torch.Tensor,
    p_row: torch.Tensor,
    weights: torch.Tensor,
    total_mass_target: float,
    total_min_eff: float,
    max_total_mass: float,
) -> torch.Tensor:
    mask = base_mask.clone()
    mass = p_row[mask].sum()
    weighted_log_mass = (p_row[mask] * torch.log(p_row[mask])).sum()
    if float(mass.item()) >= float(
        total_mass_target
    ) and _effective_count_from_stats(
        mass,
        weighted_log_mass,
    ) >= float(total_min_eff):
        return mask
    if indices.numel() == 0:
        return mask
    order = indices[
        torch.multinomial(weights / weights.sum(), int(indices.numel()), replacement=False)
    ]
    for idx in order:
        value = p_row[idx]
        if float((mass + value).item()) > float(max_total_mass):
            continue
        mask[idx] = True
        mass = mass + value
        weighted_log_mass = weighted_log_mass + value * torch.log(value)
        if float(mass.item()) >= float(
            total_mass_target
        ) and _effective_count_from_stats(
            mass,
            weighted_log_mass,
        ) >= float(total_min_eff):
            break
    return mask


def _intensity_bands(
    p_row: torch.Tensor,
    valid_row: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = torch.nonzero(valid_row, as_tuple=False).squeeze(-1)
    sorted_idx = idx[torch.argsort(p_row[idx], descending=True, stable=True)]
    cumulative = torch.cumsum(p_row[sorted_idx], dim=0)
    previous = cumulative - p_row[sorted_idx]

    high = torch.zeros_like(valid_row, dtype=torch.bool)
    medium = torch.zeros_like(valid_row, dtype=torch.bool)
    low = torch.zeros_like(valid_row, dtype=torch.bool)
    high[sorted_idx[previous < 0.45]] = True
    medium[sorted_idx[(previous >= 0.45) & (previous < 0.85)]] = True
    low[sorted_idx[previous >= 0.85]] = True
    return sorted_idx, high, medium, low


def _near_target_mask(
    mz_row_da: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    gap_da: float,
) -> torch.Tensor:
    target_mz = mz_row_da[target_mask]
    if target_mz.numel() == 0:
        return torch.zeros_like(target_mask, dtype=torch.bool)
    return (torch.abs(mz_row_da.unsqueeze(1) - target_mz.unsqueeze(0)) <= float(gap_da)).any(dim=1)


def sample_intensity_aware_masks_torch(
    peak_valid_mask: torch.Tensor,
    peak_intensity: torch.Tensor,
    peak_mz_da: torch.Tensor,
    *,
    num_target_blocks: int,
    tau: float = AWARE_MIXED_MASK_CONFIG["tau"],
    alpha: float = AWARE_MIXED_MASK_CONFIG["alpha"],
    beta_context: float = AWARE_MIXED_MASK_CONFIG["beta_context"],
    beta_target: float = AWARE_MIXED_MASK_CONFIG["beta_target"],
    eps_context: float = AWARE_MIXED_MASK_CONFIG["eps_context"],
    eps_target: float = AWARE_MIXED_MASK_CONFIG["eps_target"],
    anchor_keep: float = AWARE_MIXED_MASK_CONFIG["anchor_keep"],
    min_eff_context: float = AWARE_MIXED_MASK_CONFIG["min_eff_context"],
    min_eff_target: float = AWARE_MIXED_MASK_CONFIG["min_eff_target"],
    tail_target_mix: float = AWARE_MIXED_MASK_CONFIG["tail_target_mix"],
    local_gap_da: float = AWARE_MIXED_MASK_CONFIG["local_gap_da"],
    local_gap_probability: float = AWARE_MIXED_MASK_CONFIG["local_gap_probability"],
    min_unused_mass: float = AWARE_MIXED_MASK_CONFIG["min_unused_mass"],
) -> tuple[torch.Tensor, torch.Tensor]:
    device = peak_valid_mask.device
    batch_size, num_peaks = peak_valid_mask.shape
    context_mask = torch.zeros(batch_size, num_peaks, dtype=torch.bool, device=device)
    target_masks = torch.zeros(
        batch_size,
        int(num_target_blocks),
        num_peaks,
        dtype=torch.bool,
        device=device,
    )
    normalized = peak_intensity.float() * peak_valid_mask.float()
    normalized = normalized / normalized.sum(dim=1, keepdim=True).clamp_min(1e-12)
    entropy_terms = torch.zeros_like(normalized)
    positive = normalized > 0.0
    entropy_terms[positive] = -normalized[positive] * torch.log(normalized[positive])
    valid_count = peak_valid_mask.sum(dim=1).float().clamp_min(2.0)
    normalized_entropy = entropy_terms.sum(dim=1) / torch.log(valid_count)
    slots = torch.arange(num_peaks, device=device)

    for row_idx in range(batch_size):
        row_valid = peak_valid_mask[row_idx]
        valid_idx = torch.nonzero(row_valid, as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        p_row = normalized[row_idx]
        _, high, medium, low = _intensity_bands(p_row, row_valid)
        high_idx = slots[high]
        medium_idx = slots[medium]
        low_idx = slots[low]

        high_weights = (p_row[high_idx] + 1e-6).pow(float(alpha))
        context_mask[row_idx] |= _weighted_sample_until(
            indices=high_idx,
            p_row=p_row,
            weights=high_weights,
            mass_target=float(anchor_keep) * float(p_row[high_idx].sum().item()),
            min_eff=1.0,
        )

        h = float(normalized_entropy[row_idx].item())
        target_mass = float(torch.tensor(0.07 + 0.09 * float(tau) + 0.02 * h).clamp(0.06, 0.22).item())
        context_mass = float(torch.tensor(0.58 - 0.10 * float(tau) + 0.05 * h).clamp(0.42, 0.68).item())
        target_count_cap = max(1, int(round(float(valid_idx.numel()) * (0.12 + 0.10 * float(tau)))))

        base_target_candidates = torch.cat([medium_idx, high_idx[~context_mask[row_idx, high_idx]]])
        if float(torch.rand((), device=device).item()) < float(tail_target_mix):
            target_candidates = torch.cat([base_target_candidates, low_idx])
        else:
            target_candidates = base_target_candidates
        target_candidates = target_candidates[~context_mask[row_idx, target_candidates]]
        if target_candidates.numel() == 0:
            target_candidates = valid_idx[~context_mask[row_idx, valid_idx]]
        if target_candidates.numel() == 0:
            continue

        for target_idx in range(int(num_target_blocks)):
            reliability = (p_row[target_candidates] + 1e-6).pow(float(alpha))
            weights = (1.0 - float(eps_target)) * reliability.pow(float(beta_target))
            weights = weights / weights.sum()
            weights = weights + float(eps_target) / float(target_candidates.numel())
            target_masks[row_idx, target_idx] = _weighted_sample_until(
                indices=target_candidates,
                p_row=p_row,
                weights=weights,
                mass_target=target_mass,
                min_eff=float(min_eff_target),
                max_count=target_count_cap,
            )

        target_union = target_masks[row_idx].any(dim=0)
        blocked = target_union.clone()
        if float(torch.rand((), device=device).item()) < float(local_gap_probability):
            blocked |= _near_target_mask(
                peak_mz_da[row_idx].float(),
                target_union,
                gap_da=float(local_gap_da),
            )
        context_candidates = valid_idx[~blocked[valid_idx] & ~context_mask[row_idx, valid_idx]]
        if context_candidates.numel() == 0:
            context_candidates = valid_idx[~target_union[valid_idx] & ~context_mask[row_idx, valid_idx]]
        if context_candidates.numel() == 0:
            continue

        reliability = (p_row[context_candidates] + 1e-6).pow(float(alpha))
        weights = (1.0 - float(eps_context)) * reliability.pow(float(beta_context))
        weights = weights / weights.sum()
        weights = weights + float(eps_context) / float(context_candidates.numel())
        target_union_mass = float(p_row[target_union].sum().item())
        capped_context_mass = min(context_mass, max(0.0, 1.0 - target_union_mass - float(min_unused_mass)))
        context_mask[row_idx] = _weighted_fill_context_until(
            base_mask=context_mask[row_idx],
            indices=context_candidates,
            p_row=p_row,
            weights=weights,
            total_mass_target=capped_context_mass,
            total_min_eff=float(min_eff_context),
            max_total_mass=capped_context_mass,
        )

    return context_mask, target_masks
