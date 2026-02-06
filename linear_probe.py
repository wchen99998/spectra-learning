from __future__ import annotations

from typing import Iterable

import torch


def run_linear_probe(
    train_iter: Iterable[tuple[torch.Tensor, torch.Tensor]],
    test_iter: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    ridge: float = 1e-3,
) -> dict[str, torch.Tensor]:
    xtx = None
    xty = None
    device = None
    dtype = None

    for x, y in train_iter:
        x = x.to(dtype=torch.float32)
        y = y.to(dtype=torch.float32)
        if xtx is None:
            device = x.device
            dtype = x.dtype
            xtx = torch.zeros((x.shape[1], x.shape[1]), device=device, dtype=dtype)
            xty = torch.zeros((x.shape[1], y.shape[1]), device=device, dtype=dtype)
        xtx += x.T @ x
        xty += x.T @ y

    ridge_eye = ridge * torch.eye(xtx.shape[0], device=device, dtype=dtype)
    weights = torch.linalg.solve(xtx + ridge_eye, xty)

    total_bits = torch.zeros((), device=device, dtype=torch.float32)
    correct_bits = torch.zeros((), device=device, dtype=torch.float32)
    tanimoto_sum = torch.zeros((), device=device, dtype=torch.float32)
    samples = torch.zeros((), device=device, dtype=torch.float32)

    for x, y in test_iter:
        x = x.to(dtype=torch.float32)
        y = y.to(dtype=torch.float32)
        logits = x @ weights
        probs = torch.sigmoid(logits)
        pred = probs >= 0.5
        target = y >= 0.5

        correct_bits += (pred == target).sum().to(torch.float32)
        total_bits += torch.tensor(target.numel(), device=device, dtype=torch.float32)

        intersection = (pred & target).sum(dim=1).to(torch.float32)
        union = (pred | target).sum(dim=1).to(torch.float32)
        tanimoto_sum += (intersection / union).sum()
        samples += torch.tensor(pred.shape[0], device=device, dtype=torch.float32)

    accuracy = correct_bits / total_bits
    tanimoto = tanimoto_sum / samples
    return {"accuracy": accuracy, "tanimoto": tanimoto}
