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

    total_bits = 0
    correct_bits = 0
    tanimoto_sum = 0.0
    samples = 0

    for x, y in test_iter:
        x = x.to(dtype=torch.float32)
        y = y.to(dtype=torch.float32)
        logits = x @ weights
        probs = torch.sigmoid(logits)
        pred = probs >= 0.5
        target = y >= 0.5

        correct_bits += int((pred == target).sum().item())
        total_bits += int(target.numel())

        intersection = (pred & target).sum(dim=1).to(torch.float32)
        union = (pred | target).sum(dim=1).to(torch.float32)
        tanimoto_sum += float((intersection / union).sum().item())
        samples += int(pred.shape[0])

    accuracy = torch.tensor(correct_bits / total_bits, device=device)
    tanimoto = torch.tensor(tanimoto_sum / samples, device=device)
    return {"accuracy": accuracy, "tanimoto": tanimoto}
