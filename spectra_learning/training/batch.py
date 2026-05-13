from collections import deque
from collections.abc import Iterator

import torch

TRAIN_BATCH_KEYS = frozenset(
    {
        "peak_mz",
        "peak_intensity",
        "peak_valid_mask",
        "context_mask",
        "target_masks",
        "precursor_mz",
    }
)


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    *,
    keys: frozenset[str] = TRAIN_BATCH_KEYS,
    tensorize_non_tensors: bool = False,
) -> dict[str, torch.Tensor]:
    return {
        k: v.to(device, non_blocking=True)
        if isinstance(v, torch.Tensor)
        else torch.as_tensor(v, device=device) if tensorize_non_tensors else v
        for k, v in batch.items()
        if k in keys
    }


class BatchPrefetcher:
    def __init__(
        self,
        loader: Iterator,
        device: torch.device,
        prefetch_size: int = 1,
        keys: frozenset[str] = TRAIN_BATCH_KEYS,
        tensorize_non_tensors: bool = False,
    ) -> None:
        self._loader = loader
        self._device = device
        self._keys = keys
        self._tensorize_non_tensors = tensorize_non_tensors
        self._stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self._ready: deque = deque()
        self._exhausted = False
        for _ in range(prefetch_size):
            self._preload_one()

    def _preload_one(self) -> None:
        if self._exhausted:
            return
        batch = next(self._loader, None)
        if batch is None:
            self._exhausted = True
            return
        if self._stream is None:
            moved = move_batch_to_device(
                batch,
                self._device,
                keys=self._keys,
                tensorize_non_tensors=self._tensorize_non_tensors,
            )
            ready_event = None
        else:
            with torch.cuda.stream(self._stream):
                moved = move_batch_to_device(
                    batch,
                    self._device,
                    keys=self._keys,
                    tensorize_non_tensors=self._tensorize_non_tensors,
                )
                ready_event = torch.cuda.Event()
                ready_event.record(self._stream)
        self._ready.append((moved, ready_event))

    def next(self) -> dict[str, torch.Tensor] | None:
        if not self._ready:
            return None
        batch, ready_event = self._ready.popleft()
        if ready_event is not None:
            current_stream = torch.cuda.current_stream(device=self._device)
            current_stream.wait_event(ready_event)
            for value in batch.values():
                if isinstance(value, torch.Tensor):
                    value.record_stream(current_stream)
        self._preload_one()
        return batch
