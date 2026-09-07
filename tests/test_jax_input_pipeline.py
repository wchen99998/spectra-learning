from types import SimpleNamespace

import numpy as np
import pytest

from spectra_learning.training import pretrain_jax


def test_input_timers_include_stacking_and_device_placement(monkeypatch):
    clock = [0.0]
    batch = {"peak_mz": np.zeros((2, 4), dtype=np.float32)}

    def batches():
        clock[0] += 2.0
        yield batch

    def stack(batches):
        clock[0] += 3.0
        return {key: np.stack([batch[key] for batch in batches]) for key in batch}

    def place(batch, **kwargs):
        clock[0] += 5.0
        assert kwargs["batch_axis"] == 1
        return batch

    monkeypatch.setattr(pretrain_jax.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(pretrain_jax, "_stack_micro_batches", stack)
    monkeypatch.setattr(pretrain_jax, "numpy_batch_to_jax", place)
    loop = object.__new__(pretrain_jax._JaxTrainingLoop)
    loop.grad_accum_steps = 1
    loop.use_sjepa = False
    loop.use_sharded_step = False
    loop.timing_barriers = False
    loop.measured_start = 0.0
    loop.timing = dict.fromkeys(
        [
            "dataloader_seconds",
            "stack_seconds",
            "transfer_seconds",
            "measured_microbatches",
        ],
        0.0,
    )
    assert loop._next_accumulated_batch(iter(batches()))["peak_mz"].shape == (1, 2, 4)
    assert loop.timing == {
        "dataloader_seconds": 2.0,
        "stack_seconds": 3.0,
        "transfer_seconds": 5.0,
        "measured_microbatches": 1.0,
    }


@pytest.mark.parametrize("fail", [False, True])
def test_validation_closes_workers_without_reading_an_extra_batch(monkeypatch, fail):
    class Iterator:
        reads = 0
        closed = False

        def __iter__(self):
            return self

        def __next__(self):
            self.reads += 1
            assert self.reads == 1
            return {"x": np.zeros((2,))}

        def _shutdown_workers(self):
            self.closed = True

    iterator = Iterator()
    module = SimpleNamespace(val_loader_for_eval=lambda **kwargs: iterator)
    monkeypatch.setattr(
        pretrain_jax, "numpy_batch_to_jax", lambda batch, **kwargs: batch
    )

    def evaluate(*args):
        if fail:
            raise RuntimeError("device failure")
        return {"loss": np.asarray(2.0)}

    kwargs = dict(
        datamodule=module,
        trainable_params=None,
        static_state=None,
        eval_step=evaluate,
        max_steps=1,
        use_sharded_step=False,
        data_mesh=None,
        metric_reduction="mean",
    )
    if fail:
        with pytest.raises(RuntimeError, match="device failure"):
            pretrain_jax._evaluate_jax_validation_loss(**kwargs)
    else:
        assert pretrain_jax._evaluate_jax_validation_loss(**kwargs) == {"val/loss": 2.0}
    assert iterator.closed
    assert iterator.reads == 1
