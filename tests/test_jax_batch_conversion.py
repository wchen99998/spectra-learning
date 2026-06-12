import numpy as np


def test_numpy_batch_converts_to_jax() -> None:
    from spectra_learning.training.pretrain_jax import torch_batch_to_jax

    batch = {
        "peak_mz": np.zeros((2, 4), dtype=np.float32),
        "peak_intensity": np.ones((2, 4), dtype=np.float32),
        "peak_valid_mask": np.ones((2, 4), dtype=bool),
    }

    jax_batch = torch_batch_to_jax(batch)

    assert jax_batch["peak_mz"].shape == (2, 4)
    assert jax_batch["peak_intensity"].dtype.name == "float32"
    assert jax_batch["peak_valid_mask"].dtype.name == "bool"
