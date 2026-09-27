import json
import sys
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from retrieval_lora import LoRALinear
from spectra_learning.models.common_jax import Linear
import run_retrieval_jax as runner


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_lora_zero_initialization_and_update_equation(dtype):
    base = Linear(8, 12, compute_dtype=dtype, rngs=nnx.Rngs(1))
    layer = LoRALinear(base, rank=4, alpha=8., key=jax.random.key(2))
    x = jax.random.normal(jax.random.key(3), (2, 5, 8))
    np.testing.assert_array_equal(layer(x), base(x))
    layer.lora_b[...] = jax.random.normal(jax.random.key(4), (4, 12)) * .02
    expected = (base(x).astype(jnp.float32) + 2 * x @ layer.lora_a[...] @ layer.lora_b[...]).astype(dtype)
    np.testing.assert_allclose(np.asarray(layer(x), np.float32), np.asarray(expected, np.float32), atol=1e-6)


def test_encoder_policy_is_bound_to_saved_checkpoint(tmp_path):
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({name: str(tmp_path / name) for name in runner.INPUT_NAMES}))
    args = runner.recipe_args(["--inputs", str(inputs), "--out", str(tmp_path), "--stage", "train",
                               "--encoder-adaptation", "lora"])
    assert (args.lora_rank, args.lora_alpha, args.jepa_lr, args.lr) == (16, 16., 1e-4, 5e-4)
    saved = {"recipe": runner.RECIPE, "args": {"stage": "train", "encoder_adaptation": "full"},
             "input_identity": {}}
    with pytest.raises(ValueError, match="adaptation policy"):
        runner.validate_saved_run(saved, {}, args)
