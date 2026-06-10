import tempfile

import jax
import optax
import torch
import torchax

from spectra_learning.models.model import PeakSetJEPA
from spectra_learning.training.activation_checkpointing import apply_activation_checkpointing
from spectra_learning.training.checkpointing import (
    load_torch_checkpoint,
    save_torch_checkpoint,
)
from spectra_learning.training.torchax_pretrain import (
    _make_torchax_accumulate_grads,
    _make_torchax_grad_step,
    _make_torchax_scale_grads,
    _opt_state_checkpoint_leaves,
    _restore_torchax_state,
    _restore_opt_state,
)
from spectra_learning.training.distributed import init_distributed_from_env
from spectra_learning.training.torchax_runtime import (
    build_torchax_mesh,
    initialize_torchax_distributed,
    replicate_torchax_tree,
    shard_torchax_batch,
)


def _small_model() -> PeakSetJEPA:
    return PeakSetJEPA(
        training_mode="mae",
        model_dim=8,
        encoder_num_layers=1,
        encoder_num_heads=2,
        feature_mlp_hidden_dim=4,
        masked_latent_predictor_num_layers=1,
        masked_latent_predictor_num_heads=2,
        jepa_num_target_blocks=1,
        num_peaks=4,
        pairmixer_pair_dim=8,
        pairmixer_pair_feature_hidden_dim=4,
    )


class _ToyLossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1, bias=False)

    def forward(self, batch):
        prediction = self.linear(batch["x"])
        return {"loss": ((prediction - batch["y"]) ** 2).mean()}


def test_torchax_tensor_saves_as_portable_pt_tensor():
    torchax.enable_globally()
    tensor = torch.arange(4, dtype=torch.float32).to("jax")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/tensor.pt"
        save_torch_checkpoint({"tensor": tensor}, path)
        loaded = load_torch_checkpoint(path, map_location="cpu", weights_only=True)
    torchax.disable_globally()

    assert loaded["tensor"].device.type == "cpu"
    assert loaded["tensor"].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_torchax_opt_state_checkpoint_leaves_restore_to_fresh_structure():
    torchax.enable_globally()
    model = _small_model().to("jax")
    params = {name: param for name, param in model.named_parameters()}
    optimizer = optax.adamw(1e-3)
    opt_state = torchax.interop.call_jax(optimizer.init, params)
    leaves = _opt_state_checkpoint_leaves(opt_state)
    restored = _restore_opt_state(opt_state, [leaf.cpu() for leaf in leaves])
    torchax.disable_globally()

    assert len(jax.tree.leaves(restored)) == len(leaves)


def test_torchax_restore_accepts_plain_torch_training_checkpoint():
    source = _ToyLossModel()
    source_state = {
        name: tensor.detach().clone()
        for name, tensor in source.state_dict().items()
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/step-00000003.pt"
        save_torch_checkpoint(
            {
                "model": source_state,
                "optimizers": [],
                "schedulers": [],
                "grad_scaler": None,
                "global_step": 3,
                "epoch": 0,
                "loss": 1.0,
                "wandb_run_id": None,
            },
            path,
        )
        torchax.enable_globally()
        model = _ToyLossModel().to("jax")
        params = {name: param for name, param in model.named_parameters()}
        optimizer = optax.adamw(1e-3)
        opt_state = torchax.interop.call_jax(optimizer.init, params)

        global_step, params, opt_state = _restore_torchax_state(
            tmpdir,
            params,
            opt_state,
        )
        restored_weight = params["linear.weight"].cpu()
        torchax.disable_globally()

    assert global_step == 3
    assert torch.equal(restored_weight, source_state["linear.weight"])
    assert opt_state is not None


def test_torchax_batch_sharding_places_batch_axis_on_mesh():
    if jax.device_count() < 2:
        return
    torchax.enable_globally()
    mesh = build_torchax_mesh({"torchax_mesh_devices": 2})
    batch = {"peak_mz": torch.arange(8, dtype=torch.float32).reshape(4, 2).to("jax")}
    sharded = shard_torchax_batch(batch, mesh)

    assert str(sharded["peak_mz"].jax().sharding.spec) == "P('data', None)"
    assert sharded["peak_mz"].cpu().tolist() == [
        [0.0, 1.0],
        [2.0, 3.0],
        [4.0, 5.0],
        [6.0, 7.0],
    ]
    torchax.disable_globally()


def test_torchax_tree_replication_replicates_across_mesh():
    if jax.device_count() < 2:
        return
    torchax.enable_globally()
    mesh = build_torchax_mesh({"torchax_mesh_devices": 2})
    params = {"weight": torch.arange(4, dtype=torch.float32).reshape(2, 2).to("jax")}
    replicated = replicate_torchax_tree(params, mesh)

    assert str(replicated["weight"].jax().sharding.spec) == "P()"
    assert replicated["weight"].cpu().tolist() == [[0.0, 1.0], [2.0, 3.0]]
    torchax.disable_globally()


def test_torchax_shard_map_grad_step_replicates_gradients():
    if jax.device_count() < 2:
        return
    torchax.enable_globally()
    model = _ToyLossModel().to("jax").train()
    params = {name: param for name, param in model.named_parameters()}
    buffers = dict(model.named_buffers())
    mesh = build_torchax_mesh({"torchax_mesh_devices": 2})
    params = replicate_torchax_tree(params, mesh)
    buffers = replicate_torchax_tree(buffers, mesh)
    batch = {
        "x": torch.arange(8, dtype=torch.float32).reshape(4, 2).to("jax"),
        "y": torch.ones(4, 1, dtype=torch.float32).to("jax"),
    }
    batch = shard_torchax_batch(batch, mesh)

    loss, grads = _make_torchax_grad_step(model, mesh=mesh, params=params)(
        params,
        buffers,
        batch,
    )

    assert float(loss.cpu()) > 0.0
    assert grads["linear.weight"].jax().is_fully_replicated
    torchax.disable_globally()


def test_torchax_accumulation_helpers_run_as_jax_transforms():
    torchax.enable_globally()
    add_grads = torchax.interop.jax_jit(_make_torchax_accumulate_grads())
    scale_grads = torchax.interop.jax_jit(_make_torchax_scale_grads())
    lhs = {"weight": torch.ones(2, dtype=torch.float32).to("jax")}
    rhs = {"weight": torch.full((2,), 3.0, dtype=torch.float32).to("jax")}

    accumulated = add_grads(lhs, rhs)
    averaged = scale_grads(accumulated, 0.25)

    assert averaged["weight"].cpu().tolist() == [1.0, 1.0]
    torchax.disable_globally()


def test_activation_checkpointing_wraps_encoder_and_predictor_blocks():
    model = _small_model()
    apply_activation_checkpointing(model, {"activation_checkpoint_mode": "full"})

    assert type(model.encoder.blocks[0]).__name__ == "CheckpointWrapper"
    assert type(model.masked_latent_predictor[0]).__name__ == "CheckpointWrapper"


def test_selective_activation_checkpointing_wraps_blocks():
    model = _small_model()
    apply_activation_checkpointing(model, {"activation_checkpoint_mode": "selective"})

    assert type(model.encoder.blocks[0]).__name__ == "CheckpointWrapper"
    assert type(model.masked_latent_predictor[0]).__name__ == "CheckpointWrapper"


def test_torchax_distributed_initializer_noops_without_config_or_env(monkeypatch):
    monkeypatch.delenv("JAX_DISTRIBUTED_INITIALIZE", raising=False)
    monkeypatch.delenv("JAX_COORDINATOR_ADDRESS", raising=False)
    monkeypatch.delenv("JAX_COORDINATOR_ADDR", raising=False)
    monkeypatch.delenv("JAX_NUM_PROCESSES", raising=False)
    monkeypatch.delenv("JAX_PROCESS_COUNT", raising=False)

    initialize_torchax_distributed({})

    assert not jax.distributed.is_initialized()


def test_torchax_distributed_context_uses_jax_process_state():
    context = init_distributed_from_env("jax")
    torchax.disable_globally()

    assert context.backend == "jax"
    assert context.rank == jax.process_index()
    assert context.world_size == jax.process_count()
