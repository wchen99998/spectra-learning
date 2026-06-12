import torch

from spectra_learning.models.model import PeakSetJEPA
from spectra_learning.training.activation_checkpointing import (
    apply_activation_checkpointing as apply_pretrain_activation_checkpointing,
)
from spectra_learning.training.modules import PretrainModule
from spectra_learning.training.performance import (
    apply_activation_checkpointing,
    activation_checkpoint_mode,
    compile_forward,
    register_bf16_adamw_state_hook,
)


def _tiny_pretrain_module() -> PretrainModule:
    model = PeakSetJEPA(
        model_dim=32,
        encoder_num_layers=2,
        encoder_num_heads=4,
        attention_mlp_multiple=2.0,
        feature_mlp_hidden_dim=16,
        num_peaks=6,
        jepa_num_target_blocks=2,
        masked_latent_predictor_num_layers=1,
        masked_latent_predictor_num_heads=4,
        pairmixer_pair_dim=32,
        pairmixer_pair_feature_hidden_dim=16,
    )
    return PretrainModule(model)


def test_activation_checkpointing_preserves_state_dict_keys():
    module = _tiny_pretrain_module()
    before = set(module.state_dict())

    apply_activation_checkpointing(
        module,
        {
            "activation_checkpoint_mode": "full",
            "activation_checkpoint_preserve_rng_state": True,
        },
    )

    assert set(module.state_dict()) == before


def test_activation_checkpointing_accepts_canonical_modes():
    assert activation_checkpoint_mode({"activation_checkpoint_mode": "selective"}) == "selective"
    assert activation_checkpoint_mode({"activation_checkpoint_mode": "full"}) == "full"


def test_pretrain_activation_checkpointing_accepts_canonical_modes():
    module = _tiny_pretrain_module()

    apply_pretrain_activation_checkpointing(
        module,
        {
            "activation_checkpoint_mode": "selective",
            "activation_checkpoint_preserve_rng_state": True,
        },
    )

    assert hasattr(module.model.encoder.blocks[0], "_checkpoint_wrapped_module")
    assert hasattr(module.model.masked_latent_predictor[0], "_checkpoint_wrapped_module")



def test_activation_checkpointing_works_with_default_full_module_compile():
    module = _tiny_pretrain_module()

    apply_activation_checkpointing(
        module,
        {
            "activation_checkpoint_mode": "full",
            "activation_checkpoint_preserve_rng_state": True,
        },
    )
    compile_forward(
        module,
        {
            "compile_mode": "max-autotune",
        },
    )

    assert module._compiled_call_impl is not None


def test_block_compile_compiles_pair_mixer_blocks_only():
    module = _tiny_pretrain_module()
    assert module._compiled_call_impl is None

    compile_forward(
        module,
        {
            "compile_mode": "max-autotune",
            "compile_scope": "blocks",
        },
    )

    assert module._compiled_call_impl is None
    blocks = [*module.model.encoder.blocks, *module.model.masked_latent_predictor]
    assert all(block._compiled_call_impl is not None for block in blocks)


def test_bf16_adamw_state_hook_uses_fp32_for_non_fused_optimizer():
    param = torch.nn.Parameter(torch.ones(4))
    optimizer = torch.optim.AdamW([param], lr=1e-3, fused=False)
    register_bf16_adamw_state_hook(optimizer)

    param.sum().backward()
    optimizer.step()

    assert optimizer.state[param]["exp_avg"].dtype == torch.float32
    assert optimizer.state[param]["exp_avg_sq"].dtype == torch.float32


def test_bf16_adamw_state_hook_uses_bf16_for_fused_cuda_optimizer():
    if not torch.cuda.is_available():
        return
    param = torch.nn.Parameter(torch.ones(4, device="cuda"))
    optimizer = torch.optim.AdamW([param], lr=1e-3, fused=True)
    register_bf16_adamw_state_hook(optimizer)

    param.sum().backward()
    optimizer.step()
    torch.cuda.synchronize()

    assert optimizer.state[param]["exp_avg"].dtype == torch.bfloat16
    assert optimizer.state[param]["exp_avg_sq"].dtype == torch.bfloat16
