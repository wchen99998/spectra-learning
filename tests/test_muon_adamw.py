import torch

from optimizers.muon_adamw import MuonAdamW


def test_muon_partial_grad_fallback_accepts_fp32_grads_with_bf16_momentum(monkeypatch):
    monkeypatch.setattr(MuonAdamW, "_warmup_triton_kernels", lambda self: None)

    used = torch.nn.Parameter(torch.randn(8, 8))
    unused = torch.nn.Parameter(torch.randn(8, 8))
    optimizer = MuonAdamW(
        [
            {
                "params": [used, unused],
                "name": "ffn_2d",
                "optimizer": "muon",
                "lr": 1e-3,
                "weight_decay": 0.0,
                "momentum": 0.95,
                "nesterov": True,
                "adjust_lr_fn": "match_rms_adamw",
            }
        ]
    )

    before = used.detach().clone()
    used.square().sum().backward()
    optimizer.step()

    assert not torch.equal(before, used)
    assert optimizer._muon_state[used]["momentum_buffer"].dtype == torch.bfloat16
    assert unused.grad is None
