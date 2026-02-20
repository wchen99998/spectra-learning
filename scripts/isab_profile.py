"""Profile ISAB execution to identify optimization targets."""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks.isab_triton import TritonISAB
from networks.set_transformer_torch import ISAB

B, N, D = 512, 60, 256
DEVICE, DTYPE = "cuda", torch.bfloat16

torch.manual_seed(42)
vanilla = ISAB(dim=D, num_inducing_points=32, n_heads=8, n_kv_heads=4,
               attention_mlp_multiple=4.0).to(DEVICE, DTYPE).eval()
triton_isab = TritonISAB.from_vanilla_isab(vanilla).eval()

class Wrapper(nn.Module):
    def __init__(self, m): super().__init__(); self.isab = m
    def forward(self, x): return self.isab(x, kv_block_mask=None, q_block_mask=None)
compiled = torch.compile(Wrapper(vanilla), backend="inductor", mode="max-autotune")

x = torch.randn(B, N, D, device=DEVICE, dtype=DTYPE)

# Warmup
for _ in range(5):
    with torch.no_grad():
        vanilla(x, kv_block_mask=None, q_block_mask=None)
        triton_isab(x)
        compiled(x)
torch.cuda.synchronize()

# Profile with torch profiler
print("=== Profiling TritonISAB ===")
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    with torch.no_grad():
        for _ in range(10):
            triton_isab(x)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

print("\n=== Profiling torch.compile ===")
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    with torch.no_grad():
        for _ in range(10):
            compiled(x)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
