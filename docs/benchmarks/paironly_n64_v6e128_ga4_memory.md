# AOT memory report: 63 peaks + CLS, 128 TPU v6e chips, GA4

Per-block checkpointing, global batch 2,048, replicated bf16 Adam, no ZeRO-1.
The estimate is 30.68 GiB/chip, leaving about 0.56 GiB against the reserved
31.24 GiB budget. This is a compiler estimate, not a runtime measurement.

The total NNX parameter count includes 32 fixed Fourier frequencies excluded
from the optimizer by the existing training parameter filter.

```json
{
  "config": "configs/1p8b_fastmixer_pair_distogram_multiscale_n64_v6e128_ga4.py",
  "topology": "v6e:8x16",
  "chips": 128,
  "overrides": [],
  "zero1_shards": 0,
  "parameter_count": 1777557744,
  "trainable_parameter_count": 1777557712,
  "encoder_tokens": 55,
  "predictor_tokens": 55,
  "target_tokens": 16,
  "hbm_bytes_per_chip": 34359738368,
  "runtime_reserve_bytes": 816043787,
  "results": [
    {
      "micro_batch_per_chip": 4,
      "gradient_accumulation_steps": 4,
      "global_batch_size": 2048,
      "argument_bytes": 14220531712,
      "temp_bytes": 18718233312,
      "output_bytes": 14220513504,
      "alias_bytes": 14220497920,
      "total_bytes": 32938780608,
      "total_gib": 30.676629960536957,
      "budget_gib": 31.239999999292195,
      "fits": true
    }
  ]
}
```
