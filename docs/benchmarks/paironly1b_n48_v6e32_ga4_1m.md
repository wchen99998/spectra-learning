# 1B pair-only model on 32 v6e chips, 1M steps

Config: [`1b_fastmixer_pair_distogram_multiscale_n48_v6e32_ga4_1m.py`](../../configs/1b_fastmixer_pair_distogram_multiscale_n48_v6e32_ga4_1m.py).

Uses the [validated 1B, 48-position architecture](paironly1b_n48_v6e128_ga4.md):
47 physical peaks plus CLS, pair width 1536, 15 encoder and five predictor
blocks, metadata AdaLN-Zero, multiscale distogram loss, selective recomputation,
and replicated bf16 Adam moments.

The current input randomness contract is **v2**, recorded at JAX startup and
in checkpoint metadata. It uses a 2048-block shuffle window, shuffle seed 67,
training seed 66, independent per-host partition ordering, and sample/epoch/
operation-derived masking and metadata-dropout seeds. The eight-host shard
assignment changes with the host count; the randomness algorithm is unchanged.

Global batch **512** = 32 chips × four spectra/chip/microstep × **GA4**.
This preserves the benchmarked per-chip workload. Batch 2048 would require
GA16 at this microbatch size, with roughly four times the computation per
optimizer update; it is not the submitted setting. Batch 512 is the validated
shape transferred to 32 chips, rather than an exhaustively measured maximum.

The schedule ends at **1,000,000 optimizer steps**, totaling 512 million spectrum
presentations. Learning rate remains 6e-4 → 6e-5 with 20,000 warmup steps.
The eight-host shard plan gives at least 2,605,320 steps per epoch, so the
inherited epoch budget does not truncate the million-step schedule.

The schedule targets 1M optimizer steps, with a **72-hour training cutoff**
and **three-day TPU allocation limit**. It will stop before 1M if time expires.
At the four-chip replay's 0.859 s/update, three days allows roughly 302,000
updates before startup, validation, input loading and other overhead.
Checkpoints save every **10,000 steps**, with asynchronous Orbax keeping five;
the training time-limit exit also saves a final checkpoint.

An independent cancellation deadline 72 hours after submission bounds the
managed job across recovery attempts. Queueing time counts toward this wall-clock
limit. This prevents SkyPilot FAILOVER from extending the run after TPU expiry.
The existing automatic checkpoint restore remains available for interruptions
within that deadline.

## Submission

Run ID: `pair1b-n48-v6e32-b512-ga4-1m-3d-20260908-061857`.

Workdir:
`gs://metal-repeater-411410-spectra-checkpoints/skypilot/pair1b-n48-v6e32-b512-ga4-1m-3d-20260908-061857`.

```bash
.venv/bin/python train_sky.py \
  --config configs/1b_fastmixer_pair_distogram_multiscale_n48_v6e32_ga4_1m.py \
  --tpu-generation v6e --chips 32 --dws-run-duration 3d \
  --run-id pair1b-n48-v6e32-b512-ga4-1m-3d-20260908-061857 \
  --workdir gs://metal-repeater-411410-spectra-checkpoints/skypilot/pair1b-n48-v6e32-b512-ga4-1m-3d-20260908-061857 \
  --detach-run
```

The launcher requests eight four-chip hosts in a `4x8` topology and races the
six configured v6e regions. Its dry run verified the topology, global batch,
million-step schedule, three-day allocation duration and FAILOVER policy.

The exact 32-chip `v6e:4x8` accumulated training step compiled successfully:
29,503,186,400 bytes (**27.47698 GiB/chip**), with batch 512 and GA4.
This matches the accepted four- and 128-chip buffer estimates.

The persistent user-systemd timer `spectra-pair1b32-20260908-061857-stop.timer` requests cancellation at
**2026-09-11 06:19:40 UTC**. User lingering is enabled, so logout does not stop the timer.
The timer also fires after a reboot if its deadline was missed. Its service
uses the exact unique SkyPilot job name and retries failed cancellation commands.

Submitted successfully on 2026-09-08 as **SkyPilot managed job 74**. The superseded
seven-day submission was managed job 73 and was cancelled.
