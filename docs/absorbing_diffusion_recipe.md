# Absorbing m/z diffusion with the 1.8B encoder–predictor

`configs/1p8b_absorbing_diffusion.py` trains a fresh JAX encoder–predictor with
an absorbing diffusion objective. It preserves the student architecture,
exact trainable parameter count, and
optimization recipe from [the reference frozen-teacher JEPA run](https://wandb.ai/iclac/jepa-finalrun/runs/1p8b-ftjepa-t1p8b-s666k-perleaf-b1024-v6e32-20260905-115335).
It does not restore that run's weights or its frozen teacher. Use a fresh workdir;
resuming an existing diffusion workdir retains the usual checkpoint behavior.

## What is diffused

The categorical variables are the m/z values in the valid, ordered peak slots:
2,048 classes covering 0–1,000 Da at 0.48828125 Da resolution. This repurposes the
reference student's existing 2,048-dimensional output projection as categorical
logits, without adding a head. The reference reconstruction teacher used 2,000
bins at 0.5 Da; the new binning preserves the JEPA student's exact parameter
count. The model conditions on
the intensity vector, acquisition metadata, peak count, and slot positions.
Intensity is not diffused or reconstructed. This is a conditional model of m/z,
not a likelihood model for the complete joint spectrum or its length.

Visible m/z values are quantized to their bin midpoints before entering the
encoder. The encoder applies the same quantization during embedding extraction
and downstream probes, so it cannot use sub-bin precision unavailable in the
categorical state. Hidden m/z values cannot enter the encoder or predictor
queries. Although the conditional likelihood treats the full intensity vector
as fixed side information, this denoiser only reads intensities at visible slots.

The predictor uses all visible encoder tokens and masked-position queries.
The CLS token provides valid context even when all peaks are masked. There is
no timestep embedding or teacher network. Encoder embeddings need one clean
forward pass; this change does not add a generative reverse-diffusion sampler.
With CLS as the only visible token, the unchanged cross-attention predictor
produces identical distributions at all masked positions.
Reconstruction alone does not guarantee that CLS or mean pooling will provide
good frozen retrieval embeddings; evaluate the intended readout downstream.

## Objective and sampling

For a spectrum with `N` valid peaks, sample `K` uniformly from `1..N`, then
choose a uniformly random subset of `K` peaks to mask. All remaining valid
peaks are context; there is no unused third subset. Padding never contributes
to the mask count or loss. This includes the fully masked case `K=N`.

Average exact categorical cross-entropy over the masked peaks **within each
spectrum**, then average those per-spectrum losses over the batch. There is no
additional inverse-mask-fraction multiplier, multiscale loss, latent regression,
intensity loss, or other auxiliary objective.

This estimator integrates out the time variable of the standard continuous-time
absorbing objective, using the [RADD / any-order equivalence](https://arxiv.org/html/2406.03736v2#S3.SS3).
With masking probability `r` and independent masking at
that probability, its per-token form is

\[
\int_0^1 \mathbb E_{M\sim\mathrm{Bernoulli}(r)}
\left[\frac{1}{Nr}\sum_{i\in M}-\log p_\theta(x_i\mid x_{\bar M},c)\right]dr.
\]

For a fixed nonempty subset of size `K`, integrating its probability divided
by `r` gives `1 / (K * binom(N, K))`. Consequently the same objective is

\[
\mathbb E_{K\sim U\{1,\ldots,N\},\;|M|=K}
\left[\frac1K\sum_{i\in M}-\log p_\theta(x_i\mid x_{\bar M},c)\right].
\]

Here `c` denotes the conditioning information above. This equivalence uses a
denoiser without timestep conditioning. Multiplying the uniform-`K` estimator
by `N/K` again would change the objective. The inherited fixed context/target
fractions are unused by diffusion sampling; there is one mask/accumulation stage.

## Architecture and parameter accounting

The reference has 25 encoder blocks and five cross-attention predictor blocks,
both width 2,048 with eight attention heads (head dimension 256). Inputs retain
127 m/z-ordered peak slots plus CLS, Fourier peak embeddings, and acquisition
metadata conditioning through AdaLN-zero. The model uses the single-token
encoder track; inactive pair-track configuration fields do not create a pair
network.

| Trainable component | Reference JEPA | Absorbing diffusion |
| --- | ---: | ---: |
| Encoder | 1,516,340,736 | 1,516,340,736 |
| Five predictor blocks | 256,952,320 | 256,952,320 |
| 2,048→2,048 output projection | 4,196,352 | 4,196,352 |
| Mask token | 2,048 | 2,048 |
| **Total trainable** | **1,777,491,456** | **1,777,491,456** |

The encoder, predictor, output projection, and mask token match the reference
student's layers, dimensions, and trainable parameter counts exactly. Setting
`mae_mz_head="readout"` uses the existing output projection directly as m/z
logits; the value-head stage is a parameter-free identity. Its parameter tree
also contains 64 fixed Fourier values, giving 1,777,491,520 total parameters
including those nontrainable values. Reference
W&B counts include a frozen encoder when reporting 3,293,832,320 total parameters;
that is not the student's trainable size.

The configuration retains batch 1,024, accumulation eight, Adam with
`betas=(0.9, 0.95)`, zero weight decay, learning rate `3e-4` decaying to `3e-6`,
10,000 warmup steps, two million training steps, bf16 optimizer state, per-leaf
ZeRO-1, and full activation checkpointing. Variable masks require capacity for
all 127 targets; throughput and memory must be measured on the target TPU
configuration rather than inferred from the reference's fixed-mask timings.

## Prepare and launch

Use the existing infrastructure launcher with a fresh run ID. The following
command prepares the 32-chip v6e configuration without provisioning resources
or starting training:

```bash
RUN_ID="1p8b-absorbing-diffusion-$(date -u +%Y%m%d-%H%M%S)"
.venv/bin/python train_sky.py \
    --config configs/1p8b_absorbing_diffusion.py \
    --tpu-generation v6e --chips 32 --dws-run-duration 4d \
    --override max_duration_hours=95.0 \
    --override "jax_training_deadline_path=\"gs://metal-repeater-411410-spectra-checkpoints/skypilot/${RUN_ID}/training_deadline.txt\"" \
    --run-id "${RUN_ID}" \
    --workdir "gs://metal-repeater-411410-spectra-checkpoints/skypilot/${RUN_ID}" \
    --detach-run --dry-run
```

Remove `--dry-run` to submit that configuration. A short TPU run should verify
memory, finite updates, and throughput before committing to the full budget.
The local CPU checks and launcher dry run do not establish TPU feasibility or
representation quality.

On 2026-09-22, the launcher dry run completed successfully and resolved topology
`4x8`, eight hosts, 32 chips, batch 1,024, accumulation eight, and two million
steps. It generated
`tmp/skypilot_tasks/1p8b-absorbing-diffusion-dryrun-20260922.yaml` and skipped
credential lookup and job submission. No training run was launched by that check.

### Four-day run submitted on 2026-09-22

The user-authorized run is SkyPilot managed job **87**, named
`spectra-adiff-1p8b-ep-v6e32-4d-20260922-080322`. Its run ID is
`adiff-1p8b-ep-v6e32-4d-20260922-080322`, with W&B project `iclac/jepa-finalrun`
and workdir
`gs://metal-repeater-411410-spectra-checkpoints/skypilot/adiff-1p8b-ep-v6e32-4d-20260922-080322`.

It requests 32 v6e chips for a 96-hour DWS allocation and limits training to
95 hours to leave setup/checkpoint headroom. Rank zero persists the absolute
training deadline in `training_deadline.txt` under that workdir; recovery reuses
the deadline instead of granting another 95 hours. The existing graceful-stop
path saves and waits for the final checkpoint. Ten deadline and wall-clock-stop
tests passed before submission.

The job uses a clean source snapshot containing the runtime code, configs,
scripts, and dependency files. This avoids SkyPilot's recursive ignore-file scan
following historical results symlinks back into the repository. The first,
unsubmitted upload request was cancelled and its empty temporary bucket removed.
Submission succeeded and job 87 entered `STARTING` while requesting capacity.
Inspect current status with `sky jobs queue`, and logs with `sky jobs logs 87`.

## Validation

The 31 new masking and JAX tests pass. They check uniform mask counts/subsets,
per-spectrum loss normalization, hidden-value isolation, quantized encoder
inference, fully masked gradients, compiled accumulated optimizer updates, and
exact equality of every trainable parameter shape with the reference student.

A two-step CPU run through `train_and_evaluate_jax`, using the local HDF5 test
fixture and a width-32, one-layer encoder/predictor, completed training and
validation and wrote checkpoints at steps 1 and 2. This validates the training
pipeline, not full-size accelerator memory or throughput.

Existing masking/input/capacity checks passed (49 tests). The config and MAE
regression check passed 36 tests; its one pre-existing failure is
`test_current_pretraining_config_records_model_and_data_defaults`, whose expected
key set omits the existing `pairmixer_triangle_attention_every_n_layers` and
`pairmixer_triangle_attention_num_heads` defaults. Those unrelated defaults and
that test were left unchanged.
