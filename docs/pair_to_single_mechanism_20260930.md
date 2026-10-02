# Direct pair values for the single stream

Research date: 2026-09-30. This is an architecture investigation with a small
controlled pilot, not evidence of a production-scale pretraining improvement.

**Updated finding after longer training:** the 15k-step, three-seed
[follow-up](#longer-training-prespecified-15k-comparison) does **not establish a
convincing overall performance gain** from mean-first pair values. Mean + CLS
chemistry improves slightly on the original test set, while prediction and
retrieval are essentially unchanged. The chemistry difference is much smaller
on previously unprobed validation molecules, with an uncertainty interval that
includes zero. Keep this as an experimental interface rather than a validated
training improvement. The [efficiency follow-up](#follow-up-efficient-pair-to-single-summaries)
still establishes that averaging rows before normalization/projection is much
cheaper than learned row pooling. The earlier experiments below document the
progression and why bias and CLS were retained.

## What the existing path can and cannot transmit

The independent reference is [pair279m attention](https://wandb.ai/iclac/jepa-finalrun/runs/pair279m-attn-n32-b2048-v6e32-2d-20260910-rootcause),
a JAX pair-only model with width 640, 25 encoder blocks, and alternating triangle
multiplication/attention. Its committed step-730000 export contains directional
`[32, 32, 640]` pairs: 31 selected peaks and CLS. Its W&B summary reached step
736850; the later `crashed` status does not establish a numerical training failure.
The `rootcause` name originally concerned TPU startup, not a diagnosis of the
pair-to-single interface.

The earlier [joint run](https://wandb.ai/iclac/jepa-finalrun/runs/1b-pairmixer-rms-postnorm-v6e8x8-b2048-ga8-16-16-mae07-disto03-optfix-fp32-lr3e4-wd01-95h-noflags-longwait-20260729-014634)
used single width 1536, pair width 640, 25 blocks, 47 peaks, and
`0.7 * MAE + 0.3 * distogram`. The current compact joint path supplies **single
representations** to its masked predictor; its joint distogram also derives
from predictor/context single latents. It is distinct from the later pair-only
distogram path that directly supervises the pair stream.

For one attention head, the existing interface is

\[
a_{ij}=\operatorname{softmax}_j(q_i^T k_j/\sqrt d+b(z_{ij})),
\qquad u_i=\sum_j a_{ij}v_j.
\]

Holding singles fixed, its derivative with respect to one pair bias is

\[
\frac{\partial u_i}{\partial b_{ij}}=a_{ij}(v_j-u_i).
\]

This has two separate weaknesses:

1. Similar single values make the pair effect small; identical values make it
   exactly zero, regardless of the information in the pairs.
2. Saturated attention suppresses the routing gradient. Q/K normalization can
   address logit scale, but it cannot remove the first limitation.

There is also a per-edge compression from the pair width to one scalar per
attention head. Bias can be effective when the single values provide a rich
basis; it does not directly transport pair value vectors. These are structural
properties, not measurements proving the severity of either failure in the
historical checkpoint. No historical checkpoint intervention has been run here.

### Saved checkpoint evaluations support investigating the interface

Existing layerwise evaluation JSONs were located and copied, with original
paths and SHA-256 hashes, into `artifacts/pair_to_single_20260930/historical/`.
They are **previous evaluations**, not fresh checkpoint runs in this task.

| Checkpoint/readout | MACCS macro AUC | Nearest-train-neighbor Morgan Tanimoto | Partial cosine/Tanimoto Spearman |
| --- | ---: | ---: | ---: |
| Joint 150k, final single mean | 0.7269 | 0.1493 | 0.0301 |
| Joint 150k, final single CLS | 0.7143 | 0.1458 | 0.0258 |
| Joint 150k, final pair mean | 0.7639 | 0.1819 | 0.1010 |
| Pair-only 730k, final pair mean | 0.7721 | 0.1863 | 0.0952 |

The within-joint-checkpoint gap is especially relevant: both streams saw the
same input, but the narrower pair readout made more chemistry accessible to the
linear probe and cosine geometry. This supports transferring pair content; it
does not prove that information is entirely absent from singles, or prove the
pair stream adds information conditional on a strong nonlinear single readout.
Do not attribute the difference between the separate runs to their interfaces:
their training durations, architectures, and peak selections differ. These
historical metrics also use a different evaluation protocol from the small
pilot below and must not be compared numerically with the pilot's AUC/ranks.

## Candidate: pool pair values, then project

The experimental implementation is
[`PairToSingleReadout`](../spectra_learning/models/pair_to_single_jax.py).
For valid outgoing edges of peak i, let

\[
\bar z_{ij}=\operatorname{RMSNorm}(z_{ij}),\quad
\alpha_{ijh}=\operatorname{softmax}_j(w_h^T\bar z_{ij}),\quad
\omega_{ijh}=\tfrac12\alpha_{ijh}+\tfrac1{2n_i}.
\]

Then compute

\[
p_{ih}=\sum_{j\in\mathcal V_i}\omega_{ijh}\bar z_{ij},\quad
m_{ih}=p_{ih}V_h,\quad
s_i\leftarrow s_i+\gamma\odot\operatorname{RMSNorm}
\left(W_o\operatorname{concat}_h m_{ih}\right).
\]

Use four heads, message width 32/head at production dimensions, and a learned
channel scale initialized to 0.1. The pilot uses 16 channels/head. The uniform
half preserves a dense direct-value contribution even if the learned routing
becomes sharp. The 50:50 mixture and initial scale are hypotheses, not tuned
optima. Nonzero scale and nonzero output projections avoid blocking pair
gradients at initialization. This update is independent of single Q/K logits.

This is a separate residual branch after a joint block in the pilot. It
preserves the existing single attention and pair updates. The candidate arm
disables the old pair bias to isolate the new path. Directed rows are retained;
neither pair symmetrization nor alignment between the 31-peak pair export and a
127-peak single export is assumed. Padding is excluded on both axes, including
empty spectra. CLS shares the same compact-axis contract.

The direct propagation of edge values is related to relation-aware attention
([Shaw et al., 2018, equation 3](https://aclanthology.org/N18-2074.pdf)); this
candidate uses a separate pair readout and a mean contribution. That paper is
architectural precedent, not evidence of a gain on mass spectra.

### Why pooling first matters

For a linear value projection,

\[
\sum_j\omega_{ijh}(\bar z_{ij}V_h)
=\left(\sum_j\omega_{ijh}\bar z_{ij}\right)V_h.
\]

The implementation uses the right-hand side. It avoids materializing
`[B, N, N, H, r]` projected values or projecting every edge into the full single
width. Forward outputs and pair-input gradients match an explicit edge-value
reference in the tests.

With pair width C, single width D, H heads and width r/head, additional work is
`O(B N² C H + B N H C r + B N H r D)` and retained pair storage remains
`O(B N² C)`. This changes neither the quadratic pair-state cost nor the cubic
triangle-update cost of the trunk. It is a scalable **interface**, not a
linear-time replacement for the entire pair model. Fixed-width singles cannot
losslessly preserve an arbitrary quadratic pair tensor; the training objectives
must select useful summaries.

At C=640, D=1536, H=4, r=32 the bridge has 282624 parameters/block. Replacing
the existing 12-head bias removes 8320 parameters/block, giving a net 274304.

### Simpler alternative: read the updated diagonal

The saved joint checkpoint's diagonal mean has AUC 0.7633 versus 0.7639 for the
full pair mean, and nearest-neighbor Tanimoto 0.1856 versus 0.1819. This motivates
reading `z[i, i]` directly after the triangle updates:

\[
s_i\leftarrow s_i+\gamma\odot\operatorname{RMSNorm}
\left(W_o W_v\operatorname{RMSNorm}(z_{ii})\right).
\]

The diagonal at an intermediate layer is not merely the input self-distance.
For example, an outgoing triangle update to `z[i, i]` contracts over relations
`(i, k)`; the incoming update also aggregates neighboring relations. A gradient
test with active triangle projections confirms that this readout depends on
off-diagonal input pairs through the trunk.

The added readout work becomes `O(B N C R + B N R D)`, with only `O(B N C)`
readout activations. It retains the trunk's existing quadratic pair storage and
cubic triangle updates. The pilot keeps the row-readout parameter layout to
match projection capacity; its unused scoring weights are an experimental
control, not a proposed production parameterization. The `bias_diagonal` arm
retains routing and adds these diagonal values.

### One routing mechanism for single and pair values

The `bias_weighted` arm retains the existing single attention and uses its same
weights for the pair-value readout. In the equations above, replace the learned
pair-only \(\alpha_{ijh}\) with
\(\operatorname{softmax}_j(q_i^Tk_j/\sqrt d+b_h(z_{ij}))\). The uniform half and
post-pooling projections stay unchanged. Routing can now select a relation and
transport its vector content through the same weights. This is still a lossy
summary of the pair row, not recovery of the entire pair tensor.

The pilot reconstructs the compact attention weights from the block inputs and
updated pairs. A test checks forward output and pair gradients against the
existing compact attention, including RoPE and padding. Production integration
should return/reuse the weights within that attention operation; compiler
elimination of the duplicated calculation has not been verified here. Its
normalization and residual projection are separate from the single value path.
The experimental parameter layout retains unused pair-only scoring weights to
keep projection capacity matched. All pilot attention/readout heads number four.

## Verification and measurements

`tests/test_pair_to_single_readout.py` covers compact/padded equivalence,
permutation equivariance, invalid-edge zero gradients, empty spectra,
project-before/pool-before equivalence, and the identical-single-values
counterexample. Additional cases cover mask-matched interventions,
off-diagonal information in the updated diagonal, and shared routing parity.
All eight cases passed on azh2's H100 NVL with JAX 0.11.0 and Flax 0.12.8.
Locally seven passed; bfloat16 was skipped because this CPU backend's DotThunk
cannot execute BF16 operands with FP32 accumulator output.

The random directional-derivative experiment used B=4, N=32, C=64, D=128, four
heads, NumPy seed 41 and bridge seed 7. With identical values, the bias-only
output derivative RMS was about 3e-8 (roundoff), versus 0.118 for the bridge.
Reducing value spread from 1 to 0.01 reduced the bias derivative from 0.272 to
0.00272; the bridge was unchanged. These magnitudes depend on initialization
and residual scaling. Their limiting behavior is the relevant result, not an
absolute comparison of information quantity.

### Interface cost on H100

`scripts/benchmark_pair_to_single.py` measures synchronized forward plus
parameter/input backward, B=8, D=1536, C=640, bfloat16, 30 repeats. It excludes
the optimizer and pair trunk. The bridge arm includes ordinary single attention
plus the readout. Memory is compiler-estimated arguments + temporaries + outputs
minus aliases, not measured peak device usage. Another user's GPU process was
active, so timings should be treated as shared-device measurements.

| Positions | Bias ms | Readout ms | Bias compiler MiB | Readout compiler MiB |
| --- | ---: | ---: | ---: | ---: |
| 32 | 0.387 | 0.633 | 166.8 | 174.8 |
| 64 | 0.572 | 0.703 | 241.4 | 219.9 |
| 128 | 1.185 | 1.451 | 472.6 | 585.9 |
| 256 | 3.878 | 4.829 | 1531.5 | 2045.2 |

The additional branch is not free. Whole-model training throughput is measured
separately in the pilot.

A second measurement retained pair bias in both value-readout arms:

| Positions | Bias ms | Bias + row values ms | Bias + diagonal ms | Compiler MiB: bias / row / diagonal |
| --- | ---: | ---: | ---: | ---: |
| 32 | 0.421 | 0.544 | 0.635 | 166.8 / 196.2 / 174.8 |
| 128 | 1.336 | 2.075 | 1.299 | 472.6 / 749.0 / 482.8 |
| 256 | 4.082 | 7.135 | 4.009 | 1531.5 / 2681.3 / 1542.3 |

The diagonal's larger-N cost is near the bias baseline; small timing differences
do not establish a speedup. The row readout has substantial memory/backward
overhead at N=256. These measurements use the general attention API, whereas
training uses the compact fastmixer implementation. They do not measure the
shared-routing candidate or establish TPU performance.

## Matched training and downstream evaluation

`scripts/pair_to_single_pilot.py` uses a direct JAX/Optax loop and the existing
`PeakSetJEPAJax.forward_mae` compact encoder/predictor path. It does not change
production configurations or require a training framework.

The initial four arms are no pair feedback, existing pair bias, uniform
pair-value pooling, and learned pair-value pooling. All share the same backbone initialization,
examples, masks, optimizer, and schedule. The mean control uses identical
message/output projections; only learned routing is removed. No-feedback
retains the same parameter tree; the compiler can eliminate unused pair
computation. Its timing is not a separately tuned single-only recipe. A fifth
arm retains pair bias and adds the candidate readout, testing whether direct
content should complement rather than replace routing. It was added after the
initial uniform-pooling arm worsened validation prediction loss.
The diagonal and shared-routing arms were added after inspecting initial
results. These are exploratory experiments with repeated use of the test fold,
not a locked confirmatory comparison. A new held-out evaluation is required to
confirm a selected architecture.

Settings: single width 128, pair width 64, four encoder blocks, two predictor
blocks, 31 m/z-ordered peaks plus CLS, 2 Da bins, MAE/distogram weights 0.7/0.3,
3000 steps, batch 64, AdamW peak LR 3e-4, 100-step warmup, final LR 3e-5, weight
decay 0.01. Context is 60% of valid peaks and target is 25%, with at most eight
targets; the compact capacities are 20/28/8. Target m/z values are not supplied
as predictor queries. Original peak positions still reveal rank in m/z order.
The pilot uses `fastmixer`, which includes a single-to-pair update in every
block. The historical joint configuration used `fastmixer-dense`, which lacks
that per-block update. Thus this is a controlled test of the proposed joint
interface, not a small exact reproduction of the historical training run.

Data uses the existing reference-run export preprocessing. Training has 194119
MassSpecGym1.5 spectra, validation 19429 and test 17556. The 28929 molecular
identities were checked for zero overlap between folds. Chemical labels are
used only for frozen downstream probes, not self-supervised training.

Evaluation fixes 8192 train spectra for the ridge probe, 4096 validation spectra
for regularization selection, and the complete test fold. Features concatenate
the mean of valid physical-peak singles and CLS. MACCS AUC is macro-averaged
over eligible variable bits, with all regularization selection on validation.

Retrieval has two explicit protocols:

- Spectral-library retrieval: one fixed reference spectrum for each of 2998
  held-out molecules, with the remaining 14558 test spectra as queries. Raw
  cosine, train-mean-centered cosine, and the frozen chemical probe are reported.
- Molecule-candidate retrieval: predicted MACCS versus the actual MACCS vectors
  of those 2998 candidate molecules, using all 17556 test spectra as queries.
  This is a small fingerprint retrieval check, not the full GLACIER or
  mass-filtered production retrieval benchmark.

Raw 1 Da sqrt-intensity bins from the same 31 peaks are included as a control.
Zero/readout-shuffle interventions measure whether the trained predictor uses
the new path. Those interventions alone cannot prove the content is semantic;
the disjoint chemical/retrieval checks are required as well.

The final intervention uses cyclic cross-spectrum donors with exactly matching
compact validity masks, leaving singleton mask groups unchanged. It changes
only the direct-value branch, not the bias branch. For shared routing, routing
is kept from the recipient and only pair values are substituted. Older seed-0
mean/readout training logs used a plain batch roll; their padding-confounded
shuffle numbers are superseded by the separate `interventions.json` files.
Singleton groups dilute the intervention, so a small effect is weak evidence
for content use, not proof that the path is unused. Zeroing also changes branch
scale, so zeroing alone is insufficient evidence of content-specific use.

## Pilot results

Means over seeds 0 and 1, except the one-seed uniform mean control. Lower
prediction loss is better; higher AUC and retrieval are better. Two seeds
are insufficient to support significance claims. Prediction uses 1024 fixed
validation spectra; downstream metrics use the complete test fold described
above. The single representation has 256 dimensions in every model arm.

| Mechanism | Seeds | Prediction loss | MACCS AUC | Spectral top-1 | Molecule MACCS top-1 | Training steps/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No pair feedback | 2 | 2.83093 | 0.60479 | 10.70% | 0.228% | 29.55 |
| Pair bias | 2 | **2.78383** | 0.61646 | 11.86% | 0.253% | 26.13 |
| Uniform pair values | 1 | 2.83677 | 0.62344 | 11.69% | 0.251% | 23.86 |
| Learned pair values, no bias | 2 | 2.82433 | **0.63066** | 11.12% | 0.162% | 23.52 |
| Bias + learned pair values | 2 | 2.79043 | 0.62241 | 11.99% | 0.188% | 23.47 |
| Bias + updated diagonal | 2 | 2.80322 | 0.61350 | 11.30% | **0.293%** | 23.07 |
| Shared routing + pair values | 2 | 2.79563 | 0.61924 | **12.19%** | 0.225% | 23.08 |

The raw-spectrum control has MACCS AUC 0.60456 and spectral top-1 **30.71%**.
Thus these short pretraining runs are not competitive spectral retrieval
encoders yet. More linearly accessible chemical bits do not automatically
produce better cosine geometry or molecule identification. MACCS candidate
retrieval is weak across all arms, and its ordering differs from spectral
retrieval. The positive three-way comparison above refers to spectral-library
retrieval; it does not hold for molecule-candidate retrieval.

Replacing bias with learned values improves AUC in both paired seeds
(+0.02005, +0.00835), but worsens prediction and spectral top-1 in both. The
combined bias + learned-values arm improves all three over no feedback in both
seeds. Against bias alone, its AUC gains are +0.00148 and +0.01043; prediction
loss worsens by 0.00392 and 0.00929, while spectral top-1 changes by -0.027 and
+0.295 percentage points. Shared routing improves spectral top-1 in both seeds
but loses AUC in seed 1 and worsens prediction in both. These results do not
establish a consistently superior single recipe.

The ordinary and combined value readouts have 2,486,720 parameters versus
2,436,032 for the baseline (+2.08%). Bias + row values reduced whole-pilot
throughput by about 10.2% relative to bias. Timing is indicative because the
H100 was shared; the diagonal's cheaper isolated operation did not yield a
whole-pilot throughput advantage in these measurements.

### Does the objective use spectrum-specific pair content?

The corrected donor substitution changes the donor for 94.14% of the 1024
validation spectra. Restoring all nine readout checkpoints reproduces intact
validation loss within 0.0000061 of the training-time evaluation. The table
shows increases in prediction loss relative to the restored intact model;
positive numbers mean the intervention hurts prediction.

| Readout | Seed | Zero branch Δloss | Mask-matched shuffled values Δloss |
| --- | ---: | ---: | ---: |
| Uniform values | 0 | +0.016717 | +0.013440 |
| Learned values, no bias | 0 | +0.014122 | +0.015547 |
| Learned values, no bias | 1 | +0.012662 | +0.013317 |
| Bias + learned values | 0 | +0.006236 | +0.000914 |
| Bias + learned values | 1 | +0.006758 | +0.000638 |
| Bias + diagonal | 0 | +0.006574 | +0.000039 |
| Bias + diagonal | 1 | +0.004391 | +0.000900 |
| Shared routing + values | 0 | +0.003771 | -0.001423 |
| Shared routing + values | 1 | +0.008040 | +0.000106 |

The value-only models use spectrum-specific pair content for prediction. The
combined models show much smaller content-substitution effects than zeroing
effects. This suggests the objective can rely largely on the bias path while
using the new branch partly as a generic residual adjustment. It does not
establish that every layer ignores content or that pair information is absent
from the single embedding. Substitution is applied at every readout block and
can also change downstream pair states through single-to-pair feedback.

### Recommendation and remaining decision

Use **explicit pair values plus existing routing** as the next joint interface
to investigate. Project after pooling; use a small nonzero residual scale; keep
the compact mask/CLS contract. Do not replace the old bias based on these data,
and do not choose the diagonal solely from its cheaper asymptotic readout cost.
Sharing attention weights is elegant, but the measured gains do not justify
preferring it over the separate row readout yet.

The limiting question has moved from “can pair content reach singles?” to
“does the objective reward preserving that content in useful single geometry?”
A transport channel alone does not guarantee all three downstream outcomes.
The next decisive comparison should freeze the interface choice, use longer
matched training with the historical configuration, and evaluate chemistry,
masked prediction, and the actual intended retrieval task on a fresh locked
split. If the joint objective still weakly uses the direct branch, an aligned
pair-summary prediction/distillation target is a separate hypothesis to test,
not a result established here. No new chemistry-supervised loss was added.

## Reproducibility

Per-seed measurements are preserved in
[`pair_to_single_results_20260930.csv`](pair_to_single_results_20260930.csv).
Full checkpoints, embeddings, JSON measurements, runtime details, source hashes,
and figures are under `artifacts/pair_to_single_20260930/`. The source snapshot
includes the pre-existing uncommitted backbone files used for the experiments;
the git commit alone is insufficient to reproduce them. Training source is
isolated on azh2 under
`/home/wuhao/spectra-learning/artifacts/pair_to_single_20260930/source/` and uses
that repository's existing `.venv/bin/python`.

Example commands from the local repository root:

```sh
.venv/bin/python -m scripts.pair_to_single_pilot prepare \
  --inputs artifacts/msg15_pair279m_step730000/prepared_inputs.npz \
  --tsv /path/to/MassSpecGym.tsv \
  --output artifacts/pair_to_single_20260930/panel.npz
.venv/bin/python -m scripts.pair_to_single_pilot train \
  --inputs artifacts/pair_to_single_20260930/panel.npz \
  --output artifacts/pair_to_single_20260930/pilot_bias_readout_s0 \
  --arm bias_readout --seed 0 --steps 3000 --batch-size 64
.venv/bin/python -m scripts.pair_to_single_pilot evaluate \
  --inputs artifacts/pair_to_single_20260930/panel.npz \
  --output artifacts/pair_to_single_20260930/pilot_bias_readout_s0
.venv/bin/python -m scripts.pair_to_single_pilot interventions \
  --inputs artifacts/pair_to_single_20260930/panel.npz \
  --output artifacts/pair_to_single_20260930/pilot_bias_readout_s0
.venv/bin/python -m scripts.report_pair_to_single
```

These example paths are relative to the local repository. On azh2, the snapshot
uses `../panel.npz` and `../pilot_bias_readout_s0`, with the existing environment
at `/home/wuhao/spectra-learning/.venv/bin/python`. For GPU runs set
`XLA_PYTHON_CLIENT_PREALLOCATE=false`; the pilot used four OpenMP/OpenBLAS threads.
The uniform mean arm used seed 0 only; every other arm used seeds 0 and 1.

The GPU environment is Python 3.12.11, JAX/JAXlib 0.11.0, Flax 0.12.8, Optax
0.2.8, NumPy 2.1.3, and Torch 2.11.0+cu130. JAX, Flax, and Optax match the
local repository environment; the existing remote Python/NumPy/Torch versions
do not match all current `pyproject.toml` constraints. Training uses JAX, not
Torch. These CUDA pilots are not production-environment or TPU validation.

## Follow-up: disabling CLS

The follow-up compares no feedback, pair bias, and bias + learned pair values,
each at seeds 0 and 1 with the same 3000-step training schedule and data. The
CLS single token and all three CLS pair-token tensors are removed, so there is
no CLS row or column in the pair tensor. Every remaining initial state array
is exactly equal to its CLS-enabled counterpart. This is done by initializing
the original model first and then removing CLS, avoiding an RNG-stream shift
from conditional parameter creation.

There are still 31 physical peaks. Compact capacities decrease from 20/28/8
to 19/27/8, preserving the peak budget and removing the extra CLS slot. The
parameter reduction is only 320 scalars: 128 single channels plus three
64-channel pair tokens. No extra loss or training hyperparameter changes.

To isolate CLS's effect during training, the primary downstream comparison
uses the same **128-dimensional physical-peak mean** for both models. The
previous 256-dimensional mean + CLS readout is retained as a separate
reference. Simply comparing the old concatenation to a no-CLS mean would
confound the token ablation with readout capacity and cosine geometry.

Runs and evaluations are isolated under
`artifacts/pair_to_single_no_cls_20260930/`; the six CLS-enabled peak-mean
evaluations are saved as `evaluation_mean.json` beside the original runs.
The original concatenated evaluations remain in `evaluation.json`.
The ablation adds `--no-cls` to training and `--readout mean` to evaluation in
`scripts/pair_to_single_pilot.py`. The report is regenerated with
`.venv/bin/python -m scripts.report_pair_to_single_cls`.

The targeted ablation test confirms identical shared initialization, removal
of exactly the four CLS parameters, a 31-by-31 pair axis, padding-aware token
counts, and the peak-mean readout. It passed locally; the complete readout
test file passed all nine cases on the H100. The GPU training path
uses the preserved backbone source from the original experiment, rather than
unrelated subsequent workspace changes. This remains an exploratory comparison
on the original evaluation folds; it does not introduce a fresh held-out split.
A restored no-CLS bias + values checkpoint reproduced all recorded validation
and intervention metrics exactly.

### CLS results

**Keep CLS for this joint pilot.** All six matched runs have worse prediction
loss, chemical probe AUC, and raw spectral retrieval when CLS is removed. The
chemical differences for no-feedback and bias-only models are small; two seeds
do not establish statistical significance or a production-scale result.

Each cell below is **CLS enabled → disabled**, averaged over two seeds. Both
sides use the same 128-dimensional peak-mean readout.

| Interface | Prediction loss ↓ | MACCS AUC ↑ | Spectral top-1 ↑ |
| --- | ---: | ---: | ---: |
| No pair feedback | 2.83093 → 2.85886 | 0.60294 → 0.60246 | 11.40% → 10.85% |
| Pair bias | 2.78383 → 2.81804 | 0.60852 → 0.60744 | 11.97% → 11.02% |
| Bias + learned pair values | 2.79043 → 2.82058 | 0.61570 → 0.60588 | 11.86% → 11.20% |

The effect is especially relevant to fusion: with CLS, adding pair values to
bias raises peak-mean chemical AUC by 0.00718 on average. Without CLS, adding
pair values lowers it by 0.00155. That reversal occurs in both seeds. CLS's
benefit is therefore not explained solely by concatenating an extra vector
into the chemical probe: it also affects the physical-peak representations
learned during training.

The original 256-dimensional mean + CLS readout gives AUC 0.60479 / 0.61646 /
0.62241 for no feedback / bias / bias + values. Those numbers should not be
substituted for the matched 128-dimensional baseline in the table. The
separate molecule-MACCS retrieval averages also decline without CLS, from
0.154% / 0.251% / 0.162% to 0.134% / 0.157% / 0.125%; this remains the limited
fingerprint-candidate protocol described earlier.

For the value branch, mask-matched shuffling increases prediction loss by
0.00246 without CLS versus 0.000776 with CLS, averaged over seeds. Zeroing the
branch increases loss by 0.01342 versus 0.00650. Removing CLS makes the direct
branch somewhat more consequential to prediction, but does not improve the
final loss or embeddings. This ablation removes CLS from single attention,
pair updates, and predictor context together; it cannot isolate which of
those uses provides the benefit.

Observed steps/s increased by approximately 2.6% / 4.7% / 1.6% for the three
interfaces. These are runs at different times on a shared H100, so they are
not a controlled speedup measurement. Removing one token saves little at the
31-peak pilot size and removes only 320 parameters.

Per-seed measurements and all three readout conditions are in
[`pair_to_single_cls_results_20260930.csv`](pair_to_single_cls_results_20260930.csv).
The comparison figure is
`artifacts/pair_to_single_no_cls_20260930/cls_outcomes.png`; complete checkpoints,
embeddings, source archive, source hashes, and runtime versions are preserved
in the same artifact directory.

## Follow-up: efficient pair-to-single summaries

Two cheaper value channels retain CLS and the original pair bias. Both add
small direct pair-value residuals after each joint block. They use the same
backbone initialization, examples, masks, optimizer, and 3000-step schedule as
the earlier pilots, with seeds 0 and 1. Both have 2,485,696 pilot parameters,
versus 2,486,720 for learned row pooling and 2,436,032 for bias alone.

### Mean first, normalize once per token

For the valid pair row of token i,

\[
\mu_i=\frac{1}{|\mathcal V|}\sum_{j\in\mathcal V}z_{ij},\qquad
\Delta s_i=\gamma\odot\operatorname{RMSNorm}
\left(W_oW_v\operatorname{RMSNorm}(\mu_i)\right).
\]

Use a rank-64 message in the small pilot and rank 128 at D=1536, C=640.
The channel scale starts at 0.1. Its work is
`O(B N² C + B N C r + B N r D)`; normalization and projected activations have
only a token axis. There is no per-edge score, softmax, value projection, or
normalized pair activation. This is a direct value channel independent of the
single values' diversity and single Q/K logit scale.

**Normalization and averaging do not commute.** This changes the summary from
the previous normalized-edge readout, so its quality needs new training and
evaluation. In contrast, changing the mean's implementation from a dot product
to a reduction is a mathematically equivalent kernel optimization; tests check
its forward values and pair gradients.

The implementation uses a masked `sum(..., dtype=float32)` and then divides by
the valid count. The initial equivalent GEMM formulation produced a large
FP32 pair-gradient temporary on H100. The reduction lets XLA fuse masking,
broadcasting, and casting in its backward pass. Final reported mean runs use
this reduction implementation; an unfinished initial run was excluded and
restarted to preserve that comparison.

### Read the already-updated CLS pair edges

Let c be the actual compact position of CLS, obtained from original token
positions rather than assuming it is the last padded slot. Read

\[
r_i=[\operatorname{RMSNorm}(z_{i,c});
     \operatorname{RMSNorm}(z_{c,i})],\qquad
\Delta s_i=\gamma\odot\operatorname{RMSNorm}(W_oW_v r_i).
\]

Gather the two vectors **before** normalization. Use rank 48 in the pilot to
match the mean channel's parameter count, and rank 96 at production widths.
The added interface work is `O(B N (2 C r + r D))`, with O(B N C) readout
activations. This reuses the existing CLS relations, preserving direction.

Those relations are not merely learned input constants after the trunk updates.
For example, an outgoing triangle update to `z[i,c]` contracts information
from `(i,k)` and `(c,k)` over k. A test activates the triangle output projections
and confirms that a readout of CLS edges then has nonzero gradients to ordinary
peak-to-peak pairs. Reading the same edges before the triangle block has zero
gradient to those other pairs. This establishes a possible information path,
not that training necessarily makes optimal use of it.

Both mechanisms leave the trunk's quadratic pair storage and cubic triangle
work in place. The CLS channel has a linear **forward readout**, not a linear
whole encoder; gradients still target a dense pair tensor, so this also does
not promise linear backward memory traffic. Empty/padded spectra,
compact-position permutations, masking, BF16
gradients, and reduction equivalence are covered by eight passing tests on
both the local CPU and azh2 H100.

### Interleaved H100 interface measurements

The final benchmark randomizes condition order within each of 60 repeats,
after three warmup calls per condition. This reduces temporal load bias from
the other GPU job; timings remain shared-device estimates. Scope is unchanged:
B=8, D=1536, C=640, BF16, forward plus parameter/input backward, no optimizer
or pair trunk. Memory is the compiler estimate, not observed peak allocation.

| Tokens | Bias ms / MiB | Learned row pooling ms / MiB | Mean then normalize ms / MiB | CLS pair edges ms / MiB |
| --- | ---: | ---: | ---: | ---: |
| 32 | 0.578 / 166.8 | 0.641 / 196.2 | 0.652 / 169.4 | 0.656 / 175.5 |
| 128 | 3.492 / 501.6 | 4.192 / 749.0 | 3.585 / 511.7 | 3.607 / 511.9 |
| 256 | 8.310 / 1531.5 | 13.445 / 2681.3 | 8.429 / 1546.2 | 8.453 / 1543.3 |

At N=256, the mean channel adds about **14.7 MiB** over bias; learned row
pooling adds about **1149.8 MiB**. The mean implementation's initial GEMM version
used 2672.2 MiB in the earlier non-interleaved benchmark. Thus the reduction
implementation matters to actual memory use, not only asymptotic operation
counts. At N=32 the timing overhead remains noticeable, and these measurements
do not establish a whole-training speedup or TPU performance.
The [cost figure](../artifacts/pair_to_single_efficient_20260930/interface_cost.png)
includes timing variability and all three lengths.

The new implementation is `PairSummaryToSingleReadout` in
`spectra_learning/models/pair_to_single_jax.py`; select pilot arms
`bias_poolnorm` or `bias_cls`. Artifacts are isolated in
`artifacts/pair_to_single_efficient_20260930/`. Regenerate comparisons with
`.venv/bin/python -m scripts.report_pair_to_single_efficiency`.

### Matched training results

These are two-seed means after 3000 steps, with the same original pilot folds.
The first table uses the original 256-dimensional peak mean + CLS readout.

| Interface | Prediction loss ↓ | MACCS AUC ↑ | Spectral top-1 ↑ |
| --- | ---: | ---: | ---: |
| Pair bias | 2.78383 | 0.61646 | 11.86% |
| Bias + learned row pooling | 2.79043 | 0.62241 | 11.99% |
| Bias + mean then normalize | 2.78793 | 0.62590 | 12.13% |
| Bias + CLS pair edges | 2.78194 | 0.62035 | 11.79% |

Mean-first pooling improves chemistry and spectral retrieval over bias alone
in both seeds with this readout. Prediction improves in seed 0 and worsens in
seed 1, for an average loss increase of 0.00410 (about 0.15%). Its average
prediction loss, AUC, and retrieval are all better than learned row pooling,
although that ordering is not consistent in each seed.

The picture changes when the global CLS vector is excluded from the downstream
readout. CLS is still present during training and inference in every row below;
only the final probe/retrieval features change to 128-dimensional peak means.

| Interface | Peak-mean MACCS AUC ↑ | Peak-mean spectral top-1 ↑ |
| --- | ---: | ---: |
| Pair bias | 0.60852 | 11.97% |
| Bias + learned row pooling | 0.61570 | 11.86% |
| Bias + mean then normalize | 0.60950 | 11.82% |
| Bias + CLS pair edges | 0.60985 | 12.03% |

The mean-first chemistry gain is concentrated in the representation including
CLS. It does not reproduce learned row pooling's improvement in physical-peak
mean chemistry. CLS-edge feedback is attractive for its forward readout cost,
and improves prediction in both seeds, but its chemistry/retrieval advantage
is small and depends on readout and seed. No mechanism is an established
all-three winner.

The prediction interventions also limit the interpretation. Shuffling the
mean-first value branch at every block changes loss by -0.00031 and -0.00018;
zeroing it raises loss by 0.00465 and 0.00846. The CLS-edge branch has shuffle
increases of 0.00026 and 0.00089. Thus the prediction task shows little
dependence on which spectrum supplied these direct messages; sensitivity to
zeroing alone is insufficient evidence of content use.

Per-seed results for both readouts are in
[`pair_to_single_efficiency_results_20260930.csv`](pair_to_single_efficiency_results_20260930.csv).
These are exploratory comparisons on reused evaluation folds, with two seeds
and roughly one training epoch. They do not establish statistical significance,
production convergence, or generalization to a new untouched test set.

### Does the cheap message actually carry chemical information?

We held the chemical ridge probe fixed after fitting intact training features
with the already-selected validation hyperparameter. At inference, we replaced
each direct pair message with a different spectrum's message, matching the
complete valid-token mask. 87.62% of test spectra have a different eligible
donor within their batch; singleton groups remain unchanged. No labels or
probe weights change.

Changing only the final bridge barely affects chemical AUC: mean-first pooling
drops by 0.000061 on average, and CLS-edge feedback by 0.000442, using the
mean + CLS readout. This narrowly isolated final-step effect is small. Earlier
blocks might already have transferred information, so we additionally tested
the recommended mean-first channel at **all four blocks**.

| Mean-first readout | Intact AUC | All bridges zeroed | All bridges shuffled |
| --- | ---: | ---: | ---: |
| Mean + CLS | 0.62590 | 0.62292 | 0.62430 |
| Physical-peak mean | 0.60950 | 0.60608 | 0.60728 |

All-block shuffling lowers AUC in both seeds and both readouts. The mean + CLS
drops are 0.00203 and 0.00117; peak-mean drops are 0.00296 and 0.00148. This is
evidence of modest useful spectrum-specific content in the accumulated value
messages. It does not isolate a chemically unique factor, establish significance,
or imply that every block contributes equally. Unlike the final-only test,
all-block interventions also affect later pair updates and attention bias
through changed single states.

The all-block intact exports exactly reproduce stored test embeddings, and
all fixed probes exactly reproduce their saved AUCs on saved features. One
final-only mean seed had small restored-feature differences (maximum 0.03125,
AUC difference 1.36e-7); its shuffle effect is measured against that run's own
intact export. The conclusions do not depend on this tiny metric discrepancy.
Per-seed values and restoration checks are in
[`pair_to_single_efficiency_interventions_20260930.csv`](pair_to_single_efficiency_interventions_20260930.csv).
The exports and JSON measurements are preserved beside each model checkpoint.

These chemistry effects coexist with near-zero prediction sensitivity to
shuffling. The efficient channel is carrying some usable content, but the
current masked objectives do not yet strongly require that content. That is
the remaining limitation for the user's combined three-objective goal.

### Implementation choice and reproduction

Use `PairSummaryToSingleReadout(summary="mean", message_dim=128)` as the
default efficient candidate at single width 1536 and pair width 640. After a
joint block updates pairs, add its output to the single stream. Keep the
existing pair bias and CLS. Start the learned channel scale at 0.1, as tested.
At these widths the module adds 280,064 parameters per block. This is an
experimental module and pilot integration; production training defaults have
not been changed.

The operation order matters: masked FP32 sum and count division, then RMSNorm,
then the two low-rank projections, output RMSNorm, and channel scale. Projecting
or normalizing every pair first loses much of the memory advantage. At the
current short 32-token length the measured interface overhead is about
0.074 ms and 2.6 MiB; the larger benefit over learned pooling appears as N grows.
The timings include only the interface and remain subject to shared-GPU load.

The alternate `summary="cls"` is useful when avoiding a full row reduction is
important and the pair trunk already updates CLS relations. Its linear forward
readout is appealing, but the present experiments do not justify preferring it
for chemistry quality over mean-first pooling.

Use the preserved `source_snapshot.tar.gz` and `source_manifest.json` in the
efficiency artifact directory for reproduction. The backbone snapshot predates
unrelated workspace edits. On azh2, the extracted snapshot is `source/` beneath
that directory; run from that directory using the repository interpreter:

```sh
XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=4 \
  /home/wuhao/spectra-learning/.venv/bin/python -m scripts.pair_to_single_pilot train \
  --inputs ../../pair_to_single_20260930/panel.npz \
  --output ../reproduce_bias_poolnorm_s0 --arm bias_poolnorm --seed 0 --steps 3000
```

Repeat for seed 1 and `bias_cls`; the pilot rank and all other settings are
selected by the recorded arm. `runtime.json` records the remote versions.
The interleaved benchmark is `benchmark_interleaved.json`; the earlier
`benchmark.json` is retained for provenance and is superseded for mean pooling.

## Longer training: prespecified 15k comparison

The user requested longer training to check whether the efficient candidate's
performance advantage persists. This follow-up compares **bias alone versus
bias + mean-first values**, retaining CLS, with seeds **0, 1, and 2**. Each run
uses 15,000 steps at batch size 64, approximately 5.34 sampled epoch equivalents
over 179,847 eligible training spectra. Both arms start from fresh, matched
backbone initialization: the earlier checkpoints contain model weights but
not the optimizer state needed for faithful continuation.

The architecture, losses, data, masks, optimizer, and peak budget match the
earlier pilot. The shared cosine schedule now ends at 15k: warm up for 100 steps
to 3e-4 and decay to 3e-5. Therefore the 3k checkpoint of this longer schedule
is not equivalent to a completed earlier 3k run. Checkpoints at 3k, 7.5k, and
15k save full model/Fourier state, optimizer moments and step, NumPy RNG state,
and the original schedule horizon. A serialization check exactly reproduced
the next scheduled AdamW update and RNG samples after restoration.

Masked prediction is evaluated on all **4,096** fixed probe-validation spectra
with mask seed 42, instead of the earlier 1,024-spectrum prefix. Loss numbers
from the two protocols must not be compared directly. Intermediate checkpoints
export only probe training and validation features. The **fixed 15k endpoint**
is used for final test comparisons; checkpoints are not chosen by test results.
Both 256-dimensional mean + CLS and 128-dimensional peak-mean readouts are
evaluated. The new learning-curve scorer exactly reproduced the existing
validation AUC/alpha and raw retrieval metrics for both readouts.

The original held-out test fold has already informed this research. To add a
separate check, before any longer-run endpoint or paired model comparison was
available, we fixed a secondary set: every validation molecule absent from
the original `probe_val`. It contains **1,494 previously unprobed molecules,
3,994 spectra, and 2,500 retrieval queries**. These molecules are disjoint from
probe training, probe validation, and the original test molecules. No training
data or ridge-probe fitting changes. The exact original training/validation
features and selected ridge hyperparameters are reused when scoring this set.
It comes from the same dataset and tends to contain fewer replicate spectra;
it is not an external-dataset validation.

The frozen protocol, isolated training source, runtime versions, checkpoints,
and evaluations are in `artifacts/pair_to_single_long_20260930/`. The training
script is `scripts/pair_to_single_longrun.py`; its `--resume` option restores
full state while enforcing the same arm, seed, batch size, and schedule horizon.
`scripts/pair_to_single_long_eval.py` produces validation curves and final test
scores; `scripts/pair_to_single_confirm.py` evaluates the unused-molecule set.
`scripts/report_pair_to_single_long.py` aggregates paired seed differences.

For uncertainty, `scripts/pair_to_single_long_uncertainty.py` generates 500
paired bootstrap resamples of whole molecules. All spectra of a molecule stay
together, and all models/readouts receive the same resampling weights within
each dataset. AUC is calculated from the original frozen ridge probes; retrieval
keeps the original gallery fixed and resamples query-molecule groups. The
interval is the 2.5th–97.5th percentile of the mean paired difference across
the three trained seeds. It is conditional on these models, fitted probes,
and galleries; seed differences are reported separately. It is not a bound
on variability across all future training seeds or datasets.

AUC bits with no positive or negative weighted examples in a draw are omitted
identically for every model in that draw. The weighted/tied AUC calculation was
checked against scikit-learn and reproduced stored endpoint AUCs. The intervals
are descriptive and unadjusted for the multiple readouts/metrics. They were
added after the first endpoint pair was observed, without changing the training
protocol, endpoint, models, or evaluation sets.

### Longer-run results and decision

**The early results were too encouraging as evidence for an overall gain.**
All six 15k runs and all 18 checkpoint evaluations completed. The source used
at training start was preserved: all 131 original source files remained
unchanged, and three analysis scripts were added. All six final model and
optimizer states contain finite arrays, with optimizer counters exactly 15,000.
The comparison remains a 2.5M-parameter, 31-peak pilot, not a production-size
pretraining result.

The following are three-seed means using the original **mean + CLS** readout.
Intervals describe the paired difference, candidate minus bias, and use the
molecule-bootstrap procedure above.

| Evaluation | Metric | Bias | Mean-first values + bias | Difference [95% interval] |
| --- | --- | ---: | ---: | ---: |
| Fixed validation masks | Prediction loss ↓ | 2.89332 | 2.89397 | +0.00065; paired seed SD 0.00629 |
| Original test | MACCS AUC ↑ | 0.62861 | 0.63211 | +0.00351 [+0.00019, +0.00627] |
| Original test | Spectral top-1 ↑ | 14.93% | 15.01% | +0.078 pp [−0.204, +0.400] |
| Unused validation molecules | MACCS AUC ↑ | 0.62018 | 0.62070 | +0.00052 [−0.00344, +0.00436] |
| Unused validation molecules | Spectral top-1 ↑ | 21.64% | 21.67% | +0.027 pp [−0.513, +0.577] |

Original-test chemical AUC improves in all three seeds, by 0.00376, 0.00289,
and 0.00388. This is a small, consistent signal on the previously examined test
fold; its conditional interval only narrowly excludes zero. It is not strong
confirmation of a broader gain: the unused-molecule AUC differences are
0.00078, 0.00166, and −0.00088, and their interval spans zero.

Retrieval depends on seed. Original-test top-1 changes by **+0.728, +0.103,
and −0.598 percentage points**. On the unused-molecule set, the differences are
**+0.720, +0.680, and −1.320 points**. The third seed largely cancels the gains
in the first two. The two retrieval sets have different galleries, so their
absolute percentages should not be compared directly.

Prediction loss is effectively tied: the paired changes are −0.00519, +0.00731,
and −0.00016. Its constituent losses also remain almost unchanged on average:
MAE is 2.82781 → 2.82857, and distogram loss is 3.04617 → 3.04657.

Removing CLS only from the final readout does not produce a convincing win
either. These rows retain CLS during training and use **physical-peak means**:

| Evaluation | Metric | Bias | Mean-first values + bias | Difference [95% interval] |
| --- | --- | ---: | ---: | ---: |
| Original test | MACCS AUC ↑ | 0.62143 | 0.62179 | +0.00037 [−0.00252, +0.00342] |
| Original test | Spectral top-1 ↑ | 15.90% | 15.84% | −0.057 pp [−0.348, +0.169] |
| Unused validation molecules | MACCS AUC ↑ | 0.61180 | 0.61358 | +0.00178 [−0.00133, +0.00511] |
| Unused validation molecules | Spectral top-1 ↑ | 24.08% | 24.39% | +0.307 pp [−0.104, +0.717] |

The learning curves help explain why the short comparison was insufficient.
Within the new matched schedule, the mean + CLS validation-AUC advantage
narrows from **0.01194 at 3k** to **0.00380 at 7.5k** and **0.00167 at 15k**.
Prediction curves converge, and there is no consistent validation-retrieval
advantage. This is an observed convergence pattern, not a diagnosis of its
cause. The [learning-curve figure](../artifacts/pair_to_single_long_20260930/learning_curves.png)
shows the means and seed variation for both readouts.

The message path is used after longer training. Shuffling its content at every
block raises prediction loss by **0.01256, 0.00712, and 0.01517** in the three
seeds. Zeroing it raises loss by 0.01104, 0.01439, and 0.01386. Thus usable
spectrum-specific information travels through the branch; that dependence
does not establish an advantage over a separately trained bias-only model.

**Decision:** retain mean-first pooling as a cheap experimental pair-value
interface, but do not treat it as a demonstrated overall performance upgrade.
The original test supports a small chemical-readout gain. Prediction and
retrieval gains are not established, and the previously unprobed molecule set
does not provide clear confirmation of a chemistry gain. Production training
defaults remain unchanged by this investigation.

Per-seed endpoints are in
[`pair_to_single_long_results_20260930.csv`](pair_to_single_long_results_20260930.csv),
paired differences and intervals in
[`pair_to_single_long_deltas_20260930.csv`](pair_to_single_long_deltas_20260930.csv),
and validation curves in
[`pair_to_single_long_curves_20260930.csv`](pair_to_single_long_curves_20260930.csv).
The artifact directory also retains all optimizer/RNG checkpoints, frozen
probe features, bootstrap samples, source archives/hashes, and runtime versions.
