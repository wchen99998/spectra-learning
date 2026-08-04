# Adversarial masked-peak training

## Scope

This document specifies the training procedure implemented by
`configs/adversarial_fake_peaks_equal_50m.py`,
`spectra_learning/models/fake_peaks.py`, and
`spectra_learning/training/adversarial_fake_peaks.py`.

The task is peak-level discrimination inside a mixed spectrum. A whole
spectrum is never labeled fake. Generated peaks in selected masked positions
are fake, paired untouched peaks from the same masked region are real, and
context peaks are visible but excluded from the detection loss.

## Run contract

| Item | Value |
|---|---:|
| Optimizer steps | 50,000 |
| Effective batch per optimizer step | 1,280 spectra |
| Gradient accumulation | 16 microbatches |
| Microbatch | 80 spectra |
| Nominal spectra processed | 64,000,000 |
| Generator device | `cuda:1` |
| Discriminator device | `cuda:0` |
| Autocast | bfloat16 |
| Optimizer | fused Adam, \(\beta=(0.9,0.95)\), no weight decay |
| Global gradient clip | \(1.0\) |
| Generator parameters | 51,494,108 total; 50,935,452 trainable |
| Discriminator parameters | 51,405,531 total; 51,384,475 trainable |
| Total-parameter difference | 88,577, or \(0.1723\%\) of the discriminator |
| Seed | 66 |

One process controls both GPUs; this is not DDP. The dataset is the pinned
`novogaia/massive-v1-ms2-100m-stratified-x16` revision
`7ff47061cbde23e4cdd113378dcfb489e86b32c4`. Preprocessing keeps 47 peak
slots, drops intensities below \(10^{-4}\), and uses grouped peak filtering
with a 0.02 Da shoulder and isotope charges 1, 2, and 3. It initially orders
peaks by intensity. The trainer independently shuffles every aligned peak
field in every microbatch before either model sees it.

Both models are randomly initialized. There is no source-model checkpoint,
frozen generator, or teacher model. Resuming from this run's own checkpoint
continues that from-scratch trajectory.

The parameter-matching criterion uses constructed total parameters. The
inherited generator backbone contains a 450,000-parameter distogram head that
`DynamicPeakGenerator` does not call, and the discriminator computes a
224-parameter final pair norm whose pair output is discarded. Their
gradient-connected counts are therefore 50,485,452 and 51,384,251,
respectively, a \(1.75\%\) difference. The table's “trainable” counts mean
`requires_grad=True`, not that every parameter receives a gradient.

## Notation

For a microbatch of \(B=80\) spectra, let

- \(b\in\{1,\ldots,B\}\) index spectra;
- \(i\in\{1,\ldots,N\}\) index peak slots;
- \(m_{b,i}\in[20,1000]\) be m/z in Da and
  \(x_{b,i}=m_{b,i}/1000\) its stored value;
- \(a_{b,i}\in[0,1]\) be normalized intensity;
- \(V_b\) be the valid-peak set;
- \(C_b\subset V_b\) be the context set;
- \(T_b\subset V_b\setminus C_b\) be the masked target set;
- \(G_{b,i}=(\hat x_{b,i},\hat a_{b,i})\) be the generator output;
- \(X_{b,i}=(x_{b,i},a_{b,i})\) be the original peak; and
- \(\mathcal T=\{(b,i):i\in T_b\}\) be the target peaks pooled across the
  microbatch.

The data mask uses one random target block, context fraction \(0.35\), target
fraction \(0.50\), and no context/target overlap. With grouped preprocessing,
the random mask initially samples whole peak groups. For \(n_b\) valid groups,
the exact configured group counts are

\[
c_b
=
\min\!\left(
  \max(\operatorname{round}(0.35n_b),1),
  \max(n_b-1,0)
\right),
\qquad
t_b
=
\min\!\left(
  \max(\operatorname{round}(0.50n_b),1),
  n_b-c_b
\right).
\]

`round` is Python's ties-to-even rounding. After sampling, the most intense
valid peak is forced into \(C_b\) and removed from \(T_b\), so the base peak
is always context and never generated. That final per-peak edit can split its
original peak group.

## Generator

### Architecture

The generator is a PeakSet-JEPA/PairMixer masked predictor with

- model dimension 448;
- an 11-layer, 7-head encoder;
- pair dimension 224;
- a 3-layer, 7-head masked predictor of width 280;
- no EMA or frozen teacher;
- categorical m/z and intensity heads; and
- scalar m/z and intensity residual-shift heads.

It encodes only \(C\), uses both the encoded single and pair representations,
and predicts every position in \(T\). Non-target peak values are copied
unchanged. The shared Fourier peak-input representation is specified in
[MZ_EMBEDDER_MATHEMATICS.md](MZ_EMBEDDER_MATHEMATICS.md).

### Categorical and residual sampling

The m/z bin width is

\[
\Delta_m=0.5\ \mathrm{Da},
\qquad
K_m=\left\lceil\frac{1000}{0.5}\right\rceil=2000,
\]

and the intensity bin width is

\[
\Delta_a=0.1,
\qquad
K_a=\left\lceil\frac{1}{0.1}\right\rceil=10.
\]

For target \(i\), the generator emits categorical logits
\(\ell_i^m\in\mathbb{R}^{2000}\) and
\(\ell_i^a\in\mathbb{R}^{10}\). Bins below 20 Da are forbidden. A hard
Gumbel-softmax sample with temperature \(\tau=1\) is used in the forward pass:

\[
k_i^v
=
\operatorname*{argmax}_k
\left(\ell_{i,k}^v+\gamma_{i,k}\right),
\qquad
\gamma_{i,k}\sim\operatorname{Gumbel}(0,1),
\qquad v\in\{m,a\}.
\]

PyTorch's straight-through `hard=True` estimator supplies the backward
gradient. This lower-bin mask applies only to the generator: the
discriminator reconstruction head retains all 2,000 classes in its softmax.

Each head also emits a residual shift \(s_i^v\). With
\(\epsilon=10^{-6}\),

\[
u_{0,i}^v\sim\operatorname{Uniform}(0,1),
\qquad
u_i^v=\operatorname{clip}(u_{0,i}^v,\epsilon,1-\epsilon),
\qquad
r_i^v
=
\sigma\!\left(\operatorname{logit}(u_i^v)+s_i^v\right).
\]

The generated values are

\[
\hat m_i=(k_i^m+r_i^m)\Delta_m,
\qquad
\hat x_i=\frac{\hat m_i}{1000},
\qquad
\hat a_i=(k_i^a+r_i^a)\Delta_a.
\]

The two residual-shift heads are zero-initialized, so their initial residual
distribution is the clamped uniform base distribution.

### Reconstruction target and residual likelihood

For a scalar target \(q\), scale \(S\), and bin width \(\Delta\), define

\[
t=\frac{Sq}{\Delta},
\qquad
c=\operatorname{clip}(\lfloor t\rfloor,0,K-1),
\qquad
r=\operatorname{clip}(t-c,0,1).
\]

Here \(S=1000\) for stored m/z and \(S=1\) for intensity. The categorical
term is cross-entropy against \(c\).

For the residual term, set

\[
z=\operatorname{logit}
\left(\operatorname{clip}(r,\epsilon,1-\epsilon)\right).
\]

The logit-shifted uniform density gives the exact negative log likelihood

\[
\mathcal L_{\mathrm{res}}(s;r)
=
-\log
\frac{
  \sigma(z-s)\sigma(-(z-s))
}{
  \sigma(z)\sigma(-z)
}.
\]

At \(s=0\), this is the uniform density and the NLL is zero. Because this is
a continuous density, a valid residual NLL may be negative.

The code pools all target peaks in the microbatch before dividing. For
\(n_T=|\mathcal T|\), the generator reconstruction losses are

\[
\begin{aligned}
\mathcal L_G^m
&=
\frac{1}{n_T}
\sum_{(b,i)\in\mathcal T}
\left[
  \operatorname{CE}(\ell_{b,i}^m,c_{b,i}^m)
  +\mathcal L_{\mathrm{res}}(s_{b,i}^m;r_{b,i}^m)
\right],\\
\mathcal L_G^a
&=
\frac{1}{n_T}
\sum_{(b,i)\in\mathcal T}
\left[
  \operatorname{CE}(\ell_{b,i}^a,c_{b,i}^a)
  +\mathcal L_{\mathrm{res}}(s_{b,i}^a;r_{b,i}^a)
\right],\\
\mathcal R_G&=\mathcal L_G^m+\mathcal L_G^a.
\end{aligned}
\]

Both configured reconstruction weights are one.

## Mixed real/fake construction

After generating every target, the trainer samples a fresh uniform random
ranking over each \(T_b\). Let

\[
q_b=\left\lfloor\frac{|T_b|}{2}\right\rfloor.
\]

The first \(q_b\) ranked target peak slots form the fake set \(F_b\). The
first \(2q_b\) form the discriminator detection set \(P_b\). This partition
is per peak slot, not per peak group. Therefore

\[
F_b\subset P_b\subset T_b,
\qquad
|F_b|=|P_b\setminus F_b|=q_b.
\]

If \(|T_b|\) is odd, one target is left outside \(P_b\). It still contributes
to generator reconstruction but is hidden from the discriminator. Define the
pooled microbatch sets

\[
\mathcal F=\{(b,i):i\in F_b\},
\qquad
\mathcal P=\{(b,i):i\in P_b\}.
\]

The mixed peak values are

\[
\widetilde X_{b,i}
=
\begin{cases}
G_{b,i},&i\in F_b,\\
X_{b,i},&i\notin F_b.
\end{cases}
\]

The discriminator's visible set in spectrum \(b\) is

\[
V_{D,b}=C_b\cup P_b.
\]

Thus every discriminator input spectrum contains context plus an exactly
balanced, randomly paired set of real and generated masked peaks. The labels
are per peak:

\[
y_{b,i}=
\begin{cases}
1,&i\in F_b\quad\text{(generated/fake)},\\
0,&i\in P_b\setminus F_b\quad\text{(untouched/real)}.
\end{cases}
\]

There is no spectrum-level label. Positions in \(C_b\) are visible context but
do not enter binary cross-entropy. The original values \(X_{b,i}\) are
retained as reconstruction targets but are not encoder inputs at generated
positions.

## Discriminator

### Architecture

The discriminator uses the same 448-dimensional PairMixer family with a
13-layer, 7-head encoder and pair dimension 224. Three linear heads act on
each visible peak representation:

- one scalar fake logit \(d_i\);
- 2,000 original-m/z-bin logits; and
- 10 original-intensity-bin logits.

The heads use Xavier-normal weights and zero biases.

### Peak detection

With positive logit meaning fake, the balanced peak-detection loss is

\[
\mathcal L_D^{\mathrm{det}}
=
\frac{1}{|\mathcal P|}
\sum_{(b,i)\in\mathcal P}
\left[
y_{b,i}\,\operatorname{softplus}(-d_{b,i})
+(1-y_{b,i})\,\operatorname{softplus}(d_{b,i})
\right].
\]

### Auxiliary original-peak reconstruction

Only generated peaks \(F\) contribute to the auxiliary reconstruction heads:

\[
\begin{aligned}
\mathcal L_D^{m}
&=
\frac{1}{\max(|\mathcal F|,1)}
\sum_{(b,i)\in\mathcal F}
\operatorname{CE}(h_{b,i}^m,c_{b,i}^m),\\
\mathcal L_D^{a}
&=
\frac{1}{\max(|\mathcal F|,1)}
\sum_{(b,i)\in\mathcal F}
\operatorname{CE}(h_{b,i}^a,c_{b,i}^a),\\
\mathcal L_D^{\mathrm{rec}}
&=\mathcal L_D^m+\mathcal L_D^a.
\end{aligned}
\]

The discriminator objective is

\[
\mathcal L_D
=
\mathcal L_D^{\mathrm{det}}
+\mathcal L_D^{\mathrm{rec}},
\]

because all three configured weights are one. The mixed batch is detached
before this backward pass, so \(\mathcal L_D\) cannot update the generator.

The configured data path guarantees useful masked spectra in practice, but
the implementation assumes the pooled \(\mathcal T\), \(\mathcal P\), and
\(\mathcal F\) sets are nonempty. Generator reconstruction, discriminator
detection, and generator adversarial denominators are not clamped; only the
auxiliary discriminator reconstruction denominator is clamped as shown.

## Generator adversarial objective

For the generator pass, discriminator parameters are frozen but the
mixed-batch graph remains connected to generated peaks. The generator tries
to make only its generated masked peaks look real:

\[
\mathcal A_G
=
\frac{1}{|\mathcal F|}
\sum_{(b,i)\in\mathcal F}
\operatorname{softplus}(d_{b,i}).
\]

This is binary cross-entropy with target \(0\) (real). It does not act on
untouched real peaks, context peaks, or a whole-spectrum score. The
discriminator reconstruction heads are not part of \(\mathcal A_G\).

The scheduled scalar objective reported in metrics is

\[
\mathcal L_G^{\mathrm{reported}}(s)
=
\mathcal R_G+w(s)\mathcal A_G.
\]

The gradients are combined with the bounded procedure below rather than by
backpropagating this sum directly.

## Adversarial schedule

Let \(s\) be the optimizer step before the update. The adversarial weight is

\[
w(s)
=
0.25\,
\operatorname{clip}
\left(
\frac{s-2000}{10000},
0,
1
\right).
\]

Therefore the generator receives reconstruction-only gradients through step
2,000, a linear adversarial ramp for 10,000 steps, and full weight \(0.25\)
from step 12,000 onward.

This schedule applies only to adversarial pressure on the generator. The
discriminator trains on its full objective from the first step.

Training metrics are logged after incrementing the global step. A row labeled
\(S\) was computed with \(s=S-1\); for example, row 2,025 reports
\(w(2024)=0.0006\). Validation uses its displayed current step directly.

## Gradient accumulation and adversarial cap

Let \(K=16\) be the number of microbatches. The trainer accumulates the two
generator branches separately:

\[
g_R
=
\frac{1}{K}\sum_{j=1}^{K}
\nabla_\theta\mathcal R_G^{(j)},
\qquad
g_A
=
\frac{1}{K}\sum_{j=1}^{K}
\nabla_\theta\!\left[w(s)\mathcal A_G^{(j)}\right].
\]

Reconstruction uses ordinary backward accumulation in parameter `.grad`.
The weighted adversarial vector-Jacobian products use `torch.autograd.grad`
and a separate step-local buffer. Parameters not used by the adversarial
branch contribute no vector. Each \(\mathcal R_G^{(j)}\) and
\(\mathcal A_G^{(j)}\) is already normalized over the pooled peaks of
microbatch \(j\) before the 16 microbatch losses are averaged.

When \(w(s)=0\), the adversarial forward still runs for telemetry, but the
trainer skips `torch.autograd.grad` and merges no adversarial vector.

First clip the reconstruction branch to norm \(c=1\):

\[
\bar g_R
=
g_R\min\left(1,\frac{c}{\lVert g_R\rVert_2+\epsilon}\right).
\]

Then cap the adversarial branch to a fraction \(\rho=0.1\) of the clipped
reconstruction norm:

\[
\alpha
=
\min\left(
1,
\frac{\rho\lVert\bar g_R\rVert_2}
{\lVert g_A\rVert_2+\epsilon}
\right),
\qquad
\rho=0.1,
\qquad
\epsilon=10^{-6}.
\]

Merge the branches:

\[
g_G=\bar g_R+\alpha g_A.
\]

Finally apply the same global norm clip:

\[
\widehat g_G
=
g_G\min\left(1,\frac{c}{\lVert g_G\rVert_2+\epsilon}\right).
\]

Before the final common scaling,

\[
\frac{\lVert\alpha g_A\rVert_2}
{\lVert\bar g_R\rVert_2}
\le 0.1.
\]

The final clip scales the sum uniformly, so it does not change that branch
ratio. The adversarial direction may agree or conflict with reconstruction,
but it cannot dominate it in norm. No PCGrad projection is used.

The discriminator gradient is independently accumulated over the same 16
microbatches and clipped to norm one.

## Optimizers and learning-rate schedules

Both models use independent fused Adam optimizers:

\[
\beta_1=0.9,\qquad\beta_2=0.95,\qquad\epsilon_{\mathrm{Adam}}=10^{-8},
\]

with zero weight decay. The discriminator uses

\[
\eta_{D,\max}=10^{-4},\quad
\eta_{D,\min}=10^{-5},\quad
W_D=1000,
\]

and the generator uses

\[
\eta_{G,\max}=3\times10^{-4},\quad
\eta_{G,\min}=3\times10^{-5},\quad
W_G=2000.
\]

For warmup length \(W\), total length \(S=50000\), and
\(\delta=10^{-8}\), the schedule is

\[
\eta(s)=
\begin{cases}
\eta_{\max}
\left[
\delta+(1-\delta)\dfrac{s}{W}
\right],
&s<W,\\[6pt]
\eta_{\min}
+(\eta_{\max}-\eta_{\min})
\dfrac{1+\cos(\pi r)}{2},
&s\ge W,
\end{cases}
\]

where

\[
r=
\operatorname{clip}
\left(
\frac{s-W}{S-W},
0,
1
\right).
\]

Each optimizer and scheduler has independent state.

Update \(s\) uses \(\eta(s)\). The scheduler advances after the optimizer and
before logging, so a metric row labeled \(S\) reports \(\eta(S)\) even though
the other row metrics came from update \(s=S-1\), which used \(\eta(S-1)\).
The final update uses \(\eta(49{,}999)\); the exact minimum is reached in the
saved scheduler state at 50,000.

## One optimizer step

For each of 16 microbatches:

1. Load 80 spectra and randomly shuffle aligned peak fields.
2. Force each base peak into context.
3. Run the generator on `cuda:1` and reconstruct all masked targets.
4. Randomly choose balanced fake/real target pairs and build the mixed
   spectrum.
5. Copy the connected mixed batch to `cuda:0`; detach a second view for the
   discriminator update.
6. Backpropagate \(\mathcal L_D/16\) through the discriminator only.
7. Freeze discriminator parameters, compute
   \(\nabla[w(s)\mathcal A_G/16]\) when \(w(s)>0\), and store it outside
   `.grad`.
8. Backpropagate \(\mathcal R_G/16\) into generator `.grad`.

After all 16 microbatches:

1. Clip the discriminator gradient.
2. Clip the generator reconstruction gradient.
3. Cap and merge the accumulated adversarial gradient.
4. Clip the combined generator gradient.
5. Step both optimizers and both schedulers.
6. Synchronize both CUDA devices before timing and logging the completed
   optimizer step.

## Why shortcut discrimination is harder

The construction removes several cheap signals:

1. **No spectrum label.** Every discriminator input contains both real and
   generated masked peaks.
2. **Exact within-spectrum class balance.** Each spectrum has \(q\) fake and
   \(q\) real supervised peaks, so the class prior and peak count do not
   identify a label.
3. **Matched positions and context.** Fake and real examples come from the
   same target mask and the same source spectrum.
4. **Loss masking.** Context is visible for consistency judgments but is
   excluded from detection BCE.
5. **Random peak order.** The original intensity ordering is reshuffled every
   microbatch, so array position cannot encode the class.
6. **Random pair assignment.** The fake half of the target set is redrawn for
   every microbatch.
7. **Base-peak anchoring.** The most obvious structural peak is always
   context, never a generated-label shortcut.
8. **Joint peak generation.** Both m/z and intensity are replaced together.
9. **Stochastic outputs.** Hard Gumbel categories and continuous stochastic
   residuals prevent a fixed deterministic replacement signature.
10. **Matched constructed size.** Generator and discriminator total parameter
    counts differ by less than \(0.2\%\), with the gradient-connected caveat
    documented in the run contract.
11. **Auxiliary discriminator reconstruction.** The discriminator must model
    how a generated peak relates to the original peak, not only emit a binary
    score.

This does not make every possible shortcut mathematically impossible. It
removes the known global, ordering, count, class-prior, and unmatched-context
shortcuts while leaving the discriminator free to detect genuine
distributional defects in generated peaks.

## Validation, telemetry, and checkpoints

- Training metrics are logged every 25 optimizer steps.
- Validation runs once before fresh or resumed training begins.
- Validation runs every 500 steps for 64 microbatches.
- Validation forks CPU and both CUDA RNG states, so stochastic validation
  does not perturb the training stream.
- A local `checkpoints/last.pt` is atomically replaced every 1,000 steps and
  at the final step. Remote storage uploads do not provide the same atomic
  replacement guarantee.
- Checkpoint format 5 stores both models, both optimizers, both schedulers,
  global step, CPU and per-GPU RNG states, preprocessing, and data provenance.
- Resume requires an exact stored contract match for model settings,
  objectives, the listed masking and optimization fields, selected data
  stream fields, and seed. Preprocessing and data provenance are validated
  separately.

The checkpoint restores model, optimizer, scheduler, CPU RNG, and both CUDA
RNG states. It does not store DataLoader-worker RNG or iterator state, so a
resume is state- and contract-consistent but is not guaranteed bit-for-bit
identical. Devices, software versions, several loader settings, and
`optimizer_state_dtype` are also outside the strict training contract.

Execution hardcodes CUDA bfloat16 autocast, keeps parameters and returned
logits in float32, and uses no GradScaler. Adam state is float32 because the
parameters are float32; the `optimizer_state_dtype` config field is not read
by this PyTorch optimizer path.

The primary stability telemetry is

- reconstruction, categorical, and residual losses;
- m/z and intensity MAE and exact-bin accuracy;
- generated-versus-target means and standard deviations;
- discriminator balanced accuracy, fake recall, and real specificity;
- raw reconstruction and weighted-adversarial gradient norms;
- adversarial scale and realized gradient ratio;
- combined generator and discriminator gradient norms; and
- step time, throughput, and data-wait fraction.

`reconstruction_grad_norm` is the reconstruction norm before its first clip,
`adversarial_grad_norm` is the raw weighted adversarial norm,
`adversarial_grad_scale` is \(\alpha\), `adversarial_grad_ratio` is the
realized capped ratio, and the generator `grad_norm` is the merged norm before
the final clip.

## Entry point

The canonical invocation is

```bash
.venv/bin/python train.py \
  --config configs/adversarial_fake_peaks_equal_50m.py \
  --workdir experiments/<run-name> \
  --overrides-json '{"artifact_dir":"data/massive_v1_ms2_100m_stratified_x16"}'
```

The focused executable specification is in
`tests/test_adversarial_fake_peaks.py` and `tests/test_fake_peaks.py`.
