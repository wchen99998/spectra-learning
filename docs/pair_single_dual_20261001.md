# Pair→single information flow with pair distogram supervision

Research date: 2026-10-01. Status: waves 1–12 are complete (wave 9's R=32 run was
cancelled once R ≤ 12 settled the question). The best recipe is now the library
`training_mode="dual"` (Section 18).

**Answer (seed 0):** add a pair distogram loss on the encoder pairs, and keep
three pair→single paths: the attention bias, mean pair values, and the
edge-gated message (`dual_bias_mean_gated`). At matched pair width (C=256),
this joint model beats single-only, pair-only and the old joint recipe on
every metric, at both 2 Da and 0.1 Da (Section 8, wave 5; Section 10, wave 6).
Using 0.1 Da m/z bins adds a further +0.03–0.07 MACCS AUC.

**Current best recipe**, with the joint model's single-readout AUC on test /
confirm:

| Step | Single AUC (test / confirm) | Section |
| --- | --- | --- |
| `dual_bias_mean_gated`, C=256 | — | — |
| + 0.1 Da bins | 0.731 / 0.719 | 10 |
| + metadata as a CLS "precursor node" whose edges carry each peak's neutral loss (`cls_edge`) | 0.742 / 0.727 | 11 |

`cls_edge` also cuts MAE by 0.059 and distogram loss by 0.056, raises pair
AUC to 0.763 / 0.741, and keeps retrieval. Within-bin regression heads
(Section 12) learn sub-bin position, but at 0.1 Da they change probe scores by
at most 0.005, which is within plausible seed noise. They clearly help only
with coarse 2 Da bins. Multi-scale CE (Section 14) lowers the exact-bin m/z
CE but costs probe AUC and retrieval in all three seeds, so it stays off.
Removing the per-sublayer post norms inside every block (Section 15) raises
single AUC by 0.005–0.007 and pair retrieval by 1.6–2.9 pp in all three seeds,
even though validation CE gets worse. The final output norms of the encoder
and predictors make no difference.
None of 22 alternative m/z input encodings beats the current one: 16
learnable log-spaced Fourier frequencies (Section 16). More frequencies do not
help either: 128 single-stream frequencies gain 0.003–0.005 AUC but lose
pair retrieval, and 32–128 pair frequencies lose retrieval. Learnable frequencies
and Fourier features both matter: fixed frequencies, RBF bumps, a 0.1 Da token
embedding or no m/z features all lose probe AUC and retrieval in all three
seeds.

Code: the pilot, evaluation and report scripts
(`scripts/pair_single_dual_pilot.py`, `scripts/pair_single_dual_eval.py`,
`scripts/pair_single_dual_metadata.py`, `scripts/report_pair_single_dual.py`,
`tests/test_pair_single_dual_pilot.py`, and the `pair_to_single_*` scripts of
the earlier mechanism study) are kept at commit `78eb2c0` and removed from the
tree. The winning recipe is implemented in the library (Section 18).

## 1. Question

We moved from jointly training single and pair streams to training them
independently. The pair-only reference is
[pair279m-attn-n32-b2048-v6e32-2d-20260910-rootcause](https://wandb.ai/iclac/jepa-finalrun/runs/pair279m-attn-n32-b2048-v6e32-2d-20260910-rootcause).
The reason was poor information flow from pair to single through the
attention-logit bias. Goal: find a pair→single mechanism with which a **joint**
model beats the independently trained single-only and pair-only models on all
three fronts: masked prediction, MACCS chemistry probes and spectral retrieval.

## 2. Why the bias channel is weak

Take one head with single queries/keys/values $q_i, k_j, v_j$ and pair state
$z_{ij}\in\mathbb{R}^C$. The existing interface is

$$
a_{ij} = \operatorname{softmax}_j\!\left(\frac{q_i^\top k_j}{\sqrt d} + b(z_{ij})\right),\qquad
u_i = \sum_j a_{ij} v_j,\qquad b(z_{ij}) = w_h^\top \operatorname{RMSNorm}(z_{ij}).
$$

Holding singles fixed,

$$
\frac{\partial u_i}{\partial b_{ij}} = a_{ij}\,(v_j - u_i).
$$

1. **Content is never transported.** Pairs only route attention, and the
   transmitted content is always a single value $v_j$. When neighbour values
   are similar, the pair effect vanishes, regardless of what $z_{ij}$ encodes.
2. **Per-edge compression** from $C$ pair channels to $H$ scalars, one per head.
3. **Saturated softmax** suppresses the routing gradient.

## 3. Diagnosis: the joint objective never trains the pair stream directly

In the historical joint MAE path (`_forward_mae_fastmixer_compact`), the encoder
returns $(s, z)$, but the masked predictor reads **only singles**. The final pair
tensor $z^{(L)}$ is discarded, and the joint distogram is computed from *single*
predictor latents:

$$
\mathcal L_{\text{old}} = 0.7\,\mathcal L_{\text{MAE}}\big(P_s(s^{(L)})\big)
+ 0.3\,\mathcal L_{\text{dist}}\big(|P_s(s^{(L)})_t - P_s(s^{(L)})_j|\big).
$$

So $\nabla_z \mathcal L_{\text{old}}$ flows *only* through the bias $b(z_{ij})$,
which is exactly the weak channel above. pair279m instead supervises pairs
directly, with a pair predictor $P_z$: triangle blocks on the context pair
square, using mask tokens for unknown rows:

$$
\mathcal L_{\text{pair}} = \mathcal L_{\text{dist}}\big(P_z(z^{(L)})\big),\qquad
\mathcal L_{\text{dist}} = -\frac{1}{|\mathcal E|}\sum_{(t,j)\in\mathcal E}
\log p_\theta\!\left(\left\lfloor\tfrac{|m_t-m_j|}{\Delta}\right\rfloor \,\middle|\, P_z(z)_{tj}\right),
$$

where $\mathcal E$ is the set of pairs touching a masked target $t$, and
$\Delta = 2$ Da.

The 30 Sep pilot ([`pair_to_single_mechanism_20260930.md`](pair_to_single_mechanism_20260930.md))
tested five bridges only under $\mathcal L_{\text{old}}$, and evaluated only
single readouts. It concluded that the objective does not reward preserving
pair content, which matches this diagnosis.

## 4. Proposal: dual supervision

One context encoding feeds both predictors:

$$
(s^{(L)}, z^{(L)}) = \operatorname{Enc}_\theta(\text{context}),\qquad
\mathcal L_{\text{dual}} = 0.7\,\mathcal L_{\text{MAE}}\big(P_s(s^{(L)})\big)
+ 0.3\,\mathcal L_{\text{dist}}\big(P_z(z^{(L)})\big).
$$

The loss weights match $\mathcal L_{\text{old}}$. The only change is that the
distogram comes from the pair branch, using pair279m's predictor and head.

## 5. Interfaces tested on top of $\mathcal L_{\text{dual}}$

Each bridge is a residual update after every joint block:
$s_i \leftarrow s_i + \Delta s_i$. The output passes through
$\gamma\odot\operatorname{RMSNorm}(\cdot)$, where $\gamma$ is a learned channel
scale initialised to 0.1, and the bridge rank is $r = 64$. Notation:
$\bar z_{ij} = \operatorname{RMSNorm}(z_{ij})$, $\mathcal V$ is the set of
valid tokens, and $n = |\mathcal V|$.

- **none**: no pair→single path. The single→pair outer-product update is kept.
- **bias**: the historical routing path.
- **bias + mean-first values**:
  $\Delta s_i = \gamma\odot\operatorname{RMSNorm}\big(W_oW_v\operatorname{RMSNorm}(\tfrac1n\sum_j z_{ij})\big)$.
- **bias + attention-weighted values** (the pair term of AlphaFold IPA). It
  reuses the block's own routing $a^h_{ij}$, bias included, mixed half-and-half
  with uniform weights. The pooled vector is
  $p_{ih} = \sum_j \tfrac12(a^h_{ij}+\tfrac1n)\bar z_{ij}$, and the update is
  $\Delta s_i = \gamma\odot\operatorname{RMSNorm}(W_o\operatorname{concat}_h p_{ih}V_h)$.
- **bias + edge-gated messages** (new). Each edge gates $r$ *channels* of a
  neighbour message instead of setting $H$ scalars. A mean-first direct
  pair-value term is added:

$$
m_i = \frac1n\sum_{j\in\mathcal V}\sigma(G\bar z_{ij}+g)\odot V\operatorname{RMSNorm}(s_j)
+ P\,\frac1n\sum_{j\in\mathcal V}\bar z_{ij},\qquad
\Delta s_i = \gamma\odot\operatorname{RMSNorm}(W_o m_i).
$$

  The pair Jacobian of this bridge stays non-zero when neighbour singles are
  identical; a unit test checks this.

## 6. Protocol

- **Data:** MassSpecGym 1.5 panel, 31 m/z-ordered peaks plus CLS.
  194,119 train, 19,429 val and 17,556 test spectra, with molecule-disjoint folds.
- **Model:** single width $D=256$, pair width $C=128$, 6 fastmixer blocks,
  4 heads. The single predictor has 2 cross-attention blocks and the pair
  predictor has 2 triangle blocks; compute is bf16. Parameter counts:
  single-only and joint-old 11.10M, pair-only 3.30M, dual 11.82–12.13M.
- **Masking:** context is 60% of valid peaks and target is 25% (at most 8).
  Compact capacities are 20/28/8.
- **Optimiser:** AdamW ($\beta_2=0.95$, weight decay 0.01), peak LR
  $3\times10^{-4}$, 500-step warmup, cosine decay to $3\times10^{-5}$,
  batch 128, 20k steps.
- **Matching across arms:** every arm uses the same examples and masks.
  The joint backbone's initialisation is identical across arms.
- **Arms:** `single_only`, `joint_bias`, `pair_only`, `dual_none`, `dual_bias`,
  `dual_bias_mean`, `dual_bias_ipa`, `dual_bias_gated`. Seeds 0–2.
- **Readouts:**
  - single = [peak mean, CLS]
  - pair = [peak–peak edge mean, $\operatorname{mean}_i z_{i,\text{cls}}$,
    $\operatorname{mean}_i z_{\text{cls},i}$]
  - both = the concatenation of the two
- **MACCS probe:** ridge fit on 8,192 train spectra, with the penalty chosen on
  4,096 val spectra. It is scored on the test fold and on 1,494 unused
  validation molecules.
- **Retrieval:** cosine similarity on train-standardised features.
- **Prediction:** masked m/z CE and pair-distogram CE on 4,096 fixed val
  spectra. Bridge arms also get zero and mask-matched-shuffle interventions.

## 7. Early signal (1k steps, seed 0; not conclusive)

| Arm | val MAE CE | val pair-distogram CE | MACCS test AUC single / pair / both |
| --- | ---: | ---: | --- |
| single_only | 3.221 | — | 0.597 / 0.614 / 0.610 |
| joint_bias | 3.078 | — | |
| pair_only | — | 3.164 | — / 0.599 / — |
| dual_none | 3.299 | 3.165 | |
| dual_bias | 3.247 | 3.137 | 0.613 / 0.627 / 0.627 |
| dual_bias_mean | 3.324 | 3.155 | |
| dual_bias_ipa | 3.092 | 3.104 | |
| dual_bias_gated | 3.151 | 3.099 | |

With the distogram gradient, the joint encoder's pair stream already matches or
beats pair-only on distogram CE. The *untrained* pair readout of `single_only`
(0.614) beats pair-only (0.599). This suggests the raw m/z-difference pair
features are already informative, so that arm serves as a random-feature
control for pair readouts.

## 8. Results (20k steps, 3 seeds)

**Wave 1 answer:** giving the pair stream the distogram gradient, plus a direct
pair-value bridge, improves almost every metric. The best interface,
**bias + attention-weighted pair values** (`dual_bias_ipa`), beats the old joint
recipe in every seed on masked prediction, single chemistry and retrieval. It
also matches or beats pair-only on pair-readout chemistry. The one exception is
a 0.5% higher pair-distogram loss than pair-only.

### Per-arm means (± SD over 3 seeds)

Abbreviations: MAE = masked m/z CE (lower is better); dist = pair-distogram CE
from the pair predictor (lower is better). Old-objective arms have no pair
predictor, so they have no dist value. AUC is the macro MACCS ridge AUC.
Top-1 is spectral retrieval on train-standardised features. "confirm" means
the 1,494 previously unused validation molecules.

| Arm | MAE ↓ | dist ↓ | single AUC test / confirm | pair AUC test / confirm | single top-1 test / confirm | pair top-1 test |
| --- | ---: | ---: | --- | --- | --- | ---: |
| single_only | 2.620 | — | 0.6455 / 0.6360 | 0.6411 / 0.6291 † | 21.47% / 31.99% | 21.42% † |
| joint_bias (old) | 2.536 | — | 0.6576 / 0.6464 | 0.6837 / 0.6715 | 23.75% / 35.53% | 12.33% |
| pair_only | — | **2.272** | — | 0.7020 / 0.6889 | — | 12.73% |
| dual_none | 2.621 | 2.273 | 0.6373 / 0.6300 | 0.6983 / 0.6905 | 22.01% / 32.52% | 16.63% |
| dual_bias | 2.532 | 2.292 | 0.6679 / 0.6437 | 0.7067 / 0.6919 | 26.24% / 38.29% | 14.98% |
| dual_bias_mean | 2.527 | 2.297 | **0.6799** / 0.6643 | **0.7079 / 0.6945** | 25.55% / 37.44% | 15.55% |
| **dual_bias_ipa** | **2.518** | 2.284 | 0.6782 / 0.6641 | 0.7066 / 0.6917 | **27.36% / 41.23%** | 15.68% |
| dual_bias_gated | 2.519 | 2.286 | 0.6752 / **0.6645** | 0.7066 / 0.6922 | 25.67% / 38.60% | 15.41% |

† The single-only arm's pair stream receives no gradient. Its pair readout is
therefore a random-feature control on top of trained singles.

### Seed-matched paired differences for `dual_bias_ipa`

Each entry is the mean, with the [min, max] across seeds in brackets. All three
seeds share examples, masks and initialisation with the comparison arm.

| Versus | MAE | dist | single AUC test | single AUC confirm | top-1 test | top-1 confirm | pair AUC test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| single_only | −0.102 [−0.114, −0.091] | — | +0.033 [+0.026, +0.036] | +0.028 [+0.015, +0.036] | +5.9 pp [+5.8, +6.0] | +9.2 pp [+8.1, +10.2] | — |
| joint_bias | −0.018 [−0.021, −0.016] | — | +0.021 [+0.018, +0.022] | +0.018 [+0.011, +0.026] | +3.6 pp [+2.7, +4.1] | +5.7 pp [+5.1, +6.4] | +0.023 [+0.018, +0.028] |
| pair_only | — | +0.012 [+0.007, +0.015] | — | — | — | — | +0.005 [+0.002, +0.009] (confirm +0.003 [−0.001, +0.008]) |
| dual_bias | −0.014 [−0.025, −0.005] | −0.008 [−0.014, −0.001] | +0.010 [+0.006, +0.017] | +0.020 [+0.015, +0.024] | +1.1 pp [+0.6, +1.8] | — | — |

### What the arms show

1. **The objective was the bottleneck.** Look at `dual_bias` versus
   `joint_bias`: the architecture and loss weights are identical, and only the
   source of the distogram gradient changes. The pair-readout AUC rises by
   +0.023 and retrieval by +2.5 pp, while MAE is unchanged.

   The joint model's pair stream no longer lags the pair-only model. It now
   matches it on distogram CE: 2.273 for `dual_none` versus 2.272.

2. **Direct pair values now matter, unlike on 30 Sep.** Shuffling bridge
   content between spectra (mask-matched, every block) raises MAE CE by
   **+0.056 to +0.062**. Zeroing the bridge raises it by about +0.051. Under
   the old objective the shuffle effect was ≤ 0.015, and only 0.001–0.003
   when the bridge was combined with bias. The singles now depend on
   spectrum-specific pair content.

3. **Bias alone is insufficient even with a supervised pair stream.** Adding
   any direct-value bridge to `dual_bias` raises the confirm-set single AUC by
   **+0.020** in every seed, for all three bridges. The learning curves also
   diverge. The single val AUC of `dual_bias` plateaus: 0.671 at 15k and 0.672
   at 20k. The bridged arms keep improving, from 0.681–0.684 at 15k to
   0.685–0.689 at 20k. The 30 Sep gain shrank with training; this one does not.

4. **Interfaces compared.**
   - IPA-style values (routing shared with the attention) give the best
     prediction and retrieval.
   - Mean-first values give the best pair/single AUC, but the worst distogram
     loss.
   - Edge-gated messages sit between the two, at the highest cost.

   Throughput (steps/s, 3 runs per GH200): bias 11.9, mean 10.8, IPA 10.4,
   gated 9.9.

5. **Remaining gaps.** The single readout (0.678) is still below the pair
   readout (0.707), so not all pair chemistry reaches singles. The pair
   distogram is 0.5% worse than pair-only once singles read from pairs: the
   pair stream gives up a little distance-prediction fit to serve singles.
   Pair-readout cosine retrieval stays poor (≈15%). Single readouts are the
   right retrieval embedding.

6. **Single-only stream.** `dual_none` (pairs supervised, no feedback) leaves
   singles at single-only quality (MAE 2.621 versus 2.620). The pair→single
   path is what carries the gain into the single stream.

### Wave 2 (seed 0 only; one run per GH200)

Abbreviations used in the tables below:

- sAUC / pAUC: MACCS AUC from the single / pair readout (t = test fold, c = unused molecules)
- s@1: retrieval top-1 from the single readout
- sps: training steps per second

The single-seed wave-1 spread is about ±0.005 AUC and ±1 pp top-1.

**New interfaces (20k steps, D256/C128).** Wave-1 `dual_bias_ipa` seed 0 is
shown for reference.

| Arm | MAE | dist | sAUC t / c | pAUC t / c | s@1 t / c |
| --- | ---: | ---: | --- | --- | --- |
| dual_bias_ipa (wave 1, s0) | 2.523 | 2.294 | 0.677 / 0.660 | 0.706 / 0.690 | 28.1 / 42.3 |
| dual_gated (no bias) | 2.525 | **2.271** | **0.696 / 0.685** | 0.709 / 0.699 | 23.0 / 34.0 |
| dual_bias_mean_gated | 2.530 | 2.286 | 0.682 / 0.671 | **0.714** / 0.695 | 27.2 / 40.0 |
| dualplus_bias | 2.519 | 2.294 | 0.666 / 0.651 | 0.704 / 0.691 | 24.5 / 36.3 |
| dualplus_bias_gated | 2.518 | 2.287 | 0.672 / 0.664 | 0.703 / 0.697 | 25.4 / 37.7 |

- **Dropping the bias** (`dual_gated`, routing only through edge-gated
  channels) gives the best single-readout chemistry of any 20k run, and the
  best dual distogram. It loses about 5 pp of retrieval. Bias appears to
  shape the single cosine geometry; direct values carry the chemistry.
- **Keeping the single-latent distogram** (`dualplus`) adds nothing.

**Longer training (40k steps).** The pilot data is only about 180k spectra,
so the 40k schedule is about 28 epochs. Val MAE flattens after 30k: the 40k
endpoints sit at 2.56–2.68, worse than the 20k endpoints of 2.52–2.62. The
frozen downstream metrics still improve.

| Arm, 40k | MAE | dist | sAUC t / c | pAUC t / c | s@1 t / c |
| --- | ---: | ---: | --- | --- | --- |
| single_only | 2.654 | — | 0.657 / 0.651 | — | 23.7 / 33.9 |
| joint_bias | 2.564 | — | 0.673 / 0.661 | 0.707 / 0.697 | 27.3 / 39.7 |
| pair_only | — | 2.234 | — | 0.711 / **0.700** | — |
| dual_bias | 2.566 | 2.245 | 0.684 / 0.670 | 0.713 / 0.701 | 29.5 / 42.5 |
| dual_bias_mean | 2.569 | 2.244 | **0.695** / 0.682 | **0.724** / 0.701 | 28.2 / 43.5 |
| **dual_bias_ipa** | 2.571 | 2.235 | 0.694 / **0.684** | 0.718 / 0.698 | **30.3 / 45.3** |
| dual_bias_gated | **2.560** | 2.235 | 0.693 / 0.683 | 0.717 / 0.705 | 27.9 / 41.9 |

At 40k the distogram gap to pair-only closes: 2.235 for IPA versus 2.234.
The chemistry and retrieval gains over the old joint recipe persist:
+0.021 single AUC and +2.9 pp top-1 for IPA. Its MAE is tied with the old
joint recipe (+0.007), unlike the −0.018 seen at 20k.

**Scale (D384/C192/L8, 20k).** The bridges keep beating the old joint recipe.

| Arm, scaled | MAE | dist | sAUC t / c | pAUC t / c | s@1 t / c | params |
| --- | ---: | ---: | --- | --- | --- | ---: |
| single_only | 2.604 | — | 0.639 / 0.632 | — | 21.2 / 31.0 | 28.4M |
| joint_bias | 2.511 | — | 0.659 / 0.649 | 0.675 / 0.669 | 23.7 / 35.6 | 28.4M |
| pair_only (C192, L8) | — | **2.240** | — | 0.702 / 0.693 | — | 7.6M |
| dual_bias | 2.497 | 2.258 | 0.660 / 0.647 | 0.712 / 0.692 | 25.1 / 38.8 | 29.8M |
| dual_bias_mean | 2.496 | 2.262 | 0.676 / 0.668 | 0.712 / **0.701** | **26.4 / 40.8** | 30.3M |
| **dual_bias_ipa** | **2.493** | 2.257 | **0.686 / 0.670** | **0.715** / 0.695 | 25.9 / 38.8 | 30.3M |
| dual_bias_gated | 2.497 | 2.258 | 0.676 / 0.664 | 0.710 / 0.691 | 25.8 / 39.4 | 30.7M |

The scaled models are not better than the base models at 20k. This scale
probably needs longer training, or more data than the MSG train fold.

**Pair-only capacity controls (20k).** This is the important caveat.

| Pair-only | dist | pAUC t / c | params | sps |
| --- | ---: | --- | ---: | ---: |
| C128, L6 (wave-1 s0) | 2.282 | 0.697 / 0.689 | 3.3M | — |
| C128, L12 | 2.280 | 0.693 / 0.682 | 5.7M | 17.5 |
| C128, L18 | 2.275 | 0.695 / 0.679 | 8.1M | 12.4 |
| C192, L12 | 2.257 | 0.711 / 0.693 | 10.6M | 16.8 |
| **C256, L6** | **2.225** | **0.712 / 0.703** | 11.4M | 21.5 |

Depth does not help pair-only, but width does. Pair-only at C=256 matches the
joint models' parameter count, and trains faster than dual at C=128
(21.5 versus about 13 sps). On distogram (2.225 versus 2.27–2.29) and on
pair-readout chemistry, it equals or beats every 20k joint run at C=128.

**So the joint advantage over pair-only is not established at matched pair
width.** What the joint model clearly adds is the single readout: about
27–30% retrieval, versus 13% for pair-readout retrieval. It also adds a
cheaper O(N) embedding.

### Wave 5: joint models at matched pair width C=256 (2 Da, seed 0, 20k)

| Arm, C=256 | MAE | dist | sAUC t / c | pAUC t / c | s@1 t / c | params |
| --- | ---: | ---: | --- | --- | --- | ---: |
| single_only (D256, wave-1 s0) | 2.614 | — | 0.651 / 0.645 | — | 22.1 / 32.8 | 11.1M |
| pair_only | — | 2.225 | — | 0.712 / 0.703 | — | 11.4M |
| joint_bias | 2.540 | — | 0.664 / 0.656 | 0.696 / 0.684 | 24.7 / 35.4 | 18.5M |
| dual_none | 2.607 | **2.206** | 0.641 / 0.631 | 0.714 / 0.701 | 22.2 / 33.3 | 21.3M |
| dual_bias | 2.522 | 2.223 | 0.669 / 0.658 | 0.719 / 0.706 | 25.7 / 37.0 | 21.3M |
| dual_bias_ipa | **2.503** | 2.222 | 0.686 / 0.672 | 0.718 / 0.700 | 25.5 / 37.9 | 21.5M |
| dual_gated | 2.508 | 2.223 | **0.700 / 0.692** | 0.716 / 0.703 | 24.4 / 37.9 | 21.7M |
| **dual_bias_mean_gated** | 2.509 | 2.216 | 0.697 / 0.682 | **0.726 / 0.707** | **26.7 / 38.6** | 21.9M |

At matched pair width, **`dual_bias_mean_gated` beats both independently
trained models on every metric**:

- versus pair-only C=256: distogram −0.009, pair AUC +0.014 (t) and +0.003 (c);
- versus single-only: MAE −0.105, single AUC +0.046 / +0.037, top-1
  +4.6 / +5.8 pp;
- versus the old joint recipe at C=256: MAE −0.031, single AUC +0.033 / +0.026,
  pair AUC +0.030 / +0.023, top-1 +2.0 / +3.2 pp.

All dual arms at C=256 reach or beat pair-only on distogram (2.206–2.223
versus 2.225). With wider pairs the dual loss no longer costs any distogram
accuracy. The C=128 gap came from the pair stream sharing capacity, not from
the joint objective. The pair-AUC (confirm) and distogram margins over
pair-only are small (0.003, 0.009), and these are single-seed results.

Per-run results: `artifacts/pair_single_dual_20261001/wave1/summary.csv`, with
an `evaluation.json` file in each run directory.

## 9. Masking benchmark: Thompson-span + isotope co-masking

This is adapted from the InstaNovo-FM pretraining mask. Peaks are m/z-sorted.
Each peak gets a Thompson score from its max-normalised intensity
$\tilde I_i$:

$$
s_i \sim \operatorname{Beta}\big(1 + \kappa\tilde I_i,\; 1 + \kappa(1-\tilde I_i)\big),\qquad \kappa = 4 .
$$

The algorithm:

1. Visit anchors in order of descending $s_i$. Intense peaks are preferred,
   but the order is stochastic.
2. Each anchor $a$ adds the contiguous span $[a, a+L)$, with
   $L\sim\mathcal U\{3,4\}$. The span is trimmed to the remaining budget and
   shifted left at the spectrum's end.
3. Add the span's isotope closure, iterated twice to cover M+1/M+2 chains:

$$
\mathcal M \leftarrow \mathcal M \cup \{\, j : \exists\, i\in\mathcal M,\; k\in\{1,2\},\;
\big|\,|m_j - m_i| - k\cdot 1.003355\,\big| \le 0.002\ \text{Da} \,\}.
$$

4. An anchor is skipped if its closed group would exceed 8 targets, or would
   leave no context peak.

The target budget equals the random scheme's: $\min(\max(\lfloor 0.25n\rfloor, 1), 8)$.
Context is a uniform 60% sample of the non-target peaks.

**Tolerance.** The window is ±0.002 Da because the ¹³C spacing (1.00336 Da)
lies only 0.0045 Da from the H spacing (1.00783 Da). In MassSpecGym peak sets
the H spacing is about twice as frequent. A ±0.005 Da window would conflate
the two, and ±0.01 Da would also catch 2H at M+2.

**No m/z noise.** InstaNovo also perturbs masked m/z with
$\mathcal N(m, 10^2)$. That step is not needed here: masked targets are
latent mask tokens that never see their own m/z.

### Mask statistics (40 batches of 128 training spectra)

| | Random | Thompson-span + isotope |
| --- | ---: | ---: |
| Target fraction of valid peaks | 22.5% | 23.0% |
| Context fraction | 57.7% | 57.8% |
| Target isotope partner visible in context | 63.5% | **0%** |
| Immediate m/z neighbour of a target visible | 83.7% | **42.6%** |
| Mean target intensity | 0.139 | 0.308 |

**Runs:** all 8 core arms at seed 0, 20k steps, `--masking ts_iso`. They are
compared with the wave-1 seed-0 runs that used random masks.

All arms report `validation` with the shared random val masks, so prediction
losses stay comparable across arms. They also report `validation_ts` with
Thompson-span val masks.

### Results (seed 0, 20k steps; deltas versus random-mask seed 0)

| Arm | sAUC t / c | pAUC t / c | s@1 t / c | p@1 t |
| --- | --- | --- | --- | --- |
| single_only | 0.629 / 0.619 (−0.021 / −0.027) | — | 18.8 / 27.8 (−3.3 / −5.0) | — |
| joint_bias | 0.652 / 0.635 (−0.007 / −0.014) | 0.671 / 0.656 (−0.017 / −0.019) | 24.4 / 36.6 (+0.2 / 0.0) | 14.2 (+2.1) |
| pair_only | — | 0.685 / 0.674 (−0.012 / −0.015) | — | 13.5 (+0.9) |
| dual_none | 0.631 / 0.611 (−0.010 / −0.021) | 0.680 / 0.667 (−0.023 / −0.027) | 18.6 / 27.6 (−4.2 / −6.6) | 19.7 (+3.5) |
| dual_bias | 0.648 / 0.630 (−0.023 / −0.015) | 0.683 / 0.665 (−0.029 / −0.030) | 26.5 / 39.9 (−1.0 / +1.0) | 17.7 (+2.2) |
| dual_bias_mean | 0.666 / 0.642 (−0.011 / −0.022) | 0.685 / 0.670 (−0.025 / −0.027) | 27.0 / 41.2 (+0.1 / +2.5) | 17.9 (+2.2) |
| dual_bias_ipa | 0.659 / 0.637 (−0.018 / −0.023) | 0.685 / 0.673 (−0.021 / −0.018) | 27.1 / 41.4 (−1.0 / −0.9) | 18.0 (+2.3) |
| dual_bias_gated | 0.653 / 0.639 (−0.023 / −0.027) | 0.678 / 0.663 (−0.030 / −0.027) | **29.0 / 46.3 (+2.6 / +6.0)** | 19.2 (+3.7) |

**Verdict for this pilot: span + isotope masking hurts chemistry.**

- MACCS AUC drops by 0.01–0.03 for every arm and both readouts.
- Pair-readout cosine retrieval improves by 1–4 pp.
- Single-readout retrieval changes inconsistently across arms.
- On the random-mask val set, MAE plateaus by 10k at about 2.93 (versus
  2.52), and even its own task stays harder (Thompson-span val MAE about 3.1).
- The ranking of arms is unchanged: dual with bridges > old joint > single-only.

A plausible cause is that intensity-preferring anchors bias targets toward
the base-peak region. MACCS-relevant structure may sit in weaker fragments.
The isotope co-masking itself touches only about 8% of peaks. Not adopted.

## 10. m/z bin-size benchmark: 0.1 Da and 0.2 Da

Both the masked m/z head and the distogram head classify into bins of
$\Delta$ Da:

$$
y^{\text{MAE}}_t = \lfloor m_t/\Delta \rfloor,\qquad
y^{\text{dist}}_{tj} = \min\!\big(\lfloor |m_t-m_j|/\Delta\rfloor,\; K-1\big),\qquad
K = \lceil 1000/\Delta \rceil .
$$

All earlier runs used $\Delta = 2$ Da ($K = 500$). The two new settings,
$\Delta = 0.2$ and $0.1$ Da, give $K$ = 5,000 and 10,000. At 0.1 Da this adds
about 2.6M parameters to the m/z head ($D\times K$). The distogram head grows
by about 1.3M parameters on the pair branch ($C\times K$), or 2.6M on the
single-latent branch.

2 Da bins only test nominal mass to within ±1 Da. 0.1–0.2 Da bins also
resolve the **mass defect**, the fractional part of m/z. The mass defect
carries elemental-composition information: halogens and S/P shift it
negative, and H-rich fragments shift it positive. The bins remain too coarse
to separate near-isobaric neutral losses: C₂H₄ (28.0313 Da) and CO
(27.9949 Da) differ by only 0.036 Da. The cross-entropy values are not
comparable across bin sizes,
because $K$ differs. The comparison therefore relies on the frozen downstream
probes and retrieval, and on rankings between arms at a fixed $\Delta$.

**Runs:** all 8 core arms at seed 0, 20k steps, with `--mz-bin-size 0.1` and
`0.2`. They are compared with the wave-1 seed-0 runs at 2 Da.

### Results (seed 0, 20k steps, D256/C128)

CE values are not comparable across $\Delta$, but they are comparable across
arms at the same $\Delta$.

| Arm | Δ | MAE CE | dist CE | sAUC t / c | pAUC t / c | s@1 t / c |
| --- | --- | ---: | ---: | --- | --- | --- |
| single_only | 2 | 2.614 | — | 0.651 / 0.645 | — | 22.1 / 32.8 |
| | 0.2 | 3.812 | — | 0.692 / 0.681 | — | 24.1 / 35.3 |
| | 0.1 | 4.295 | — | 0.691 / 0.681 | — | 23.7 / 35.5 |
| joint_bias | 2 | 2.540 | — | 0.659 / 0.650 | 0.688 / 0.675 | 24.2 / 36.6 |
| | 0.2 | 3.755 | — | 0.712 / 0.694 | 0.720 / 0.698 | 26.5 / 38.6 |
| | 0.1 | 4.236 | — | 0.729 / 0.709 | 0.728 / 0.707 | 26.5 / 37.9 |
| pair_only | 2 | — | 2.282 | — | 0.697 / 0.689 | — |
| | 0.2 | — | 3.267 | — | 0.726 / 0.711 | — |
| | 0.1 | — | **3.488** | — | 0.722 / 0.707 | — |
| dual_none | 0.1 | 4.290 | 3.467 | 0.691 / 0.679 | 0.735 / 0.721 | 22.9 / 33.0 |
| dual_bias | 0.1 | 4.193 | 3.536 | 0.720 / 0.708 | 0.737 / 0.723 | 27.9 / 42.0 |
| dual_bias_mean | 0.2 | 3.712 | 3.264 | 0.721 / 0.707 | 0.740 / 0.717 | 27.0 / 40.1 |
| | 0.1 | 4.173 | 3.494 | **0.731 / 0.712** | **0.741** / 0.718 | 27.7 / 42.4 |
| dual_bias_ipa | 0.2 | 3.701 | 3.279 | 0.723 / 0.708 | 0.737 / 0.719 | 26.8 / 39.5 |
| | 0.1 | **4.160** | 3.501 | 0.728 / 0.711 | 0.739 / **0.725** | 26.7 / 40.0 |
| dual_bias_gated | 0.2 | 3.699 | 3.274 | 0.712 / 0.700 | 0.736 / 0.718 | 25.5 / 37.0 |
| | 0.1 | 4.159 | 3.506 | 0.726 / 0.710 | 0.740 / 0.722 | **28.2 / 43.0** |

**Bin size is the largest lever found in this study.** Going from 2 Da to
0.1–0.2 Da raises MACCS AUC for every arm and both readouts:
+0.04 for single-only, +0.05–0.07 for the old joint recipe's single readout,
+0.03 for pair-only, and +0.03–0.05 for the dual arms. Retrieval changes less,
from −1 to +2.6 pp. Resolving the mass defect is evidently chemically
informative.

At 0.1 Da the joint-versus-independent picture changes:

- **The old joint recipe's single readout catches up with dual:** 0.729 versus
  0.726–0.731. Its single-latent distogram now carries 10,000-bin targets.
- **Dual keeps the edge on MAE:** −0.076 CE for IPA versus the old joint recipe.
- **Pair-readout chemistry beats pair-only** by +0.017 (t) and +0.018 (c) for
  IPA. Dual distogram CE sits within 0.006–0.048 of pair-only (3.488): IPA is
  3.501 and the edge-gated arm 3.506, and `dual_none` is better (3.467).
- **Retrieval:** dual is +1.4 to +5 pp better than the old joint recipe for
  the edge-gated arm (28.2 / 43.0 versus 26.5 / 37.9). It is tied for IPA.

0.1 and 0.2 Da are close for single-only and pair-only. For the joint arms,
0.1 Da is better by about 0.005–0.017 AUC.

### Wave 6: matched pair width C=256 at 0.1 Da (seed 0, 20k)

| Arm, C=256, Δ=0.1 | MAE | dist | sAUC t / c | pAUC t / c | s@1 t / c | params |
| --- | ---: | ---: | --- | --- | --- | ---: |
| single_only | 4.295 | — | 0.691 / 0.681 | — | 23.7 / 35.5 | 16.0M |
| pair_only | — | 3.3645 | — | 0.737 / 0.721 | — | 13.8M |
| joint_bias | 4.212 | — | 0.723 / 0.707 | 0.727 / 0.710 | 26.6 / 38.0 | 23.4M |
| dual_bias | 4.150 | 3.3596 | 0.721 / 0.705 | 0.752 / 0.732 | 26.5 / 39.1 | 26.2M |
| dual_bias_mean | 4.139 | **3.3553** | **0.735** / 0.718 | 0.752 / 0.734 | 25.9 / 38.7 | 26.4M |
| dual_bias_ipa | **4.125** | 3.3642 | 0.732 / 0.720 | 0.750 / 0.730 | 26.1 / 38.6 | 26.4M |
| dual_gated | 4.152 | 3.3635 | **0.735 / 0.721** | **0.752 / 0.736** | 26.2 / 39.7 | 26.5M |
| **dual_bias_mean_gated** | 4.157 | 3.3569 | 0.731 / 0.719 | 0.748 / 0.732 | **26.8 / 40.2** | 26.7M |

At 0.1 Da with matched pair width, **every dual arm beats both independent
models on MAE, single AUC, pair AUC and retrieval, and matches or beats
pair-only on distogram.**

- Pair AUC gains over pair-only grow to +0.011 to +0.015 (t) and +0.009 to
  +0.015 (c).
- Single AUC gains over single-only are +0.03 to +0.04.

Against the old joint recipe at the same width, `dual_bias_mean_gated` is
better on every metric:

- MAE −0.055;
- single AUC +0.008 (t) and +0.012 (c);
- pair AUC +0.021 (t) and +0.022 (c);
- top-1 +0.2 (t) and +2.2 (c) pp.

The other dual arms trade 0.1–0.7 pp of test top-1 for larger AUC gains.

`dual_bias_mean_gated` is the only arm that wins every comparison at both bin
sizes (wave 5 at 2 Da, and here). It is therefore the recommended interface.
Its distogram margin over pair-only is small (−0.008 here, −0.009 at 2 Da) and
seed 0 only. Multi-seed confirmation of the distogram and confirm-set pair-AUC
margins is the remaining check.

## 11. Metadata injection (wave 7; seed 0, C=256, Δ=0.1 Da)

**Starting point.** The pilot so far gives the encoder only one piece of
metadata: the precursor $P$, inside the pair features. These are
$d_{ij}/P$, the complement $m_i+m_j-P$, and $m_i/P$. Production is uneven.
Pair-only (pair279m) applies AdaLN-Zero to every pair block. The joint
fastmixer path, however, passes no condition to its blocks, so
`metadata_conditioning="adaln_zero"` reduces to a modulation of the final
single norm only. The pair stream never sees it.

**Condition vector.** The production `massive_v2_acquisition_v1` layout
(28 values) is filled from MassSpecGym:

- precursor m/z;
- collision energy ($/100$, clipped; present in 53% of spectra);
- instrument family (Orbitrap / QTOF / unknown);
- constant charge 1 and positive polarity.

Training uses the production 10% per-field dropout
(`scripts/pair_single_dual_metadata.py`). An embedding
$c = W_2\,\mathrm{SiLU}(W_1 m)$ feeds one zero-initialised head per site, so
every arm starts exactly at the no-metadata model (tested). The sites are:

| Site | Update |
| --- | --- |
| `token` (production additive) | $s_i \mathrel{+}= H c$ for all peaks and CLS, before the pair embedder reads singles |
| `cls` | $s_{\text{cls}} \mathrel{+}= Hc$ |
| `pair_in` | $z_{ij} \mathrel{+}= Hc$ |
| `film_single`, `film_pair` | at every block input, $x \leftarrow x\odot(1+\gamma_\ell(c)) + \beta_\ell(c)$ |
| `final` (production joint AdaLN) | the same modulation after the final single norm |
| `cls_edge` (precursor node) | $z_{i,\text{cls}}, z_{\text{cls},i} \mathrel{+}= \mathrm{MLP}\big(\phi(P-m_i),\,(P-m_i)/1000,\,m_i/P\big) + Hc$ |

`cls_edge` turns CLS into the precursor. Each of its edges carries that peak's
neutral loss $\ell_i = P - m_i$, with Fourier features down to 0.01 Da. A
triangle update over $(i,\text{cls},j)$ then compares the losses directly,
since $\ell_i - \ell_j = m_j - m_i$.

**Evaluation additions.** All arms are scored with metadata present.

- *Blind*: the same model is also scored with every field absent, so its
  features carry no metadata.
- *Isomer retrieval*: candidates are restricted to library molecules within
  10 ppm of the query's neutral mass, so precursor mass cannot solve the task.
  Only 4,719 test queries and 478 confirm queries have two or more candidates,
  and the sets are tiny (top-10 is always 100%). This metric is therefore
  noisy.
- *Control*: the raw metadata vector is appended to the base model's readouts.

| Arm | MAE | dist | sAUC t / c | pAUC t / c | s@1 t / c | blind MAE | blind sAUC t |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: |
| base (no metadata) | 4.157 | 3.357 | 0.731 / 0.719 | 0.748 / 0.732 | 26.8 / 40.2 | — | — |
| precursor removed from pair features | 4.157 | 3.373 | 0.716 / 0.703 | 0.730 / 0.720 | 23.5 / 33.6 | — | — |
| `final` | 4.147 | 3.360 | 0.739 / 0.725 | 0.752 / 0.730 | 26.7 / 39.8 | 4.154 | 0.733 |
| `token` | 4.128 | 3.340 | **0.745** / 0.729 | 0.759 / 0.736 | 24.0 / 33.8 | 4.161 | 0.736 |
| `cls` | 4.131 | 3.337 | 0.743 / **0.735** | 0.760 / 0.737 | 24.0 / 35.3 | 4.163 | 0.732 |
| `pair_in` | 4.151 | 3.341 | 0.740 / 0.728 | 0.759 / 0.737 | 24.0 / 33.9 | 4.184 | 0.731 |
| `token+pair_in` | 4.138 | 3.345 | 0.744 / 0.727 | 0.757 / 0.740 | 25.0 / 37.6 | 4.168 | 0.732 |
| `film_single` | 4.123 | 3.355 | 0.743 / 0.727 | 0.759 / 0.740 | 24.1 / 34.8 | 4.142 | 0.729 |
| `film_pair` | 4.125 | 3.356 | 0.743 / 0.728 | 0.762 / 0.739 | 25.1 / 36.4 | 4.155 | 0.731 |
| `film_single+film_pair` | 4.137 | 3.345 | 0.742 / 0.727 | 0.760 / 0.741 | 24.5 / 34.3 | 4.170 | 0.732 |
| **`cls_edge`** | **4.098** | **3.301** | 0.742 / 0.727 | **0.763 / 0.741** | **26.9** / 39.7 | 4.158 | 0.729 |
| pair-only, native AdaLN-Zero | — | 3.360 | — | 0.735 / 0.714 | — | — | — |
| pair-only, no metadata | — | 3.364 | — | 0.737 / 0.721 | — | — | — |

Control, with metadata appended to the base readouts: single 0.735 / 0.723
(+0.004), pair 0.750 / 0.734 (+0.002), retrieval unchanged. Metadata alone
scores 0.637 / 0.624.

Findings:

1. **Every site helps chemistry by more than the control:** +0.008 to +0.014
   single AUC and up to +0.015 pair AUC, against +0.002 to +0.004 from simply
   appending metadata. The encoder uses the condition to interpret fragments.
   It is not merely forwarding the condition to the probe.
2. **`cls_edge` is best overall.** It gives the lowest MAE (−0.059), the lowest
   distogram loss (−0.056) and the best pair AUC, and it keeps retrieval.
3. **Retrieval is the trade-off for the other sites.** `token`, `cls`,
   `pair_in` and FiLM lose 2–6 pp top-1. Collision energy and instrument
   differ between spectra of the same molecule, so injecting them into every
   token pulls those embeddings apart. Blind features recover 26–27% top-1.
   `cls_edge` and `final` keep retrieval because the condition stays on CLS
   edges or on one final affine map.
4. **Missing metadata is safe.** Blind MAE and AUC return to the base model's
   level, an effect of the 10% field dropout.
5. **The precursor-relative pair features matter:** removing them costs
   −0.015 AUC and −3 pp retrieval.
6. **Production's per-block AdaLN-Zero does not help pair-only at 20k steps.**
   With `cls_edge`, the joint model now leads pair-only on distogram by 0.06.
7. **GPU runs are deterministic.** The rerun base reproduced wave 6 exactly, so
   there is no run-to-run noise. Seed variance is still unmeasured.

**Speed note.** Step time is about 60–65 ms on a GH200 for batch 32, 64 and
128, and 121 ms for batch 256. Host masking takes 1.3 ms. The step is
latency-bound, so data parallelism over 2 GPUs would not shorten it. Running
two runs per GPU nearly doubles throughput.

## 12. Within-bin regression heads (wave 8; seed 0, C=256, base + `cls_edge`)

A value $x$ in bin $k=\lfloor x/\Delta\rfloor$ also has a within-bin position
$u = x/\Delta - k \in [0,1)$. Classification discards $u$; at 0.1 Da it
carries mass-defect detail down to the instrument's mDa precision. A
`BinOffsetHead` predicts

$$
\hat u_k(h) = \sigma\!\big(w_k^\top g(h) + b_k\big),\qquad
\mathcal L_{\text{off}} = \lambda\,\mathbb E\,|\hat u_{k^\star}(h) - u| ,
$$

with per-bin rows $w_k, b_k$ on a shared feature $g(h)=\mathrm{SiLU}(Ah)$.
Only the true bin's row $k^\star$ is gathered, so the cost does not grow
with $K$. Zero rows start every prediction at the bin centre. The `mz` head
reads the predictor's masked-peak latents. The `dist` head reads the
distogram pair features of each supervised pair.

Offset errors are in bin units. Predicting the bin centre for uniformly
placed values scores 0.25. A per-bin bias alone can already beat that, so
part of the gain is a per-bin prior.

| Arm | MAE | dist | m/z offset | dist offset | sAUC t / c | pAUC t / c | s@1 t / c |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| base + `cls_edge`, 0.1 Da | 4.098 | 3.301 | — | — | 0.742 / 0.727 | 0.763 / 0.741 | 26.9 / 39.7 |
| + mz | 4.099 | 3.322 | 0.147 | — | 0.743 / 0.728 | 0.761 / 0.741 | 26.6 / 40.4 |
| + dist | 4.100 | 3.311 | — | 0.122 | 0.744 / 0.731 | 0.761 / 0.743 | 25.5 / 36.6 |
| + both, λ=0.3 | 4.104 | 3.303 | 0.150 | 0.126 | **0.747 / 0.731** | **0.766** / 0.745 | 26.2 / 38.8 |
| + both, λ=1 | 4.104 | 3.309 | 0.144 | 0.122 | 0.741 / 0.728 | 0.764 / **0.747** | 27.0 / 41.0 |
| + both, λ=3 | 4.114 | 3.332 | 0.140 | 0.117 | 0.747 / 0.728 | 0.764 / 0.741 | 26.3 / 39.6 |
| base + `cls_edge`, 0.2 Da | 3.642 | 3.105 | — | — | 0.738 / 0.723 | 0.761 / 0.741 | 26.1 / 38.6 |
| + both, λ=1 | 3.636 | 3.101 | 0.104 | 0.081 | 0.740 / 0.723 | 0.763 / 0.741 | 25.9 / 39.5 |
| base + `cls_edge`, 2 Da | 2.475 | 2.187 | — | — | 0.698 / 0.683 | 0.728 / 0.708 | 25.4 / 38.5 |
| + both, λ=1 | **2.453** | **2.162** | 0.167 | 0.180 | **0.707 / 0.697** | **0.740 / 0.726** | **26.8 / 39.9** |
| pair-only, 0.1 Da, + dist | — | 3.378 | — | 0.127 | — | 0.738 / 0.720 | — |
| base + `cls_edge` + adduct field | 4.134 | 3.321 | — | — | 0.741 / 0.729 | 0.761 / 0.743 | 25.6 / 38.0 |
| base + `cls_edge` + `film_pair` | 4.108 | 3.304 | — | — | 0.741 / 0.723 | 0.762 / 0.742 | 27.4 / **42.9** |

(Bold marks the best value within each bin size.)

Findings:

1. **The heads learn sub-bin position.** At 0.1 Da, the m/z offset error is
   0.14–0.15 bins (about 14 mDa) and the distance offset error is 0.12 bins
   (about 12 mDa), against 0.25 for bin-centre guessing.
2. **At fine bins they barely change the representation.** Both heads move
   AUC by at most +0.005 and retrieval by ±1.5 pp. These effects are within
   plausible seed noise. The distance head alone costs 3 pp of confirm
   retrieval. λ=0.3–1 is the safe range. At λ=3 the distogram loss worsens
   by +0.031.
3. **At coarse bins they clearly help.** At 2 Da the regression adds
   +0.009 / +0.014 single AUC, +0.012 / +0.017 pair AUC, +1.4 pp retrieval, and
   lowers both cross-entropies. That recovers only about 20–30% of the gap to
   0.1 Da bins (0.742 / 0.763), so fine bins remain the main lever.
4. **Pair-only does not benefit:** pair AUC 0.738 / 0.720 against
   0.737 / 0.721.
5. **Metadata follow-ups.** Adding the adduct field does not help
   (MAE +0.036, −1.3 pp retrieval). `cls_edge + film_pair` keeps AUC and
   raises confirm retrieval by 3.2 pp. That is worth a seed check.
6. **The end-to-end m/z mDa error logged in this wave is not usable.** It
   averaged over misclassified bins, which can be tens of Da off. The metric
   is now computed only over correctly binned targets, together with
   `mz_bin_accuracy`.

**Recommendation.** Keep 0.1 Da bins and `cls_edge`. The both-heads variant at
λ=0.3–1 is optional: it is cheap (no measurable slowdown) and adds sub-bin
m/z / distance predictions, but its probe gains are not yet distinguishable
from noise. The next step is multi-seed confirmation of
base → `cls_edge` → `cls_edge` + heads.

Throughput: with two runs per GPU, each runs at about 7.1 sps, against 12 sps
alone. That is 1.18× aggregate at C=256, less than the 1.7× seen at C=128.

## 13. Register tokens (wave 9; seed 0, base + `cls_edge`, C=256, 0.1 Da)

R learned register tokens are appended to the joint encoder axis after
compaction, at RoPE positions 32, 33, …. They get learned pair rows and
columns ($z_{i,r}$, $z_{r,j}$, and a shared $z_{r,r'}$) and join every block.
As with ViT registers, they are dropped before the predictors and readouts,
so they can only serve as scratch space for the peaks and CLS. They use their
own RNG stream, so the rest of the model initialises exactly as without
registers. The R=0 run comes from wave 7.

The training encoder axis is the 20-token compact capacity (19 visible peaks
plus CLS). The export axis is 32 (31 peaks plus CLS), so both cannot be
aligned at once.

| R | Train / export tokens | MAE | dist | sAUC t / c | pAUC t / c | s@1 t / c | sps |
| ---: | --- | ---: | ---: | --- | --- | --- | ---: |
| 0 | 20 / 32 | 4.098 | **3.301** | 0.742 / 0.727 | **0.763** / 0.741 | **26.9 / 39.7** | 11.9 |
| 1 | 21 / 33 | **4.085** | 3.308 | 0.743 / 0.730 | 0.761 / 0.744 | 26.3 / 38.5 | 10.5 |
| 2 | 22 / 34 | 4.114 | 3.326 | 0.743 / **0.732** | 0.761 / 0.744 | 23.5 / 33.8 | 10.8 |
| 4 | 24 / 36 | 4.104 | 3.305 | 0.743 / 0.724 | 0.760 / 0.738 | 26.6 / 39.0 | 10.5 |
| 8 | 28 / 40 | 4.115 | 3.307 | **0.746** / 0.731 | 0.758 / 0.744 | 25.9 / 37.1 | 8.0 |
| 12 | **32** / 44 | 4.095 | 3.315 | 0.742 / 0.725 | 0.761 / 0.742 | 25.2 / 38.2 | 7.3 |
| 32 | 52 / **64** | cancelled | | | | | 2.6 |
| 44 | **64** / 76 | out of memory (68 GB allocation at batch 128) | | | | | — |

Findings:

1. **Registers do not help.** Every R ≤ 12 lies within ±0.005 AUC of R=0,
   with no trend in R. That is within plausible seed noise.
2. **Retrieval drops** by 0.3–3.4 pp (test) and 0.7–5.9 pp (confirm), with no
   offsetting gain in MAE or distogram loss.
3. **Registers cost speed:** −12% at R=1–4, −33% at R=8 and −39% at R=12.
   Even R=1 is slower, because every block now carries an extra pair row
   and column.
4. **Aligned totals give no speed-up.** 32 training tokens (R=12) is slower
   than 28 (R=8). Step time grows with token count: the pair tensor scales
   as $T^2$ and the triangle updates as $T^3$. The earlier "latency-bound"
   measurement (Section 11) varied only batch size, at T=20.

Registers are not adopted. In this architecture the pair stream already
provides $O(T^2)$ scratch state, and CLS (with `cls_edge`) is a global token,
which plausibly covers what registers supply in ViTs.

## 14. Multi-scale cross entropy (wave 10; seeds 0–2, base + `cls_edge`, C=256, 0.1 Da)

Multi-scale CE adds, for each width $w$, the cross entropy of the coarse bin
formed by $w$ adjacent fine bins, with the coarse probability being the sum of
the fine probabilities:

$$
\mathcal L = \mathrm{CE}_1 + \lambda \cdot \tfrac{1}{|W|}\sum_{w\in W}
-\log \sum_{k \in G_w(k^\star)} p_k .
$$

The library implements it for the masked m/z head
(`mae_mz_multiscale_widths`, `mae_mz_multiscale_weight`) and the distogram
(`distogram_multiscale_widths`, `distogram_multiscale_weight`). Earlier
library configs used $W = \{2, 4, 8, 16, 32, 64\}$ and $\lambda = 1$, which
at 0.1 Da spans 0.2–6.4 Da groups. Waves 1–9 ran with it off. Here the pilot's
`--multiscale mz|dist|mz+dist`, `--multiscale-weight` and
`--multiscale-widths` flags set those four library fields. The rerun base
seed 0 reproduces the wave-7 run exactly.

The MAE and dist columns are the exact-bin (fine) cross entropies. The
multi-scale term is not included in them.

| Arm | Seeds | MAE | dist | m/z bin acc. | sAUC t / c | pAUC t / c | s@1 t / c | p@1 t |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: |
| base | 0–2 | 4.112 | **3.317** | **24.0%** | **0.744 / 0.729** | **0.762 / 0.743** | **25.8 / 38.0** | **19.7** |
| + mz+dist, λ=1, W=2–64 | 0–2 | **3.907** | 3.323 | 23.8% | 0.736 / 0.725 | 0.756 / 0.737 | 24.3 / 35.4 | 18.8 |
| + mz, λ=1 | 0 | 3.904 | 3.337 | 23.5% | 0.734 / 0.721 | 0.754 / 0.737 | 25.0 / 35.8 | 18.3 |
| + dist, λ=1 | 0 | 4.089 | 3.287 | 24.2% | 0.744 / 0.726 | 0.761 / 0.741 | 26.3 / 39.3 | 19.4 |
| + mz+dist, λ=0.3 | 0 | 3.964 | 3.311 | 24.2% | 0.741 / 0.728 | 0.762 / 0.736 | 26.5 / 39.8 | 19.7 |
| + mz+dist, λ=1, W=2,4,8 | 0 | 3.933 | 3.295 | 24.4% | 0.740 / 0.732 | 0.761 / 0.742 | 25.5 / 37.7 | 18.3 |
| base, seed 0 only | 0 | 4.098 | 3.301 | 24.1% | 0.742 / 0.727 | 0.763 / 0.741 | 26.9 / 39.7 | 20.0 |

(The first two rows are means over seeds 0–2. Seed SD of the base is
0.0015–0.0018 AUC and 1.8 / 2.4 pp single retrieval.)

Seed-matched differences, multi-scale on both heads (λ=1) minus base:

| Metric | Seed 0 | Seed 1 | Seed 2 | Mean |
| --- | ---: | ---: | ---: | ---: |
| MAE (exact CE) | −0.205 | −0.216 | −0.194 | −0.205 |
| distogram (exact CE) | +0.015 | +0.010 | −0.008 | +0.006 |
| single AUC test / confirm | −0.002 / −0.002 | −0.012 / −0.005 | −0.012 / −0.005 | −0.009 / −0.004 |
| pair AUC test / confirm | −0.004 / −0.004 | −0.007 / −0.008 | −0.006 / −0.007 | −0.005 / −0.006 |
| single top-1 test / confirm (pp) | −1.5 / −2.6 | −1.4 / −1.5 | −1.4 / −3.5 | −1.4 / −2.5 |
| pair top-1 test (pp) | −1.6 | −0.4 | −0.6 | −0.9 |

Findings:

1. **Multi-scale CE does not improve the representation.** On both heads it
   lowers every probe metric in every seed: −0.009 / −0.004 single AUC,
   −0.005 / −0.006 pair AUC, −1.4 / −2.5 pp single retrieval. The AUC losses
   are 3–5× the base seed SD.
2. **It improves calibration, not accuracy.** Exact-bin m/z CE falls by 0.205
   in every seed, but top-1 bin accuracy does not change (23.8% vs 24.0%).
   The coarse terms move probability mass onto neighbouring bins, which makes
   misses cheaper without making hits more frequent.
3. **The m/z term causes the loss.** m/z alone costs 0.008 / 0.006 single AUC
   and 1.9 / 3.9 pp retrieval (seed 0). The distogram term alone is
   neutral: within ±0.002 AUC, −0.6 pp retrieval, and −0.014 distogram CE.
4. **Weaker or narrower variants are neutral at best.** λ=0.3 and W=2,4,8
   keep AUC within about 0.006 of the base, but still lose 0.4–2.0 pp of
   retrieval on at least one fold.

**Recommendation.** Keep multi-scale CE off for both heads, as in the 1B
configuration. A distogram-only term is the only variant that is not worse,
and it shows no gain to adopt.

## 15. Output norms (wave 11; seeds 0–2, base + `cls_edge`, C=256, 0.1 Da)

The model has two kinds of output norm:

- **Final norms.** These normalise what the encoder and predictors emit: the
  encoder's single `final_norm` (RMSNorm without scale) and pair
  `final_pair_norm`, the MAE predictor's `predictor_final_norm`, and the pair
  predictor's `predictor_pair_norm`.
- **Block post norms.** Every residual sublayer adds a normalised update,
  $h \leftarrow h + \mathrm{RMSNorm}(f(h))$ (the scale starts at 1). These are
  `tri_mul_out_post_norm`, `tri_mul_in_post_norm`,
  `pair_transition_post_norm`, `single_attention_post_norm` and
  `single_transition_post_norm` in the encoder fastmixer blocks;
  `attention_post_norm` and `ffn_post_norm` in the MAE predictor's
  cross-attention blocks; and the triangle and transition post norms in the
  pair-predictor blocks.

The pilot flag `--no-norms final|post|final+post` sets the three
`*_apply_final_norm` settings to False, replaces every `*_post_norm` in the
encoder and both predictors with identity, or does both. The pre-norms inside
each sublayer are kept. The base runs are the wave-10 base runs (the flag is
off, so the computation is unchanged).

| Arm | MAE | dist | m/z bin acc. | sAUC t / c | pAUC t / c | s@1 t / c | p@1 t / c |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| base | **4.112** | **3.317** | 24.0% | 0.744 / 0.729 | **0.762** / 0.743 | 25.8 / 38.0 | 19.7 / 29.1 |
| no final norms | 4.128 | 3.331 | 24.2% | 0.744 / 0.728 | 0.761 / **0.744** | 26.1 / 38.1 | 19.6 / 29.4 |
| no block post norms | 4.145 | 3.341 | **24.4%** | **0.749** / 0.735 | 0.761 / 0.742 | 26.2 / 38.3 | 21.3 / 31.2 |
| no final + no post | 4.185 | 3.397 | 23.8% | **0.749 / 0.736** | 0.760 / **0.744** | **27.1 / 40.0** | **21.9 / 31.9** |

(Means over seeds 0–2. Base seed SDs: 0.0015 single AUC, 0.0018 pair AUC,
1.8 / 2.4 pp single retrieval, 0.4 / 0.3 pp pair retrieval.)

Seed-matched differences from the base (seeds 0 / 1 / 2, then the mean):

| Metric | no final | no post | no final + no post |
| --- | --- | --- | --- |
| MAE (val CE) | +0.029 / +0.008 / +0.014 → +0.017 | +0.054 / +0.044 / +0.002 → +0.033 | +0.108 / +0.068 / +0.044 → +0.073 |
| distogram (val CE) | +0.015 / +0.046 / −0.020 → +0.014 | +0.029 / +0.051 / −0.006 → +0.025 | +0.112 / +0.102 / +0.026 → +0.080 |
| single AUC test | −0.000 / +0.000 / +0.001 → +0.000 | +0.006 / +0.004 / +0.006 → +0.005 | +0.009 / +0.003 / +0.003 → +0.005 |
| single AUC confirm | +0.001 / −0.005 / +0.003 → −0.001 | +0.008 / +0.004 / +0.007 → +0.006 | +0.016 / +0.005 / −0.000 → +0.007 |
| pair AUC test | +0.000 / −0.000 / −0.004 → −0.001 | −0.005 / +0.004 / −0.001 → −0.001 | −0.005 / +0.001 / −0.001 → −0.002 |
| pair AUC confirm | +0.002 / +0.000 / +0.000 → +0.001 | +0.003 / −0.003 / −0.004 → −0.001 | +0.004 / +0.001 / −0.004 → +0.000 |
| single top-1 test (pp) | +1.1 / −0.6 / +0.7 → +0.4 | +0.0 / −1.4 / +2.7 → +0.4 | +0.6 / +0.4 / +2.8 → +1.3 |
| single top-1 confirm (pp) | +1.5 / −1.7 / +0.5 → +0.1 | +0.2 / −2.4 / +3.1 → +0.3 | +0.7 / +1.6 / +3.8 → +2.1 |
| pair top-1 test (pp) | −0.3 / −0.2 / +0.4 → −0.0 | +1.0 / +1.5 / +2.3 → +1.6 | +2.3 / +2.1 / +2.4 → +2.3 |
| pair top-1 confirm (pp) | +0.1 / +0.2 / +0.8 → +0.3 | +2.2 / +1.6 / +2.6 → +2.1 | +2.9 / +3.0 / +2.8 → +2.9 |

Findings:

1. **The final output norms do not matter.** Without them, every probe
   difference stays within one base seed SD and changes sign between seeds.
   Validation CE rises by at most 0.017.
2. **The block post norms hurt the representation.** Without them:
   - Single AUC rises by 0.005 / 0.006 in every seed, 3–4× the seed SD.
   - Pair top-1 retrieval rises by 1.6 / 2.1 pp in every seed, about 5× the
     seed SD.
   - Pair AUC is unchanged, and single retrieval is mixed: seed 1 loses
     1.4–2.4 pp.
3. **Removing both gives the largest gains.**
   - Pair top-1 rises by 2.3 / 2.9 pp and single top-1 by 1.3 / 2.1 pp, all
     six differences positive.
   - Single AUC rises by 0.005 / 0.007; five of six differences are positive,
     and seed 2 confirm is flat.
   - Without the post norms, the final norms are the only normalisation
     between the encoder's last residual stream and the readout. Removing them
     too helps retrieval slightly, although alone they are neutral.
4. **Pretraining loss and representation quality diverge again.** Removing
   both kinds of norm raises validation CE in every seed (+0.073 MAE, +0.080
   distogram), yet improves the probes. As with multi-scale CE (Section 14),
   the exact-bin CE is not a reliable guide to downstream quality.

A plausible mechanism, not tested here: with post norms, every sublayer adds
an update of about unit RMS, however little it contributes. Over 6 blocks × 4–5
sublayers, these fixed-size increments may dilute the earlier,
peak-identity-bearing content of the residual stream.

**Caveats and recommendation.** This is a 27M-parameter, 6-layer, 20k-step
pilot. Post (sandwich) norms are mainly a stability device for deep, large
models: the 1B and 2.3B runs have 21 and 36 encoder blocks and bf16 compute, and those
depths were not tested without them. The library has no switch for them yet;
the post norms are always built. Before changing the production recipe, the
next steps are:

- Add a `*_post_norm` switch to the library.
- Check stability and probes without post norms at depth (a 12–24-layer pilot
  or the 100M configuration).
- Split encoder versus predictor post norms, to see which carries the effect.

## 16. Input m/z representation (wave 12; seeds 0–2, base + `cls_edge`, C=256, 0.1 Da)

The encoder sees m/z in two places:

- **Single stream.** `PeakFeatureEmbedder` maps each peak's m/z (in Da)
  through `FourierFeatures`: $[\cos 2\pi m b_k, \sin 2\pi m b_k]$ with
  wavelengths $1/b_k$ log-spaced over 0.003–1000 Da, 16 of them. An MLP then
  maps these to half the token width; the raw scalar m/z and the intensities
  feed the other half.
- **Pair stream.** `PairFeatureEmbedder` applies a second Fourier block
  (0.01–1000 Da, 16 wavelengths) to the difference $d$, $|d|$ and the
  complement $m_i + m_j - M$, and a relative block (0.001–1, 16) to
  $d / M$, where $M$ is the precursor mass.

The frequencies $b_k$ are `nnx.Param`s, so they are trained. The 1B and 2.3B
configurations use exactly this setup.

New pilot flags:

- `--mz-features log|gauss|rbf|harmonic|rbf_harmonic|token|none` with
  `--mz-freqs`, `--mz-band`, `--mz-sigma`, `--mz-harmonics` and
  `--mz-token-bin`.
- `--mz-learnable off`, which turns every encoder frequency into a fixed
  buffer.
- `--freq-decay off`, which excludes frequencies from AdamW weight decay.
- `--pair-band` and `--pair-freqs`.

Only the m/z features and the MLP's input width change; the MLP widths stay
the same. `harmonic` uses harmonics $k = 1..16$ of 1 Da, which follow the
nominal-mass lattice and so encode the mass defect, plus 16 log-spaced
wavelengths from 2 to 1000 Da. RBF centres sit on an even 0–1000 Da grid,
with width equal to the spacing.

A base rerun with an explicit default flag reproduces the wave-10 base
exactly: training losses are bit-identical, and every evaluation difference
is 0.0000. The pilot is deterministic, so the differences below are effects
of the encoding, and seed is the only noise.

| Arm | MAE | dist | m/z acc. | sAUC t / c | pAUC t / c | s@1 t / c | p@1 t / c |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| base: log Fourier, 16 freqs, 0.003–1000 Da, learnable | 4.112 | 3.317 | 24.0% | 0.744 / 0.729 | 0.762 / 0.743 | 25.8 / 38.0 | 19.7 / 29.1 |
| frequencies fixed (single and pair) | 4.152 | 3.345 | 24.0% | 0.739 / 0.726 | 0.756 / 0.740 | 23.8 / 34.9 | 18.2 / 26.4 |
| learnable, no weight decay (as the library) | 4.102 | 3.293 | 24.4% | 0.743 / 0.728 | 0.760 / 0.743 | 25.3 / 36.8 | 19.2 / 28.4 |
| log Fourier, 32 freqs | 4.124 | 3.322 | 24.1% | 0.744 / 0.728 | 0.761 / 0.740 | 25.1 / 36.7 | 18.9 / 27.9 |
| log Fourier, 64 freqs | 4.126 | 3.319 | 24.0% | 0.742 / 0.726 | 0.761 / 0.739 | 26.0 / 38.7 | 19.3 / 28.5 |
| log Fourier, 128 freqs | 4.121 | 3.301 | 24.7% | 0.747 / 0.734 | 0.763 / 0.746 | 25.2 / 38.1 | 18.8 / 28.4 |
| log Fourier, 256 freqs (library default) | 4.163 | 3.317 | 23.6% | 0.744 / 0.731 | 0.759 / 0.744 | 25.5 / 38.7 | 19.3 / 28.9 |
| log band 0.001–1000 Da | 4.097 | 3.317 | 24.7% | 0.743 / 0.730 | 0.762 / 0.742 | 24.7 / 36.5 | 17.9 / 26.6 |
| log band 0.01–1000 Da | 4.099 | 3.312 | 24.3% | 0.741 / 0.730 | 0.758 / 0.741 | 24.5 / 35.8 | 17.6 / 26.4 |
| log band 0.1–1000 Da | 4.104 | 3.312 | 24.4% | 0.741 / 0.726 | 0.760 / 0.741 | 24.4 / 35.7 | 18.6 / 27.8 |
| Gaussian RFF, 64, σ=1 /Da | 4.119 | 3.335 | 24.0% | 0.745 / 0.729 | 0.760 / 0.743 | 25.5 / 38.2 | 18.7 / 27.3 |
| Gaussian RFF, 64, σ=10 /Da | 4.138 | 3.320 | 24.0% | 0.741 / 0.727 | 0.758 / 0.741 | 25.4 / 38.3 | 18.5 / 27.0 |
| 1 Da harmonics k=1–16 + 16 log 2–1000 Da | 4.107 | 3.297 | 24.5% | 0.743 / 0.728 | 0.761 / 0.742 | 26.5 / 39.0 | 19.2 / 28.0 |
| RBF, 256 centres (3.9 Da) | 4.237 | 3.395 | 21.4% | 0.738 / 0.726 | 0.757 / 0.740 | 24.1 / 35.3 | 18.9 / 27.2 |
| RBF 256 + 1 Da harmonics | 4.183 | 3.313 | 22.6% | 0.743 / 0.732 | 0.761 / 0.745 | 24.9 / 37.2 | 19.5 / 28.8 |
| token embedding, 0.1 Da bins | 4.214 | 3.354 | 23.0% | 0.740 / 0.726 | 0.757 / 0.741 | 23.9 / 34.9 | 18.2 / 26.6 |
| no m/z features (raw scalar only) | 4.484 | 3.421 | 15.7% | 0.745 / 0.727 | 0.758 / 0.738 | 18.4 / 27.6 | 18.1 / 26.1 |
| pair band 0.001–1000 Da | 4.106 | 3.325 | 24.1% | 0.741 / 0.725 | 0.760 / 0.738 | 24.3 / 35.6 | 17.9 / 26.8 |
| pair band 0.1–1000 Da | 4.120 | 3.336 | 23.8% | 0.740 / 0.724 | 0.759 / 0.738 | 24.4 / 36.3 | 18.6 / 28.2 |
| pair Fourier, 32 freqs | 4.096 | 3.338 | 24.1% | 0.743 / 0.728 | 0.762 / 0.744 | 25.8 / 38.0 | 18.5 / 27.5 |
| pair Fourier, 64 freqs | 4.099 | 3.322 | 24.4% | 0.741 / 0.727 | 0.761 / 0.744 | 24.9 / 36.5 | 18.2 / 27.3 |
| pair Fourier, 128 freqs | 4.102 | 3.327 | 23.9% | 0.743 / 0.726 | 0.762 / 0.745 | 24.5 / 35.8 | 17.9 / 27.0 |
| single and pair Fourier, 64 freqs each | 4.124 | 3.317 | 24.2% | 0.744 / 0.725 | 0.760 / 0.740 | 25.7 / 38.4 | 18.8 / 27.8 |

(Means over seeds 0–2. Base seed SDs: 0.0015 single AUC, 0.0018 pair AUC,
1.8 / 2.4 pp single top-1, 0.4 / 0.3 pp pair top-1.)

Seed-matched mean differences from the base. ³ marks differences with the
same sign in all three seeds:

| Arm | MAE | dist | sAUC t / c | pAUC t / c | s@1 t / c (pp) | p@1 t / c (pp) |
| --- | ---: | ---: | --- | --- | --- | --- |
| frequencies fixed (single and pair) | +0.040³ | +0.028³ | −0.0052³ / −0.0023 | −0.0053³ / −0.0035³ | −1.9³ / −3.1³ | −1.4³ / −2.6³ |
| learnable, no weight decay (as the library) | −0.010 | −0.024 | −0.0007 / −0.0007 | −0.0020³ / +0.0001 | −0.4 / −1.1 | −0.5 / −0.7 |
| log Fourier, 32 freqs | +0.012³ | +0.005 | −0.0001 / −0.0001 | −0.0005 / −0.0036³ | −0.7 / −1.3 | −0.8³ / −1.2³ |
| log Fourier, 64 freqs | +0.015 | +0.003 | −0.0016 / −0.0026 | −0.0006 / −0.0039³ | +0.2 / +0.7 | −0.3 / −0.6 |
| log Fourier, 128 freqs | +0.009 | −0.016 | +0.0026³ / +0.0054³ | +0.0009 / +0.0028³ | −0.6 / +0.1 | −0.9³ / −0.7 |
| log Fourier, 256 freqs (library default) | +0.051³ | −0.000 | −0.0002 / +0.0029 | −0.0022 / +0.0002 | −0.2 / +0.7 | −0.3 / −0.2 |
| log band 0.001–1000 Da | −0.014³ | +0.000 | −0.0006 / +0.0016³ | −0.0001 / −0.0012 | −1.0 / −1.4 | −1.7³ / −2.5³ |
| log band 0.01–1000 Da | −0.012 | −0.004 | −0.0030 / +0.0012 | −0.0033³ / −0.0022 | −1.3 / −2.2³ | −2.0³ / −2.7³ |
| log band 0.1–1000 Da | −0.008 | −0.005 | −0.0027 / −0.0020³ | −0.0022 / −0.0026³ | −1.4³ / −2.2³ | −1.0³ / −1.3³ |
| Gaussian RFF, 64, σ=1 /Da | +0.008 | +0.018 | +0.0009 / +0.0003 | −0.0018³ / −0.0007 | −0.3 / +0.2 | −0.9³ / −1.8³ |
| Gaussian RFF, 64, σ=10 /Da | +0.026³ | +0.003 | −0.0033³ / −0.0016 | −0.0037 / −0.0027³ | −0.4 / +0.3 | −1.2³ / −2.1³ |
| 1 Da harmonics k=1–16 + 16 log 2–1000 Da | −0.005 | −0.020 | −0.0009³ / −0.0002 | −0.0003 / −0.0018³ | +0.7 / +1.0 | −0.4³ / −1.1³ |
| RBF, 256 centres (3.9 Da) | +0.125³ | +0.078³ | −0.0063³ / −0.0025 | −0.0043 / −0.0029³ | −1.7³ / −2.6³ | −0.7 / −1.9³ |
| RBF 256 + 1 Da harmonics | +0.072³ | −0.004 | −0.0011³ / +0.0033³ | −0.0009 / +0.0018 | −0.9 / −0.8 | −0.2 / −0.3 |
| token embedding, 0.1 Da bins | +0.103³ | +0.038 | −0.0045³ / −0.0027 | −0.0051³ / −0.0020 | −1.8 / −3.1 | −1.5³ / −2.5³ |
| no m/z features (raw scalar only) | +0.372³ | +0.105³ | +0.0008 / −0.0013³ | −0.0036 / −0.0051³ | −7.4³ / −10.4³ | −1.6³ / −3.0³ |
| pair band 0.001–1000 Da | −0.006 | +0.008 | −0.0031³ / −0.0038³ | −0.0015 / −0.0051³ | −1.4³ / −2.4³ | −1.8³ / −2.3³ |
| pair band 0.1–1000 Da | +0.008³ | +0.019³ | −0.0045³ / −0.0043³ | −0.0028 / −0.0055³ | −1.3³ / −1.6³ | −1.0³ / −0.9³ |
| pair Fourier, 32 freqs | −0.016³ | +0.022 | −0.0012 / −0.0005 | +0.0005 / +0.0004³ | +0.1 / +0.1 | −1.2³ / −1.5³ |
| pair Fourier, 64 freqs | −0.013 | +0.005 | −0.0033 / −0.0017 | −0.0003 / +0.0003 | −0.8 / −1.5³ | −1.4³ / −1.8³ |
| pair Fourier, 128 freqs | −0.010 | +0.011 | −0.0015 / −0.0021³ | −0.0001 / +0.0019 | −1.3³ / −2.2³ | −1.7³ / −2.1³ |
| single and pair Fourier, 64 freqs each | +0.013 | +0.001 | −0.0004 / −0.0034³ | −0.0016 / −0.0035³ | −0.0 / +0.4 | −0.9³ / −1.3³ |

Findings:

1. **The current encoding is at least as good as every alternative.** No
   arm improves probe AUC or retrieval consistently. Eighteen of the 22 arms
   lose pair top-1 in all three seeds on at least one fold, by 0.4–3.0 pp.
   The other four are within noise: 64 and 256 log frequencies,
   RBF + harmonics, and no frequency decay.
2. **Fourier features are needed.**
   - With only the raw scalar m/z, MAE rises by 0.37 and bin accuracy falls
     by 8.3 pp. Single retrieval drops by 7.4 / 10.4 pp, although single
     MACCS AUC does not change, because the substructure probe barely uses
     fine m/z.
   - RBF bumps and the 0.1 Da token embedding both lose 0.004–0.006 single
     AUC and 1.7–3.1 pp single retrieval. Neither interpolates between
     neighbouring masses the way sinusoids do.
3. **Learnable frequencies matter.** Fixing them costs 0.005 single and pair
   AUC and 1.4–3.1 pp retrieval, in all three seeds on most metrics.
   - The learned wavelengths show where the learning happens. In the base
     runs, wavelengths of 2 Da and above move a lot and differently in each
     seed (1000 → 650–680 Da, 183 → 170–195 Da).
   - Wavelengths of 0.5 Da and below all stretch by exactly 3.3%, the same in
     every seed. That is the pilot's AdamW decay on every parameter,
     $\exp(-0.01 \sum \eta) \approx 0.967$.
   - The library decays only 2-D `weight` leaves, so its frequencies never
     decay. Running the pilot that way (`--freq-decay off`) is within noise
     of the base: −0.002 pair test AUC, −0.4 to −1.1 pp retrieval, with mixed
     signs. The decay is therefore an unimportant pilot artefact, and the
     production setup is equivalent.
4. **Changing the band trades CE for retrieval.**
   - Moving the single-stream band edge to 0.001, 0.01 or 0.1 Da lowers MAE
     by 0.008–0.014 and raises bin accuracy by 0.3–0.7 pp in all three
     seeds.
   - Each of these changes also costs 1.0–2.7 pp pair retrieval in all three
     seeds, so lower CE again does not mean a better representation.
   - The band edge sets the spacing of all 16 wavelengths, so these arms do
     not isolate the short wavelengths themselves. Those are numerically
     noisy in float32 on GPU: the maximum cos error over m/z up to 1000 Da is
     0.67 at 0.001 Da, 0.22 at 0.003 Da, 0.08 at 0.01 Da and < 0.01 above
     0.1 Da.
5. **The pair band is already right.** Widening the pair band to 0.001 Da or
   narrowing it to 0.1 Da costs 0.002–0.005 AUC and 0.9–2.4 pp retrieval,
   mostly in all three seeds. More pair frequencies do not help either: 32,
   64 and 128 change AUC by at most 0.003 but lose 1.2–2.1 pp pair retrieval
   in all three seeds, and the loss grows with the count.
6. **Alternatives built on the 1 Da lattice tie the base.**
   - Harmonics of 1 Da lower distogram CE by 0.020 and raise bin accuracy by
     0.5 pp. They gain 0.7 / 1.0 pp single retrieval but lose 0.4 / 1.1 pp
     pair retrieval.
   - Adding them to the RBF grid recovers most of what the RBF alone loses,
     and gives the only consistent AUC gain here: +0.0033 single confirm
     AUC, 3/3 seeds. MAE stays 0.07 worse.
   - Neither is a clear improvement.

7. **More single-stream frequencies do not help on balance.** With the same
   0.003–1000 Da band:
   - 32 and 64 frequencies gain nothing: AUC changes by 0 to −0.004 and
     pair retrieval by −0.3 to −1.2 pp.
   - 128 frequencies are the only arm in the wave with AUC gains in all
     three seeds: +0.0026 / +0.0054 single and +0.0028 pair confirm AUC.
     Distogram CE drops by 0.016, but pair test top-1 falls by 0.9 pp in all
     three seeds.
   - 256 (the library default) raises MAE by 0.051 in all three seeds and
     is otherwise within noise.
   - The AUC changes do not trend with the count (64 is below 16, 128 above,
     256 level), so the 128 gain may not survive more seeds or scale.
   - Raising both streams to 64 frequencies loses 0.003 AUC on the confirm
     fold and 0.9 / 1.3 pp pair retrieval in all three seeds.

**Recommendation.** Keep the current m/z encoding, with 16 learnable
frequencies, for single and pair inputs. No input representation tested here
is worth adopting. 128 single-stream frequencies are the only candidate worth
a confirmation run, and only if MACCS AUC matters more than pair retrieval. The most promising lead is to add 1 Da harmonics to the
current log grid rather than replacing it, which this wave did not test.

## 17. Reproduction

The pilot scripts live at commit `78eb2c0` (`git checkout 78eb2c0` first).

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python -m scripts.pair_single_dual_pilot \
  --arm dual_bias_gated --seed 0 --steps 20000 \
  --inputs artifacts/pair_to_single_20260930/panel.npz \
  --confirmation artifacts/pair_to_single_long_20260930/confirmation_panel.npz \
  --output runs/dual_bias_gated_s0
.venv/bin/python -m scripts.pair_single_dual_eval --inputs artifacts/pair_to_single_20260930/panel.npz runs/*_s*
.venv/bin/python -m scripts.report_pair_single_dual runs --csv pair_single_dual.csv
```

Current best recipe with metadata, optional offset heads and registers:

```bash
.venv/bin/python -m scripts.pair_single_dual_metadata \
  --panel artifacts/pair_to_single_20260930/panel.npz \
  --output artifacts/pair_single_dual_20261001/msg_metadata.npz
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python -m scripts.pair_single_dual_pilot \
  --arm dual_bias_mean_gated --seed 0 --steps 20000 --pair-dim 256 --mz-bin-size 0.1 \
  --metadata artifacts/pair_single_dual_20261001/msg_metadata.npz \
  --meta-fields precursor+ce+instrument --meta-sites cls_edge \
  [--offset-heads mz+dist --offset-weight 0.3] [--registers 4] \
  --inputs artifacts/pair_to_single_20260930/panel.npz \
  --confirmation artifacts/pair_to_single_long_20260930/confirmation_panel.npz \
  --output runs/best_s0
.venv/bin/python -m scripts.pair_single_dual_eval --inputs artifacts/pair_to_single_20260930/panel.npz \
  --metadata artifacts/pair_single_dual_20261001/msg_metadata.npz runs/best_s0
```

Isambard launcher: `~/spectra-p2s/p2s_queue.sbatch` (`SPECS="arm:seed …"`,
`SLOTS` runs per GPU). Outputs for each wave are under
`/scratch/u6rb/wuhao.u6rb/p2s/` (`wave1`, `wave2`, `wave3_ts`, `wave4_bins`,
`wave5_pd256`, `wave6_pd256_bin01`, `wave7_meta_sites`, `wave8_offsets`,
`wave9_registers`, `wave10_multiscale`, `wave11_norms`, `wave12_mz`). The wave-10 to
wave-12 pilot flags (`--multiscale*`, `--no-norms`, `--mz-*`, `--freq-decay`,
`--pair-band`, `--pair-freqs`) are not in `78eb2c0`; apply
`artifacts/pair_single_dual_20261001/pilot_78eb2c0_waves10-12.patch`
(`git apply`) after checking out that commit. Evaluation JSON files are copied to
`artifacts/pair_single_dual_20261001/<wave>/`. Tests and evaluation run as
Slurm jobs, never on the login node.

## 18. Library implementation (`training_mode="dual"`)

The best recipe (`dual_bias_mean_gated`, C=D, 0.1 Da bins, `cls_edge`
metadata) is a `PeakSetJEPAJax` training mode, trained by the standard JAX
pretraining loop:

| Pilot | Library |
| --- | --- |
| joint MAE model + separate pair-only model without encoder | one model: fastmixer encoder (pair path), `masked_latent_predictor` (cross-attention, MAE), `pair_predictor` (PairOnlyBlocks), `pair_mask_token`, `distogram_head` |
| `BridgedBlock` with `mean` + `gated` bridges | `PairMixerBlock.pair_to_single` (`PairToSingleBridge`, `pair_to_single_rank`) applied after each fastmixer block, in both the compact and the scattered-single paths |
| `MetadataInjection` with site `cls_edge` | `encoder_metadata_conditioning="precursor_edge"` (`PrecursorEdgeEmbedder`, condition MLP `MetadataEmbedder`), `massive_v2_acquisition_v1` vectors from the collator with `spectrum_metadata_dropout_probability` |
| `dual_forward`: 0.7 MAE + 0.3 distogram | `forward_dual`: `mae_loss_weight` + `distogram_loss_weight` |

With the pilot's parameters copied into the library model, the loss terms and
the CLS readout are identical (checked with perturbed, non-zero edge heads).
Tests: `tests/test_dual_jax.py`.

Not carried over: Thompson-span masking, within-bin offset heads, register
tokens, the adduct field, the IPA readout, `dualplus`, and the token / CLS /
FiLM / `pair_in` / `final` metadata sites.

The 1B GH200 configuration is `configs/1b_dual_n32_n64_gh200x16.py`. It uses
peak-count stages (`num_peaks_schedule`): 31 peaks plus CLS first, then 63 plus
CLS, in one run.

GPU throughput work for that run, measured on Isambard GH200 nodes at 128
spectra per GPU:

- `nnx.remat` gave no memory saving in the sharded (multi-GPU) step: the
  stage-1 step needed 104 GiB with full remat and 99 GiB without. XLA deferred
  the weight-gradient GEMMs, which feed only the gradient all-reduce, and kept
  every layer's recomputed activations alive. `activation_checkpoint_mode =
  "ordered"` (`spectra_learning/models/common_jax.py`) recomputes after the
  cotangent arrives and releases the input cotangent together with the
  parameter gradients: 18 GiB at 31 peaks, and 63 peaks fits as well.
- Pair-path dots return f32 from bf16 inputs. Their f32 cotangents made XLA
  run the backward GEMMs in fp32, 30% of GPU time. On GPU they now take bf16
  cotangents (`_acc_dot_general` in `pairmixer_jax.py`): +30% throughput.
- The fastmixer triangle contraction runs as a batched GEMM on GPU. The
  broadcast-multiply-reduce form ran out of memory, and a fused reduction was
  10x slower at 39 or more tokens.
- The 0.1 Da distogram has 10,000 classes. At stage 2 that is about
  112k scored pairs per GPU, or 4.5 GB of f32 logits.
  `distogram_ce_chunk_rows = 8192` fuses the head with the cross entropy over
  row chunks and recomputes each chunk in the backward pass. The step's peak
  is in the predictor backward, so it saves only 0.5 GB (72.7 → 72.2 GB per
  GPU at 63 peaks), with no throughput change.
- `fastmixer_predictor_capacity_multiple = 4` pads the stage-2 predictor
  capacity from 55 to 56 tokens. One pair-predictor block runs in 150 ms
  instead of 163 ms. The full step gains 0.5% (139.2 → 139.9 spectra/s per
  node) and peak memory rises by 1.5 GB.

- From 4 nodes up, the gradient all-reduce dominated: 16 GPUs ran 199
  spectra/s (5.2 s per step, 64 per GPU) and 32 GPUs about 9 s per step.
  XLA's default 30 MB buckets ran at about 1 GB/s bus bandwidth inside the
  step. The same network gives 85-89 GB/s on buckets of 128 MB or more.
  Isambard's recommended NCCL settings alone gave 233; `NCCL_ALGO=Ring`,
  `NCCL_PROTO=Simple` and 256 MB buckets
  (`--xla_gpu_all_reduce_combine_threshold_bytes`) restore linear scaling.
  `scripts/isambard_gh200_train.sbatch` sets all of them. Buckets of 1 GB or
  more segfault.

| Peaks | Spectra/s per node | Peak memory | 16 GPUs | 32 GPUs (batch 4096) |
| --- | ---: | ---: | ---: | ---: |
| 31 | 454 | 26 GB | 1793 (99%) | 3506 (97%) |
| 63 | 137 | 73 GB | 541 (99%) | 984 (90%)* |

\* Measured with Ring alone, before `NCCL_PROTO=Simple`.

Without remat (`activation_checkpoint_mode = "none"`) the step fits at most
64 spectra per GPU at 31 peaks (499 spectra/s per node) and 16 at 63 peaks
(126). `"ordered_saved"` keeps the forward residuals and only orders the
backward. It fits 128 and 32 spectra per GPU and runs 509 and 131 spectra/s
per node: +12% at 31 peaks (1990 on 16 GPUs) but slower at 63 peaks, and at
90 GB peak memory batch sizes near the limit fail unpredictably (96 per GPU
ran out of memory where 128 fit).

Disabling Triton GEMMs cost 30%; disabling command buffers changed nothing.
The remaining overhead is mostly layout transposes around the channel-batched
triangle GEMMs (19% of kernel time).
