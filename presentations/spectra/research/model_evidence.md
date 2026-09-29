# Model evidence for Novogaia presentation

Read-only inspection of source at commit `86c1a6795c1b7c747f0c7f6a70ecd5b6b6a071d8` (workspace may also contain uncommitted changes). Paths below are relative to `/home/wuhao/spectra-learning/`. Runtime config resolution was verified with the repository `.venv/bin/python`; installed JAX 0.11.0, Flax 0.12.8, PyTorch 2.13.0+cpu.

## Safe main narrative

1. A raw tandem spectrum is a set of peaks `x={(m_i,I_i)}` plus acquisition metadata `a`.
2. Stage 1 trains an MAE to recover missing m/z bins from visible peaks. The concrete 1.8B teacher recipe uses one fine categorical m/z prediction with coarse marginal losses.
3. Stage 2 freezes this pretrained encoder. The student sees only context peaks, while the teacher sees all valid peaks. A predictor learns to recover teacher representations at masked target positions.
4. Pair and triangle models are a separate implemented research branch. Do not silently depict the two-stage single-stream teacher/student recipe as containing pair features.

## Stage 1: MAE

Teacher config: `configs/1p8b_singlemixer_dense_adam_metadata_adaln_multiscale_n128_b6144_666k_d2048.py`; width 2048, 25 encoder layers, 5 predictor layers, 127 peaks plus CLS, m/z ordering, 60% context and 25% target, 666k configured training steps. Current frozen-teacher configs point to the 666000 checkpoint. The model family name 1.8B includes predictor side; the frozen encoder alone is approximately 1.516B according to `configs/1b_mae_teacher_jepa.py:49` and existing audit docs.

Let `C` be visible context, `T` masked targets, `q_i` a learned mask query with original ordinal position, `p_i` the predicted m/z distribution, and `b_i=floor(m_i/0.5 Da)` the target class among 2000 bins covering the supported 1000 Da range. Then

```tex
h_C=f_\phi(x_C,a),\qquad p_i=\operatorname{softmax}(D_\psi(h_C,q_i)),
\quad i\in T.
P_i^{(w)}(g)=\sum_{c:\lfloor c/w\rfloor=g}p_i(c),
\quad \mathcal W=\{2,4,8,16,32,64\}.
\mathcal L_{\mathrm{MAE}}=
-\frac1{|T|}\sum_{i\in T}\left[
\log p_i(b_i)+\frac1{|\mathcal W|}\sum_{w\in\mathcal W}
\log P_i^{(w)}(\lfloor b_i/w\rfloor)\right].
```

Plain language: first learn where the missing peak belongs; a near miss receives useful supervision at broader resolutions. All scales come from the same softmax, not independent heads. The shown teacher recipe has **intensity reconstruction weight zero**. Intensity is still an input. Do not say it reconstructs both mass and intensity in this specific run.

Evidence: `docs/multiscale_ce_and_grouped_jepa_cls_invariance.md:20-164`; `spectra_learning/models/model_jax.py:780-845` (MAE flow), `:1766-1797` (multiscale CE), `:1823-1859` (targets/objective), `:1499-1529` (queries contain mask token and ordinal position, not target m/z); `configs/1b_singlemixer_dense_muon_metadata_adaln.py:94-125`; `configs/1b_singlemixer_dense_muon_metadata_adaln_multiscale_n128_b2048_1m.py:12-14`.

## Stage 2: frozen-teacher JEPA

```tex
y_i=\operatorname{sg}\!\left[f_{\phi^*}(x,a)_i\right],
\qquad \hat y_i=g_\theta(f_\omega(x_C,a),q_i),
\qquad
\mathcal L_{\mathrm{JEPA}}=
\frac1{|T|d}\sum_{i\in T}\|\hat y_i-y_i\|_2^2.
```

`sg` means the teacher receives no gradient. The teacher remains fixed; this is **not EMA**. Current configs use raw teacher targets (`jepa_target_normalization="none"`), identity target projector (`target_projector_dim=-1`), and MSE, not L1. The actual prediction is target peak latent, not necessarily a pooled/global CLS vector. Predictor output is mapped to the teacher's feature dimension (2048 for the teacher above). Student width can be 1536 (`1b_mae_teacher_jepa`) or 2048 (`1p8b_mae_teacher_jepa`). Both use 25 encoder layers and 5 cross-attention predictor layers.

Evidence: `configs/1b_mae_teacher_jepa.py:17-45,59-67`; `configs/1p8b_mae_teacher_jepa.py:13-39`; `spectra_learning/models/model_jax.py:603-614` (stop-gradient and loss), `:1265-1283` (all valid teacher input), `:1299-1337` (teacher vs student inputs), `:1600-1603` (dimension-mean squared error), `:2117-2127` (EMA guarded by use_ema_teacher); `spectra_learning/training/pretrain_jax.py:124-133` excludes teacher modules from optimization.

Presentation caveats: fixed targets prevent teacher/student jointly drifting to a constant, but do not prove all student dimensions remain useful or guarantee downstream improvement. Raw MSE weights high-scale latent directions heavily. Acquisition modulation can change target magnitudes after final RMSNorm. The internal audit explicitly finds JEPA loss insufficient by itself to select retrieval checkpoints (`docs/frozen_jepa_salt_audit_20260905.md:1-76`).

## Metadata: explain the acquisition context

The 28-D vector contains precursor m/z, collision energy, precursor charge, mass accuracy, retention time fraction, precursor intensity z-score, polarity, DDA/DIA acquisition type, isolation-window offsets, and instrument family. Scalar fields have presence indicators; categorical fields have an unknown category. Instrument family has seven categories including unknown. This is not arbitrary sample identity or molecule labels.

Metadata is embedded to width 256 and modulates blocks via AdaLN-Zero; dropout probability 0.10 discourages complete dependence on metadata. A simple visual equation can use

```tex
\operatorname{AdaNorm}(h,a)
=(1+\gamma(a))\odot\operatorname{RMSNorm}(h)+\beta(a).
```

The actual implementation also predicts residual gates and uses zero-initialized modulation. Explain as: the same spectral peak should be interpreted in light of how it was measured.

Evidence: `spectra_learning/models/spectrum_metadata.py:16-33,227-275,291-312`; `spectra_learning/models/encoder_jax.py:170-185,298-315`; `configs/1b_mae_teacher_jepa.py:19-22`.

## Pair representation and triangle operations

`z_ij` describes the relationship between peaks i and j. Raw features include signed and absolute m/z difference, precursor-relative difference, complement `m_i+m_j-m_precursor`, individual masses relative to precursor, intensities, and Fourier features. These are spectrum-peak relationships, not chemical bonds, atom distances, or residue contacts.

Schematic outgoing/incoming multiplication (gates, normalization, and final projection omitted):

```tex
\Delta z_{ij}^{\rm out}=\sum_k A(z_{ik})\odot B(z_{jk}),
\qquad
\Delta z_{ij}^{\rm in}=\sum_k A(z_{ki})\odot B(z_{kj}).
```

Starting-node triangle attention uses the third edge as an attention bias:

```tex
\alpha_{ijk}=\operatorname{softmax}_k
\left(q(z_{ij})^\top k(z_{ik})/\sqrt{d}+b(z_{jk})\right),
\qquad
\Delta z_{ij}=\sum_k\alpha_{ijk}v(z_{ik}).
```

Plain language: whether peaks i and j belong together can be informed by how each relates to a third peak k. Inspired by AlphaFold's pair communication, adapted to mass relationships rather than 3D geometry. Triangle operations are implemented; whether they capture chemically useful fragmentation interactions or improve retrieval is an experimental hypothesis.

Evidence: `spectra_learning/models/pairmixer_jax.py:300-436` (pair features), `:509-566` (multiplication), `:923-986` (attention), `:989-1009` (pair-only multiplication), `configs/279m_paironly_d640_l25_triattn_n32_v6e32_ga4_1m.py:1-21,35-53` (alternation: odd multiplication, even attention; 25 encoder + 5 predictor blocks; 279M parameters).

The pair-only pretext task masks peaks and predicts **absolute m/z differences**, discretized in 0.5 Da bins, with multiscale CE, for pairs involving a target. This is a spectral “distogram”; avoid describing it as spatial molecular-distance prediction. `spectra_learning/models/model_jax.py:964-1160`; pair-only config `:65-103`.

### Critical distinction checked by executable config resolution

`spectra_learning/models/settings.py:209-219` activates the pair track only when explicitly requested, for pair_distogram, or for distogram/contrastive settings. In the current configs:

| Config | Active pair track | Active pretext loss |
|---|---:|---|
| 1.8B multiscale MAE teacher | No | m/z multiscale CE |
| 1B frozen-teacher JEPA | No | Masked latent MSE |
| 1.8B frozen-teacher JEPA | No | Masked latent MSE |
| 279M pair-only triangle-attention run | Yes | Pair m/z-distance CE |

Do not imply the two-stage recipe or a reported 33% retrieval necessarily used triangle modeling. Match every result to its actual checkpoint/config.

## Data-scale wording

The README documents a MassIVE-v2 exact-MS2 conversion retaining provenance, group-safe splits, training eligibility, and 65 train + 3 validation shards. This alone does not verify that more than one billion spectra were used in the specific training run. It is safe to distinguish the billion-scale **raw data opportunity** from the filtered, selected pretraining corpus. Do not equate spectra count to independent molecules or labeled spectra.

