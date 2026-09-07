# m/z reconstruction objectives and grouped JEPA CLS invariance

## Scope

This document specifies three training objectives used in this
repository:

1. multi-scale m/z cross-entropy for masked peak reconstruction;
2. sparse ordinal soft-target cross-entropy for masked peak reconstruction; and
3. grouped-spectrum JEPA training, which makes CLS representations invariant
   across spectra assigned to the same dataset entity.

The equations describe the implemented reductions and gradient flow. The m/z
losses are implemented in
[`models/model.py`](../spectra_learning/models/model.py) and
[`models/model_jax.py`](../spectra_learning/models/model_jax.py). Grouped JEPA
is implemented in
[`models/grouped_jepa_jax.py`](../spectra_learning/models/grouped_jepa_jax.py),
with its EMA update and target pipeline in
[`training/pretrain_jax.py`](../spectra_learning/training/pretrain_jax.py).

## 1. Multi-scale m/z cross-entropy

### Fine m/z targets

Let \(x_{b,n}\in[0,1]\) be the stored normalized m/z of peak \(n\) in
spectrum \(b\), let \(M=1000\,\mathrm{Da}\) be the supported m/z range, and
let \(\Delta\) be the fine output-bin width in Da. The fine categorical
target is

\[
y_{b,n}
=
\operatorname{clip}
\left(
  \left\lfloor\frac{M x_{b,n}}{\Delta}\right\rfloor,
  0,
  C-1
\right),
\qquad
C=\left\lceil\frac{M}{\Delta}\right\rceil.
\]

The predictor can produce several target views for each spectrum. Write its
fine-bin logits as

\[
z_{b,k,n,c},
\qquad
p_{b,k,n,c}
=
\frac{\exp z_{b,k,n,c}}
     {\sum_{c'=0}^{C-1}\exp z_{b,k,n,c'}},
\]

where \(k\) indexes the target view and \(c\in\{0,\ldots,C-1\}\) indexes a
fine m/z bin. Let \(a_{b,k,n}\in\{0,1\}\) indicate that the peak is an active
prediction target, and define

\[
Z=\max\left(\sum_{b,k,n}a_{b,k,n},1\right).
\]

The ordinary masked fine-bin cross-entropy is

\[
\mathcal{L}_{\mathrm{fine}}
=
-\frac{1}{Z}
\sum_{b,k,n}
a_{b,k,n}\log p_{b,k,n,y_{b,n}}.
\]

### Coarse marginals from the same softmax

For a scale width \(w\in\mathbb{N}\), measured in fine bins, define the
coarse-bin map

\[
g_w(c)=\left\lfloor\frac{c}{w}\right\rfloor.
\]

The probability of coarse bin \(q\) is the exact marginal of the fine
distribution:

\[
P^{(w)}_{b,k,n,q}
=
\sum_{\substack{0\le c<C\\g_w(c)=q}}
p_{b,k,n,c}.
\]

The cross-entropy at scale \(w\) is therefore

\[
\mathcal{L}_{w}
=
-\frac{1}{Z}
\sum_{b,k,n}
a_{b,k,n}
\log P^{(w)}_{b,k,n,g_w(y_{b,n})}.
\]

In code this is evaluated stably as a `logsumexp` over adjacent groups of
fine-bin log-probabilities. If \(C\) is not divisible by \(w\), the final group
is padded with \(-\infty\), which adds no probability mass.

For a configured set of widths \(\mathcal{W}\), all coarse resolutions receive
equal weight:

\[
\mathcal{L}_{\mathrm{multi}}
=
\frac{1}{|\mathcal{W}|}
\sum_{w\in\mathcal{W}}\mathcal{L}_{w},
\]

and the m/z objective is

\[
\boxed{
\mathcal{L}_{m/z}
=
\mathcal{L}_{\mathrm{fine}}
+\lambda_{\mathrm{multi}}\mathcal{L}_{\mathrm{multi}}
}.
\]

There is one fine-resolution output head and one softmax. The coarse losses
do not introduce additional heads or independently normalized predictions;
they sum the fine probabilities. Consequently, a prediction receives partial
credit when it misses the exact fine bin but remains in the correct coarse
neighborhood.

If intensity reconstruction is enabled, the complete per-mask reconstruction
loss is

\[
\mathcal{L}_{\mathrm{recon}}
=
\mathcal{L}_{m/z}
+\lambda_I\mathcal{L}_{I},
\]

where \(\mathcal{L}_{I}\) is the masked intensity-bin cross-entropy. The MAE
training path contributes
\(\lambda_{\mathrm{MAE}}\mathcal{L}_{\mathrm{recon}}\) to the model loss. The
same reconstruction loss can instead be used as the auxiliary
\(\lambda_{\mathrm{JEPA\text{-}MAE}}\mathcal{L}_{\mathrm{recon}}\) term in a
JEPA run.

### Current multi-scale recipe

The multi-scale configs use

\[
\Delta=0.5\,\mathrm{Da},
\qquad
\mathcal{W}=\{2,4,8,16,32,64\},
\qquad
\lambda_{\mathrm{multi}}=1.
\]

These widths correspond to physical resolutions of
\(1,2,4,8,16,32\,\mathrm{Da}\). The logged values are
`mae_mz_loss` for \(\mathcal{L}_{\mathrm{fine}}\),
`mae_mz_ce_{w}bins` for each \(\mathcal{L}_w\), and
`mae_mz_multiscale_loss` for \(\mathcal{L}_{\mathrm{multi}}\).

## 2. Sparse ordinal soft-target cross-entropy

The m/z objective is selected by `mae_mz_objective`, whose supported values
are `exact_ce`, `multiscale_ce`, and `sparse_ordinal_ce`. Legacy configs that
do not contain this field resolve to `multiscale_ce` when their multiscale
widths and weight are active, and to `exact_ce` otherwise.

For target bin \(y\), radius \(R\), and offset \(d\in\{-R,\ldots,R\}\), define

\[
w_d=(R+1-|d|)\,\mathbf{1}[0\le y+d<C],
\qquad
q^{\mathrm{local}}_{y+d}=\frac{w_d}{\sum_j w_j}.
\]

Invalid neighbors at the lower and upper class boundaries are discarded and
the remaining triangular weights are renormalized. Mixing the local target
with exact-bin supervision gives

\[
q=(1-\epsilon)\delta_y+\epsilon q^{\mathrm{local}}.
\]

The implementation evaluates the per-token loss as

\[
\boxed{
\mathcal L_{\mathrm{sparse\text{-}ord}}
=
\operatorname{LSE}(z)
-
\sum_{d=-R}^{R}q_{y+d}z_{y+d}
}.
\]

It computes one global `logsumexp` and gathers only \(K=2R+1\) logits. It does
not construct a dense \(C\)-class target. The initial experiment uses
\(R=2\), \(\epsilon=0.2\), and 0.5 Da bins, giving interior local weights
`[1, 2, 3, 2, 1] / 9` over \(\pm1\) Da.

## 3. Grouped JEPA CLS invariance

### Group construction

A training batch contains \(G\) entity groups. Each group contains \(R\)
spectra sampled without replacement from the same MassIVE v2 entity, whose
assigned key is `(massive_id, global_group_id)`. The randomly ordered spectra
are split into

\[
T\text{ teacher spectra}
\quad\text{and}\quad
S=R-T\text{ student spectra}.
\]

Every spectrum is encoded with all of its valid peaks visible. Let

\[
E_{\theta}(X)\in\mathbb{R}^{(N+1)\times D}
\]

denote the online encoder output for spectrum \(X\), including the final CLS
token. Its CLS embedding is

\[
h_{\theta}(X)=E_{\theta}(X)_{\mathrm{CLS}}\in\mathbb{R}^{D}.
\]

### Teacher target

Let \(\bar\theta\) denote the teacher parameters. For group \(g\), the target
is the mean teacher CLS embedding over its \(T\) teacher spectra:

\[
t_g
=
\operatorname{sg}
\left[
  \frac{1}{T}
  \sum_{i=1}^{T}
  h_{\bar\theta}(X^{(t)}_{g,i})
\right],
\]

where \(\operatorname{sg}\) is stop-gradient. Averaging makes one group-level
target from several experimental spectra rather than choosing one spectrum as
the canonical view.

The teacher can be formed in either of two ways:

- **EMA teacher:** \(\bar\theta\) is a non-trainable copy of the online encoder.
  After the student optimizer update, it is updated as

  \[
  \bar\theta_{u+1}
  =
  m\bar\theta_u+(1-m)\theta_{u+1}.
  \]

- **Shared online encoder:** \(\bar\theta=\theta\), but the group target is
  detached. Thus the invariance loss sends no gradient through the teacher
  spectra even though both branches use the same encoder parameters.

### Student prediction and invariance loss

For each student spectrum, the predictor receives every encoded token, not
only its CLS token. A learned query cross-attends to the valid peak tokens and
the CLS token, and a linear readout produces

\[
\hat t_{g,j}
=
P_{\psi}\!\left(E_{\theta}(X^{(s)}_{g,j})\right)
\in\mathbb{R}^{D},
\qquad j=1,\ldots,S.
\]

Each student spectrum in the group independently predicts the same stopped
teacher target. The implemented invariance loss is mean-squared error over
embedding dimensions, students, and groups:

\[
\boxed{
\mathcal{L}_{\mathrm{inv}}
=
\frac{1}{GSD}
\sum_{g=1}^{G}
\sum_{j=1}^{S}
\left\|\hat t_{g,j}-t_g\right\|_2^2
}.
\]

Gradients from \(\mathcal{L}_{\mathrm{inv}}\) update the online encoder
\(\theta\) through the student spectra and update the predictor \(\psi\).
They do not update the teacher target branch.

### Optional distribution regularization

Grouped JEPA can add the VisReg regularizer used by
[`configs/1b_grouped_jepa_visreg.py`](../configs/1b_grouped_jepa_visreg.py).
For each view \(v\), collect CLS embeddings over the \(B\) groups into
\(H_v\in\mathbb{R}^{B\times D}\). With distributed gathering enabled, \(B\)
is the global number of groups. Define

\[
\mu_v=\frac{1}{B}\sum_{b=1}^{B}H_{v,b},
\qquad
\sigma_{v,d}
=
\sqrt{
  \frac{1}{B}
  \max\left(
    \sum_{b=1}^{B}(H_{v,b,d}-\mu_{v,d})^2,
    10^{-12}
  \right)
}.
\]

The center and scale penalties are

\[
\mathcal{L}_{\mathrm{center}}
=
\operatorname{mean}_{v,d}\mu_{v,d}^2,
\qquad
\mathcal{L}_{\mathrm{scale}}
=
\operatorname{mean}_{v,d}(\sigma_{v,d}-1)^2.
\]

For the shape penalty, standardize with a stopped scale,

\[
\widetilde H_{v,b,d}
=
\frac{H_{v,b,d}-\mu_{v,d}}
     {\operatorname{sg}(\sigma_{v,d})},
\]

project onto \(K\) fixed random unit directions
\(r_k\in\mathbb{R}^{D}\), sort each projection across groups, and compare it
with standard-normal quantiles:

\[
q_b=\Phi^{-1}\!\left(\frac{b}{B+1}\right),
\qquad
\mathcal{L}_{\mathrm{shape}}
=
\frac{1}{V B K}
\sum_{v,b,k}
\left(
  \operatorname{sort}_{b}
  \langle\widetilde H_{v,b},r_k\rangle
  -q_b
\right)^2.
\]

The regularizer and total grouped-JEPA objective are

\[
\mathcal{L}_{\mathrm{visreg}}
=
\alpha_c\mathcal{L}_{\mathrm{center}}
+\alpha_s\mathcal{L}_{\mathrm{scale}}
+\alpha_h\mathcal{L}_{\mathrm{shape}},
\]

\[
\boxed{
\mathcal{L}_{\mathrm{group\text{-}JEPA}}
=
\lambda_{\mathrm{inv}}\mathcal{L}_{\mathrm{inv}}
+\lambda_{\mathrm{visreg}}\mathcal{L}_{\mathrm{visreg}}
}.
\]

Without an EMA teacher, VisReg is applied to the raw CLS embeddings of both
teacher and student spectra; unlike the stopped invariance target, those
teacher-view embeddings remain differentiable through VisReg. With an EMA
teacher, only the online student CLS embeddings enter VisReg.

### Current grouped-JEPA recipes

The base recipe in
[`configs/1b_grouped_jepa.py`](../configs/1b_grouped_jepa.py) uses

\[
R=8,
\qquad
T=3,
\qquad
S=5,
\qquad
G=256,
\qquad
m=0.9992,
\]

with \(\lambda_{\mathrm{inv}}=1\) and
\(\lambda_{\mathrm{visreg}}=0\). Its `lookahead` pipeline precomputes the
next batch's target before the current optimizer and EMA updates. Except for
the first batch, the target used for a batch is therefore one EMA update old;
this is reported by `teacher_target_age_steps = 1`.

The VisReg recipe disables the EMA teacher, uses same-step stopped targets from
the shared online encoder, and sets

\[
\lambda_{\mathrm{inv}}=0.9,
\qquad
\lambda_{\mathrm{visreg}}=0.1,
\qquad
K=256,
\qquad
\alpha_c=\alpha_s=\alpha_h=1.
\]

The principal logged losses are `group_jepa_loss` for
\(\mathcal{L}_{\mathrm{inv}}\), `group_jepa_invariance_term` for its weighted
term, `visreg_loss` for \(\mathcal{L}_{\mathrm{visreg}}\), and
`group_jepa_visreg_term` for its weighted term.
