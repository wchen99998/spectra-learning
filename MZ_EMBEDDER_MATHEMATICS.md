# Mathematical specification: Fourier vs. tokenized m/z

## Scope

This document specifies the two peak-input variants in
`configs/mz_token_ablation.py`. They have the same 52M-parameter PairMixer
backbone, masked prediction target, optimizer, data, masks, and output heads.
Only the per-peak m/z representation changes.

The comparison reported here uses:

1. the newest synchronized saved checkpoints, evaluated at the same optimizer
   step; and
2. the same-step validation curve and its component metrics.

It does not compare unmatched training batches. Those are noisy and the two
single-GPU jobs can differ by roughly one logging interval in wall-clock
progress.

## Notation and dimensions

For peak \(i\), let

- \(x_i\in[0,1]\) be the stored normalized m/z;
- \(m_i=Mx_i\) be m/z in Da, with \(M=1000\);
- \(a_i\in[0,1]\) be normalized peak intensity;
- \(D=448\) be the model dimension;
- \(D_m=D_r=224\) be the m/z-branch and raw-branch dimensions;
- \(H=896\) be the feature-MLP hidden dimension;
- \(\sigma(u)=u\,\operatorname{sigmoid}(u)\) be SiLU.

Both variants ultimately produce one vector \(e_i\in\mathbb{R}^{448}\) per
peak.

## Variant 1: continuous Fourier m/z

### Fixed Fourier features

The Fourier branch uses \(F=64\) wavelengths logarithmically spaced from
\(\lambda_{\min}=0.003\) Da to \(\lambda_{\max}=1000\) Da:

\[
\lambda_j
=
10^{
  \log_{10}\lambda_{\min}
  + \frac{j}{F-1}
    \left(\log_{10}\lambda_{\max}-\log_{10}\lambda_{\min}\right)
},
\qquad j=0,\ldots,F-1.
\]

The corresponding frequency is \(b_j=1/\lambda_j\). For a physical mass
\(m_i\), the fixed feature map is

\[
\phi_F(m_i)
=
\left[
  \cos(2\pi m_i b_0),\ldots,\cos(2\pi m_i b_{63}),
  \sin(2\pi m_i b_0),\ldots,\sin(2\pi m_i b_{63})
\right]
\in\mathbb{R}^{128}.
\]

These wavelengths are buffers, not learned parameters.

### Four-layer m/z network

The 128 Fourier features pass through a four-linear-layer MLP:

\[
\begin{aligned}
u_i^{(1)} &= \sigma(W_1\phi_F(m_i)+b_1),
&W_1&\in\mathbb{R}^{896\times128},\\
u_i^{(2)} &= \sigma(W_2u_i^{(1)}+b_2),
&W_2&\in\mathbb{R}^{896\times896},\\
u_i^{(3)} &= \sigma(W_3u_i^{(2)}+b_3),
&W_3&\in\mathbb{R}^{896\times896},\\
h_i^{F} &= W_4u_i^{(3)}+b_4,
&W_4&\in\mathbb{R}^{224\times896}.
\end{aligned}
\]

Thus \(h_i^F\in\mathbb{R}^{224}\). Every learned weight in this branch is
shared over the entire m/z range. Nearby masses receive related deterministic
inputs, while the shortest wavelengths retain sensitivity to fine mass
differences.

## Variant 2: flat tokenized m/z

### Discretization

The token width is \(\Delta_{\mathrm{in}}=0.02\) Da. The vocabulary size is

\[
V=\left\lceil\frac{M}{\Delta_{\mathrm{in}}}\right\rceil=50{,}000.
\]

The token ID is

\[
k_i
=
\operatorname{clip}
\left(
  \left\lfloor\frac{m_i}{0.02}\right\rfloor,
  0,
  49{,}999
\right).
\]

All masses in the half-open interval
\([0.02k,\,0.02(k+1))\) therefore have exactly the same per-peak m/z input.

### Factorized learned embedding

Let

\[
E\in\mathbb{R}^{50{,}000\times38}
\]

be a learned embedding table. The token branch is

\[
z_i=E_{k_i}\in\mathbb{R}^{38},
\qquad
h_i^T=Pz_i+c\in\mathbb{R}^{224},
\]

where \(P\in\mathbb{R}^{224\times38}\). This is a factorized,
ALBERT-style version of a BERT token embedding: the vocabulary table is
low-dimensional and a learned projection maps it to the branch width.

The rows are initialized independently:

\[
E_{k,d}\sim\mathcal{N}(0,0.02^2).
\]

There is no mathematical coupling between \(E_k\) and \(E_{k+1}\). Any
relationship between adjacent 0.02 Da bins must be learned indirectly from
data. The token branch contains no continuous residual within a bin.

## Raw intensity branch and final peak embedding

Both variants have the same two-layer raw branch, but its first input differs.
Define

\[
s_i^F=[x_i,\ a_i,\ \log(1+a_i)]
\]

for the Fourier model, and

\[
s_i^T=[0,\ a_i,\ \log(1+a_i)]
\]

for the token model. The zero is deliberate: it prevents continuous m/z from
leaking into the token peak embedder.

For \(v\in\{F,T\}\),

\[
r_i^v
=
R_2\,\sigma(R_1s_i^v+d_1)+d_2
\in\mathbb{R}^{224},
\]

with \(R_1\in\mathbb{R}^{896\times3}\) and
\(R_2\in\mathbb{R}^{224\times896}\).

The complete per-peak embedding is

\[
e_i^v
=
O
\begin{bmatrix}
h_i^v\\
r_i^v
\end{bmatrix}
+o
\in\mathbb{R}^{448},
\qquad
O\in\mathbb{R}^{448\times448}.
\]

All linear weights use Xavier-normal initialization and zero bias. The shared
raw branch, output projection, backbone, predictor, and output heads have
identical initial values in a paired seed.

## Parameter accounting

The Fourier-specific m/z branch has

\[
(128\cdot896+896)
+2(896^2+896)
+(896\cdot224+224)
=1{,}923{,}936
\]

trainable parameters.

The token-specific m/z branch has

\[
50{,}000\cdot38
+(38\cdot224+224)
=1{,}908{,}736
\]

trainable parameters.

The difference is 15,200 parameters:

| Quantity | Fourier | Token | Difference |
| --- | ---: | ---: | ---: |
| m/z-specific branch | 1,923,936 | 1,908,736 | -15,200 |
| Complete model | 52,044,810 | 52,029,610 | -15,200 |
| Estimated FLOPs / optimizer step | 159,881,656,320 | 159,834,961,920 | -0.029% |

Thus the comparison is parameter- and compute-matched at the complete-model
level. It is not depth-matched inside the m/z branch: Fourier uses a
four-layer shared MLP, whereas token uses a table and one linear projection.

## Shared pairwise pathway

The token model is free of continuous m/z only in its **per-peak embedder**.
Both variants retain the same pairwise mass pathway.

For peaks \(i,j\), define

\[
d_{ij}=m_j-m_i,\qquad
\delta_{ij}=|d_{ij}|,\qquad
\rho_{ij}=\frac{d_{ij}}{R},\qquad
c_{ij}=m_i+m_j-R,
\]

where \(R\) is precursor mass when present, otherwise the maximum valid peak
mass. The raw pair vector contains

\[
\left[
\frac{d_{ij}}M,\frac{\delta_{ij}}M,\rho_{ij},\frac{c_{ij}}M,
\frac{m_i}R,\frac{m_j}R,
a_i,a_j,a_ia_j,\log(1+a_i),\log(1+a_j),
\operatorname{sign}(d_{ij}),\mathbf{1}_{i=j},\mathbf{1}_{d_{ij}>0}
\right].
\]

It is augmented by 16-frequency Fourier features of \(d_{ij}\),
\(\delta_{ij}\), and \(c_{ij}\), plus 16-frequency relative Fourier features
of \(\rho_{ij}\). This gives 142 raw pair features.

Let \(g_{\mathrm{raw}}\) be the shared \(142\rightarrow448\rightarrow224\)
MLP. The pair embedder also uses the single-peak embeddings:

\[
q_{ij}
=
\left[e_i,e_j,e_i\odot e_j,e_j-e_i\right]\in\mathbb{R}^{1792},
\]

followed by a shared \(1792\rightarrow448\rightarrow224\) MLP
\(g_{\mathrm{single}}\). The initial pair representation is

\[
p_{ij}=g_{\mathrm{raw}}(\text{raw}_{ij})
      +g_{\mathrm{single}}(q_{ij}).
\]

The pair architecture and raw continuous/Fourier inputs are identical across
variants. The \(g_{\mathrm{single}}\) contribution naturally differs because
it consumes the different per-peak embeddings.

## Shared prediction target, loss, and accuracy

The masked m/z target uses a coarser
\(\Delta_{\mathrm{out}}=0.5\) Da grid:

\[
y_i
=
\operatorname{clip}
\left(
\left\lfloor\frac{m_i}{0.5}\right\rfloor,
0,1999
\right).
\]

There are therefore 2,000 output classes and 25 input token bins per output
class. Except at the upper clamp,

\[
y_i=\left\lfloor\frac{k_i}{25}\right\rfloor.
\]

For predictor output \(\hat z_{bvi}\in\mathbb{R}^{448}\), the shared linear
m/z head produces logits

\[
\ell_{bvi}=A_{\mathrm{mz}}\hat z_{bvi}+c_{\mathrm{mz}}
\in\mathbb{R}^{2000}.
\]

If \(T_{bvi}\in\{0,1\}\) denotes the target mask, the reported m/z loss is

\[
\mathcal{L}_{\mathrm{mz}}
=
\frac{
\sum_{b,v,i}T_{bvi}
\left[-\log\operatorname{softmax}(\ell_{bvi})_{y_i}\right]
}{
\sum_{b,v,i}T_{bvi}
}.
\]

Exact-bin top-1 accuracy is

\[
\operatorname{Acc}_{\mathrm{mz}}
=
\frac{
\sum_{b,v,i}T_{bvi}
\mathbf{1}\!\left[\arg\max_c\ell_{bvi,c}=y_i\right]
}{
\sum_{b,v,i}T_{bvi}
}.
\]

The MAE also predicts ten 0.1-wide intensity classes and a 2,000-class
0.5 Da pairwise distogram. With all configured weights equal to one,

\[
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{mz}}
+\mathcal{L}_{\mathrm{intensity}}
+\mathcal{L}_{\mathrm{distogram}}.
\]

The target definitions, masks, prediction heads, and loss weights are
identical for Fourier and token.

## How the live 100K run is compared

The live experiment runs seed 66 from scratch:

| Setting | Value |
| --- | ---: |
| Optimizer steps | 100,000 |
| Global batch | 512 |
| Spectra / model at completion | 51,200,000 |
| Optimizer | fused Adam, weight decay 0 |
| Learning rate | 3e-4 to 3e-5 cosine decay |
| Warmup | 250 steps |
| Validation interval | 2,500 steps |
| Validation size / model | 8,192 spectra |
| Checkpoint interval | 10,000 steps |

At an intermediate time, “latest” has two meanings:

- **latest synchronized saved checkpoint:** the greatest step for which both
  `step-XXXXXXXX.pt` files exist;
- **latest synchronized validation:** the greatest same-step validation row
  in both metric CSV files.

Checkpoint comparison should use the first definition and give both saved
models the same held-out spectra and masks. Curve comparison should use the
second definition. At completion, each `last.pt` will be evaluated on the
same larger paired holdout.

## Current training curve

Snapshot taken after both runs completed validation at step 15,000. Positive
accuracy delta means token is better; lower cross-entropy is better.

| Step | LR | Fourier m/z accuracy | Token m/z accuracy | Token − Fourier | Fourier m/z CE | Token m/z CE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,500 | 0.00029966 | 0.4516% | 0.4851% | +0.0335 pp | 6.9652 | 6.9628 |
| 5,000 | 0.00029849 | 0.7325% | 0.6180% | -0.1145 pp | 6.9296 | 6.9197 |
| 7,500 | 0.00029650 | 0.5978% | 0.5036% | -0.0941 pp | 6.8650 | 6.8608 |
| 10,000 | 0.00029369 | 0.6765% | 0.5076% | -0.1689 pp | 6.7670 | 6.8267 |
| 12,500 | 0.00029008 | 0.6798% | 0.6066% | -0.0732 pp | 6.7934 | 6.8118 |
| 15,000 | 0.00028569 | 0.9188% | 0.6523% | -0.2665 pp | 6.7388 | 6.7791 |

Other validation metrics:

| Step | Fourier total loss | Token total loss | Fourier intensity accuracy | Token intensity accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 2,500 | 13.9829 | 13.9365 | 83.269% | 85.817% |
| 5,000 | 13.8768 | 13.8444 | 85.587% | 86.669% |
| 7,500 | 13.8241 | 13.7826 | 83.518% | 85.493% |
| 10,000 | 13.6399 | 13.7053 | 86.185% | 86.719% |
| 12,500 | 13.7122 | 13.6969 | 84.011% | 86.010% |
| 15,000 | 13.5961 | 13.6098 | 85.438% | 87.311% |

The newest synchronized **saved checkpoint** is currently step 10,000. Its
scheduled validation favors Fourier on m/z accuracy and cross-entropy:

\[
\Delta\operatorname{Acc}_{\mathrm{mz}}
=0.5076\%-0.6765\%
=-0.1689\text{ percentage points},
\]

\[
\Delta\mathcal{L}_{\mathrm{mz}}
=6.8267-6.7670
=+0.0597.
\]

The newest synchronized **validation state** is step 15,000. It also favors
Fourier:

\[
\Delta\operatorname{Acc}_{\mathrm{mz}}
=0.6523\%-0.9188\%
=-0.2665\text{ percentage points},
\]

\[
\Delta\mathcal{L}_{\mathrm{mz}}
=6.7791-6.7388
=+0.0403.
\]

Token continues to predict intensity more accurately, so total loss can look
closer or occasionally favor token even when the requested m/z metric favors
Fourier. The run is still incomplete; the final checkpoint and complete curve
remain the decision point.

## Source locations

- Per-peak implementations: `spectra_learning/models/peak_features.py`
- Shared pair implementation: `spectra_learning/models/pairmixer.py`
- Target and losses: `spectra_learning/models/model.py`
- Canonical ablation config: `configs/mz_token_ablation.py`
- Live Fourier metrics:
  `experiments/mz_token_ablation_100k/seed66_fourier_100k_1xh100/csv_logs/metrics.csv`
- Live token metrics:
  `experiments/mz_token_ablation_100k/seed66_token_100k_1xh100/csv_logs/metrics.csv`
