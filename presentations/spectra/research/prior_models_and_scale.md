# Prior models and data scale: verified presentation notes

Verified 2026-09-18 from primary papers, official repositories, and existing local git objects. Do not present DiffMS as the current global state of the art: call it the 2025 reference / previous state-of-the-art baseline.

## DiffMS: two important corrections to the requested shorthand

DiffMS already adopts the **MIST formula transformer**. Its encoder is pretrained for Morgan fingerprint prediction, while its graph decoder is separately pretrained on fingerprint–structure pairs. Final supervised fine-tuning connects the spectrum encoder embedding to graph diffusion end to end. Consequently, a diagram should distinguish **fingerprint-supervised pretraining** from **the continuous spectrum embedding used to condition the final graph decoder**.

Given a formula, heavy-atom counts/types are fixed. The generated variable is the bond adjacency tensor, with five edge classes: absent, single, double, triple, aromatic. Hydrogen placement is implicit, so exact hydrogen counts are not guaranteed by the heavy-atom constraint.

Useful equations:

\[
X=X(F),\qquad A_{ij}\in\{\varnothing,1,2,3,\mathrm{aromatic}\},
\]
\[
p_\theta(A^0\mid S,F)=\sum_{A^{1:T}}p(A^T)\prod_{t=T}^{1}p_\theta(A^{t-1}\mid A^t,X(F),h_\phi(S)).
\]

Visual: fixed colored atom nodes on all three graph snapshots; only edges change from noise to structure. Plain English: “The formula supplies the pieces; diffusion learns how they connect.”

Sources: [DiffMS, §3.1–3.3](https://arxiv.org/html/2502.09571v2), [official implementation](https://github.com/coleygroup/DiffMS).

## MIST and the Novogaia autoregressive route

MIST turns peak formula annotations into fingerprints with a formula transformer that models pairwise neutral-loss relationships. Predicted fingerprints summarize possible substructures; they are not complete structures. Source: [Goldman et al., Nature Machine Intelligence 2023](https://www.nature.com/articles/s42256-023-00708-3), [official MIST repository](https://github.com/samgoldman97/mist).

**The requested autoregressive model exists on the `ar` branch**, despite the main README describing masked diffusion. The existing clone at `/home/wuhao/oracle-benchmark/md4` has remote `https://github.com/Novogaia/md4.git`; its local `origin/ar` is commit `e804f677c0a2a44cc5388cc303d1c9c51183091f`. Public and authenticated connector reads returned 404, but local git objects permitted direct verification without changing the checkout.

The model is JAX/Flax, uses causal attention, shifts targets for teacher-forced next-token prediction, ignores padding in cross entropy, and generates one token per step. Fingerprints enter through cross-attention after reshaping into a short memory sequence. The evaluation path can preserve a molecular-formula prefix and generate the separator and SMILES continuation. The Morgan configuration supports 4096-bit radius-2 fingerprints. Other configs support SIRIUS fingerprints, so avoid presenting every configuration as MIST-specific.

\[
\hat f=E_{\mathrm{MIST}}(S),\qquad
p_\theta(s\mid\hat f,F)=\prod_{t=1}^{L}p_\theta(s_t\mid s_{<t},\hat f,F),
\]
\[
\mathcal L_{\rm AR}=-\sum_{t=1}^{L}\log p_\theta(s_t\mid s_{<t},f,F).
\]

For mathematical accuracy, the implementation actually applies cross entropy to all non-padding sequence tokens, including the formula prefix during training; the equation above displays the conditional SMILES portion for communication.

Visual: spectrum → MIST → a strip of fingerprint bits; formula prefix below → causal decoder → growing `C`, `CC`, `CCO`. Use C2H6O / CCO as a simple illustrative output, not a claimed measured spectrum.

Comparative tradeoffs (reasoned architectural implications, not benchmark claims):

| Graph diffusion | Autoregressive SMILES |
|---|---|
| Native atom/bond representation and fixed heavy-atom inventory | Familiar next-token objective and efficient use of structure-only sequences |
| Repeated global denoising, quadratic edge representation | Sequential generation; early choices constrain later tokens |
| Heavy-atom formula guaranteed; chemical validity still requires checks | Formula prefix is conditioning, not hard atom-count enforcement |
| No arbitrary SMILES serialization | Several strings can encode the same graph; syntax and valence need validation |

Both routes can generate candidates absent from a retrieval library; both can suffer information loss or error propagation from fingerprint conditioning. Do not claim AR inference is faster without measurements: the checked AR sampler recomputes model logits in each loop and does not show KV caching.

Branch links: [AR implementation](https://github.com/Novogaia/md4/blob/ar/md4/models/diffusion/md4.py), [AR sampling](https://github.com/Novogaia/md4/blob/ar/md4/sampling.py), [formula/fingerprint inference](https://github.com/Novogaia/md4/blob/ar/eval_msg_sp.py). Local source snapshots are in `md4_ar_sources/` alongside this note.

## MassIVE scale and the DreaMS precedent

The **one-billion-plus repository opportunity** is supported by an official GNPS/MassIVE dataset reporting searches over approximately 1.2 billion public MS/MS spectra from 2,706 metabolomics projects. This is a repository-scale count, not evidence that Novogaia has already trained on all of them. [Official MassIVE dataset MSV000091795](https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?task=c4f2a2160dbd47148d703d2f519a103a).

DreaMS mined around 700 million spectra, filtered and controlled redundancy, and pretrained on GeMS-A10; its official tutorial reports 23,517,534 spectra. Its self-supervised tasks predict masked spectral peaks and chromatographic retention order. Therefore describe the contrast as **richer acquisition metadata plus latent prediction**, not “DreaMS uses no metadata.” Its 201-million-spectrum Atlas is distinct from its pretraining subset.

Sources: [DreaMS primary paper](https://www.nature.com/articles/s41587-025-02663-3), [official repository](https://github.com/pluskal-lab/DreaMS), [official GeMS-A10 size documentation](https://dreams-docs.readthedocs.io/en/latest/tutorials/msdata.html).

Visual: large reservoir `1B+ raw spectra` → quality / deduplication funnel → unlabeled pretraining → small labeled head. Label the first quantity “available opportunity,” and show a separate annotation “DreaMS precedent: 23.5M selected training spectra.” Do not conflate raw acquisitions, filtered spectra, distinct molecules, or actual samples used by a given Novogaia checkpoint.
