# Evidence for the 44.10% retrieval presentation

Checked against the completed canonical evaluation and maintained recipe on **24 September 2026**. This is local experimental evidence, not a literature review. The earlier presentation and its 36.94% result remain historical artifacts.

## Authoritative sources

1. [Complete mathematical recipe](../../../docs/retrieval_recipe.md): implemented architecture, equations, inputs, training, limitations and reproduction.
2. [Final canonical results](../../../results/retrieval_selected_4602_20260924/canonical_test/attempt2/results.json): completed held-out metrics and MCES.
3. [Full-test comparison](../../../results/retrieval_selected_4602_20260924/canonical_test/attempt2/comparison.json): every saved rank and top prediction agrees with the original-runtime evaluation.
4. [Original full-test results](../../../results/retrieval_selected_4602_20260924/results.json): hit counts, optional compound-weighted metrics, frozen-baseline comparison and MCES protocol.
5. [Historical 32.47% metrics](../../../results/retrieval_selected_4602_20260924/historical_gine_qk_3247/metrics.json) and [manifest](../../../results/retrieval_selected_4602_20260924/historical_gine_qk_3247/manifest.json): original wider GINE + Q/K and its JEPA source.
6. [Historical 36.94% metrics](../../../results/atom_flare_gine3m_qk_global_glacier_s0_20260918/run/metrics.json), [manifest](../../../results/atom_flare_gine3m_qk_global_glacier_s0_20260918/run/manifest.json) and [existing-deck evidence](../../spectra/research/latest_retrieval_result.md): the previous Global + GLACIER result.
7. [Selected encoder metadata](../../../results/retrieval_selected_4602_20260924/selected_encoder_metadata.json), [selected manifest](../../../results/retrieval_selected_4602_20260924/selected_manifest.json) and [training progress](../../../results/retrieval_selected_4602_20260924/selected_progress.json): selected model and training lineage.
8. [Canonical evaluation MCES protocol](../../../results/retrieval_selected_4602_20260924/canonical_test/attempt2/full_results/mces_protocol.json) and [MCES summary](../../../results/retrieval_selected_4602_20260924/canonical_test/attempt2/full_results/mces_summary.json).
9. [Canonical runner](../../../scripts/run_retrieval_jax.py), [fixed input paths](../../../configs/retrieval_4602.json) and [cleanup record](../../../results/retrieval_selected_4602_20260924/cleanup_report.md).

## Exact headline and test statistics

Selected recipe: `crossattn_warm_joint`, human joint epoch **2** / stored epoch index **1**, selected from **8 completed joint epochs** using strict improvement in optimistic validation Hit@1. Stable and optimistic selected metrics agree. The checkpoint was chosen before the held-out evaluation.

| Quantity | Exact value | Slide rounding |
|---|---:|---:|
| Validation spectra | 5,000 | 5,000 |
| Validation Hit@1 | 0.4602 = 2,301 correct | 46.02% |
| Test spectra | 17,556 | 17,556 |
| Test candidate pairs | 2,909,859 | 2.91M |
| Test Hit@1 | 0.4410457963089542 = 7,743 correct | 44.10% |
| Test Hit@2 | 0.5763841421736159 = 10,119 correct | 57.64% |
| Test Hit@5 | 0.7357028935976304 = 12,916 correct | 73.57% |
| Test Hit@10 | 0.835725677830941 = 14,672 correct | 83.57% |
| Test Hit@20 | 0.9223057644110275 = 16,192 correct | 92.23% |
| Test MRR | 0.5750935334110906 | 0.5751 |
| Test median rank | 2 | 2 |
| Test mean rank | 7.377933470038733 | 7.38 |
| Test mean MCES@1 | 4.954887218045113 | 4.9549 |
| Test median MCES@1 | 2 | 2 |
| MCES failures | 0 | 0 |

The primary unit is **one spectrum**: all 17,556 spectra have equal weight. Test candidates are the complete original supplied truth-containing formula pools, in original order. Stable and optimistic ranks agree, and there are no missing candidate entries. Candidate pools contain at most 256 molecules; do not state that all pools contain 256.

The optional **57.37692616098685% compound-weighted Hit@1** averages correctness within each InChIKey14 group and then over 2,998 groups. It is not the primary test headline and cannot be compared directly with spectrum-weighted 32.47%, 36.94% or 44.10%.

MCES uses `myopic-mces 1.2.2`, `pulp 3.3.2`, RDKit `2026.3.5`, threshold 15 and CBC with one solver thread, no time limit and `catch_errors=False`. Values below threshold are exact; values above may be certified lower bounds. Do not claim every distance is unrestricted exact MCES, clip above 15, or equate zero MCES with the benchmark's exact SMILES identity rule.

## Historical comparison: matched denominator, combined changes

| Historical model | Validation Hit@1 | Test Hit@1 | Test Hit@5 | Test MRR | Median rank |
|---|---:|---:|---:|---:|---:|
| Wider GINE + asymmetric Q/K | 33.10% | 32.473228525860104% | 64.29141034404192% | 0.4690802948862312 | 3 |
| Global context + GLACIER, prior presentation | 38.06% | 36.94463431305536% | 67.06539074960127% | 0.5081385844610372 | 2 |
| **Selected recipe** | **46.02%** | **44.10457963089542%** | **73.57028935976304%** | **0.5750935334110906** | **2** |

All test rows in this table are spectrum weighted with **17,556 observations**. Historical wider GINE + Q/K had 5,701 correct; the selected model has 7,743: **+2,042 spectra, +11.631351105035314 percentage points**. From the prior presentation's 36.94% model, the difference is **+7.159945317840055 percentage points**. These comparisons describe accumulated architecture/modelling/training changes; they are **not** per-component ablations. Do not plot each added component as an independently measured accuracy increment.

The later frozen-encoder uniform baseline in the selected audit is a different model from the prior 36.94% run. It has 42.45272271588061% test Hit@1 and MCES@1 4.977472089314195. Relative to it, the selected model changes Hit@1 by +1.65185691501481 percentage points and MCES by −0.02258487126908193. If used, name this baseline explicitly; do not relabel it as the original wider GINE or the previous presentation's model.

## Which changes are new relative to which baseline?

| Component | Historical 32.47% | Previous presentation 36.94% | Selected 44.10% |
|---|---|---|---|
| Original JEPA student at step 1,190,000 | Yes, frozen | Yes, frozen | Same starting checkpoint; jointly adapted after head training |
| Raw 2,048D tokens → learned 256D MLP; no PCA | Yes | Yes | Yes |
| Three-layer width-496 GINE | Yes | Yes | Yes |
| Asymmetric Q/K256 | Yes | Yes | Yes |
| Whole-molecule mean MLP broadcast | No | Yes | Yes |
| Learned GLACIER fragment tokens and gate | No; atom tokens | Yes | Yes |
| Eight formula negatives | First 16 eligible bank | First 16 eligible bank | Uniform over entire eligible supplied pool |
| Physical/readout changes | Fractional mass + atom MaxSim | Fragment gated MaxSim | Precise-ion bonus + forward log-mixture + gated reverse maximum |
| Candidate-conditioned 32D cross-attention | No | No | Yes |
| Warm joint JEPA/head adaptation | No | No | Yes |

The original JEPA source is run `1p8b-ftjepa-t1p8b-s666k-perleaf-b1024-v6e32-20260905-115335`, student step **1,190,000**. It is **not** the later periodic-teacher checkpoint and **not** a pair-stream model. The inherited directory names contain historical labels and do not imply a MACCS loss or new encoder source.

The prior 36.94% reduction is verified directly in its [preserved scorer source](../../../results/atom_flare_gine3m_qk_global_glacier_s0_20260918/source/spectra_learning/probes/massspec/glacier_fragments_jax.py):

\[
S^{\rm old}_{pf}=q_p^\top k_f+e^{\theta_\beta}\exp[-((r_p-\rho_f)/e^{\theta_\sigma})^2],
\qquad
F_{\rm old}(s,G)=\tfrac12\left[\frac1{|P_s|}\sum_p\max_fS^{\rm old}_{pf}+\sum_f\pi_f\max_pS^{\rm old}_{pf}\right].
\]

It used a learned fractional-mass Gaussian with initial sigma 0.01 and strength 0.5, without hydrogen-shift tolerance or the new factor 1.5. The fragment gate entered only the reverse maximum term. Its forward hard maximum had no mixture temperature. Therefore “changed aggregation” includes both the new prior-weighted forward log-mixture and the 0.5/0.5 → 0.65/0.35 reduction weights; do not portray either as an isolated controlled experiment.

## Architecture and equation facts for diagrams

- JEPA: 25 layers, width 2,048, eight heads, 28D acquisition metadata, AdaLN-zero conditioning; retrieval uses up to 127 physical-token outputs and drops final CLS. Retrieval remains JAX/Flax with a direct training loop.
- GINE: three residual message-passing layers, width 496, epsilon fixed zero; 49 atom features and 8 bond features; real-atom masked BatchNorm statistics update in both training phases.
- Global context: \(\tilde h_a=h_a^3+L_2\operatorname{ReLU}(L_1\bar h_G)\), \(\bar h_G=|V_G|^{-1}\sum_ah_a^3\). It is a broadcast molecule-wide summary, not an additional long-range pair-attention layer or a graph-distinguishability guarantee.
- Fragment proposals: at most 128 including intact anchor; fixed offline GLACIER memberships, at least three atoms and mass fraction at least 0.05 for predicted fragments. Pool all real atoms, including atoms beyond the old 96-token readout cap. Member and cut-boundary means each contribute 496 dimensions.
- Fragment input \(c_f\in\mathbb R^{1010}\): member mean; boundary mean (zero if empty); cut-count embedding8; kind embedding8; mass fraction; \(\log|V_f|\). A normalized MLP produces \(z_f\in\mathbb R^{256}\). Gate \(g_f=L[c_f;I_f;\log(1+I_f)]\), prior \(\pi_f=\operatorname{softmax}_fg_f\). Intensity enters only the gate; no induced-subgraph GNN per fragment.
- Peak MLP: layer normalization → linear2048→256 → exact GELU/dropout0.1 → linear256→256 → unit normalization. No PCA.
- Q/K: \(q_p=p_pW_Q\), \(k_f=z_fW_K\), separate bias-free 256×256 identity-initialized maps. No post-Q/K normalization, and no \(1/\sqrt{256}\) multiplier in their dot product.
- Base matrix: \(S_{pf}=q_p^\top k_f+B_{pf}\). The physical bonus uses a fixed 0.01 Da Gaussian ion-mass residual, rounded/clipped hydrogen shifts in \([-2,2]\), observed protonated/sodiated adduct and a half-weight protonated alternative. Initial strength \(1.5e^{\theta_\beta}\) with \(e^{\theta_\beta}=0.5\). At selection \(e^{\theta_\beta}\approx1.6802522\). The retained-fraction fallback is inactive for all recorded spectra.
- Cross-attention: project and RMS-normalize \(u_p=\operatorname{RMS}(q_pU)\), \(v_f=\operatorname{RMS}(k_fV)\), each 32D. Then \(T_{pf}=\operatorname{softmax}_f(S_{pf}+\log\pi_f)\), \(c_p=\sum_fT_{pf}v_f\), \(b_p=\tanh[u_pA_q+c_pA_c+(u_p\odot c_p)A_x+b]\), \(\delta_{pf}=(b_pO)^\top v_f/\sqrt{32}\), and \(\widehat S=S+\delta\). The softmax temperature is 1. Fresh adapter output matrix \(O\) starts at zero; joint adaptation restores the trained adapter. Total adapter parameters 20,512.
- Sole final score:

\[
F(s,G)=0.65\,\frac1{|P_s|}\sum_p0.1\log\sum_f\pi_f\exp(\widehat S_{pf}/0.1)
+0.35\sum_f\pi_f\max_p\widehat S_{pf}.
\]

  Forward peaks are uniformly weighted. The same fragment prior appears inside the forward mixture, in candidate cross-attention, and as reverse weights. There is no independent score fusion, fingerprint head, pair-stream input, ensemble or test-time reranker.

## Training facts and the finetuning limitation

Dataset: MassSpecGym 1.5; 194,119 training rows, 19,429 validation rows, 17,556 test rows. Model selection uses the fixed sorted seed-0 subset of 5,000 validation rows. Eight formula negatives are sampled without replacement from the full eligible training pool, refreshed at each visit; use all if fewer than eight exist. Exclude exact supplied truth SMILES for formula negatives. Off-diagonal in-batch masks use InChIKey14. Do not claim these are identical identity criteria.

The objective is symmetric InfoNCE with \(\tau=0.05\): spectrum→molecule has in-batch candidates plus sampled formula negatives; molecule→spectrum has in-batch candidates only, with reverse weight 1. This is not exact full-pool CE. No hard mining is used.

| Phase/comparison | Frozen JEPA? | Head LR | JEPA LR | Selected validation Hit@1 |
|---|---|---:|---:|---:|
| Fresh complete retrieval head, selected human epoch13 | Yes | 5e−4 | — | 45.04% |
| Matched head-only continuation, selected epoch2 | Yes | 1e−5 | — | 45.94% |
| Selected joint continuation, selected epoch2 | No | 1e−5 | 3e−6 | 46.02% |

Shared seed 0, batch 128, dropout 0.1. Adam has no schedule, weight decay or clipping. Head phase maximum 60 epochs / patience 8; joint maximum 8 epochs / patience 8. Joint starts fresh Adam moments but preserves trained head and GINE BN state. JEPA uses bf16 compute and per-layer activation checkpointing; parameters/moments are fp32. The raw token boundary remains fp16→fp32. Joint adaptation is not from-scratch pretraining.

The matched head-only and joint selected validation values differ by **0.08 percentage points / four spectra**. This is a single-seed comparison; do not present a reliable finetuning gain, a confidence interval or a causal explanation for the full 32.47→44.10 improvement. No held-out scores for the 45.04/45.94 checkpoints are reported here.

Head parameters **4,121,707**; trainable JEPA parameters **1,516,340,736**; joint total **1,520,462,443**. The adapter is 32D, but the main Q/K score is 256D. The historical 32.47% head had 3,103,586 parameters; old 36.94% head had 4,101,195. Head parameter growth is another change, not a controlled constant.

## Provenance and limitations to preserve

- Selected checkpoint SHA-256: `324679e57439fd3294cce1cc379f1332a197598b1a0a15fe6fa5b35c7a6f6b99`.
- Final canonical Modal app: `ap-GGtsFE4jAulaC5OU12stT8`; 302 pinned source files. Final canonical evaluation reproduces every stable/optimistic rank and top prediction over the full test set. It performs no training.
- The [completed test audit](../../../results/retrieval_selected_4602_20260924/canonical_test/attempt2/test_audit.json) pins two archive representations of the same arrays: native archive `963d80ee5a504b469ee128da4d490223613d9c118b672606fe835d1c9f81e7ef` and compact compressed archive `7113945ac79dc3c516953ff8c99ef5d9831354d7d94708869d16341e8e2211f0`. Their distinct byte hashes reflect archive representation, not different ranks or stale evidence. Original-runtime and canonical evaluations agree in each representation.
- Results are exact saved-rank/prediction reproduction, not a universal GPU bitwise-score determinism guarantee. Earlier small probes and failed strict tolerances remain in the [numerical audit](../../../results/retrieval_selected_4602_20260924/canonical_parity/equivalence_audit.json).
- GLACIER proposals were regenerated because the upstream full-fragment dump was unavailable. Pinned generator commit `67ba0f202305d4bed3134991dd4562fe1edb1763`, checkpoint SHA `5a47cecca707d3abd5a49c7dbac99d100aa2a586d5d4f848f1e8e35140d7db11`, fixed [M+H]+/Orbitrap and collision energies 20/30/40. Exact equality with the missing upstream dump and proposal-model training-set disjointness have not been established. The selected full cache has 43 intact-only molecules including 20 unsupported inputs; do not reuse the old 36.94% cache's smaller counts.
- This recipe uses supplied formula pools containing the truth. It is not open-world retrieval, de novo structure generation, or a test of a candidate-generation step.
- Diagrams are explanatory and synthetic unless directly plotting the tabulated metrics. Label peak/fragment matrix axes and any illustrated attention weights. Spectral traces must not imply experimentally measured matches to illustrated molecules. Historical bars require the same spectrum-weighted test label; validation belongs in a separate panel.
