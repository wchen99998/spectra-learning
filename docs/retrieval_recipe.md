# Selected retrieval recipe: Global + GLACIER + candidate cross-attention

This is the recipe selected as **`crossattn_warm_joint`: 46.02% validation Hit@1**, or **2,301 / 5,000 spectra**, at joint-adaptation epoch **2** (stored epoch index `1`). It combines a JEPA spectrum encoder with a GINE molecular encoder and learned GLACIER fragment tokens, and produces **one compatibility score per spectrum–molecule pair**. Retrieval training and inference use JAX/Flax and a direct training loop.

The selected model has **1,520,462,443 trainable parameters**: **1,516,340,736** in JEPA and **4,121,707** in the retrieval head. The main query/key space is **256D**; the candidate-conditioned cross-attention adapter is **32D**.

This document describes this selected recipe only. Its completed held-out evaluation gives **44.1046% test Hit@1 (7,743 / 17,556)** and **MCES@1 4.9548872**, with no failed MCES queries. The 46.02% figure identifies the selected validation checkpoint. The [test audit](../results/retrieval_selected_4602_20260924/test_audit.json) and [independent audit](../results/retrieval_selected_4602_20260924/independent_test_audit.json) confirm the original full test population, candidate counts and checkpoint identity.

## Recorded results and what changed

| Model/checkpoint | Fixed 5,000-spectrum validation Hit@1 | Full 17,556-spectrum test Hit@1 |
|---|---:|---:|
| Historical wider GINE + Q/K, selected epoch 22 | 33.10% | 32.4732% (5,701 correct) |
| This architecture, freshly trained head with frozen JEPA, selected epoch 13 | 45.04% | Not reported here |
| Matched head-only continuation, selected continuation epoch 2 | 45.94% | Not reported here |
| **Selected joint continuation, epoch 2** | **46.02%** | **44.1046% (7,743 correct)** |

The selected validation checkpoint also has Hit@5 **71.70%**, Hit@10 **80.84%**, Hit@20 **90.86%**, MRR **0.5791552117**, and median rank **2**. Optimistic and stable aggregate metrics agree. The joint and matched head-only continuations differ by **0.08 percentage points: four correctly ranked validation spectra**. This single-seed comparison does not establish a reliable benefit from encoder finetuning.

| Selected model, spectrum-level test | Result |
|---|---:|
| Spectra / scored candidate pairs | 17,556 / 2,909,859 |
| Hit@1 / Hit@2 | 44.1046% / 57.6384% |
| Hit@5 / Hit@10 / Hit@20 | 73.5703% / 83.5726% / 92.2306% |
| MRR | 0.5750935334 |
| Median / mean rank | 2 / 7.3779334700 |
| MCES@1 mean / median | 4.9548872180 / 2 |

Stable and optimistic test ranks agree. The separately computed compound-weighted Hit@1 is **57.3769% across 2,998 InChIKey14 groups**; it must not be compared directly with the spectrum-weighted 44.1046%. Saved [test ranks](../results/retrieval_selected_4602_20260924/run/test_ranks.npz), [MCES summary](../results/retrieval_selected_4602_20260924/full_results/mces_summary.json) and [per-query MCES](../results/retrieval_selected_4602_20260924/full_results/mces_per_query.npz) support these figures. The MCES artifact labels this selected model `global_glacier`.

The final maintained CLI was also run over the entire test set after cleanup. It reproduced **every saved rank and top prediction**, with the same SHA-256 for the complete rank archive and identical aggregate metrics. This verifies the deployed pipeline on all 17,556 spectra, independently of the small numerical checks described below. The [canonical evaluation and comparison](../results/retrieval_selected_4602_20260924/canonical_test/attempt2/comparison.json) record the source snapshot, invocation and checkpoint identity; the completed Modal app is `ap-GGtsFE4jAulaC5OU12stT8`.

Relative to the historical 32.47% recipe ([retained metrics](../results/retrieval_selected_4602_20260924/historical_gine_qk_3247/metrics.json)), the substantive changes are:

1. Add a learned whole-molecule mean context to each atom representation.
2. Replace individual atom retrieval tokens with learned GLACIER fragment tokens, including boundary information and an intact-molecule anchor.
3. Replace the atom fractional-mass/MaxSim readout with the precise-ion bonus, learned fragment prior, forward log-mixture and reverse gated maximum specified below.
4. Sample eight formula negatives from the **entire eligible supplied pool**, instead of sampling eight from the first sixteen eligible candidates.
5. Add a 32D candidate-conditioned cross-attention residual to the compatibility matrix.
6. After training the retrieval head, jointly adapt the original JEPA encoder and head with separate learning rates.

The raw-token learned projection and asymmetric 256D Q/K projections were already present in the 32.47% model. Both start from the same JEPA student at step 1,190,000, as verified in the [historical manifest](../results/retrieval_selected_4602_20260924/historical_gine_qk_3247/manifest.json). The table is a historical progression, not a controlled attribution of gains to these changes. In particular, **32.47% test and 46.02% validation are not directly comparable**.

## Inputs, identity and fixed preprocessing

Use MassSpecGym **1.5**, dataset `roman-bushuiev/MassSpecGym` at revision `d2e86d0c3bd905a6d578c0dd6053ed2bd41f9c2a`. Its 231,104 spectra comprise **194,119 training**, **19,429 validation**, and **17,556 test** rows. Model selection uses the persisted, sorted **5,000-row validation subset**, sampled with seed 0. Its row-array SHA-256 is `0959c5bbedf2c7efee626c49cf279d869edd20470297bf67420437ca5da0106b`.

The authoritative spectral input is the existing `massspecgym_maccs_mz_order_metadata_n128_v2/prepared_massspecgym.npz`; the directory's historical MACCS name does not imply a fingerprint loss. The selected retrieval recipe does not use MACCS targets. Keep its row order, physical-peak columns, valid masks and acquisition metadata unchanged. Preprocessing normalizes intensities by the spectrum maximum, retains relative intensities **≥0.0001** and m/z **20–1,000 Da**, applies **no precursor-exclusion window**, retains up to **127** peaks by intensity, renormalizes retained intensities, then sorts valid peaks by m/z. m/z and precursor m/z are scaled by 1,000. The checkpoint metadata supplies the preprocessing configuration; do not substitute current defaults when rebuilding the archive.

The original JEPA **student at step 1,190,000**, run `1p8b-ftjepa-t1p8b-s666k-perleaf-b1024-v6e32-20260905-115335`, receives the physical peaks plus **28D acquisition metadata**. Its saved configuration specifies **25 layers, width 2,048, eight attention heads**, and `adaln_zero` metadata conditioning with condition dimension 256 and schema `massive_v2_acquisition_v1`. The complete architecture configuration is retained in [selected_encoder_metadata.json](../results/retrieval_selected_4602_20260924/selected_encoder_metadata.json). It uses its CLS token internally; retrieval discards the final CLS and uses the **127 × 2,048** physical-token outputs. This is the original single-stream encoder, not the later periodic-teacher or pair-stream checkpoint. Raw 2,048D tokens enter a learned MLP; **there is no PCA**.

During frozen-head training those tokens are the persisted float16 cache. During joint training they are recomputed by the trainable encoder with bfloat16 compute, cast through **float16 → float32**, then passed to the same readout. These casts preserve the original cache boundary and remain differentiable. Parameters and Adam state are float32; the runner sets `jax_default_matmul_precision="highest"` for float32 matrix products. Fourier-frequency buffers named `b` remain fixed. Every one of the 25 transformer blocks received nonzero finite first-update gradients in the recorded run. The online input container has shape `[B,127,32]` for spectral values and metadata; its last dimension is **not** the JEPA embedding dimension.

Molecule identity follows the existing benchmark paths exactly:

- Ground truth and formula-negative exclusion use **exact supplied SMILES**. Candidate order is the original supplied formula-pool order.
- Off-diagonal in-batch exclusions use **InChIKey14**. These are different criteria; this recipe does not merge candidate stereoisomers or tautomers.
- The prepared graph universe contains **5,304,587 molecules**. Training negative pools exclude exact truth and have 0–255 entries. Evaluation uses the complete supplied pools, at most 256 candidates, with no missing entries or candidate truncation.

The fixed validation set contains **817,641 spectrum–candidate pairs**; the completed full test contains **2,909,859**. These counts, row identities and candidate order must agree before comparing scores.

## Molecular encoder and fragment tokens

All affine maps below include a bias unless explicitly stated. Let `LN` use epsilon \(10^{-5}\), `Drop` have probability 0.1 during training, and

\[
\operatorname{unit}(x)=x/\sqrt{\max(\sum_d x_d^2,10^{-24})}.
\]

**Graph inputs.** RDKit parses the original SMILES without adding explicit hydrogen nodes or 3D coordinates. Each bond gives two directed edges; there are no added self-edges. The **49 atom features** are one-hot element, degree, formal charge, hydrogen count, hybridization and chirality, plus aromaticity, ring membership and atomic mass/100. Category sizes are 15, 8, 6, 6, 7 and 4 respectively, including unknown categories. The **8 bond features** are five bond-type categories plus conjugation, ring membership and nonzero stereo. The frozen graph-preparation source pins the exact vocabularies and atom order.

**Three residual GINE updates, width 496.** For atom \(a\), directed edge \(u\to a\), and layer \(\ell=0,1,2\):

\[
h_a^0=L_{49\to496}(x_a),\qquad e_{ua}=L_{8\to496}(b_{ua}),
\]
\[
m_a^\ell=\sum_{u\to a}\operatorname{ReLU}(h_u^\ell+E_\ell(e_{ua})),
\]
\[
h_a^{\ell+1}=h_a^\ell+
\operatorname{Drop}\!\left[\operatorname{ReLU}\!\left(
\operatorname{BN}_\ell\!\left(L_{\ell,2}\operatorname{ReLU}
  \left(L_{\ell,1}(h_a^\ell+m_a^\ell)\right)\right)\right)\right].
\]

The edge maps and both MLP maps are 496→496. GINE epsilon is fixed at zero. BatchNorm excludes padded nodes, uses population variance for training normalization, epsilon \(10^{-5}\), and running-stat momentum 0.1 with unbiased variance correction. **Its running statistics update during both training phases** and are restored for evaluation.

**Global context.** Pool every real atom in the molecule, transform that mean, and broadcast-add it:

\[
\bar h_G=|V_G|^{-1}\sum_a h_a^3,\qquad
\tilde h_a=h_a^3+L_2\operatorname{ReLU}(L_1\bar h_G).
\]

Both maps are 496→496. There is no normalization of \(\tilde h\) before fragment pooling. This gives each atom access to a pooled whole-molecule context; it is not a guarantee of distinguishing every graph topology. The GLACIER path uses **all real atoms**: the legacy `eval_atom_cap=96` does not truncate fragment memberships or the global mean. The prepared maximum is 134 atoms, within the configured 160-atom capacity.

**Fixed GLACIER proposals.** Each molecule has up to **128 fragments including the intact-molecule anchor**. Predicted atom sets are deduplicated, kept in supplied decreasing-intensity order, and filtered to at least three atoms and retained monoisotopic mass fraction at least 0.05. The anchor is always first, with fraction 1, no cut boundary, zero cuts, and intensity 1. For fragment \(f\), \(B_f\subseteq V_f\) contains member atoms bonded to atoms outside the fragment. Its cut count is the number of crossing bonds, capped at 7. Canonical proposal indices are remapped to the graph's original RDKit atom order.

The missing upstream full-fragment dump was regenerated offline with GLACIER commit `67ba0f202305d4bed3134991dd4562fe1edb1763`, checkpoint SHA-256 `5a47cecca707d3abd5a49c7dbac99d100aa2a586d5d4f848f1e8e35140d7db11`, using [M+H]+, Orbitrap and collision energies 20/30/40. Atom sets are united across predictions and receive the maximum predicted intensity. Unsupported proposal inputs retain the intact anchor. The final cache has 43 intact-only molecules, including 20 unsupported inputs. Exact equality to the unavailable upstream dump and the proposal model's training-set disjointness have not been established. Proposal generation is offline; its weights and memberships do not train with retrieval.

**Learned fragment representation.** With empty boundary means defined as zero,

\[
c_f=\left[
|V_f|^{-1}\sum_{a\in V_f}\tilde h_a;
|B_f|^{-1}\sum_{a\in B_f}\tilde h_a;
\operatorname{Embed}_8(\min(\mathrm{cuts}_f,7));
\operatorname{Embed}_8(\mathrm{kind}_f);
\rho_f;\log|V_f|
\right]\in\mathbb R^{1010}.
\]

Kind 0 is intact and 1 is predicted; the embedding table has four entries. \(\rho_f=M_f/M_G\) uses the same cached monoisotopic atom masses, including implicit hydrogens, as the physical score. Then

\[
z_f=\operatorname{unit}\!\left(L_{496\to256}\operatorname{Drop}
  \left[\operatorname{GELU}_{\rm exact}
  \left(L_{1010\to496}\operatorname{LN}(c_f)\right)\right]\right),
\]
\[
g_f=L_{1012\to1}[c_f;I_f;\log(1+I_f)],\qquad
\pi_f=\operatorname{softmax}_{f\in\text{valid fragments}}(g_f).
\]

Predicted intensity \(I_f\) enters **only the gate**, not \(z_f\). The fragment vectors pool intact/global atom states; this recipe does not run an induced-subgraph GNN inside each fragment.

## Spectrum–candidate compatibility and the sole retrieval score

For a physical JEPA peak token \(t_p\in\mathbb R^{2048}\),

\[
p_p=\operatorname{unit}\!\left(L_{256\to256}\operatorname{Drop}
  \left[\operatorname{GELU}_{\rm exact}
  \left(L_{2048\to256}\operatorname{LN}(t_p)\right)\right]\right),
\qquad q_p=p_pW_Q,\quad k_f=z_fW_K.
\]

\(W_Q,W_K\in\mathbb R^{256\times256}\) are separate, bias-free, identity-initialized matrices. Their outputs are **not** renormalized, and \(q_p^\top k_f\) has no \(1/\sqrt{256}\) scaling.

**Physical ion bonus.** Let \(x_p\) be the observed peak m/z in Da, \(H=1.00782503207\), \(a_H=H-0.000548579909065\), and \(a_s\) the observed precursor-adduct mass: \(a_H\) for [M+H]+, or \(22.989769282-0.000548579909065\) for [M+Na]+. Define

\[
\Delta_{pf}(a)=x_p-(M_G\rho_f+a),\qquad
j_{pf}(a)=\operatorname{clip}(\operatorname{rint}(\Delta_{pf}(a)/H),-2,2),
\]
\[
K_{pf}(a)=\exp\!\left[-\left(\frac{\Delta_{pf}(a)-j_{pf}(a)H}{0.01\ {\rm Da}}\right)^2\right].
\]

For eligible [M+H]+ / [M+Na]+ spectra,

\[
B_{pf}=1.5\,e^{\theta_\beta}\max\{K_{pf}(a_s),0.5K_{pf}(a_H)\},
\qquad S_{pf}=q_p^\top k_f+B_{pf}.
\]

Unsupported adducts use the preserved fallback \(B_{pf}=1.5e^{\theta_\beta}\exp[-((r_p-\rho_f)/e^{\theta_\sigma})^2]\), where \(r_p\) is the aligned cached peak retained fraction. **All 231,104 rows in this run are eligible**, so the fallback is inactive. Initial \((e^{\theta_\sigma},e^{\theta_\beta})=(0.01,0.5)\); at the selected checkpoint they are approximately \((0.01,1.6802522)\). The 0.01 Da ion width is fixed, independently of the fallback sigma.

**Candidate-conditioned cross-attention residual, width 32.** All attention normalizations exclude padded fragments. Let \(\operatorname{RMS}(x)=x/\sqrt{\operatorname{mean}(x^2)+10^{-6}}\). For each particular candidate molecule,

\[
u_p=\operatorname{RMS}(q_pU),\quad v_f=\operatorname{RMS}(k_fV),\quad
T_{pf}=\operatorname{softmax}_f(S_{pf}+\log\pi_f),\quad
c_p=\sum_f T_{pf}v_f,
\]
\[
b_p=\tanh\!\left(u_pA_q+c_pA_c+(u_p\odot c_p)A_x+b\right),
\qquad \delta_{pf}=\frac{(b_pO)^\top v_f}{\sqrt{32}},\qquad
\widehat S_{pf}=S_{pf}+\delta_{pf}.
\]

Here \(U,V\) are 256×32; \(A_q,A_c,A_x,O\) are 32×32; \(b\) is 32D. The attention temperature is **1.0**. Fresh initialization uses Glorot for \(U,V,A_q,A_c,A_x\) and zeros for \(O,b\), so it initially preserves the base matrix. Warm joint training restores the **already trained** adapter; it does not zero it again. There are **20,512 adapter parameters**. The residual changes the same compatibility matrix used in both reductions, rather than adding a separately trained retrieval score.

**Final score.** If \(P_s\) denotes valid physical peaks,

\[
F(s,G)=0.65\,\frac1{|P_s|}\sum_{p\in P_s}
 \left[0.1\log\sum_{f\in G}\pi_f\exp(\widehat S_{pf}/0.1)\right]
 +0.35\sum_{f\in G}\pi_f\max_{p\in P_s}\widehat S_{pf}.
\]

This is the **only ranking score**. Peaks receive uniform weight in the forward average. There is no ensemble, score fusion, pair-stream feature, fingerprint score or test-time reranking. A custom maximum VJP preserves the forward maximum and splits gradients equally between exact tied winners; it avoids a documented JAX 0.11 recomputation-gradient issue. It does not change inference scores.

## Training objective, sampling and the two phases

For a minibatch of \(B\le128\) paired spectra/molecules, set \(f_{ij}=F(s_i,G_j)\). Exclude off-diagonal pairs with equal InChIKey14. For each spectrum sample up to eight exact-truth-excluded candidates **uniformly without replacement from its complete training formula pool**, freshly at each visit. Fewer than eight available negatives are all used. Repeated paired truths remain separate graph instances; appended decoy graphs are deduplicated among themselves and encoded with current GINE weights.

Let \(J_i\) be eligible in-batch columns, \(N_i\) the sampled formula negatives, and \(\tau=0.05\). The loss is

\[
\ell_i^{s\to G}=\log\left[\sum_{j\in J_i}e^{f_{ij}/\tau}
                  +\sum_{G\in N_i}e^{F(s_i,G)/\tau}\right]-f_{ii}/\tau,
\]
\[
\ell_i^{G\to s}=\log\sum_{j:\,i\in J_j}e^{f_{ji}/\tau}-f_{ii}/\tau,
\qquad
\mathcal L=\frac1{2B}\sum_i(\ell_i^{s\to G}+\ell_i^{G\to s}).
\]

Formula negatives occur only in the spectrum→molecule denominator; reverse weight is **1.0**. This is sampled symmetric InfoNCE, **not exact full-pool cross-entropy**. “Full-pool eight” describes sampling support, not the number of scored formula candidates in training. There is no hard mining. Training shuffles all 194,119 rows per epoch, including the final partial minibatch, using the persisted NumPy/JAX RNG states.

| Setting | Fresh retrieval-head phase | Selected joint-adaptation phase |
|---|---|---|
| Encoder | Frozen original student 1,190,000 | Same student, trainable |
| Head initialization | Fresh; Q/K identity; adapter output zero | Complete selected fresh head and GINE BN state restored |
| Head Adam learning rate | \(5\times10^{-4}\) | \(10^{-5}\) |
| JEPA Adam learning rate | — | \(3\times10^{-6}\) |
| Maximum epochs / patience | 60 / 8 | 8 / 8 |
| Selected human epoch | 13 | 2 |
| Seed / batch / dropout | 0 / 128 / 0.1 | 0 / 128 / 0.1 |

Adam uses \(\beta_1=0.9,\beta_2=0.999,\epsilon=10^{-8},\epsilon_{\rm root}=0\), with no weight decay, learning-rate schedule or gradient clipping. Joint adaptation starts **fresh Adam moments**, restores the complete retrieval head including the learned adapter, and uses per-layer activation checkpointing in JEPA. It is not a from-scratch 1.52B model training run. The source head's zero-update validation is eligible for selection. Historical selection updates the checkpoint only for strictly higher **optimistic** validation Hit@1; the selected optimistic and stable metrics agree. The eight-epoch joint curve was 45.90, 46.02, 45.72, 45.82, 45.68, 45.80, 45.74, 45.62%.

Fresh retrieval affine kernels **and biases** use uniform initialization \([-1/\sqrt{d_{\rm in}},1/\sqrt{d_{\rm in}}]\), matching the upstream PyTorch Linear convention. Fragment category embeddings use normal standard deviation 1; normalization scales/biases initialize to one/zero. Adapter and Q/K initialization are the explicit exceptions above.

| Trainable component | Parameters |
|---|---:|
| JEPA encoder, excluding fixed Fourier buffers | 1,516,340,736 |
| Raw-token peak readout | 594,432 |
| GINE + global context + fragment pooler/gate | 3,375,689 |
| Q/K projections | 131,072 |
| Cross-attention adapter + two physical scalars | 20,514 |
| **Retrieval head / joint total** | **4,121,707 / 1,520,462,443** |

## Evaluation and MCES conventions

Restore `best.msgpack` **including BatchNorm statistics**, disable dropout, and score every candidate in each supplied pool. Encode the molecular fragments and peaks, apply Q/K, cache those projected vectors in float16, then cast to float32 for scoring. Gate logits, fragment fractions and molecular masses remain float32. Use candidate padding only for shapes; exclude it from ranks. The selected native settings are molecule encode batch **512**, online JEPA encode batch **128**, query blocks **2,000**, candidate capacity **256**, atom capacity **160**, node/edge capacities **65,536 / 147,456**, and training score-query chunk **16**.

For truth at supplied column \(t\),

\[
r_{\rm optimistic}=1+\#\{c:F_c>F_t\},\qquad
r_{\rm stable}=r_{\rm optimistic}+\#\{c<t:F_c=F_t\}.
\]

These comparisons use the actual native scores, with **no score rounding or tie tolerance**. Top prediction is the first maximum in supplied candidate order. Report spectrum-level Hit@1/5/10/20, MRR, median rank and, when derived from saved ranks, mean rank. Each spectrum has equal weight. Optional per-compound results first average correctness among spectra sharing InChIKey14, then average compounds equally; these have a different denominator and must be labelled separately.

For **MCES@1**, compare each spectrum's saved top-1 SMILES with its truth SMILES and average the distances over spectra. Identical strings contribute zero. The pinned protocol uses `myopic-mces==1.2.2`, `pulp==3.3.2`, RDKit 2026.3.5, `threshold=15`, `PULP_CBC_CMD`, one solver thread, no time limit and `catch_errors=False`. Distances are exact below the algorithm's threshold and can be certified lower bounds above it; **do not describe every returned value as exact or clip values above 15**. MCES zero does not replace the retrieval identity convention. Compatible cached pair distances are reused only with recorded package/provenance checks; new pairs and solver modes are journaled. Test ranks, predictions, MCES protocol and per-query distances must be retained together.

## Reproduction and artifact identities

The maintained entrypoint is the fixed-recipe `scripts/run_retrieval_jax.py`, with paths supplied by `configs/retrieval_4602.json`. The seven inputs are the raw-token root, prepared graphs/rows, fragment cache, legacy peak fractions, original JEPA checkpoint, encoder metadata, and prepared raw spectra. The example uses the recorded `/results/...` mounted paths; these inputs must exist in the execution environment. Architecture and optimization settings are fixed by its stage; use the repository's `.venv`:

```sh
.venv/bin/python scripts/run_retrieval_jax.py --inputs configs/retrieval_4602.json --out results/retrieval_head/run --stage head
.venv/bin/python scripts/run_retrieval_jax.py --inputs configs/retrieval_4602.json --out results/retrieval_joint/run --stage joint --head-checkpoint results/retrieval_head/run/best.msgpack
.venv/bin/python scripts/run_retrieval_jax.py --inputs configs/retrieval_4602.json --out results/retrieval_joint/run --stage eval
```

Compute MCES from that run's saved predictions:

```sh
.venv/bin/python scripts/evaluate_atom_flare_mces.py --root results/retrieval_joint --workers 16
```

The evaluator reads `run/test_ranks.npz` and writes `full_results/`. It needs no historical model or seed cache. Optional `--seed-cache PATH` reuses certified compatible distances, and optional `--baseline ROOT` adds an explicitly labelled comparison.

For an already selected canonical model, evaluate a directory containing its `best.msgpack`, `manifest.json`, `progress.json` and completed `validation_complete.json`; do not rerun training to reproduce test metrics. The separately hash-pinned historical model importer accepts the selected original checkpoint with its original manifest/progress. `--resume` restores a training continuation, including optimizer and RNG state. The historical joint invocation used a previously verified zero-update state and therefore included `--resume`; simply copying that flag to a fresh directory is not equivalent.

On the existing Modal volume, the native selected-checkpoint evaluation command is:

```sh
.venv/bin/modal run --detach scripts/modal_retrieval_jax.py --stage eval --out /results/retrieval_selected_4602_20260924/run
```

Preparation of the graph/row cache is fixed to seed 0, validation size 5,000, full formula-candidate support and atom capacity 160:

```sh
.venv/bin/python scripts/prepare_atom_flare_jax.py --input-root /results/jepa_fps_ftjepa_1p8b_step1190000_flare_raw --out /results/atom_flare_gine3m_qk_fullpool8_s0_20260918/prepared --workers 16
```

This prepares chemistry and row selection; it does not fit PCA or regenerate JEPA tokens or GLACIER proposals. The maintained runner rejects changed input identities on resume, requires completed validation selection before canonical test evaluation, and accepts a historical checkpoint only when its bytes and recorded selected inputs match the pinned 46.02% model.

The historical source of truth is the [selected manifest](../results/retrieval_selected_4602_20260924/selected_manifest.json), [selected progress](../results/retrieval_selected_4602_20260924/selected_progress.json), [selection lock](../results/retrieval_selected_4602_20260924/selection_lock.json), and the immutable [395-file runtime](../results/global_glacier_crossattn_joint_runtime_20260924/). The exact historical argument list is [joint_baseline_adaptation_arguments.json](../results/global_glacier_crossattn_joint_followup_s0_20260924/joint_baseline_adaptation_arguments.json); the source head's manifest is retained alongside its checkpoint.

| Artifact | SHA-256 |
|---|---|
| Selected joint checkpoint, 6,081,876,317 bytes | `324679e57439fd3294cce1cc379f1332a197598b1a0a15fe6fa5b35c7a6f6b99` |
| Source head checkpoint, 45.04% validation | `c66d5c7ce0d8073d31045e9fdbfdaf0dfac7e986334b5ca60f14cc4a97c5c5cc` |
| Frozen runtime manifest | `59e6c1c7dd7ccad6c45fcd9e0df77fbb6d14c2f0470aeae76ca6c4fa87f78984` |
| JEPA configuration metadata | `f3f2c451203a59312c2552058ebba464c3b5d8a97c5c56cd3b4f23d10f3dca4e` |
| Prepared spectral NPZ | `6637b4da92e2d8f177aa24f29f85baccc72cf8abb1039f035051a3838aa89f07` |
| MassSpecGym TSV | `50cfdd1d22f79543c59555f9ce43c6893bd788a19a42b21fb4e2e3a54673c06a` |
| Graph-store ordered SMILES | `55d207330edbc42651e2346b4fb5a7d8aa485662796de83196e09f2cedb54c18` |
| Fragment cache metadata | `e845cce4844298342d7116d8fed116803cbb3bf91c9a16b07803fa08c166ca75` |
| Prepared rows NPZ | `4b74f8c40ef75344c9ff84826c263a18d9f71889d30ef5c3418e3699a8cf9c3d` |
| Supplied formula candidate JSON | `454636d2346aebb3cd87edc2efc3fa1bf91eb8e0219edff5880f02aa537ac954` |
| Selected native test-rank archive | `963d80ee5a504b469ee128da4d490223613d9c118b672606fe835d1c9f81e7ef` |

Historical implementation lineage records upstream pre-fragment commit `55dfbdf940b6a5052026de0153941e0bba2d65d8` and fragment-recipe commit `4c076818eb3486491a415cfd8ac6b8c2d34f7e3d`; the GLACIER proposal-generator commit above is a separate dependency. The recorded training environment used Python **3.14.7**, JAX/jaxlib **0.11.0**, Flax **0.12.8**, Optax **0.2.8**, NumPy **2.5.2**, and RDKit **2026.3.5**. Preserve the repository dependency lock and installed environment when reproducing numerical results.

Recorded pre-adaptation checks established exact online/frozen initial rankings and top IDs over all 5,000 validation rows and 817,641 pairs, exact raw-token parity at the export batch shape, trained-adapter/head/BN restoration, checkpoint resume, finite selected parameters/statistics, and gradients reaching all JEPA blocks. The selected held-out evaluation subsequently verified all 17,556 test rows and 2,909,859 candidates without training or changing the selected checkpoint. MCES completed for every row with finite nonnegative distances and no failures.

The cleaned implementation passed **51 focused preparation/runner checks**, including a real two-epoch training/session-resume comparison. Resumed and uninterrupted training produced exactly equal parameters, BatchNorm state, Adam moments, JAX/NumPy RNG state and selected checkpoint bytes. The checks also cover full-pool sampling support, complete molecular graphs, token/CLS alignment, exact tie handling, checkpoint input identities, source recording and rejection of other historical checkpoints.

Separate numerical checks verified selected-model initialization, the 4,121,707-parameter head, score/input/parameter gradients, padding, BatchNorm/dropout and one Adam update against the immutable training source. Independent scalar equations and PyTorch pooling/GINE references also passed. Shared graph/fragment/SALT boundaries passed 34 checks, and the selected-only MCES reporting CLI passed three checks.

[Production-shape trace checks](../results/retrieval_selected_4602_20260924/model_trace_parity.json) found identical JAX computation graphs for the molecule encoder, Q/K projections, peak MLP and final scorer, with all 91 shared JEPA source files unchanged. A [controlled H100 comparison](../results/retrieval_selected_4602_20260924/canonical_parity/equivalence_audit.json) produced bit-exact scores for all 5,396 candidate pairs when given identical cached embeddings, and identical ranks/top predictions for all 32 validation spectra.

This does **not** establish bitwise GPU reproducibility. Re-encoding the same molecules with the unchanged original code also changes some floating-point outputs: in the controlled check, original-versus-repeat and original-versus-cleaned maximum score differences were both \(4.8161\times10^{-5}\). An earlier separate-process comparison had a larger 0.01003 difference and one low rank changed from 43 to 42, with no top-prediction change; that larger excursion was not fully attributed. The failed strict checks remain in the audit. The separate full-test evaluation through the cleaned CLI nevertheless reproduced all saved ranks and top predictions exactly. That is a result-level comparison; it does not assert that all unretained raw scores were bitwise identical.

The supported retrieval interface is `scripts/run_retrieval_jax.py`; alternate architectures, objectives and post-hoc rerankers have been removed from the working pipeline. PCA preparation is removed. [The cleanup record](../results/retrieval_selected_4602_20260924/cleanup_report.md) and [file inventory](../results/retrieval_selected_4602_20260924/cleanup_applied.json) document the retired research files. Selected provenance, necessary data, SALT archives and unrelated pretraining work are preserved. Removed research files are recoverable outside the repository at `/mnt/data/retrieval-research-archive-20260924`.
