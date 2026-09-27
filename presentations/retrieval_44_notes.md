# Novogaia — 44.10% retrieval recipe: presenter notes

**Scope:** the selected `crossattn_warm_joint` recipe, evaluated on 24 September 2026. These are diagram-led speaking prompts for 12 main slides and two optional appendices, roughly 12–15 minutes before questions. Use the measured results and qualifications in [the evidence sheet](research/retrieval_44_evidence.md). The complete equations and reproduction contract live in [the method document](../docs/retrieval_recipe.md).

**Delivery:** distinguish three historical model families: 32.47% wider GINE + Q/K; the 36.94% Global + GLACIER result in the existing presentation; and the selected 44.10% result. These are spectrum-level test numbers. The 46.02% figure is validation. Synthetic spectra, molecular sketches, vector colors, attention maps and candidate scores explain computation; they are not experimental matched pairs or measured activations.

## 01 — From local atom matching to 44.10% retrieval

Lead with the completed held-out result: 7,743 of 17,556 spectra rank the correct candidate first. The selected checkpoint reached 46.02% on the fixed 5,000-spectrum validation set; it reached 44.10% on the test set. Keep those denominators visible.

**Point and pause:** trace one spectrum and its formula-matched candidate pool into one compatibility score for each molecule. The method ranks supplied candidates; it does not generate a new molecular structure. Median test rank is two.

**Transition:** explain how this differs from the architecture in the earlier presentation.

## 02 — What changed since the previous presentation

The historical wider GINE + Q/K model achieved 32.47% test Hit@1. Adding global atom context and GLACIER fragments produced the 36.94% result shown previously. The selected system retains those additions and changes how it learns from candidates and reads out the fragment evidence: full-pool negative support, a precise-ion bonus, a learned forward mixture, a candidate-conditioned cross-attention residual, and joint encoder/head adaptation.

The original JEPA student checkpoint, raw token MLP and asymmetric Q/K projections are common foundations. We did not obtain this result by switching to the newer periodic-teacher or pair-stream encoder.

**Point and pause:** read the arrows as historical development, not a controlled attribution of percentage points to individual components. The aggregate improvement from the previous slide deck is 7.16 percentage points on the same spectrum-level test denominator.

## 03 — Two encoders meet in a peak–fragment matrix

The spectrum branch reads up to 127 physical peaks plus acquisition metadata. The original JEPA student emits 2,048-dimensional peak tokens; a learned MLP produces normalized 256-dimensional vectors. There is no PCA. Retrieval discards the final CLS token.

The molecular branch applies three width-496 GINE updates, broadcasts a learned whole-molecule mean context, and pools up to 128 GLACIER fragment tokens, including the intact-molecule anchor. Separate 256-dimensional Q/K maps build an asymmetric peak–fragment compatibility matrix. The physical term and 32-dimensional cross-attention adapter modify that same matrix before a single score is produced.

**Point and pause:** identify rows as physical peaks and columns as fragments of this particular candidate. Fragment tokens are not atom tokens or learned JEPA pair-stream embeddings.

## 04 — Give each atom whole-molecule context · retained foundation

Three local GINE updates propagate information through at most three graph edges. Increasing width alone does not extend that radius. We mean-pool the final real-atom states, transform that mean with an MLP, and add the resulting vector back into every atom.

The broadcast supplies a learned summary of the entire molecular graph to each fragment pool. It does not replace the local atom states, and mean pooling does not guarantee that every distinct topology becomes distinguishable.

**Point and pause:** follow the atom-to-mean arrows, then the mean-to-every-atom arrows. All real atoms participate; the historical 96-atom readout cap does not truncate this fragment path.

## 05 — Match evidence to fragments with a cut boundary · retained foundation

GLACIER proposes fragment atom memberships offline. For each retained fragment, the trainable pooler combines the member-atom mean, the mean of member atoms touching cut bonds, cut-count and fragment-kind embeddings, mass fraction, and log atom count. This becomes a learned 256-dimensional fragment vector.

The intact molecule is always the first anchor. Proposal intensity informs a separate learned fragment gate, normalized into a prior over valid fragments. Intensity does not enter the fragment embedding itself. The model pools globally contextualized full-molecule atom states; it does not rerun GINE on a separate induced graph for every fragment.

**Point and pause:** show member atoms and cut-boundary atoms with labels as well as color. The illustrated fragment must be a valid subset of the pictured graph. The proposal network is frozen preprocessing; retrieval gradients update GINE and the pooler, not GLACIER memberships.

## 06 — Combine learned compatibility with precise ion mass · changed readout

Start with the unscaled dot product between separately projected peak queries and fragment keys. Add a physical ion-mass bonus using a fixed 0.01 Da width, the precursor adduct, and hydrogen shifts from minus two to plus two. A half-weight protonated alternative is included.

This physical term supplies a useful compatibility signal; it does not force every peak into one fragment assignment. All spectra in this recorded dataset use the supported protonated or sodiated path. The old fractional-mass fallback is inactive for this run.

**Point and pause:** show learned compatibility and mass agreement entering the same matrix entry. The main 256D Q/K product has no square-root dimension scaling and is not renormalized after the projections.

## 07 — Let the candidate change the peak interpretation · new adapter

Each peak attends over this candidate's fragments using the base compatibility matrix and the learned fragment prior. It receives a weighted 32D fragment-value summary. The adapter combines the peak state, that candidate summary, and their elementwise interaction, then adds a learned residual to every peak–fragment compatibility.

This is the actual candidate-conditioned cross-attention step. Asymmetric Q/K alone creates learned similarities but does not perform this softmax-and-value aggregation. The adapter has 20,512 parameters and changes the one matrix used for retrieval; it does not introduce a separate reranker.

**Point and pause:** trace the attention row into the weighted sum, then back into the residual matrix. The pictured weights are synthetic. A different candidate supplies different fragment values and therefore a different peak context.

## 08 — Read one matrix in two directions · changed aggregation

The forward term asks how well each observed peak can be explained by the candidate's fragment mixture. It uses a prior-weighted log-sum-exp at temperature 0.1 and averages valid peaks uniformly. The reverse term asks whether each likely fragment has supporting spectral evidence; it takes the best peak for each fragment and averages with the learned fragment prior.

The final score is 0.65 times the forward term plus 0.35 times the reverse term. Both terms use the cross-attention-adjusted matrix. There is one scalar score per candidate, with no fingerprint fusion, ensemble or test-time reranking.

The previous 36.94% model averaged a forward hard maximum and the gated reverse maximum with equal 0.5 weights. Its prior entered only the reverse term. The current forward mixture now uses that prior too, and the reduction weights are 0.65/0.35.

**Point and pause:** sweep across rows for the forward mixture, then down columns for the reverse maximum. Label the mixture and maximum explicitly; neither arrow denotes another neural encoder.

## 09 — Keep eight negatives; remove the fixed-bank restriction · new support

Previously, each training spectrum sampled eight formula negatives from the first sixteen eligible candidates. Now it samples up to eight uniformly without replacement from the entire eligible supplied pool, refreshed at each visit. The number of sampled negatives is unchanged; their support expands.

Formula negatives augment the spectrum-to-molecule denominator. The reverse direction still uses in-batch examples, with equal loss weight. This remains sampled symmetric InfoNCE with temperature 0.05, not exact full-pool cross-entropy and not hard-negative mining.

**Point and pause:** show a highlighted first-sixteen region before, and eight dispersed selections across the pool after. The drawing is schematic. Evaluation always scores the complete supplied pool, with no candidate truncation or missing entries.

## 10 — Train the head, then adapt the same encoder · new continuation

The first phase trains the retrieval head against frozen original JEPA features. Its selected checkpoint reaches 45.04% validation Hit@1. The next phase restores the complete learned head, adapter and GINE BatchNorm state, then jointly adapts the same JEPA encoder and head with fresh Adam moments and separate learning rates.

The selected joint epoch is the second, at 46.02%. A matched head-only continuation reaches 45.94%. That difference is just four correct validation spectra, or 0.08 percentage points. This single-seed comparison does not establish a reliable gain from encoder finetuning. We should not credit joint adaptation for the entire improvement over the historical model.

**Point and pause:** distinguish a freshly trained head from a pretrained encoder. This is not from-scratch training of a 1.52-billion-parameter JEPA model. Training and inference remain JAX/Flax direct loops.

## 11 — Full-pool held-out retrieval: the complete result

Report the spectrum-weighted test numbers together: Hit@1 44.10%, Hit@5 73.57%, Hit@10 83.57%, Hit@20 92.23%, MRR 0.5751, median rank two, and mean rank 7.38. We scored 2,909,859 candidate pairs across 17,556 spectra. Mean MCES@1 is 4.9549 and its median is two, with no failed queries.

MCES describes the graph distance of the top prediction; lower is better. Its pinned threshold-15 protocol returns exact values below threshold and can return certified lower bounds above it. It is a separate statistic from exact candidate identity. Stable and optimistic ranks agree in this test.

**Point and pause:** point to the spectrum-level label. The 57.38% equal-compound result uses a different weighting and must not replace the 44.10% headline. Historical model comparisons are descriptive progression, not isolated causal ablations or a multi-seed uncertainty estimate.

## 12 — One supported, verified retrieval pipeline

The maintained entrypoint now fixes this recipe. The selected checkpoint and inputs are pinned, validation chooses the checkpoint before test evaluation, and the full cleaned pipeline reproduced every saved test rank and top prediction. The method document records exact equations, input identity, training state, evaluation conventions and reproduction commands.

Retired research code and experiment directories were archived outside the repository. Necessary data, checkpoint lineage, MCES provenance and unrelated work remain available. The headline is a completed result with a reproducible inference path; it is not a claim that every floating-point GPU intermediate is universally deterministic.

**Point and pause:** follow the three operations: train head, joint adaptation, evaluate selected checkpoint. Finish with the measured outcome and the scope: 44.10% Hit@1 on formula-constrained, spectrum-level retrieval.

## 13 — Appendix: the physical ion compatibility term

Let \(M_f=M_G\rho_f\), observed peak mass-to-charge ratio be \(x_p\), neutral hydrogen mass \(H=1.00782503207\), and adduct mass be \(a\). Define

\[
\Delta_{pf}(a)=x_p-(M_f+a),\quad
j_{pf}(a)=\operatorname{clip}\{\operatorname{rint}(\Delta_{pf}(a)/H),-2,2\},
\]
\[
K_{pf}(a)=\exp\!\left[-\left(\frac{\Delta_{pf}(a)-j_{pf}(a)H}{0.01\;\mathrm{Da}}\right)^2\right],\quad
B_{pf}=1.5e^{\theta_\beta}\max\{K_{pf}(a_s),0.5K_{pf}(a_H)\}.
\]

The observed precursor adduct \(a_s\) is protonated or sodiated; \(a_H\) is the proton mass. The hydrogen-shift adjustment uses neutral hydrogen mass. Do not silently substitute the proton mass in both places. All recorded rows take this eligible-ion path. The fixed 0.01 Da width differs from the inactive retained-fraction fallback's learned sigma.

**Point and pause:** the kernel is a compatibility bonus, not a normalized likelihood over peaks. Show the mass residual after the clipped integer hydrogen adjustment. The fragment proposals remain offline and fixed.

## 14 — Appendix: data, training and parameter budget

Training uses 194,119 rows and a fixed seed-0 subset of 5,000 validation spectra drawn from 19,429 validation rows. Test remains the full 17,556 spectra. Candidate identities and supplied order are preserved; formula negatives exclude exact truth SMILES, while in-batch exclusions use InChIKey14.

The head has 4,121,707 trainable parameters: 594,432 in the peak MLP, 3,375,689 in GINE/global/fragment pooling and gate, 131,072 in Q/K, and 20,514 in cross-attention plus physical scalars. The jointly trainable JEPA encoder adds 1,516,340,736, for a total of 1,520,462,443. The 32D adapter itself has 20,512 parameters.

Shared settings are seed zero, batch 128, dropout 0.1 and symmetric InfoNCE temperature 0.05. The fresh head uses Adam at 5e−4; the joint phase uses 1e−5 for the head and 3e−6 for JEPA, with fresh moments. Formula-negative support spans the full eligible pool, but each visit scores only up to eight sampled formula negatives alongside the in-batch denominator. No hard mining, weight decay, schedule or gradient clipping is used.

**Point and pause:** separate the retrieval head from the pretrained JEPA encoder. Restoring a trained head for joint adaptation is intentional; it is not another randomly initialized model or a different checkpoint family.

## Additional mathematical detail for questions

The exact global residual is \(\tilde h_a=h_a^3+L_2\operatorname{ReLU}(L_1|V|^{-1}\sum_bh_b^3)\). The fragment representation pools these states; it has no independent induced-subgraph message passing.

The base score is \(S_{pf}=q_p^\top k_f+B_{pf}\). Attention is \(T_{pf}=\operatorname{softmax}_f(S_{pf}+\log\pi_f)\) at temperature one. With 32D RMS-normalized projected states \(u_p,v_f\), context \(c_p=\sum_fT_{pf}v_f\), and \(b_p=\tanh(u_pA_q+c_pA_c+(u_p\odot c_p)A_x+b)\), the residual is \(\delta_{pf}=(b_pO)^\top v_f/\sqrt{32}\). The trained score matrix is \(\widehat S=S+\delta\).

The only retrieval score is

\[
F(s,G)=0.65\frac1{|P_s|}\sum_p0.1\log\sum_f\pi_f\exp(\widehat S_{pf}/0.1)
+0.35\sum_f\pi_f\max_p\widehat S_{pf}.
\]

Candidate cross-attention temperature one, forward-mixture temperature 0.1, and contrastive-loss temperature 0.05 are three distinct operations. Exact equations for the physical kernel, gate, loss and trainable parameter counts are in the linked method document.
