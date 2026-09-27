# Novogaia — presenter notes

**Duration: 30:00 for 24 main slides; questions and appendix are additional.** These are speaking prompts, not a script to read verbatim. Follow the diagrams, explain each equation in words, and use the indicated pauses. Aim for about 115–125 spoken words per minute during the mathematical sections. The time windows below are the timing authority; they include transitions and time for the audience to inspect the figures.

**Evidence vocabulary:** “Implemented” describes repository behavior; “reported result” describes the specified completed benchmark; “analytic illustration” describes a simplified mathematical model; “proposed” describes an experiment not claimed as completed. A mathematical mechanism is not proof of an empirical improvement.

**Visual delivery:** The main slides now show spectra, molecular structures, feature coordinates and matrices, with boxes reserved for model operations. Point to each scientific object before explaining its equation. Spectral traces, fingerprint shades, feature-tile colors and compatibility matrices are synthetic illustrations unless explicitly identified as a reported result; pictured spectra and molecules are not experimental matched pairs. Keep fuller explanations and qualifications here. Visual cues fit within the existing time windows, and source footnotes need not be read aloud.

## 01 — Learning to decode molecular structure

**00:00–01:00 · 60 seconds · Research overview**

Our problem is to identify a molecule from the fragments we measure in a mass spectrometer. This talk follows three ideas: generating structures from a compact chemical signature, learning richer representations from raw spectra, and using those representations to rank candidate structures.

Follow the bottom diagram from measured peaks, through learned vectors, to a molecule. The middle representation is the main research object: what information does it keep, and how easily can a downstream model use that information? We will show the training design, a mathematical explanation of its dynamics, and a concrete retrieval result of 36.94 percent correct at rank one.

**Point and pause:** Spend a few seconds on the three verbs: predict, represent, retrieve. This is a confidential research presentation; the footer remains throughout.

**Transition:** Begin with why going from a molecule to a spectrum is easier than reversing that process.

## 02 — A spectrum is evidence, not a structure

**01:00–02:15 · 75 seconds · Problem definition**

The forward arrow is fragmentation: a molecule produces ions whose mass-to-charge ratios and abundances form the spectrum. The reverse arrow is the inference problem. A peak does not directly name an atom or identify one bond; multiple structures and fragmentation pathways can produce similar observations.

Read the notation once. Capital S is the spectrum. Each pair, m and I, is a peak position and its intensity. G is the molecular graph: V lists atoms and A describes bonds. F is a supplied molecular formula. The conditional probability asks which graphs are plausible given the spectrum and formula.

Knowing the formula is like knowing the ingredients and their counts. We still need to determine how they connect.

**Point and pause:** Start at phenol on the left, trace the reverse arrow, then compare the three actual candidate structures on the right. Phenol, the cyclic dienone and the acyclic cumulene aldehyde all have C6H6O, verified from their SMILES; their connections differ. They illustrate structural ambiguity, not their frequency in real candidate pools. Read argmax as “choose the most plausible graph,” not certainty. The spectrum is synthetic and is not an experimental match to any pictured structure.

**Transition:** There are two useful ways to turn this evidence into an identity.

## 03 — Two routes from evidence to identity

**02:15–03:15 · 60 seconds · Organizing framework**

The upper route is supervised generation. An encoder learns a compact representation with chemical supervision, and a decoder proposes a molecule. A fingerprint describes chemical features or substructures; it does not fully specify a unique structure.

The lower route begins with large collections of spectra that do not require known structures. Pretraining learns representations of the spectral evidence. A downstream task then teaches us how to use those features, here by ranking a supplied set of molecules.

Generation can propose structures; retrieval answers a narrower question: which of these candidates is the best explanation? Keep that difference in mind when we reach the numerical result.

**Point and pause:** Follow the actual peak trace through the upper operations to the generated C–C–O graph. On the lower row, compare C–C–O with C–O–C: ethanol and dimethyl ether share C2H6O but differ in connectivity. The underline marks an illustrative choice, not a benchmark outcome. The lower argmax is restricted to the supplied candidate set C(F); the upper probability can propose graphs. These spectra and molecules are separate illustrations, not measured pairs.

**Transition:** First, the graph-generation approach used by DiffMS.

## 04 — DiffMS: fix the atoms, denoise the bonds

**03:15–04:45 · 90 seconds · Published prior work**

DiffMS is a strong prior approach to structure generation from tandem mass spectra. Its spectral encoder is MIST-derived and pretrained with fingerprint supervision. An important detail is that the diffusion decoder is conditioned on the encoder's learned embedding, rather than requiring a literal predicted fingerprint vector at inference.

The molecular formula fixes the heavy-atom inventory. Diffusion then models bond types between those atoms. Read the three matrices as an initially noisy bond pattern, an intermediate pattern, and a final graph. Each reverse step uses the spectrum representation and the formula to improve its bond prediction.

The product in the equation means “perform all the denoising steps”; the sum accounts for possible intermediate paths. No one needs to calculate it during the talk.

**Point and pause:** Trace the spectrum into the encoder and its explicit coordinate vector h(S). The small binary fingerprint strip is a pretraining-supervision branch, not the decoder input. Then point to the fixed C,C,C,O inventory from C3H8O and follow the three bond matrices. Their atom order stays fixed, while synthetic categorical shades denote no, single, double, triple or aromatic bond. They are not probabilities or measured denoising states. Heavy atoms are fixed, but validity and implicit hydrogen handling still matter; the spectrum is not a matched measurement of the illustrated structure.

**Transition:** Our alternative changes the decoder's output language.

## 05 — Our alternative: write the molecule as SMILES

**04:45–06:00 · 75 seconds · Implemented AR architecture; qualitative pipeline comparison**

Here the MIST encoder predicts a fingerprint, and an autoregressive transformer generates a SMILES string. SMILES is a compact textual encoding of a molecular graph. Letters name atoms; other symbols express bonds, branches, and ring closure.

Point to the formula prefix, then read the next-token idea from left to right. At each step, the model sees the tokens already generated, the formula information, and the fingerprint. It predicts a distribution for the next token. The product in the equation says that the probability of a complete string is assembled from these successive choices.

The displayed string is an illustrative encoding of phenol. The relevant implementation is the repository's `ar` branch; its name alone should not be taken to imply this is the default diffusion model.

**Point and pause:** Follow the spectrum into MIST, then inspect the fingerprint-coordinate strip before the AR transformer. White-to-black shades illustrate predictions from zero to one; they are synthetic, not measured MIST outputs. Trace the output brace down to the full phenol token sequence, starting after the C6H6O prefix. The spectrum and phenol string are not an experimental pair. Formula conditioning supplies information; it does not guarantee formula-correct output.

**Transition:** Changing the language moves the constraints and computational work.

## 06 — Two decoders. Two kinds of constraints.

**06:00–07:05 · 65 seconds · Qualitative comparison**

Graph diffusion operates directly on atoms and bonds. Many bonds can be updated in one step, but sampling requires repeated denoising. Autoregressive SMILES uses familiar sequence models with explicit next-token probabilities, but tokens must be generated sequentially. The verified AR sampler does not implement a KV cache; do not imply a measured speed advantage.

Their constraints differ. Fixing the heavy-atom inventory is natural for DiffMS. SMILES requires attention to grammar, matching ring closures, valid chemistry, and the final formula. Both approaches still need validation.

The shared issue is conditioning quality. If the representation loses useful spectral evidence, the decoder must rely more on chemical prior knowledge. For our AR path, the fingerprint is an explicit intermediate. DiffMS uses a richer encoder embedding trained with fingerprint supervision, so these are not identical information bottlenecks.

**Point and pause:** Contrast the two matrix snapshots with C → C → O → EOS. Both illustrate ethanol, using fixed atom order C,C,O on the left. The noisy matrix is an intermediate state, not a required chemically valid molecule. Point to X=X(F), then to the question mark over formula agreement on the right. This is architectural reasoning, not a measured speed or accuracy comparison.

**Transition:** That motivates learning directly from the much larger pool of raw spectra.

## 07 — Use the unlabeled reservoir

**07:05–08:10 · 65 seconds · Published precedent and data opportunity**

Public mass-spectrometry data offers a billion-scale opportunity even though only a fraction comes with confidently assigned molecular structures. The official MassIVE source describes searching approximately 1.2 billion public MS/MS spectra. DreaMS demonstrates the value of learning from large unlabeled spectral collections and transferring those representations to molecular tasks.

DreaMS mined around 700 million spectra and used the selected GeMS-A10 subset of 23,517,534 spectra for pretraining. Its objectives include masked-peak prediction and chromatographic retention-order prediction. The 201-million-spectrum DreaMS Atlas is a different quantity. Retention information is already part of that precedent; our distinction is richer acquisition conditioning and the latent-prediction objective.

We follow that broad motivation, with acquisition metadata and a different two-stage objective. The first stage learns spectral reconstruction; the second predicts learned teacher features. A structure label is unnecessary for either pretraining stage.

The scale on this slide describes the public raw-data opportunity, not a verified count consumed by one training run, and certainly not a count of unique molecules. Our MassIVE pipeline filters eligible MS2 spectra and preserves acquisition and group identity when preparing splits.

**Point and pause:** Start at the reservoir, move through quality control and duplicate handling, then follow the branches. Point to the DreaMS stack of spectral records and its rounded “23.5M selected” caption: this is its selected training subset, not our run size. Both stacks contain schematic record icons, not experimental traces. Below, follow metadata into our encoder and its latent target. The shared funnel does not imply identical filtering or examples. The retention-order notation names a prediction target; the model is not given both timestamps to answer the task.

**Transition:** More spectra are useful when the model also knows how they were acquired.

## 08 — The same molecule can be measured differently

**08:10–09:20 · 70 seconds · Implemented metadata conditioning**

The peak inputs provide mass and intensity. Fourier mass features let the encoder express relationships at different mass scales. The output is a contextual vector per peak, plus a CLS vector that can summarize the spectrum.

Acquisition conditions help explain why related measurements differ. The metadata includes precursor information, collision energy, instrument family, retention-time information, polarity, and acquisition mode. The 28 dimensions include presence indicators and categorical encodings; they are not 28 independent physical measurements.

Read the equation as “normalize a feature, then let metadata adjust its scale and offset.” Gamma supplies the scale and beta supplies the offset. The implementation uses RMS normalization and also residual gates, omitted in the schematic. Metadata dropout helps the model work when fields are unavailable.

**Point and pause:** Follow the acquisition fields into the explicit tuple (a1, a2, …, a28). These are encoded coordinates, including presence and category information, not 28 separate measurements. Follow the upward gamma/beta arrow into the encoder, then the peak trace into the output feature bars h1,…,hN and CLS. Bars denote vectors, not measured magnitudes. The scale-and-shift equation omits residual gates and does not claim that acquisition effects are perfectly removed.

**Transition:** With the inputs defined, here is the two-stage training process.

## 09 — First learn a teacher. Then learn from it.

**09:20–10:25 · 65 seconds · Implemented two-stage design**

Stage one is a masked autoencoder, or MAE. We hide some peaks and learn to reconstruct their mass bins using the visible peaks. This produces a teacher that has learned regularities in spectra.

Stage two freezes that teacher. The teacher reads the full spectrum and produces target vectors. A separate student sees partial context, and a predictor learns to recover the teacher's vectors at selected missing peaks. JEPA stands for joint embedding predictive architecture; here the important idea is prediction in representation space.

The training target changes from a measured quantity to a learned feature vector. Freezing makes the target function stable while the student learns. It does not by itself guarantee useful student features or superior retrieval.

**Point and pause:** Follow the visible peak trace into the MAE, then its probability histogram over mass bins. Pause at “freeze” before comparing the teacher and student feature arrays by MSE. The spectrum, histogram and tile shades are synthetic, not model measurements; close shades do not demonstrate successful training. This visible trace shares the pre-mask intensity normalization used on slide 11. Hidden peaks are absent, with no marks exposing their masses. Point to the zero teacher derivative: the vertical comparison does not update the teacher.

**Transition:** First examine what the teacher learns to reconstruct.

## 10 — Reconstruct mass, at more than one resolution

**10:25–11:50 · 85 seconds · Implemented MAE objective**

Dark tokens are visible peaks; question marks indicate positions whose values are withheld. The encoder reads visible peaks, and the decoder receives masked queries for the positions to predict. Hidden target masses are not supplied as query values.

For each target, the model predicts one probability distribution over fine m/z bins. We then add neighboring probabilities to obtain coarser distributions. Read the first equation as “a broad-bin probability is the total probability inside that bin.” Cross-entropy rewards assigning probability to the correct location.

The second equation combines the fine and coarse errors. Fine bins are 0.5 m/z units; coarser scales supervise neighborhoods from the same prediction. This supplies useful signal when the model finds approximately the right location before the exact bin. The loss is averaged over selected target peaks.

**Point and pause:** Trace two adjacent fine bins into one coarse bin, then four into a broader bin. The white-to-black key means probability on a common scale; broad-bin values are sums, not separate predictions. These bands are illustrative probabilities. Intensity is an input, but intensity reconstruction is disabled in this teacher recipe.

**Transition:** Once this encoder is trained, we can make its contextual features the next target.

## 11 — Predict what the teacher sees in the missing peaks

**11:50–13:20 · 90 seconds · Implemented frozen-teacher JEPA**

Follow the upper path: the frozen teacher sees all valid peaks. We select its feature vectors at target positions; these are Y. Follow the lower path: the student sees the visible context C, produces representation Z, and the predictor uses Z and target-query information Q to predict those vectors.

The target query identifies where a prediction is needed, rather than revealing its hidden mass. The loss is the average squared difference between predicted and target coordinates. M is the set of target peaks, and d is the vector width. The stop-gradient symbol means no learning signal changes the teacher.

The current recipe adds no target normalization: the normalization operator is the identity. There is no moving-average teacher update. At deployment we export student features and train task-specific readouts; the pretraining predictor is unnecessary.

**Point and pause:** Find the two circled peaks in the full teacher spectrum, then confirm that both are absent from the student context. The traces share one pre-mask intensity scale, also used on slide 9; masking does not rescale surviving peaks. No circles, placeholders or hidden mass hints enter the context. Follow the student into its coordinate strip Z, then the predictor and target arrays. Traces and tile colors are synthetic. The vertical MSE comparison is not a two-way learning arrow: only student and predictor weights learn, and the student encoder is exported.

**Transition:** The next slides explain what this loss encourages—and what it leaves undetermined.

## 12 — Same teacher target. Different student codes.

**13:20–14:40 · 80 seconds · Numerical illustration**

Start at the top: for this complete spectrum, the frozen teacher gives a feature value of four. Its weights stay fixed, so the same input keeps the same target.

Now compare the two student codes. Code A is two, and its predictor multiplies by two. Code B is eight, and its predictor divides by two. Both give four. Their final predictions match even though their internal codes differ. These are two possible encoder–predictor combinations, not successive training steps.

That is the distinction: training compares the predictor's answer with the teacher target. The student encoder has room to organize its own representation. The numbers stand for one feature of a much larger vector; they are not measured peak intensities or masses.

**Point and pause:** Read “two times two” and “eight divided by two,” then point to the identical outputs. Freezing stabilizes each input's target. It does not, by itself, guarantee a different or better student representation. The general coordinate-change argument is in Appendix I.

**Transition:** The student also sees less of the spectrum than the teacher. What should it predict when that missing information matters?

## 13 — When the answer is uncertain, predict its average

**14:40–15:55 · 75 seconds · Equal-probability toy example**

Point to the left trace: this is all the student sees. The visible peaks, query and acquisition conditions are the same in all three alternatives. The dashed branches represent possible complete spectra, not additional inputs available to the student.

Each complete spectrum adds a different hidden peak. The same frozen teacher maps these different inputs to feature values of eight, ten or twelve. Its weights have not changed; the inputs have. These are learned feature values in arbitrary units, not masses or intensities.

If the three possibilities are equally likely, squared-error training favors their average: ten. Choosing eight would overcommit to one hidden possibility. The expectation symbol below is simply the average answer given the evidence available.

**Point and pause:** Trace the unchanged visible peaks, the three hidden alternatives and their average. This illustrates why predicting teacher features need not preserve every detail the teacher sees. It does not establish that the predictable part is always chemical or that the student will outperform the teacher. The optimum shown assumes a predictor able to express the conditional mean.

**Transition:** Prediction errors now have to change the student. The predictor controls which parts of its code receive that feedback.

## 14 — The predictor directs the student’s learning

**15:55–17:25 · 90 seconds · Numerical feedback example**

The student code has two entries: one and five. For this example, the predictor uses only the first entry and doubles it. Its answer is two, while the fixed teacher target is four, so the error is minus two.

Follow the return arrow. Changing the first entry changes the answer; changing the second entry has no direct effect on this predictor. To reduce the error, the first entry must increase toward two. The lower equation says the same thing in calculus: the gradient is minus four for the first entry and zero for the second. Gradient descent subtracts the gradient, so a negative gradient raises the first entry.

**Point and pause:** Trace the full loop before reading any symbols. The main message is that the predictor translates prediction error into feedback for the student. The teacher target alone does not dictate every coordinate of the code.

The predictor is held fixed only for this toy calculation. In real training both student and predictor learn, and shared encoder parameters can also move other code directions. “No direct feedback” does not mean a coordinate can never change. Appendix I gives the general formula.

**Transition:** Across many examples, some predictive relationships become established sooner than others.

## 15 — Some predictable patterns are learned earlier

**17:25–18:55 · 90 seconds · Analytic learning-speed illustration**

Read the vertical axis as the fraction of one context-to-target relationship learned by the encoder and predictor together. All three examples start at the same small fraction. The difference is the strength of each context-to-target relationship in this linear model.

Pause every curve at the same time, marked by the vertical line. The stronger link is nearly fully learned, the intermediate link is close, and the weaker link is still early. The bars show the same comparison: approximately 100, 92 and 14 percent. These are calculated fractions in an ideal model, not retrieval accuracies or measured transformer learning curves.

The equation at the lower right gives the idea in one line: halving predictive strength doubles the time to reach halfway, under the common initialization used here. The teacher targets can remain fixed while the student and predictor develop these relationships at different speeds.

**Point and pause:** Trace the common time marker to the three bars. “Predictable” is not the same as “chemically important”; a weaker relationship may still be valuable for identification. The assumptions are centered and whitened inputs, aligned modes, balanced factors and equal learning rates. Appendix C gives the full equation.

**Transition:** That is why we evaluate the exported student representation directly instead of relying only on prediction loss.

## 16 — Same loss does not mean the same representation

**18:55–20:10 · 75 seconds · Proposed diagnostics and controlled experiments**

First, measure whether exported encoder features help actual tasks. A lower prediction loss is useful evidence, but it is not itself a retrieval result.

Second, distinguish representation progress from predictor progress. Freeze an encoder checkpoint and fit the same fresh readout each time. This reduces the risk of confusing a stronger co-trained predictor with a better encoder.

Third, monitor whether representations vary across spectra and retain multiple usable directions. Effective rank is one way to summarize how broadly variation is distributed; it is not a complete chemical quality metric.

The proposed experiments on the right vary predictor learning speed separately from depth, use simple readouts, and compare matched compute with fixed masking. These are recommendations from the analysis, not completed improvements. Acquisition-only and query-only baselines can also reveal shortcuts.

**Point and pause:** Start with the schematic context spectrum and follow it through a frozen encoder to the coordinate strip Z. At the fork, the existing predictor gives JEPA loss while a standardized fresh readout tests retrieval; the downward branch measures covariance and effective rank across spectra. Spectrum and feature tiles are illustrative, not measurements. Each depth-by-rate grid point is a separate proposed pretraining ablation, not a measured result or an additional readout input. Fixed masks and matched compute make comparisons interpretable.

**Transition:** A complementary research direction changes what relationships the architecture can store explicitly.

## 17 — Give each peak pair its own memory

**20:10–21:25 · 75 seconds · Implemented pair-model branch**

A standard contextual peak vector stores information about one peak in its surrounding spectrum. A pair representation gives every ordered pair of peaks a separate vector. This creates a table of relationships: N peaks become N by N entries, each with its own feature channels.

The raw cues include signed and absolute mass differences, relationships to precursor mass, and endpoint intensities. In the illustration, 100 to 118 and 144 to 162 both have an absolute m/z gap of 18 Th. Equal gaps need not have equal meaning because their endpoints and surrounding context differ. An 18 Th gap alone is not a verified water loss or neutral-mass assignment.

These entries are relationships between spectral peaks. They are not labels for molecular bonds or confirmed fragment assignments. Single-stream models already learn interactions implicitly; the pair branch stores relationships more explicitly and incurs quadratic storage in the number of peaks.

**Point and pause:** Trace the two 18 Th braces, then move from their peak rows to cells in the pair table. Each cell is a vector with dp channels; the uniform fills are not a scalar heatmap. Read the phi expression as “combine endpoint and relative evidence,” with later blocks adding context. N×ds becoming N×N×dp makes the storage cost explicit.

**Transition:** Once relationships have their own states, they can exchange evidence through triangles.

## 18 — A third peak can clarify the relationship

**21:25–22:55 · 90 seconds · Implemented computation; hypothesized chemical benefit**

To update the relationship between peaks i and j, consult a third peak k. The relation from i to k and the relation from j to k together can provide evidence about the first pair. This is the computational analogy to AlphaFold's triangle updates.

In the equation, a and b are learned transformations. The circled multiplication combines corresponding feature coordinates, the sum aggregates third peaks, and W maps the result back into the pair state. The actual implementation adds gates, masks, and normalization; it also contains triangle attention, which uses the third edge to influence which relations receive attention.

The signed-gap equation is an elementary consistency relation among measured coordinates. It motivates relational reasoning; it does not prove a particular fragmentation pathway. Our indices are spectral peaks, not residues or atoms, and we are not predicting 3D distances here.

**Point and pause:** Trace the two relations to k, then the downward aggregation arrow into the updated ij edge. The edges are relationships, not physical fragmentation arrows. With delta_ij=mi−mj, subtract delta_jk from delta_ik to recover delta_ij. Compare the multiplication equation with attention's weighted sum; both are schematic and omit implementation details. Chemical benefit remains a hypothesis to test downstream.

**Transition:** The pair branch has its own masked pretraining target.

## 19 — Train relations by predicting missing mass gaps

**22:55–23:55 · 60 seconds · Implemented separate pretext task**

The encoder observes context pairs, and the predictor infers distributions over mass gaps for pairs involving masked target peaks. The target is the absolute difference between two m/z coordinates, not a spatial distance between atoms.

The loss uses the same multi-resolution idea as the MAE: one fine distribution is grouped into broader bins and supervised at several scales. Cross-entropy rewards probability on the observed gap. The reduction averages the valid supervised pairs.

This is a separate research branch. Its triangle computation is implemented, but it is not part of the single-stream frozen-JEPA system behind the 36.94 percent result we are about to discuss. We should establish its value with matched downstream comparisons rather than infer it from the analogy.

**Point and pause:** Highlight the masked row and column, follow the encoder/predictor arrow, then point to a probability bin in the schematic gap distribution. The bar heights are illustrative, not a measured prediction. Keep “implemented objective” separate from “demonstrated improvement,” and point explicitly to its separation from the reported 36.94 percent system.

**Transition:** Now turn to the retrieval system that produced the reported result.

## 20 — Match local spectral evidence to candidate atoms

**23:55–25:05 · 70 seconds · Historical JEPA–FLARE retrieval system**

The spectrum encoder is frozen. It provides up to 127 peak slots, each with a 2048-dimensional contextual vector. A trainable readout combines these features with measured peak scalars and maps them into 512-dimensional comparison vectors. Invalid padded slots are excluded.

On the molecular side, a three-layer GINE graph network reads atom and bond features and produces a contextual vector for each atom. This molecular network and the spectral readout are trained for retrieval; the historical recipe does not use a pretrained molecular backbone.

Both sides therefore enter a shared comparison space, but the final score retains local peak and atom information instead of reducing everything to one global vector. The CLS vector helps during initial warmup; the final candidate score does not include a CLS term.

**Point and pause:** Follow the actual peak trace through frozen JEPA and the trainable readout to the peak-vector matrix; follow phenol through GINE to the atom-vector matrix. Both feed the distinct peak-by-atom compatibility matrix, Q A_G transpose plus B. Tiles are coordinate containers and only schematic subsets of the labeled dimensions. The 0–1 grayscale values are synthetic, not probabilities, an asserted real-score range or a retrieval result; spectrum and molecule are not experimentally paired. Here Q denotes peak vectors rather than the earlier target-query information, and A_G denotes atom vectors rather than bond adjacency.

**Transition:** The compatibility matrix shows how those local vectors become a candidate score.

## 21 — Ask which atoms explain which peaks

**25:05–26:35 · 90 seconds · Implemented historical scorer**

Each matrix entry compares one peak vector with one candidate atom vector. Their dot product measures learned compatibility. Before taking any maxima, a soft mass-agreement term adds a bonus when approximate retained-mass fractions agree. This is a chemical heuristic, not a complete fragmentation simulator.

The score combines two directions. First, each peak chooses its best matching atom. Peaks that distinguish candidates receive larger weights. Second, each atom chooses its best matching peak, and those matches are averaged uniformly. The two directional contributions have equal weight.

The weights depend on the entire candidate set: a peak that matches all alternatives equally provides little discrimination. Several peaks may select the same atom, so this is not a one-to-one assignment or verified fragment interpretation. The heatmap is illustrative.

**Point and pause:** Continue with the same illustrative five-by-seven compatibility matrix from the previous slide, now marking each row maximum in white. Its seven columns correspond to the phenol example's heavy atoms. Trace a row and then a column to explain both matching directions. The exponential bonus rewards closer mass agreement; high entropy means a peak supports alternatives similarly and receives less discriminative weight. The 0–1 scale is illustrative, not a probability normalization or real-score bound. Changing the candidate pool can change the score.

**Transition:** This scorer is trained through a sequence of increasingly task-specific stages.

## 22 — Train the matcher to resolve harder alternatives

**26:35–27:40 · 65 seconds · Completed historical four-stage recipe**

First, contrastive warmup aligns local peak and atom features and also global representations. Second, candidate training focuses on molecules with the supplied formula and emphasizes informative peaks. Third, hard-negative mining selects alternatives the current scorer confuses with the truth. Fourth, mass agreement is introduced and the retrieval components continue training.

The JEPA encoder stays frozen throughout. Readout and graph-network parameters are selected by validation Hit at one. Only after training and selection do we evaluate the chosen system on full test candidate pools.

The equation simply selects the highest-scoring candidate from the formula-constrained set. This history explains what produced the result; it is not an isolated ablation proving that each individual stage is necessary or that its effect is independent of continued training.

**Point and pause:** Trace the four mini-diagrams: the positive-pair diagonal during warmup; a common peak versus a discriminative peak; up to 32 sampled negatives reduced to the hardest eight; and two nearby retained-mass fractions receiving a soft bonus. Each stage continues from the validation-selected predecessor. These are training schematics, not measured similarities or an ablation chart. Then return to the frozen-JEPA/trainable-matcher path and the candidate-selection equation.

**Transition:** This was the historical training recipe. The latest fresh run uses global molecular context and GLACIER fragment matching; here are its test results.

## 23 — Global context + GLACIER: results from 18 Sep 2026

**27:40–29:00 · 80 seconds · Latest completed benchmark result**

These results are from 18 Sep 2026. The methodology for this latest system is not included in these slides; the earlier retrieval slides explain the historical foundation.

The latest system achieves test Hit@1 of 36.94%, Hit@5 of 67.07%, Hit@10 of 78.61%, Hit@20 of 88.89%, MRR of 0.50814, mean MCES@1 of 6.14830, and median rank of 2 on 17,556 spectra. Hit@1 means the correct molecule is ranked first; Hit@5 and Hit@10 count whether it appears within the first five or ten candidates; Hit@20 extends this to the first twenty. MRR averages the reciprocal of the correct molecule's rank; higher is better.

The supplied formula-matched candidate pools contain the truth. These are per-spectrum retrieval metrics, not de novo or open-world identification.

The system adds a whole-molecule mean-pooling residual to the GINE atom features and uses GLACIER fragment pooling and gated MaxSim. Frozen JEPA features remain fixed; GLACIER proposals are generated offline with a frozen released model. The earlier slides explain the historical local-matching foundation, not the exact latest architecture or its training schedule.

The run is one seed, selected at epoch 24 using the fixed 5,000-query validation set. This result does not isolate the contribution of individual components. GLACIER checkpoint training-set disjointness from this benchmark has not been established; do not claim absence of pretraining overlap.

Mean MCES@1 measures structural distance between the top-ranked candidate and the true molecule; lower is better. The reported protocol uses myopic-mces 1.2.2 with threshold 15: distances are exact below the threshold and may be lower bounds above it, rather than globally exact or simply clipped. Zero MCES need not be an exact benchmark hit because not all stereochemistry is encoded. All query distances are included. Median rank 2 describes the middle of the true-molecule rank distribution; lower is better.

**Point and pause:** Start with 36.94% and explain that this is the fraction ranked correctly at first place. Scan the seven metric rows, explaining the larger retrieval windows and reciprocal rank. Finish with the evaluation scope: 17,556 spectra and supplied formula-matched pools containing the target.

**Transition:** Close with what this establishes and which research questions remain.

## 24 — Learn the evidence. Make chemistry easier to read.

**29:00–30:00 · 60 seconds · Conclusions and proposed next steps**

The generation route uses predicted chemical features to condition a molecular decoder. The representation route learns first from unlabeled spectra, with a frozen MAE teacher providing stable latent prediction targets. Retrieval then tests whether those features can support a real structure-ranking task.

Our analysis explains why prediction loss alone cannot settle representation quality: the predictor can change how information is encoded, which directions receive gradient, and which predictable patterns are learned first. The next controlled tests should measure accessible downstream information directly.

Pair states and triangle operations add an explicit way to represent relationships among peaks. Their implementation is established; their downstream advantage remains to be shown. The reported 36.94 percent comes from frozen single-stream JEPA features plus global context and GLACIER fragment matching.

**Point and pause:** Follow the three visual columns: fingerprint to AR tokens; frozen teacher and student meeting at the latent loss; then the solid student-to-global-context-and-GLACIER path leading to 36.94 percent. Separately trace the dashed triangle-to-FLARE arrow and its question mark. Solid marks the documented system; dashed marks a proposed downstream test. The pair branch is not silently added to the reported result.

**Closing:** The research objective is a representation that makes chemical evidence easier to use. Stop at 30:00 and invite questions; use the appendix for derivations and protocol detail.

## Appendix discussion cues (untimed)

The main talk still ends at 30:00. These pages support questions; coordinate strips are symbolic containers, not measured features.

- **25 — Conditional means.** The first loss term is unpredictable target variation; the second is predictive information lost when C becomes Z. If the conditional mean given C,Q equals the mean given Q alone, context contributes no conditional-mean information and a constant encoder can be optimal. This is the unrestricted squared-error result; coordinatewise absolute error instead leads to conditional medians. A frozen teacher removes joint target drift, not every route to an uninformative student.
- **26 — Exact gradient flow.** Read the two residual-decay terms as encoder and predictor contributions for fixed masked examples and unregularized continuous gradient descent. The small vector diagram shows the differential relation dŷ = Jg dz; its squared norm induces Gz = JgᵀJg. These are local differential statements. Jacobians can vary during training. Raw gradient norms are not an exact attribution under AdamW, momentum or weight decay.
- **27 — A solvable linear model.** Follow C through W to Z, then P to the predicted teacher target. Vector strips show representative coordinates, not a fixed dimensionality or measured amplitude. Centered data and whitened context yield the Frobenius factorization objective. Singular-vector alignment separates the modes; equal learning rates and balanced factors give the displayed logistic solution for ui = ai bi. The best rank-d statement describes the optimum of this idealized linear problem, not an unconditional guarantee for transformer training. In slide 15, all normalized mode products start at the same fraction rho = 0.015, so ui(0) = rho times sigma_i. The plotted quantity is ui/sigma_i for the joint encoder–predictor mapping; with balanced factors, each factor’s normalized amplitude is its square root. The half-learning time is log[(1−rho)/rho]/(2 eta sigma_i). Here sigma_i is a singular value of the context–target cross-covariance R = E[YCᵀ], so it depends on target scale as well as predictive association. It is not a pure predictability or chemical-relevance score.
- **28 — Adaptation versus capacity.** H is the Hessian of the regularized joint objective. At a differentiable inner optimum with positive-definite predictor block, eliminating predictor coordinates gives the Schur-complement profile Hessian. Predictor adaptation can relax local constraints on encoder coordinates. The lower diagram is a proposed experiment: use a weak, simple auxiliary readout on the embedding actually deployed. A universal readout could invert an unstructured code and solve the task itself; faster adaptation and larger computational capacity are different interventions. This proposal did not produce the reported 33% result.
- **29 — Implementation contract.** The cited path uses 2,000 mass bins at 0.5 Th spacing and coarse marginal groups of 2, 4, 8, 16, 32 and 64. Intensity is an input, with zero active reconstruction weight. Both the target projector and target normalization are identity; the teacher is frozen, with no EMA. Masked queries use a learned mask token and ordinal position without hidden target m/z or intensity content. The 1.8B family has 25 encoder layers, 5 predictor layers, width 2,048 and 127 m/z-ordered peak slots plus CLS. A proposed query-and-metadata-only control measures the added value of context, including information carried by ordinal positions.
- **30 — Retrieval details.** For each candidate atom, compare peak/precursor m/z fractions with up to six retained mass fractions from incident bridge cuts and the intact graph. This is a soft compatibility heuristic, not exhaustive fragment enumeration or a physical fragment assignment. The dashed mark represents a synthetic observed peak fraction; the arrow selects its nearest solid candidate fraction. The two probability glyphs share one vertical scale: a height of 0.45 drawing units means probability one. Their values are (1/3, 1/3, 1/3) and (0.0778, 0.8889, 0.0333), with the latter values rounded for description. Their entropy gives low and high relative peak weight. Normalize across valid peaks, use floors and the singleton-pool guard, and retain the two directional terms with weight one-half each. Candidate pools affect weighting. Test Hit@1 counts exact target-SMILES rank-one outcomes over 17,556 spectra after checkpoint selection on validation data.
- **31–32 — References and provenance.** These are lookup pages, with linked papers, repository versions and bundled evidence. They do not add claims to the main talk.
- **33 — Mathematical companion.** For an invertible A, the compositions Af and g∘A⁻¹ produce identical predictions if the respective model classes can represent them. General shears and rescalings can alter similarity geometry, while a pure orthogonal rotation preserves cosine similarity. For a fixed example or stacked batch under unregularized continuous gradient descent, direct latent feedback is Jgᵀr and the encoder moves as ż = −ηe Jf Jfᵀ Jgᵀr. A predictor-nullspace direction has zero direct latent-gradient component, but shared encoder parameters can still move it. These equations do not directly describe Adam optimizer-state dynamics.

## Timing checkpoints

| After slide | Elapsed | Topic completed |
|---|---:|---|
| 06 | 07:05 | DiffMS and autoregressive generation |
| 11 | 13:20 | Data, metadata, and two-stage pretraining |
| 16 | 20:10 | Mathematical interpretation and diagnostics |
| 19 | 23:55 | Pair and triangle modeling |
| 23 | 29:00 | Retrieval method and benchmark |
| 24 | 30:00 | Research direction |

## Short answers for likely questions

- **Does a fixed teacher prevent collapse?** It prevents jointly moving the target toward a trivial solution. It does not guarantee a high-rank student or good retrieval. Query/metadata shortcuts and unused latent directions still need checks.
- **Why can the student outperform the teacher?** Predicting from partial context can favor reproducible signal over unpredictable variation, and the student has its own representation. This is a possible mechanism, not a guaranteed outcome.
- **Is this identical to SALT?** It shares the reconstruction-pretrained frozen-teacher idea. The spectral task, MSE objective, masking, architecture, and optimizer recipe differ.
- **Are the triangle models responsible for 36.94 percent?** No. The documented result uses single-stream JEPA peak features and global context and GLACIER fragment matching.
- **Is a peak–atom match a fragment assignment?** No. The scorer permits many-to-one maxima and learns compatibility; it does not enforce a physical fragment explanation.
- **What does “mass agreement” mean?** It softly compares peak-to-precursor m/z fractions with a limited set of graph-component mass fractions from incident bridge cuts. It does not fully model adducts, protonation, charges, or arbitrary fragmentation.
- **What does the result establish?** Global context + GLACIER achieves 36.94% test Hit@1 in the specified formula-constrained retrieval setting. This single-seed result does not establish each component's individual contribution.
- **Can the retrieval system identify a molecule absent from the candidate list?** This benchmark does not test that. A ranking system cannot select a missing candidate; generation and open-set evaluation are different tasks.

Source details remain in the slide footnotes, `research/model_evidence.md`, `research/prior_models_and_scale.md`, and the repository's `docs/original_flare_recipe_math.md`. Keep the actual, illustrative, and proposed labels when adapting these notes.
