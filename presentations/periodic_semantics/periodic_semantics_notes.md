# Presenter notes — periodic teacher promotion

Suggested timing: 20–25 minutes for slides 1–17, then optional appendix. The five-slide overview takes about five minutes. The measured question is whether this endpoint improves chemical semantics over the actual initial teacher; it is not whether every bootstrapping algorithm works.

1. **Verdict.** Lead with the tradeoff. There is positive evidence for global ordering and dropout stability, but most chemical endpoints decline. The headline −0.020 is an absolute AUROC difference, not a percentage change in accuracy.
2. **Three definitions.** Separate decodability, metric geometry and robustness. A supervised probe can recover a weak signal from coordinates whose cosine distances are unhelpful.
3. **Lineage.** The initial teacher is already JEPA-trained. The periodic student starts randomly; it does not continue the initial teacher's student weights. Copying every 50k includes encoder buffers and does not reset the optimizer. The latest saved step exceeds the last W&B log by 250 updates.
4. **Loss.** These are real logs, not analytic sketches. The scale falls across generations, but the targets also change. Do not use the loss reduction to conclude that more chemistry was learned.
5. **Depth sweep.** This is existing evidence, reanalyzed rather than rerun. Include random and MAE baselines so retention of meaningful information is visible. Layer 19 is an exploratory internal readout, not a validation-selected deployment winner. Legacy pair ranks and pair normalization have caveats documented in the appendix.
6. **New molecule-level probes.** Explain why 2,998 test molecules is the relevant uncertainty unit despite 17,556 test spectra. Arithmetic centroids use molecular identity and are intentionally distinguished from single-spectrum deployment. Regularization is selected on validation. Only 151 MACCS bits have both test classes.
7. **Geometry.** Raw cosine is the most direct test of naturally organized distances. Centering, scaling and whitening are unsupervised postprocessing fitted on training molecules. All variants retain weaker local neighbors for the latest student. Meanwhile raw partial global correlation rises; the objectives differ. The physical precursor window is ±5 Th after undoing input normalization; see the report for the additional single-spectrum control.
8. **Intervals.** The chemical-probe and neighbor differences are paired on identical molecules. They exceed this sampling uncertainty. These intervals do not capture random training seeds, the full source of dataset bias, or multiple-testing uncertainty.
9. **Covariance.** Avoid saying “the model collapsed.” Mean effective rank increases, while CLS variance becomes concentrated in one principal component. This is a pooling-specific finding; global isotropy and chemical discriminability are not equivalent.
10. **Positive evidence.** Give the global correlation increase explicit attention. Also show how standardization changes the CLS result. The same checkpoint can improve a broad similarity ranking while losing fine local distinctions.
11. **Label efficiency.** One percent means 227 labeled training molecules. All three budgets favor the initial teacher. This is a fixed deterministic subset per budget, not a seed sweep.
12. **Properties.** All seven final-mean property regressions favor the initial teacher. TPSA and H-bond acceptor scores move less than ring-related properties. These descriptors are proxies for aspects of molecular structure.
13. **Robustness.** Peak dropout and acquisition-zeroing give opposite results. Describe acquisition dependence as a hypothesis supported by a perturbation, not proven causal shortcut learning. The dropout removes visible peaks without compacting their positions. The metadata intervention zeroes charge/polarity too, so it can be out of distribution. Actual repeat-spectrum matching decreases for mean pooling and increases modestly for CLS.
14. **Attention readout.** Existing two-seed attentive training does not reverse the teacher/student ordering. Small marks are seed observations, not 95% intervals. Mean-linear token pooling includes CLS and differs from the valid-peak mean in the new ridge audit.
15. **Mechanism.** The conditional-mean equation is exact only for fixed targets and an unrestricted squared-error predictor. It is not a theorem about convergence of this finite network. Promotion does not provide a separate molecular objective, and a constant code is a possible fixed point of the unconstrained formulation.
16. **Boundary of the conclusion.** Never turn the endpoint comparison into an isolated causal claim. The initial and periodic models have different training histories, and no matched never-promoted fresh-student run is available. Pretraining overlap was not audited.
17. **Next experiment.** Present the matched three-arm experiment as a proposal. Shared batches, masks and seeds matter. Save pre/post-promotion states and evaluate across time; do not select a rule on this repeatedly examined test set.
18. **Protocol appendix.** Distinguish new measurements from reused supporting evidence. Training remained JAX; existing frozen encoder exports were evaluated in PyTorch without any checkpoint updates.
19. **Single spectrum.** Repeating local-neighbor comparisons with only the first measured spectrum per molecule preserves the teacher advantage. The physical ±5 Th window narrows the gap; this is not an identity-averaging artifact. No spectrum is chosen by chemical outcome.
20. **Controls.** Random neighbors, binned spectra and shuffled labels put the chemical signal in context. Correlation-adjustment controls are not a universal causal deconfounder. Single-spectrum checks address one centroid limitation, not every benchmark limitation.
21. **References.** Papers provide context about self-distillation, masking and covariance control. The answer about this run comes from the saved measurements, not analogy to vision models.

## Likely questions

**“So is the original idea true?”** There is a plausible mechanism and one positive global-geometry result here. The stronger claim of automatic or broad improvement is not supported. Chemistry remains present, with measurable tradeoffs.

**“Is the initial teacher simply better because of more pretraining?”** The histories differ, which is precisely why the endpoint audit cannot isolate promotion's causal effect. A matched fresh-student control is necessary.

**“Would whitening solve it?”** Whitening helps local neighbor chemistry in both models, but the periodic endpoint remains weaker. It also reduces the random-pair global correlation. It is a coordinate intervention, not recovery of all lost information.

**“Could the projection be misleading?”** No low-dimensional projection is used as evidence. Every displayed curve is a measured metric in the original or explicitly transformed representation space.

**“Do error bars show robustness across training runs?”** No. They resample test molecules with the trained models fixed. Only the reused attentive probes have two downstream-training seeds.
