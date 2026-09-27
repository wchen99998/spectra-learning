# Wider GINE + Q/K with global context and GLACIER fragments

One fresh seed-0 combined run adds a whole-molecule mean-pooling MLP residual to the three-layer width-496 GINE and replaces atom matching with learned GLACIER fragment pooling and gated MaxSim. All 4,101,195 retrieval parameters initialize afresh. The raw frozen JEPA tokens, original first-16 negative bank with eight sampled negatives, symmetric loss, and validation selection remain the original recipe.

| Run | Parameters | Best validation Hit@1 | Test Hit@1 | Hit@5 | Hit@10 | Hit@20 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original wider GINE + Q/K | 3,103,586 | 33.10% | 32.4732% | 64.2914% | 76.8398% | 86.5573% | 0.469080 |
| Global context + GLACIER | 4,101,195 | 38.06% | 36.9446% | 67.0654% | 78.6113% | 88.8927% | 0.508139 |

Selected epoch **24 of 32** on the fixed 5,000 validation queries. The full 17,556 test spectra, truth identities, and candidate counts match the original run.

Paired test Hit@1 changes by **+4.4714 percentage points**: 2,487 gains and 1,702 losses. There are 0 queries with different stable and optimistic truth ranks.

This is a combined architecture change with more trainable parameters; the comparison does not isolate the pooled context from fragment readout, gating, or mass-agreement changes. Mean pooling gives each atom molecule-wide summary information but does not distinguish all global topologies. The new evaluator uses all atoms for fragment pooling and retains float32 fragment masses/gates, while projected Q/K vectors are cached in float16. The original atom readout scored at most 96 atoms.

Proposals were regenerated with the pinned official MSG1 GLACIER release at collision energies 20/30/40, fixed [M+H]+ and Orbitrap conditions. They are not claimed byte-identical to the unavailable upstream frags_full.npz. The proposal network runs frozen in a separate PyTorch preprocessing environment; retrieval training and gradients remain in JAX. Checkpoint training-set disjointness has not been established, so this comparison does not support such a claim.

The complete graph universe contains 4 explicitly unsupported GLACIER molecules and 15 total intact-only rows. All rows remain in retrieval; unsupported prediction rows use the intact anchor. These are single-seed results selected without test-based tuning.

- [Recipe](../../docs/atom_flare_global_glacier_recipe.md)
- [Metrics and history](../../results/atom_flare_gine3m_qk_global_glacier_s0_20260918/run/metrics.json)
- [Run manifest](../../results/atom_flare_gine3m_qk_global_glacier_s0_20260918/run/manifest.json)
- [Completion audit](../../results/atom_flare_gine3m_qk_global_glacier_s0_20260918/completion_audit.json)


Presentation provenance: copied from the completed repository benchmark report. Source links above refer to the repository docs/results tree; metric values were also supplied explicitly by Wuhao. Slides 20–22 retain the historical matcher explanation and are labeled accordingly.

## Expanded displayed results

Latest Global + GLACIER: Hit@1 36.94%; Hit@5 67.07%; Hit@10 78.61%; Hit@20 88.89%; MRR 0.50814; mean MCES@1 6.14830; median rank 2. Supplied by Wuhao and checked against `results/atom_flare_gine3m_qk_global_glacier_s0_20260918/full_results/report.md`. MCES uses myopic-mces 1.2.2, threshold 15; exact below threshold, potentially lower bounds above it. All query distances included; zero does not establish exact stereochemical identity.
