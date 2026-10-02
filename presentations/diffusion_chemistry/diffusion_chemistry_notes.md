# Diffusion supplement: speaker notes

Primary final-token MACCS AUROC: 0.724376, 95% descriptive interval [0.7141743070952841, 0.734622605591546].

The protocol preserves physical peak tokens and fits the same downstream attention and mean-control budgets on each frozen native encoder.

## Paired final-token differences

- Diffusion minus parent_1190k: -0.0744 [-0.0825, -0.0661].
- Diffusion minus periodic_1209k: -0.0506 [-0.0583, -0.0427].

All intervals resample molecules jointly across checkpoints. They do not include pretraining-seed variation and are not multiplicity adjusted.

## Sparse decomposition

Raw state 19, k=32, seed 0: FVE versus the training mean is 0.927768. 3 candidates were frozen on training/validation data before test confirmation. All three fitted dictionaries per encoder are shown, including sparsity and seed controls.

This is not uniformly superior reconstruction: original-unit relative L2 is 0.259499 for diffusion, versus 0.087566 for parent and 0.053802 for periodic. Different representation means and variance scales change the denominators. The slide specifically claims higher explained centered variance, not lower error under every normalization.

Chemical-association points average all frozen candidates for a concept; absence remains a gap. Feature IDs belong to their own dictionary. Better reconstruction does not by itself establish better chemistry; no diffusion feature intervention was run.

The strongest hydroxyl candidate (f4682) has test AUROC 0.744, but also mass correlation 0.599; strict matched-pair ordering is 0.517.

The f4682 summary is a descriptive illustration of the strongest test-associated hydroxyl candidate among the already frozen list. It was not used to tune the feature list or inference protocol. Matched donor pairs reuse molecules and are not independent counterfactual observations.

## Training and provenance

All three W&B histories start at global step zero. The periodic student was freshly trained; the parent is its teacher initialization, not its own weight initialization. Nominal presentations multiply global step by global batch 1,024, with accumulation already included. These are not unique spectra counts.

The dataset revision, manifest hash and shard-plan hash are shared. Diffusion has a different objective, shorter exposure, no frozen teacher and native m/z midpoint quantization at 0.48828125 Da. This is an endpoint comparison, not a clean causal estimate of the diffusion loss.

Only train.py is archived as historical W&B source. The current native forward source is separately hashed and its actual remote execution is attested; exact native graph and cache parity are verified.

## Experiment entry points

The presentation build uses only frozen compact inputs. The original experiment is separately orchestrated by `scripts/run_encoder_chemistry_diffusion.py` in three phases:

```sh
.venv/bin/python -m scripts.run_encoder_chemistry_diffusion prepare
.venv/bin/python -m scripts.run_encoder_chemistry_diffusion fit
.venv/bin/python -m scripts.run_encoder_chemistry_diffusion download
```

Preparation requires authenticated pinned checkpoint access and is protected once the committed native cache exists. Fitting schedules the prescribed GPU readouts and SAEs; downloading retrieves their saved outputs. Rebuilding this PDF invokes none of these experiment phases.
