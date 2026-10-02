# Diffusion encoder supplement

Seven-slide supplement to the existing native encoder-chemistry study, using the shared presentation theme.

- [PDF report](diffusion_chemistry.pdf)
- [Editable LaTeX source](diffusion_chemistry.tex)
- [Speaker notes and metric definitions](diffusion_chemistry_notes.md)
- [Evidence index and source hashes](research/evidence_index.json)
- [Source and compact evidence bundle](diffusion_chemistry_source.zip)

The new encoder is the generation-pinned latest committed checkpoint at step 831,000 from `adiff-1p8b-ep-v6e32-4d-20260922-080322`. It is evaluated on the same molecule panel and final physical-token readout protocol as parent 1,190k and periodic 1,209k. The raw-block-19 sparse analysis is a separate endpoint; reconstruction quality does not establish chemical semantics.

Diffusion final-token attention MACCS AUROC is 0.7244, below parent 0.7987 and periodic 0.7750. Its primary sparse autoencoder explains 92.8% of held-out variance relative to the training mean. This does not mean uniformly lower reconstruction error: original-unit relative L2 is higher than both references. Only three learned chemical candidates passed the frozen selection rules; the strongest hydroxyl association also tracks molecular mass.

The editable TikZ diagram is in `figures/native_token_sae.tex`. Its arrows represent native computation and activation export, not chemical fragments or causal feature interventions.

`research/results.tex` is generated from completed analysis outputs before the deck compiles. Missing results prevent a final report build. The prior deck, shared theme and existing analysis artifacts remain unchanged.

From the repository root:

```sh
bash presentations/diffusion_chemistry/build.sh
```

The build reads frozen compact measured inputs, regenerates all measured vector figures, compiles the editable diagram and shared-theme LaTeX, inspects all PDFs at 180 dpi, then packages the source. It does not run models or allocate GPUs. It requires the repository's existing `.venv`, LuaLaTeX and Poppler utilities.

The self-contained [plotting code](make_figures.py) preserves the completed analyzer's plotting functions, with its exact source hash recorded. [Compact inputs](research/plot_inputs.json) retain every plotted value, interval and numerical caption, along with original artifact hashes. Rebuilding needs no full artifact directory, checkpoint state or activation arrays; those are excluded from the small presentation bundle.

Primary AUROC averages three seed logits before scoring on the same fixed 145 MACCS targets. Paired molecule bootstrap intervals are descriptive and conditional on the fitted checkpoints/readouts; 980 of 1,000 MACCS resamples retain every fixed target. Feature-association intervals likewise condition on the frozen feature candidates. There is no diffusion feature intervention, validated fragment assignment or isolated comparison of training objectives.
