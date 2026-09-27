# Novogaia: learning to decode molecular structure

**General method overview:** [Four-slide PDF](retrieval_44_overview.pdf) · [Editable source](retrieval_44_overview.tex). A short visual explanation of candidate matching, broader negative sampling and joint adaptation, without detailed equations.

**New retrieval follow-up:** [The 44.10% recipe](retrieval_44_README.md) — a separate 14-slide deck explaining the changes from this presentation's 36.94% result, with architecture/scoring diagrams, full test metrics, editable sources and speaker notes. [Open the PDF](retrieval_44.pdf).

A 16:9 LaTeX Beamer presentation for a **30-minute English talk**: 24 main slides and 9 optional mathematical/reference slides. The main talk is diagram-led, with short labels and equations; extended explanations live in the speaker notes and appendix.

Data flows use the scientific objects themselves: spectral peak traces, molecular structures, fingerprint strips, feature arrays, SMILES tokens and peak–atom compatibility matrices. Boxes denote model operations. Toy spectra, feature colors and matching values illustrate the computation; they are not experimental measurements or learned activations.

The dynamics plot uses a transparent background in the slides to match the Novogaia canvas; its standalone publication export retains a white background.

## Deliverables

- `novogaia_spectra.pdf` — compiled presentation.
- `dynamics_explainer.pdf` — four-page excerpt of the revised dynamics story (slides 12–15).
- `novogaia_spectra.tex` — editable content and TikZ diagrams.
- `novogaia-theme.sty` — Novogaia typography, colors, logo and page footer.
- `speaker_notes.md` — slide-by-slide presenter prompts, timed to exactly 30 minutes, plus untimed appendix cues and Q&A.
- `assets/` — vector logo, analytic plot, website fonts, and open Liberation Sans figure fonts.
- `figures/` — 27 standalone vector figures at Nature publication widths, with editable sources.
- `figure_standards.md` — sourced Nature artwork requirements and audit criteria.
- `page_review.md` — findings and revisions for every page, including the appendix.
- `research/` — source evidence and the supplied dynamics analysis.
- `novogaia_spectra_source.zip` — portable source, assets, notes and research bundle.

## Build

From the repository root:

```sh
presentations/build.sh
```

Requires LuaLaTeX with Beamer, TikZ, fontspec, unicode-math and adjustbox. Liberation Sans is bundled. PP Neue Montreal is licensed and not tracked in git: place its `PPNeueMontreal-{Book,Medium,Bold,Italic}.otf` files in `assets/` before building. Compiled PDFs, `figures/*.pdf` and the `*_source.zip` bundles are build outputs and are gitignored. On Ubuntu, the required TeX components are available through `texlive-luatex texlive-latex-extra texlive-fonts-recommended`. The build runs twice and places the final PDF in this directory. Auxiliary files stay in `build/`.

To regenerate the analytic plot or rendered contact sheets, use this repository's existing virtual environment:

```sh
.venv/bin/python presentations/make_dynamics_figure.py
.venv/bin/python presentations/inspect_pdf.py
.venv/bin/python presentations/export_figures.py
.venv/bin/python presentations/verify_artifacts.py
```

Plot generation uses Matplotlib and NumPy. Figure export/auditing uses PyMuPDF and Pillow. Inspection also requires Poppler (`pdftoppm` and `pdftotext`). Editable scientific figures and all diagrams remain vector graphics in the PDF. Images in `build/preview/` are only review previews.

## Scientific scope

- **DiffMS:** its own MIST-derived encoder is pretrained with fingerprints; final graph diffusion is conditioned on its learned spectrum embedding. Formula fixes heavy-atom counts and types. Implicit hydrogen counts can vary.
- **Novogaia AR:** the requested autoregressive implementation was verified in the existing local `Novogaia/md4` repository's **`ar` branch**, commit `e804f677c0a2a44cc5388cc303d1c9c51183091f`. Formula-prefix conditioning is not a hard formula guarantee.
- **Scale:** 1B+ is the raw public-data opportunity, not an asserted count used by a particular Novogaia checkpoint. DreaMS's selected 23.5M-spectrum training subset is labeled separately.
- **Two-stage training:** masked mass reconstruction first; then a fixed MAE teacher supervises masked per-peak latent MSE. The cited recipe uses identity target normalization and no EMA teacher. Exported student features are evaluated separately from the predictor.
- **Dynamics:** mathematical curves are analytic illustrations, not observed training telemetry. Exact gradient-flow/linear-model results have explicit assumptions. Proposed readout/optimizer experiments are labeled as proposals.
- **Pairs:** triangle multiplication and attention are implemented in a separate pair-model direction. The 33% result does not use pair-stream features.
- **Result:** Global context + GLACIER achieves **36.94% test Hit@1**, **67.07% Hit@5**, **78.61% Hit@10**, **88.89% Hit@20**, **0.50814 MRR**, **6.14830 mean MCES@1**, and **median rank 2** on 17,556 spectra. Formula-constrained retrieval from truth-containing pools; one validation-selected seed. Earlier matcher slides describe the historical foundation. See `research/latest_retrieval_result.md`.

Pages 12–15 explain the dynamics through numerical codes, alternative hidden spectra, explicit feedback and normalized learning-speed curves. Their general coordinate and gradient equations are collected on optional slide 33.

The main talk ends at slide 24. Appendix slides provide conditional-mean analysis, exact gradient flow, deep-linear modes, predictor profiling, implementation contracts, retrieval equations and linked references.

## Visual identity

The deck uses the actual Novogaia website logo, PP Neue Montreal titles and wordmark, standard sans-serif figure labels, near-black `#0C0C0C`, and off-white `#F6F7F8`. Source URLs are in `research/asset_sources.md`. Website fonts/logo are bundled for this requested Novogaia presentation; their original rights remain with their owners.

Confidentiality notices have been removed from every slide.

## Verification

The deck was built with LuaLaTeX and visually reviewed using high-resolution renders. The alignment pass uses named TikZ anchors for pipeline connectors, explicit equation baselines, consistent page margins, and clear separation of body content, takeaways, citations and page footers. Standalone exports are inspected separately because their font metrics differ. Standalone scientific figures follow the linked Nature production guidance: white background, standard sans-serif labels, live embedded text, print-size typography and consistent vector strokes. The separate figure audit records actual dimensions, font sizes, stroke widths and absence of raster art; Nature’s editors retain final layout decisions. The final audit checks 33 pages, the absence of confidentiality notices, no text outside the page margins, no missing glyphs or overfull boxes, and vector-only PDF artwork. No model/training source files were changed.

Result-slide annotation: results dated **18 Sep 2026**; **methodology for these results is not included in these slides**. Earlier retrieval diagrams describe the historical foundation.
