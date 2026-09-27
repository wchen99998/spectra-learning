# The 44.10% retrieval recipe

A separate **14-slide, 16:9 Novogaia Beamer presentation**: 12 main slides and two technical appendix slides. It reuses the existing logo, bundled fonts, monochrome palette, page grid and vector-diagram format. The original `novogaia_spectra` presentation is preserved.

- [Presentation PDF](retrieval_44.pdf)
- [Editable deck](retrieval_44.tex) and [context/fragment diagrams](retrieval_44_context.tex)
- [Speaker notes](retrieval_44_notes.md)
- [Verified numbers, equations and source evidence](research/retrieval_44_evidence.md)
- [Portable source bundle](retrieval_44_source.zip)
- [Complete method document](../docs/retrieval_recipe.md)

The story starts from the **36.94% test result in the previous presentation**. Global GINE context and GLACIER fragments are identified as retained foundations. The new changes are full-pool negative sampling, precise-ion scoring with a forward log-mixture, candidate-conditioned cross-attention, and warm joint encoder/head adaptation. The wider-GINE 32.47% result appears only as an earlier historical milestone.

The selected checkpoint achieved **46.02% validation** and **44.10% held-out test Hit@1**. Test results weight all 17,556 spectra equally. The presentation distinguishes historical combined changes from controlled ablations and explicitly notes the small validation difference between matched head-only and joint continuation. It includes MRR, mean/median rank and the threshold-15 MCES convention.

## Rebuild

From the repository root, with LuaLaTeX and the same TeX packages as the original deck:

```sh
presentations/build_retrieval_44.sh
.venv/bin/python presentations/inspect_retrieval_44.py
```

Build auxiliaries and 180dpi review previews stay under `presentations/build/retrieval_44/`. The final PDF is `presentations/retrieval_44.pdf`. Python inspection uses the repository's existing virtual environment and installed PyMuPDF/Pillow. No training code or dependencies are changed.

The source archive can be extracted into a fresh directory. It includes the presentation sources, shared theme, required fonts/logo, build and inspection scripts, speaker notes and evidence. Run the bundled build script from that directory. Research links in the evidence/notes point into the original repository; they are not needed to compile the deck.

## Diagrams and review

All diagrams are editable TikZ and remain vector graphics with embedded, searchable text in the PDF. Molecular drawings, peak traces, attention bars, sampling grids and compatibility heatmaps are explicitly illustrative. Only the historical test bars, joint-validation curve and test-retrieval bars represent measured values.

Every slide is rendered and inspected for internal overlaps, arrow endpoints and caption clearance. The automated audit checks page boundaries, embedded fonts, vector artwork and LaTeX warnings. The final [page review](retrieval_44_page_review.md) records the visual checks; the machine-readable audit is saved under `build/retrieval_44/layout-audit.json`.
