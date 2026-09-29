# Presentations

Each standalone LaTeX presentation has its own directory, source, PDF and `build.sh`. Supporting notes, figures, evidence and build output live with their deck. The two overviews have their own directories; the periodic overview reuses figures and measured values from the full report.

| Presentation | PDF | Source and notes |
|---|---|---|
| Spectrum representation learning | [33-slide deck](spectra/spectra.pdf) | [Directory](spectra/README.md) |
| Retrieval: the 44.10% recipe | [14-slide deck](retrieval_44/retrieval_44.pdf) | [Directory](retrieval_44/README.md) |
| Retrieval method overview | [Four-slide deck](retrieval_44_overview/retrieval_44_overview.pdf) | [Directory](retrieval_44_overview/README.md) |
| Retrieval failure analysis | [17-slide deck](retrieval_failures/retrieval_failures.pdf) | [Directory](retrieval_failures/README.md) |
| Periodic teacher semantics | [21-slide deck](periodic_semantics/periodic_semantics.pdf) | [Findings and sources](periodic_semantics/README.md) |
| Periodic teacher overview | [Five-slide deck](periodic_semantics_overview/periodic_semantics_overview.pdf) | [Directory](periodic_semantics_overview/README.md) |

`shared/` contains the unbranded presentation theme, fonts, figure-production guide and common build helper. Headers and PDF metadata contain no company label. Original external source URLs and asset provenance remain intact.

Build all six decks from the repository root:

```sh
bash presentations/build.sh
```

Or build one deck:

```sh
bash presentations/periodic_semantics/build.sh
```

Use the repository's `.venv/bin/python` for figure and analysis scripts. LuaLaTeX and the existing fonts are required; see [figure standards](shared/figure_standards.md). PDFs, local build output and source bundles are generated artifacts. Each source bundle preserves the directory layout and includes the shared files it needs.

Git tracks the deck sources, shared build tools, notes and small measured inputs for the periodic-semantics plots. Compiled PDFs, generated plot SVGs, archives, logs, TeX auxiliaries and licensed PP Neue Montreal fonts stay local. Regenerate the periodic-semantics figures with `.venv/bin/python presentations/periodic_semantics/make_periodic_semantics_figures.py` before building that deck from a fresh checkout; the other decks' figure-generation requirements are documented in their own READMEs.
