# MIST as a second spectrum modality for the 44.10% retrieval model

[16-slide PDF](mist_modality.pdf) · [Editable LaTeX](mist_modality.tex) · [Figure script](make_figures.py) · [Measured inputs](research/)

This deck covers the 2 October 2026 experiments that add FRIGID's MassSpecGym MIST checkpoint (`mist_msg.pt`) to the single-model 44.10% formula-candidate retrieval recipe. There are two routes:

- **Late score fusion:** both models frozen; one similarity type and one weight are chosen on validation.
- **Trained-in auxiliary modality:** MIST h0, fp or h0+fp features enter the retrieval head through a zero-initialized term.

Fusion with Bernoulli similarity at w = 0.9 reaches 61.52% test Hit@1, against 44.10% for the base model (+17.42 pp, 95% CI 14.80–20.12). Every trained-in variant is 0.2–2.1 pp below its control and leaves less for fusion to add.

Scope and caveats:

- **Seeds and stages:** each head run is a single seed (0). Only the head stage was trained. The joint stage was skipped because no auxiliary head beat the control.
- **Evaluation:**
  - Selection uses the fixed 5,000-spectrum validation subset.
  - Test (17,556 spectra) was evaluated once at the selected setting.
  - Ranks are pessimistic: ties count against the truth.
- **Slide 13 hypothesis:** MIST is sharper on its own training fold, so the trained-in term over-trusts it. This is consistent with the curves but was not isolated by a separate experiment. In particular, out-of-fold MIST features were not tried.
- **Control head:** it sits 1.28 pp below the original source head on validation, which is seed and hardware variance.

Build from the repository root:

```sh
bash presentations/mist_modality/build.sh
```

The build uses the repository `.venv`, Matplotlib, LuaLaTeX and the shared `../shared/` theme and fonts. `make_figures.py` reads only the JSON files in `research/` and writes five vector figures into `figures/`. No GPU, training outputs or network access is needed. As with the other decks, the compiled PDF and build output are local artifacts and are not committed. The 16-page build was visually reviewed page by page.

Slide sources (page numbers):

| Pages | Content | Source |
|---|---|---|
| 1, 15 | Headline numbers and decision | `research/fusion_sel4602.json` and `research/fusion_head_*.json` |
| 3 | MIST encoder | FRIGID `SpectraEncoderGrowing` MSG config |
| 4 | Export and fold checks | `research/mist_export.json` and `research/overlap_sel4602.json` |
| 5–9 | Quality, scoring, fusion curve and Hit@k | `research/fusion_sel4602.json` |
| 10 | Complementarity grid | `research/overlap_sel4602.json` |
| 12–13 | Head-stage curves and table | `research/head_curves.json` |
| 14 | Head runs plus fusion | `research/fusion_head_{control,mist_h0fp,mist_h0,mist_fp}.json` |

The fusion procedure and bootstrap on page 7 follow `fuse_mist_scores.py`, which clusters by the first InChIKey block (1,000 resamples). The diagrams on pages 2, 3, 6 and 11 are schematic. Their bit, probability, token and spectrum shading is illustrative, not measured output.

The MIST code, feature export and fusion scripts are kept out of this repository. They live in the separate `mist-modality` workspace (`export_mist_features.py`, `fuse_mist_scores.py`, `make_aux_inputs.py`, `overlap_mist.py`; summary in `results/RESULTS.md`). Isambard outputs are under `/scratch/u6rb/wuhao.u6rb/mist-modality/{runs,scores,fusion_*}`; the head stage is Slurm job 7007408. Only the small measured JSON reports are copied here.
