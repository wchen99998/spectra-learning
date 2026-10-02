# Pair-to-single information flow with pair distogram supervision

[14-slide PDF](pair_single_dual.pdf) · [Editable LaTeX](pair_single_dual.tex) · [Research worklog](../../docs/pair_single_dual_20261001.md) · [Notion worklog](https://app.notion.com/p/3ec53a6114a681929bc7c72c5dc8b894)

This presentation covers the completed waves 1–6 dated 1 October 2026: the bias-channel bottleneck, dual supervision, three-seed bridge evidence, pair-width controls, span/isotope masking, finer m/z bins, and the final `dual_bias_mean_gated` candidate. Wave-1 results are three-seed means; matched-width and other follow-up results use seed 0. Small distogram and confirm-set pair-AUC margins still need multi-seed confirmation. Matching pair width does not match total parameters or compute.

Build from the repository root:

```sh
bash presentations/pair_single_dual/build.sh
```

The build uses the repository `.venv`, Matplotlib, LuaLaTeX and the existing `../shared/` theme/fonts. `make_figures.py` reads the worklog tables directly to generate four vector PDF figures. No training outputs or network access are needed. As with the other decks, compiled PDFs and build files are generated local artifacts and are not committed.

The Codex standalone preview compiler cannot resolve the shared theme and external figures; use the project build command above for the PDF. The 14-page local export was compiled and visually reviewed, its fonts are embedded, and its two matched-width result tables were checked numerically against the worklog.

Slide sources: 1–5 use the worklog overview and §§2–6; 6–9 use §8; 10 uses §9; 11–12 use §10; 13 uses the wave-2 follow-ups in §8; 14 uses the concluding recommendation. Tables on slides 9 and 12 reproduce the selected wave-5 and full wave-6 rows. Plot values come from the worklog; no seed standard deviations are inferred from the heading because its table supplies means only. Slide 7 reports paired seed ranges, not confidence intervals.

The presentation summarises the recorded experiments; the underlying runs were not rerun or independently audited. The worklog's rounded tables can differ slightly from its deltas calculated using unrounded results. The older C128 capacity caveat is presented before the C256 results that address it. Finer bins also enlarge the heads, so the bin-width comparison does not isolate resolution from parameter count.

Candidate reproduction command. The pilot scripts were removed after the recipe became `training_mode="dual"`; check out commit `78eb2c0` to run it (flags checked against `scripts/pair_single_dual_pilot.py` there; source worklog retains its original wave-1 example):

```sh
git checkout 78eb2c0
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python -m scripts.pair_single_dual_pilot \
  --arm dual_bias_mean_gated --single-dim 256 --pair-dim 256 \
  --mz-bin-size 0.1 --masking random --seed 0 --steps 20000 \
  --inputs artifacts/pair_to_single_20260930/panel.npz \
  --confirmation artifacts/pair_to_single_long_20260930/confirmation_panel.npz \
  --output runs/dual_bias_mean_gated_c256_bin01_s0
```

The command requires the experiment scripts and data referenced by the worklog; the presentation build only needs its committed Markdown source and the existing presentation dependencies.
