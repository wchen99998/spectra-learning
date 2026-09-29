# Where formula-candidate retrieval fails: presenter notes

**Scope:** failure analysis of the validation-selected 15-member ensemble on MassSpecGym formula-candidate retrieval, 29 September 2026. These prompts cover 16 main slides and one appendix, about 15 minutes before questions. Every number is in [the evidence sheet](research/retrieval_failures_evidence.md); the full tables are in [ANALYSIS.md](../../results/retrieval_failure_analysis_20260929/ANALYSIS.md).

**Delivery:** test figures are post hoc diagnostics. The model was selected on validation before this analysis. Validation is shown next to test on most charts. Uncertainty is molecule-clustered, so quote intervals, not spectrum counts, when comparing strata.

## 01 — Where retrieval fails

The ensemble ranks the true molecule first for 47.33% of 17,556 test spectra. The 95% interval, resampling molecules, is 43.6–51.2%. Four headline numbers:
- 9,246 spectra are wrong.
- 468 more correct spectra would reach 50%.
- 72% of misses sit in pools that contain a close analogue of the truth.
- 86% of misses fit one of three failure modes.

**Transition:** first, how to read the numbers.

## 02 — Protocol

The subject is one fixed model. Test spectra cluster heavily, about six per molecule, so a spectrum-level interval (±0.7 points) overstates precision. The molecule-clustered interval is about ±4 points. Differences below about one point between single runs cannot be resolved here.

**Point and pause:** "replicates on validation" is the credibility check for each pattern.

## 03 — The failure map

Read left to right:
- **Mass:** accuracy peaks at 67.6% for 300–400 Da and falls to about a third above 500 Da.
- **Adduct:** sodium adducts reach 33.6%.
- **Instrument:** QTOF reaches 32.6%.
- **Spectrum size:** spectra with five or fewer peaks reach 32.3%.

Open circles are validation; the patterns match.

**Transition:** these groups overlap. Are they one effect?

## 04 — Mass and acquisition fail independently

- **Sodium:** costs 18 points below 500 Da but only 5 above, because large protonated spectra are already at 37%.
- **QTOF:** costs 26 points for small molecules.
- **Mass alone:** large Orbitrap spectra (36%) are still far below small ones (63%).

## 05 — What predicts a miss

The logistic model holds the other factors fixed. The most similar decoy is the strongest predictor: every +0.1 of Tanimoto multiplies the odds of success by 0.64. Mass, sodium, QTOF and peak count keep independent effects. Training similarity does not.

## 06 — Distribution shift

- **Mass:** training is centred at 327 Da, test at 498 Da. Half the test set lies where training has 17% of its spectra.
- **QTOF:** the counter-example. It is 25% of training but 14% of test, so its weakness is not missing data.
- **Reweighting:** training toward the validation mix did not help.

## 07 — Pool difficulty

When every decoy is dissimilar (Tanimoto below 0.5), the model is right 81% of the time. With a decoy at 0.7–0.9 it drops to 26%. Half the test set is in that hard regime, and it contains 72% of the misses.

## 08 — Not memorization

Molecules with a close training analogue look terrible (14%). The drop comes entirely from decoys:
- without a close decoy, they score 98%;
- with one, 3.7%.

The model is not recalling training answers. Only 0.4% of misses choose a training molecule. Analogue series simply put near-duplicates into the same pool.

## 09 — Most misses are near misses

- **Rank:** a quarter of misses put the truth second, and 56% within the top five.
- **Similarity:** 29% choose a structure with Tanimoto at least 0.7 to the truth, often sharing its scaffold.
- **Hard floor:** these are the cases MS/MS alone may not separate.

## 10 — Three representative failures

The examples were selected by fixed rules, not by eye:
- **Near-isomer:** a cage xanthone with a rearranged ring fusion.
- **Uninformative spectrum:** three peaks, no fragment ion matching either molecule.
- **Large sodiated glycoside:** the wrong triterpene isomer explains 96% of the intensity, against 60% for the truth.

**Point and pause:** each needs a different fix.

## 11 — Fragment evidence

The scorer's mass kernel shows that on 78% of misses the wrong molecule explains at least as much intensity as the truth: strictly more in 40%, a tie in 38%. Even when the model is right, the truth explains only about half the intensity of large or sodiated spectra.

## 12 — The generator is the limit

On validation:
- **More proposals:** keeping all proposals raises coverage by 2.5 points, and plausibility-weighted ranking does not move.
- **Exhaustive enumeration:** explains more but ranks far worse.
- **Conditioning:** GLACIER regenerated with the query's own adduct produces almost the same fragments.

## 13 — What was tried

Ten single-head changes are shown against one control: regularization, readouts, a global term, intermediate encoder layers and more negatives. The best test gain is +0.6 points, and validation gains of up to +1.1 did not transfer. The spectrum tokens are not the bottleneck.

## 14 — Headroom

- **Hard core:** a third of the test set defeats all 15 members, because they share the molecule side.
- **Where the points are:** bringing large molecules up to the rest would give 59%, and close-decoy pools 70%. Sodium, QTOF or sparse spectra alone only reach about 50%.

## 15 — Three failure modes

| Mode | Share of misses | Matching lever |
|---|---:|---|
| Fragment-indistinguishable | 78% | a learned, adduct-conditioned fragmenter |
| Near-isomer | 29% | decoy-aware pairwise scoring and analogue hard negatives |
| Uninformative spectrum | 25% | abstention or priors |

These levers are proposals, not results.

## 16 — Conclusions

The failures are concentrated, independent, near and fragment-limited. The 50% target is 2.7 points away, inside the uncertainty interval. A real improvement needs to beat about one validation point and replicate on test.

## A — Definitions

Use it for questions on fingerprints, the explained-intensity kernel, the bootstrap or the regression covariates.

## Likely questions

- **Is 50% within noise?** It lies inside the 95% molecule interval of the current result. That makes the target plausible, but a claimed improvement still has to be measured against the same interval.
- **Why not reweight training toward large molecules?** It was tried at two strengths and did not improve validation.
- **Could the near-isomers be undecidable from MS/MS?** Some likely are; 29% of misses pick a structure with Tanimoto ≥0.7. That share bounds how many misses could be undecidable for this reason.
