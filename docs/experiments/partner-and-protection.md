# Experiment: partner selection vs. multi-person crop protection

*October 2026. Code: `tools/partner_experiment.py` (failure checks: `tools/crop_failures.py`).
Raw results: [`partner_experiment.json`](partner_experiment.json).*

## Question

The [embedding comparison](embedding-comparison.md) showed that the main crop weakness is the people
around the main dancer: when the model's chosen partner isn't the one the photographer kept, the
photographer's couple can be cut (heads, hands or feet cut in about 12% of crops), because the no-cut
rule only protects the chosen pair. Two fixes were tested, separately and combined, before considering a
larger redesign (group selection plus candidate-crop scoring):

| Variant | Change |
|---|---|
| **A** current | main dancer + partner from the subject classifier and a distance rule; protect that pair |
| **B** pairwise partner | a classifier trained on candidate pairs ("does this person belong with the main dancer?") using distance, relative size, vertical overlap, hand-to-hand and hip distance, relative sharpness, subject probability; crop regressor retrained on the new couple boxes |
| **C** multi-person protection | A's crop, but every other person with subject probability >= tau is also protected, unless that grows the crop beyond `cap` x its area (then only the couple is protected) |
| **D** | B + C |

## Protocol

* 7,987 cropped photos, 84 events; `GroupKFold(5)` by event, identical splits for every variant. The
  subject classifier, the pairwise classifier and the crop regressor are all trained out-of-fold; CLIP
  B/32 embedding and LightGBM settings as production, no hyperparameter tuning.
* C and D were run on a grid (tau 0.5 / 0.7 / 0.9 x cap 1.25 / 1.5 / 2 / unlimited), reported in full.
  The headline result picks a setting **per test fold using only the other four folds**, with a rule
  fixed before running: minimise couple truncation, provided mean IoU drops by at most 0.01 and
  too-loose crops rise by at most 2 points versus the baseline; otherwise fall back to the baseline.
* Paired per-event differences vs A with 95% bootstrap confidence intervals (2,000 resamples over
  events). Numbers only, run offline (zero network attempts), no photos viewed.

## Results

Share of photos (held-out events). "Couple" = the one or two largest people the photographer kept.

| | Couple cut (any) | Head | Hands | Feet | Partner missed | Other kept person sliced | Too tight | Too loose | Mean IoU |
|---|---|---|---|---|---|---|---|---|---|
| **A** current | 12.2% | 8.1% | 9.0% | 7.3% | 10.8% | 19.6% | 10.2% | 8.1% | 0.748 |
| **B** pairwise partner | 12.7% | 8.2% | 9.4% | 7.4% | 10.7% | 19.3% | 10.1% | 8.1% | 0.747 |
| **C** chosen out-of-fold | **7.9%** | 6.0% | 5.9% | 4.4% | 8.7% | 15.8% | 7.6% | 9.5% | **0.751** |
| **D** chosen out-of-fold | 8.1% | 6.2% | 5.9% | 4.6% | 8.9% | 16.2% | 7.6% | 9.3% | 0.751 |
| *Your own crops* | *-* | *1.1%* | *2.3%* | *4.1%* | | | | | |

Chosen settings: C used tau 0.7 with no cap in 4 folds (tau 0.5, cap 1.5 in one); D used tau 0.7, no
cap in all 5.

Paired per-event change vs A (positive = better):

| | Couple cut | IoU | Too loose | Too tight | Partner missed | Other kept person sliced |
|---|---|---|---|---|---|---|
| **B** | -0.1 pt [-0.9, +0.9] | +0.001 [-0.003, +0.005] | -0.2 pt [-0.7, +0.2] | +1.2 pt [-0.9, +4.1] | +0.2 pt [-0.4, +0.6] | +0.4 pt [-1.0, +2.0] |
| **C** | **+4.8 pt** [+3.4, +6.5]; 62 events better, 0 worse | **+0.006** [+0.003, +0.008] | -1.0 pt [-1.4, -0.7] | +3.2 pt [+1.6, +6.1] | +1.8 pt [+1.3, +2.4] | +6.3 pt [+3.6, +10.2] |
| **D** | +5.0 pt [+3.5, +6.9] | +0.007 [+0.003, +0.012] | -1.0 pt [-1.6, -0.6] | +3.2 pt [+1.1, +6.2] | +1.6 pt [+1.0, +2.2] | +4.5 pt [+2.5, +7.4] |

The full grid (in the JSON) shows the trade-off: lowering the threshold to 0.5 protects more people
(couple cut 6.1%) but makes more crops too loose (11.0%); 0.9 changes little (10.6%, 8.3%). The cap
mostly matters at tau 0.5.

Runtime: the protection step costs 0.01-0.03 ms per photo; the pairwise classifier is similarly
negligible. Neither changes the plugin's run time noticeably.

## Conclusions

* **Better partner picking (B) doesn't help.** The pairwise classifier, with richer pair features, picks
  essentially the same partners; the confidence intervals all include zero. "Who is the partner" is not
  where the errors come from.
* **Protecting every confidently-kept person (C) does.** Couple truncation falls by about a third
  (12.2% -> 7.9%), better in 62 events and worse in none, and crops actually match the photographer's
  slightly better (IoU +0.006, fewer too-tight crops). The cost is about 1 point more too-loose crops.
* **D brings nothing over C**, so the simpler C is preferred.
* The remaining gap to the photographer's own crops (heads cut 6% vs 1%) suggests the group-aware
  candidate-crop approach could still pay off, but C captures much of the gain with no new model.

**Recommendation:** adopt C (protect every person with subject probability >= 0.7, no area cap) in the
plugin's cropping.

## Limitations

* Same as the embedding comparison: slider/crop agreement is measured against the photographer's past
  crops; "couple" and "kept" are inferred from those crops and pose keypoints, which are imperfect in
  dim, crowded scenes; crops treated as axis-aligned for these checks.
* "Other kept person sliced" includes legitimate artistic choices, so it's an upper bound on real errors.
* The grid and the selection rule were defined before running; the headline setting was chosen on other
  folds, but the out-of-fold models for those folds were trained on data that includes the test fold
  (an indirect and very small effect on a two-parameter rule choice).

## Reproduce

```
python tools/partner_experiment.py <data_dir>
```
