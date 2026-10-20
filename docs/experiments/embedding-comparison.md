# Experiment: which image embedding should the models use?

*October 2026. Code: `tools/embed_models.py`, `tools/compare_embeddings.py`, `tools/crop_failures.py`.
Raw results: [`embedding_comparison.json`](embedding_comparison.json) and anonymised per-event results
[`embedding_comparison_events.csv`](embedding_comparison_events.csv).*

## Question

The tone and crop models combine hand-engineered features (brightness histogram, clipping, colour cast,
EXIF, pose keypoints, etc.) with a compressed image embedding from CLIP ViT-B/32, a small, older vision
model. Would a stronger backbone help, and does the embedding help at all?

Variants compared:

| Variant | Model | Params | Input | Embedding |
|---|---|---|---|---|
| `none` | hand-engineered features only | - | - | - |
| `clip_b32` (production) | open_clip ViT-B/32, `laion2b_s34b_b79k` | 151 M | 224 px | 512-d |
| `clip_l14` | open_clip ViT-L/14, `laion2b_s32b_b82k` | 428 M | 224 px | 768-d |
| `dinov2_b14` | timm `vit_base_patch14_dinov2.lvd142m` (CLS token) | 87 M | 518 px | 768-d |

## Protocol

* **Held-out events.** `GroupKFold(5)` by event folder: every photo of a test event is unseen in
  training. 10,359 edited photos / 94 events for tone (82 events with enough photos to be scored);
  7,987 cropped photos / 84 events for crops.
* **Identical for every variant:** the same folds, the same randomly chosen key photos per event (3, seed
  0), the same LightGBM parameters, the same subject-classifier probabilities for crops.
* **No leakage from test events.** The PCA compressing each embedding (32 dims for tone, 16 for crop) is
  fitted on the training folds only. The subject classifier's probabilities are out-of-fold. There was
  **no hyperparameter tuning or model selection** in this experiment; all settings were fixed beforehand.
* **What differs:** only the components that use the embedding, i.e. the per-photo exposure and white
  balance model and the crop regression. Synced sliders, subject choice and the no-cut rule are shared.
* **Uncertainty:** paired per-event differences, with 95% bootstrap confidence intervals over events
  (2,000 resamples), and counts of events that improved or got worse.
* Everything ran offline (zero network attempts recorded) and on numbers only; no photos were viewed.

## Results

### Tone: per-photo exposure and white balance (non-key photos, 3 key photos per event)

| | Exposure error (stops) | White balance error (mired) | Normalised error |
|---|---|---|---|
| One fixed preset | 0.567 | 23.2 | 0.690 |
| Sync from key photos (no model) | 0.359 | 12.6 | 0.410 |
| Features only (`none`) | 0.285 | 11.3 | 0.342 |
| **CLIP B/32 (production)** | **0.276** | 11.2 | **0.334** |
| CLIP L/14 | 0.282 | **11.1** | 0.338 |
| DINOv2 B/14 | 0.278 | 11.3 | 0.337 |

*Normalised error: mean of the exposure and white balance absolute errors, each divided by that
setting's spread.*

Paired per-event differences in normalised error (positive = the first variant is better):

| Comparison | Mean improvement [95% CI] | Events better / worse |
|---|---|---|
| CLIP B/32 vs features only | +0.0084 [+0.0031, +0.0139] | 49 / 33 |
| CLIP L/14 vs features only | +0.0060 [+0.0005, +0.0114] | 49 / 33 |
| DINOv2 vs features only | +0.0066 [+0.0008, +0.0133] | 45 / 37 |
| CLIP L/14 vs CLIP B/32 | -0.0024 [-0.0073, +0.0023] | 38 / 44 |
| DINOv2 vs CLIP B/32 | -0.0018 [-0.0059, +0.0028] | 36 / 46 |

Any embedding helps slightly and reliably (about 2% lower error than features alone). Neither bigger
model beats CLIP B/32; their confidence intervals include zero. Per event, the differences between
backbones are small and go both ways (10th-90th percentile of the per-event gain vs B/32: about
-0.03 to +0.02).

### Crops

| | Mean IoU with my crop | IoU > 0.7 | Cuts into the picked couple |
|---|---|---|---|
| Features only (`none`) | 0.744 | 66.8% | 0.1% |
| **CLIP B/32 (production)** | 0.747 | 67.8% | 0.1% |
| CLIP L/14 | 0.749 | 68.2% | 0.1% |
| DINOv2 B/14 | **0.751** | **68.9%** | 0.1% |

| Comparison (IoU) | Mean improvement [95% CI] | Events better / worse |
|---|---|---|
| CLIP B/32 vs features only | +0.0012 [-0.0043, +0.0061] | 52 / 32 |
| CLIP L/14 vs CLIP B/32 | -0.0029 [-0.0065, +0.0003] | 43 / 41 |
| DINOv2 vs CLIP B/32 | +0.0049 [+0.0006, +0.0103] | 51 / 33 |

For crops the CLIP B/32 embedding adds nothing measurable over the pose features. DINOv2 is slightly
and statistically better than B/32, but by 0.005 IoU (0.4 percentage points), which is not noticeable
in practice.

### Crop failures beyond IoU (share of photos, held-out events)

Defined in `tools/crop_failures.py`; "your crops" is the same check applied to the photographer's own
crop, as a reference.

| Failure | Features only | CLIP B/32 | CLIP L/14 | DINOv2 | Your crops |
|---|---|---|---|---|---|
| **Subject selection** | | | | | |
| Wrong main dancer | 1.9% | 1.9% | 1.9% | 1.9% | - |
| Partner you kept mostly outside the crop | 10.8% | 10.8% | 10.8% | 10.3% | - |
| Another person you kept sliced by the crop | 19.3% | 19.6% | 19.9% | 19.9% | - |
| **Truncated limbs (your couple)** | | | | | |
| Head (face/ears) cut | 8.4% | 8.2% | 8.3% | 7.9% | 1.1% |
| Hand (wrist) cut | 9.3% | 9.1% | 9.1% | 9.0% | 2.3% |
| Foot (ankle) cut | 7.3% | 7.3% | 7.3% | 7.2% | 4.1% |
| **Insufficient margin** (inside, but under 3% of couple height) | | | | | |
| Above the head | 0.2% | 0.2% | 0.3% | 0.3% | 0.6% |
| Below the feet | 7.0% | 7.1% | 7.1% | 7.2% | 8.7% |
| **Composition vs your crop** | | | | | |
| Portrait / landscape flipped | 3.2% | 3.2% | 3.1% | 3.5% | - |
| Shape class differs (incl. square vs not) | 45.9% | 44.1% | 44.0% | 42.7% | - |
| Too tight (< 2/3 of your crop's area) | 10.8% | 10.2% | 10.1% | 9.4% | - |
| Too loose (> 1.5x your crop's area) | 9.0% | 8.1% | 7.9% | 8.0% | - |
| Couple off-centre (> 15% of width) | 6.9% | 6.8% | 7.0% | 6.7% | - |

What this shows:

* **The main weakness is choosing the partner, not the backbone.** When the person the model pairs with
  the main dancer isn't the one you kept (about 11% of photos), your actual partner can be cut, which
  explains most of the head/hand/foot truncation (7-9% vs 1-4% in your own crops). The no-cut rule only
  protects the people the model picked. All backbones are equal here, because subject choice uses pose
  features only.
* **Margins match your habits.** Tight feet margins occur about as often as in your own crops.
* **Composition errors are moderate:** about 1 in 10 crops is clearly tighter or looser than yours,
  portrait/landscape is flipped in about 3%; the large "shape class differs" figure is mostly
  square-vs-nearly-square, borderline cases.
* "Another person sliced" is often a judgment call (a dancer half in frame behind the couple), so it's
  an upper bound on real failures.

### Cost (RTX 3080, fp32, batch 32, 13,198 previews)

| Model | GPU time / photo | Total time / photo (incl. JPEG decoding) | Peak GPU memory | Load time |
|---|---|---|---|---|
| CLIP B/32 | 1.5 ms | 7.3 ms | 0.66 GB | 4.1 s |
| CLIP L/14 | 11.3 ms | 14.6 ms | 2.0 GB | 7.1 s |
| DINOv2 B/14 (518 px) | 25.2 ms | 31.4 ms | 2.1 GB | 1.2 s |

For a 200-photo event all are fast (DINOv2 adds about 5 seconds), so cost isn't the deciding factor.

## Decision

**Keep CLIP ViT-B/32 in production.** It's the best for tone, and no alternative improves crops by a
meaningful amount (DINOv2: +0.005 IoU). The embedding itself contributes little: most of the skill
comes from the hand-engineered light and pose features.

**Next improvement:** partner selection, e.g. training the classifier on pairs (who belongs with the
main dancer) and protecting every confidently-kept subject with the no-cut rule.

## Limitations

* Error in slider values is not error in the rendered image; different slider combinations can give a
  similar look.
* The key photos are random in this test; a photographer picks representative ones.
* Design-level reuse: the no-cut margins and subject-classifier features were designed earlier using
  aggregate results on this same library. That affects all variants equally (so the comparison is fair),
  but the absolute numbers may be slightly optimistic.
* Crop checks treat crops as axis-aligned and rely on pose keypoints, which are sometimes wrong in dim,
  crowded scenes.
* "Your couple" is defined as the one or two largest people you kept in frame, which can occasionally be
  a large bystander.

## Reproduce

```
python tools/update_models.py --clip ViT-L-14:laion2b_s32b_b82k --timm vit_base_patch14_dinov2.lvd142m
python tools/embed_models.py <data_dir>
python tools/compare_embeddings.py <data_dir>
```
