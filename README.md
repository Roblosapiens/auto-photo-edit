# auto-photo-edit

Learn a photographer's Lightroom editing style from their existing catalog and apply it to new photos.

The model predicts Lightroom develop settings (white balance, tone, presence, sharpening, noise
reduction, vignette, dehaze, later crop) from the unedited photo and its EXIF data. A Lightroom
plugin applies the predicted settings, so every edit stays non-destructive and adjustable.

Target: Lightroom Classic 7.0 (process version 2012 sliders), Windows.

## Plan

1. **Extract** - read develop settings + EXIF from a *copy* of the `.lrcat` catalog; render a neutral
   "before" image for each photo (DNG embedded preview).
2. **Explore** - which sliders are actually used, and how much settings vary within vs. between events.
3. **Features** - histogram / clipping / colour-cast stats, pretrained image embeddings (CLIP or
   DINOv2), EXIF, and per-event context.
4. **Train** - gradient-boosted trees or a small MLP per slider; white balance predicted relative to
   as-shot. Evaluate on held-out *events*, against preset and per-event-average baselines.
5. **Apply** - Lightroom plugin calls `photo:applyDevelopSettings()` with the predictions.
6. **Hybrid mode** - photographer edits 2-3 key shots per event; the model adjusts the rest relative
   to them.

## Data

Nothing personal is committed: catalogs, photos, extracted datasets (`data/`) and models
(`models/`) are git-ignored. Always work from a catalog copy - never write to the live catalog while
Lightroom is open.

## Tools

`tools/` holds catalog maintenance scripts (diagnose and relink missing folders).
