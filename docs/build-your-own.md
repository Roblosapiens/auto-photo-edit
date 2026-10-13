# Build your own auto-edit plugin

Advice for photographers who want to do this for their own catalog, plus a phased prompt to give a
coding assistant.

## Advice

**Before starting**

- **The simplest route is to reuse this project's code** and retrain it on your own catalog. That is
  far more reliable than rebuilding from a prompt. Rebuilding makes sense if your setup differs:
  Mac, another camera, another kind of photography.
- **Check you have enough data:** a few thousand photos you have edited and cropped yourself,
  ideally mostly from one camera with a consistent workflow. An NVIDIA GPU and 50+ GB of free disk
  space help a lot.
- **If you have a recent Lightroom Classic,** try Imagen or Aftershoot first. They do much of this
  commercially, with no coding needed.

**Protecting your catalog and photos**

- Back up the catalog first. Always work on a copy, and never let anything write to the catalog
  while Lightroom is open.
- **Set the privacy rule before any work starts:** the assistant must never open your photos.
  Viewing an image uploads it to the AI provider, and the people in the photos haven't agreed to
  that. Results can be checked with numbers, with anonymised drawings, or by you looking locally.

**Working with the assistant**

- **Don't expect one shot to do everything.** Even with a good prompt, this project took several
  rounds of testing and fixing. The prompt below is split into phases with checkpoints so you can
  steer at each step.
- **Insist on honest testing.** Always test on whole events the model hasn't seen, and compare
  against simple baselines (one preset, copying the key photos). Otherwise the results look better
  than they are.
- **Expect the same lessons:** a fully automatic overall look doesn't work; "edit a few key photos,
  let the model adjust the rest" does. Straightening from line detection was unreliable; GeoCalib
  worked better.
- **Check licences if you'll share the result.** Ultralytics YOLO is AGPL-licensed, which is fine
  for personal use but matters if you distribute it.

## The prompt

Replace the `[BRACKETED]` parts first.

````text
I want to build a Lightroom Classic plugin that learns my personal editing style from my existing
catalog and applies it to new photos: tone settings, crop and straightening. Work in phases, stop at
each CHECKPOINT to show me results and wait for my go-ahead.

MY SETUP
- Lightroom Classic [VERSION] on [Windows/macOS]; catalog at [CATALOG PATH]
- Camera: [CAMERA MODEL]; raw files are [DNG converted with Adobe DNG Converter / native raw]
- Photos are on [PHOTO DRIVE]; put large data files on [DATA DRIVE WITH SPACE], outside the repo
- GPU: [GPU or "none"]
- Subject of my photos: [e.g. couples dancing at swing dance socials]; events are one folder each
- My workflow: cull, import, edit white balance/tone/presence/sharpening/NR/vignette/dehaze, crop,
  straighten, export with watermark

HARD RULES
1. PRIVACY: never view my photos or any image made from them (previews, contact sheets, debug
   images) - viewing uploads them to your provider and the people in them have not consented.
   Verify with numbers; if you must see something, render anonymised versions (pose skeletons and
   boxes on a blank canvas, or heavy blur), or save the image locally and ask me to describe it.
   If no method is clearly safe, ask me. Never upload my data anywhere. Everything must run
   locally. Write this rule into the repo's CLAUDE.md first.
2. CATALOG SAFETY: only ever read a COPY of the .lrcat. Never write to the catalog while Lightroom
   runs (check the process and the .lock file); back up before any write.
3. HONEST EVALUATION: group cross-validation by event folder (GroupKFold), always against simple
   baselines. Report plain-language results.
4. Set up a git repo (personal data, catalogs, photos, models and the data folder git-ignored), a
   Python venv, and commit after each phase.

PHASE 1 - EXTRACT (read-only)
- The catalog is SQLite. Develop settings live in Adobe_imageDevelopSettings.text as a Lua table
  that only lists NON-DEFAULT values (fill defaults). Join Adobe_images, AgLibraryFile,
  AgLibraryFolder, AgLibraryRootFolder, AgHarvestedExifMetadata (+ interned camera/lens tables).
- hasDevelopAdjustments may be NULL: infer "edited" from core sliders/white balance.
- Crop = CropLeft/Top/Right/Bottom (normalised, sensor orientation) + CropAngle (degrees,
  clockwise about the crop centre); also CropConstrainAspectRatio.
- Summarise: edited count per camera/year, which sliders I use, % cropped, crop aspect ratios,
  and how much of each slider's variance is explained by the event.
CHECKPOINT.

PHASE 2 - "BEFORE" IMAGES
- Use the embedded preview in each raw (for DNG: parse the TIFF IFDs/SubIFDs and read only the
  JPEG preview bytes + AsShotNeutral and Orientation - do NOT decode whole raws, it is ~100x slower).
  Keep previews ~1024 px on the data drive. Verify previews are unedited (uncropped, no edits baked in).
- As-shot white balance: map camera multipliers to Lightroom Kelvin using photos left at "As Shot".
CHECKPOINT.

PHASE 3 - TONE MODEL
- Features: luminance histogram/percentiles, clipping, colour cast, EXIF (ISO, shutter, aperture,
  flash, focal length, time), as-shot WB, CLIP image embeddings (PCA), event-level means and
  per-photo deviations.
- Compare: one preset; fully automatic model; "sync" = mean of K key photos per event; "hybrid" =
  sync + model's per-photo difference (train LightGBM on within-event deviations). Choose per slider
  whichever wins (in my case only exposure and white balance benefited from the model).
CHECKPOINT.

PHASE 4 - CROP MODEL
- Run a pose model (e.g. YOLO11-pose) on all previews; also save the person boxes and keypoints.
  Use small GPU batches - Lightroom shares the GPU.
- Learn a SUBJECT CLASSIFIER from my crops: people mostly inside my crop = subject; features incl.
  size, position, frame-edge contact, keypoint visibility and sharpness relative to the image
  (blurry foreground bystanders).
- Pick the subject(s) by classifier probability; express my crop RELATIVE to the subject box
  (centre offset, log width/height); predict with LightGBM.
- Add a rule: never cut into a confidently detected subject (push out only the offending edge, with
  minimum margins taken from my own crops).
- Report IoU vs my crops against baselines (no crop, my mean crop, a fixed template around the
  subject) and the % of crops cutting the subject. Skip camera-rotated (portrait) frames at first.
- To check crops visually, follow the privacy rule (skeleton/box drawings or I look locally).
CHECKPOINT.

PHASE 5 - STRAIGHTENING
- Try GeoCalib (github.com/cvg/GeoCalib) camera-roll estimates with a focal-length prior; compare
  to my CropAngle on photos I rotated (check the sign). Tune a scale factor and a small dead zone
  on my data; report error and wrong-direction rate vs never rotating. (Line-segment heuristics
  were unreliable on 1024 px previews in a previous attempt.)
CHECKPOINT.

PHASE 6 - LIGHTROOM PLUGIN
- Lua plugin (Info.lua + shared core module), menu items under Library > Plug-in Extras:
  "Auto Edit + Crop" and "tone only".
- Flow: user edits 1-3 key photos and gives them a green label, selects the whole event, runs the
  menu item. The plugin collects path, label, EXIF (batchGetRawMetadata/batchGetFormattedMetadata),
  current develop settings, writes request JSON (hand-written JSON encoder - no library), runs the
  venv's python predict script via LrTasks.execute (on Windows wrap the whole command in an extra
  pair of quotes), reads a simple TSV response, shows a confirm dialog with counts, then applies
  everything with photo:applyDevelopSettings inside ONE catalog:withWriteAccessDo (one undo step).
- Never change key photos; never re-crop photos already cropped; booleans must arrive as booleans.
- Test predict.py end-to-end on a real event before I try it in Lightroom; syntax-check the Lua
  with a standalone Lua interpreter. Write install/usage/retraining steps in the README.
CHECKPOINT: I test it in Lightroom on a copy of an event and report back.
````
