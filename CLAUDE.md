# CLAUDE.md

## Privacy: user data stays on this computer

The photos show dancers (and their children) who have not consented to their images being sent to
AI services. Treat all user data — photos, previews, catalogs, file/folder names, EXIF, GPS — as private.

- **Never view photos or anything rendered from them** (previews, contact sheets, crop/debug images)
  with Read, screenshots, SendUserFile or Artifacts. Doing so uploads them to Anthropic's servers.
- **Obfuscate before anything leaves the machine.** If you need to inspect data yourself, use a quick,
  reliable safe method:
  - verify with numbers (metrics, geometry checks, summary statistics) instead of images;
  - render anonymised visuals: pose skeletons / boxes / crop rectangles on a blank canvas, or a heavy
    blur that leaves no one identifiable;
  - replace names and paths with IDs in printed output where they aren't needed;
  - or save the output locally and ask the user to open it and describe what they see.
- **If no such method is clearly safe, stop and discuss with the user first.**
- Never upload user data to external services (APIs, web tools, cloud storage). Downloading public
  models/packages is fine; sending data out is not.
- The plugin and all training must keep running fully locally.

## Project notes

- Lightroom Classic 7.0 on Windows. Live catalog:
  `C:\Users\Robert\OneDrive\Pictures\Lightroom\Lightroom Catalog - Copy.lrcat` — work on a copy; never
  write to it while Lightroom is running (check `Lightroom.exe` and the `.lock` file).
- Data and intermediate files go in `S:\auto-photo-edit-data` (J: is nearly full). S: is an old RAID 0
  that has returned a read error: nothing irreplaceable there.
- Python env: `.venv` (`.venv\Scripts\python`). Pipeline and retraining steps: see README.
- Evaluate changes on held-out events (GroupKFold by folder), against simple baselines.
