"""Summarise the extracted edits: what is edited, which sliders are used, event vs. per-photo variation.

Usage: python src/explore.py <edits.parquet>
"""
import offline  # noqa: E402  must run before torch/open_clip/ultralytics are imported

offline.enable()
import sys

import pandas as pd

from extract import SLIDERS

df = pd.read_parquet(sys.argv[1])
print(f"photos in catalog: {len(df)}")
print("\nfile formats:\n", df.fileFormat.value_counts().to_string())

raw = df[df.fileFormat.isin(["RAW", "DNG"])].copy()
# hasDevelopAdjustments is always NULL in LR 7 catalogs, so infer "edited" from the sliders.
CORE = ["Exposure2012", "Contrast2012", "Highlights2012", "Shadows2012", "Whites2012", "Blacks2012",
        "Clarity2012", "Dehaze", "Vibrance", "Saturation"]
raw["edited"] = (raw[CORE] != 0).any(axis=1) | ~raw.WhiteBalance.isin(["As Shot", None])
print(f"\nraw/DNG photos: {len(raw)}, with develop adjustments: {raw.edited.sum()}")
print("\nedited raw photos by camera:\n", raw[raw.edited].camera.value_counts().head(8).to_string())

ed = raw[raw.edited].copy()
ed["year"] = ed.captureTime.str[:4]
ed["has_crop"] = (ed.CropTop > 0) | (ed.CropLeft > 0) | (ed.CropBottom < 1) | (ed.CropRight < 1) | (ed.CropAngle != 0)
print("\nedited raw photos by year:\n", ed.year.value_counts().sort_index().to_string())
print("\nwhite balance mode:\n", ed.WhiteBalance.value_counts(dropna=False).to_string())
print(f"\ncropped: {ed.has_crop.mean():.0%}")

# Per-slider usage and how much variation is explained by the event (folder)
events = ed.groupby("folder").filter(lambda g: len(g) >= 10)
print(f"\nevents with >=10 edited photos: {events.folder.nunique()} ({len(events)} photos)")
rows = []
for k, default in SLIDERS.items():
    if k.startswith("Crop") or k.startswith("Incremental"):
        continue
    s = ed[k].astype(float)
    if default is not None:
        used = (s.round(3) != default).mean()
    else:
        used = s.notna().mean()
    e = events[k].astype(float).dropna()
    total_var = e.var()
    within_var = events.loc[e.index].groupby("folder")[k].transform(lambda x: x - x.mean()).astype(float).var()
    rows.append({
        "slider": k, "used_%": round(100 * used), "median": s.median(),
        "p10": s.quantile(.1), "p90": s.quantile(.9),
        "event_explains_%": round(100 * (1 - within_var / total_var)) if total_var and total_var > 0 else None,
    })
print("\n" + pd.DataFrame(rows).to_string(index=False))
