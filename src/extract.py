"""Extract develop settings + EXIF for every photo in a Lightroom Classic catalog copy.

Usage: python src/extract.py <catalog.lrcat> <out.parquet>

Reads the catalog read-only. Each row is one photo: file path, event folder, EXIF, and the
develop-slider values (Lightroom only stores non-default values; missing ones are filled with
Lightroom's defaults).
"""
import re
import sqlite3
import sys

import pandas as pd

# Sliders we care about, with Lightroom's default value (process version 2012+).
SLIDERS = {
    "Temperature": None, "Tint": None,                    # absolute WB (raw files)
    "IncrementalTemperature": 0, "IncrementalTint": 0,    # relative WB (JPEG etc.)
    "Exposure2012": 0.0, "Contrast2012": 0, "Highlights2012": 0, "Shadows2012": 0,
    "Whites2012": 0, "Blacks2012": 0,
    "Clarity2012": 0, "Dehaze": 0, "Vibrance": 0, "Saturation": 0,
    "Sharpness": 25, "SharpenRadius": 1.0, "SharpenDetail": 25, "SharpenEdgeMasking": 0,
    "LuminanceSmoothing": 0, "ColorNoiseReduction": 25,
    "PostCropVignetteAmount": 0,
    "CropTop": 0.0, "CropLeft": 0.0, "CropBottom": 1.0, "CropRight": 1.0, "CropAngle": 0.0,
}
STRINGS = ["WhiteBalance", "CameraProfile", "ProcessVersion"]

TOP_LEVEL = re.compile(r'^(\w+) = ("(?:[^"\\]|\\.)*"|-?[\d.]+(?:e-?\d+)?|true|false)\s*,?\s*}?$', re.M)


def parse_settings(text):
    out = {}
    for key, val in TOP_LEVEL.findall(text or ""):
        if val.startswith('"'):
            out[key] = val[1:-1]
        elif val in ("true", "false"):
            out[key] = val == "true"
        else:
            out[key] = float(val)
    return out


QUERY = """
SELECT i.id_local AS image_id, i.captureTime, i.fileFormat, i.rating, i.pick,
       i.fileWidth, i.fileHeight, i.orientation,
       rf.absolutePath || fo.pathFromRoot AS folder, fi.idx_filename AS filename,
       ds.hasDevelopAdjustments, ds.croppedWidth, ds.croppedHeight, ds.whiteBalance AS wb_mode,
       ds.text AS settings,
       ex.isoSpeedRating AS iso, ex.shutterSpeed AS shutter_apex, ex.aperture AS aperture_apex,
       ex.focalLength, ex.flashFired, cm.value AS camera, ln.value AS lens
FROM Adobe_images i
JOIN AgLibraryFile fi ON i.rootFile = fi.id_local
JOIN AgLibraryFolder fo ON fi.folder = fo.id_local
JOIN AgLibraryRootFolder rf ON fo.rootFolder = rf.id_local
LEFT JOIN Adobe_imageDevelopSettings ds ON ds.image = i.id_local
LEFT JOIN AgHarvestedExifMetadata ex ON ex.image = i.id_local
LEFT JOIN AgInternedExifCameraModel cm ON ex.cameraModelRef = cm.id_local
LEFT JOIN AgInternedExifLens ln ON ex.lensRef = ln.id_local
"""


def main(cat, out):
    con = sqlite3.connect(f"file:{cat}?mode=ro", uri=True)
    df = pd.read_sql_query(QUERY, con)
    for c in ("croppedWidth", "croppedHeight"):  # 'uncropped' when there is no crop
        df[c] = pd.to_numeric(df[c], errors="coerce")
    parsed = df.pop("settings").map(parse_settings)
    for k, default in SLIDERS.items():
        df[k] = parsed.map(lambda s: s.get(k, default))
    for k in STRINGS:
        df[k] = parsed.map(lambda s: s.get(k))
    df["n_settings_keys"] = parsed.map(len)
    df.to_parquet(out, index=False)
    print(f"{len(df)} photos -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
