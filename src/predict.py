"""Predict develop settings for one event from its key photos. Called by the Lightroom plugin.

Usage: python src/predict.py <request.json> <response.tsv> [model.pkl]

request.json (written by the plugin):
  {"photos": [{"id": 123, "path": "J:/...dng", "key": true, "iso": 6400, "shutter": 0.005,
               "aperture": 2.8, "focal": 35, "flash": false, "time": 700000000.0,
               "settings": {"Exposure2012": 1.2, ...}   # key photos only
              }, ...]}
response.tsv: one line per setting for every non-key photo:  <id> TAB <setting> TAB <value>

Synced sliders get the mean of the key photos. Exposure and white balance get the key photos'
mean plus the model's prediction of how this photo differs from the key photos.
"""
import json
import math
import os
import pickle
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from features import clip_embed, exif_features, image_stats
from model import HYBRID_TARGETS, SYNC_SLIDERS, add_derived, design
from previews import load_preview

warnings.filterwarnings("ignore", message="X does not have valid feature names")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COCOA_EPOCH = pd.Timestamp("2001-01-01")  # Lightroom's dateTimeOriginal is seconds since 2001
INTEGER_SLIDERS = set(SYNC_SLIDERS) - {"SharpenRadius"}
LIMITS = {"Exposure2012": (-5, 5), "Temperature": (2000, 50000)}


def build_frame(photos):
    with ThreadPoolExecutor(8) as pool:
        loaded = list(pool.map(lambda p: load_preview(p["path"]), photos))
    images = [im for im, _, _ in loaded]
    df = pd.DataFrame({
        "id": [p["id"] for p in photos],
        "key": [bool(p.get("key")) for p in photos],
        "folder": "event",
        "iso": [p.get("iso") or 100 for p in photos],
        "shutter_apex": [-math.log2(p["shutter"]) if p.get("shutter") else np.nan for p in photos],
        "aperture_apex": [2 * math.log2(p["aperture"]) if p.get("aperture") else np.nan for p in photos],
        "focalLength": [p.get("focal") for p in photos],
        "flashFired": [float(bool(p.get("flash"))) for p in photos],
        "captureTime": [str(COCOA_EPOCH + pd.Timedelta(seconds=p["time"])) if p.get("time") else None for p in photos],
        "wb_r": [r for _, r, _ in loaded], "wb_b": [b for _, _, b in loaded],
    })
    stats = pd.DataFrame([image_stats(im) for im in images])
    feats = pd.concat([exif_features(df.assign(image_id=df.id)).drop(columns="image_id"), stats], axis=1)
    return pd.concat([df[["id", "key", "folder", "captureTime"]], feats], axis=1), clip_embed(images)


def main(req_path, resp_path, model_path):
    photos = json.load(open(req_path, encoding="utf-8"))["photos"]
    keys = [p for p in photos if p.get("key")]
    if not keys:
        sys.exit("No key photos: give 1-3 edited photos in the selection a green label.")
    m = pickle_load(model_path)

    df, emb = build_frame(photos)
    df = add_derived(df, m["coef"])
    _, Xd = design(df, m["feat_cols"], m["pca"], emb)
    key_mask = df.key.to_numpy()

    # Targets observed on the key photos
    ks = pd.DataFrame([p["settings"] for p in keys])
    observed = {"Exposure2012": ks.get("Exposure2012", pd.Series(0.0, index=ks.index)).astype(float).to_numpy()}
    if "Temperature" in ks:
        observed["temp_mired_delta"] = 1e6 / ks.Temperature.astype(float).to_numpy() - df.asshot_mired[key_mask].to_numpy()

    out = {}
    for t in HYBRID_TARGETS:
        if t not in observed:
            continue
        p = m["models"][t].predict(Xd)
        out[t] = observed[t].mean() + (p - p[key_mask].mean())
    sync = {s: float(ks[s].astype(float).mean()) for s in SYNC_SLIDERS if s in ks}

    with open(resp_path, "w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            if row.key:
                continue
            vals = dict(sync)
            if "Exposure2012" in out:
                vals["Exposure2012"] = out["Exposure2012"][i]
            if "temp_mired_delta" in out:
                vals["Temperature"] = 1e6 / (row.asshot_mired + out["temp_mired_delta"][i])
                vals["WhiteBalance"] = "Custom"
            for k, v in vals.items():
                if isinstance(v, float):
                    lo, hi = LIMITS.get(k, (-math.inf, math.inf))
                    v = min(max(v, lo), hi)
                    v = int(round(v)) if k in INTEGER_SLIDERS or k == "Temperature" else round(v, 2)
                f.write(f"{row.id}\t{k}\t{v}\n")
    print(f"{len(keys)} key photos -> settings for {len(df) - len(keys)} photos")


def pickle_load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0], args[1], args[2] if len(args) > 2 else os.path.join(ROOT, "models", "autoedit.pkl"))
