"""Predict develop settings (and crops) for one event from its key photos. Called by the Lightroom plugin.

Usage: python src/predict.py <request.json> <response.tsv> [--no-crop] [--models DIR]

request.json (written by the plugin):
  {"photos": [{"id": 123, "path": "J:/...dng", "key": true, "cropped": false, "iso": 6400,
               "shutter": 0.005, "aperture": 2.8, "focal": 35, "flash": false, "time": 700000000.0,
               "settings": {"Exposure2012": 1.2, ...}   # key photos only
              }, ...]}
response.tsv: one line per setting for every non-key photo:  <id> TAB <setting> TAB <value>

Synced sliders get the mean of the key photos. Exposure and white balance get the key photos'
mean plus the model's prediction of how this photo differs from the key photos. Photos that are not
cropped yet (and were shot with the camera upright) get a crop framed around the dancers, unstraightened.
"""
import argparse
import json
import math
import os
import pickle
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from crop_features import analyse, load_pose_model
from crop_model import TARGETS as CROP_TARGETS
from crop_model import SUBJECT_P, contain_subject, person_features, photo_features, targets_to_crop
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
    images = [im for im, _, _, _ in loaded]
    df = pd.DataFrame({
        "id": [p["id"] for p in photos],
        "key": [bool(p.get("key")) for p in photos],
        "cropped": [bool(p.get("cropped")) for p in photos],
        "upright": [up for _, _, _, up in loaded],
        "folder": "event",
        "iso": [p.get("iso") or 100 for p in photos],
        "shutter_apex": [-math.log2(p["shutter"]) if p.get("shutter") else np.nan for p in photos],
        "aperture_apex": [2 * math.log2(p["aperture"]) if p.get("aperture") else np.nan for p in photos],
        "focalLength": [p.get("focal") for p in photos],
        "flashFired": [float(bool(p.get("flash"))) for p in photos],
        "captureTime": [str(COCOA_EPOCH + pd.Timedelta(seconds=p["time"])) if p.get("time") else None for p in photos],
        "wb_r": [r for _, r, _, _ in loaded], "wb_b": [b for _, _, b, _ in loaded],
    })
    stats = pd.DataFrame([image_stats(im) for im in images])
    feats = pd.concat([exif_features(df.assign(image_id=df.id)).drop(columns="image_id"), stats], axis=1)
    frame = pd.concat([df[["id", "key", "cropped", "upright", "folder", "captureTime"]], feats], axis=1)
    return frame, images, clip_embed(images)


def tone_settings(df, keys, emb, m):
    """Per-photo tone settings for the non-key photos: {row index: {setting: value}}."""
    df = add_derived(df, m["coef"])
    _, Xd = design(df, m["feat_cols"], m["pca"], emb)
    key_mask = df.key.to_numpy()
    ks = pd.DataFrame([p["settings"] for p in keys])
    observed = {"Exposure2012": ks.get("Exposure2012", pd.Series(0.0, index=ks.index)).astype(float).to_numpy()}
    if "Temperature" in ks:
        observed["temp_mired_delta"] = 1e6 / ks.Temperature.astype(float).to_numpy() - df.asshot_mired[key_mask].to_numpy()
    hybrid = {}
    for t in HYBRID_TARGETS:
        if t in observed:
            p = m["models"][t].predict(Xd)
            hybrid[t] = observed[t].mean() + (p - p[key_mask].mean())
    sync = {s: float(ks[s].astype(float).mean()) for s in SYNC_SLIDERS if s in ks}

    out = {}
    for i, row in df.iterrows():
        if row.key:
            continue
        vals = dict(sync)
        if "Exposure2012" in hybrid:
            vals["Exposure2012"] = hybrid["Exposure2012"][i]
        if "temp_mired_delta" in hybrid:
            vals["Temperature"] = 1e6 / (row.asshot_mired + hybrid["temp_mired_delta"][i])
            vals["WhiteBalance"] = "Custom"
        for k, v in vals.items():
            if isinstance(v, float):
                lo, hi = LIMITS.get(k, (-math.inf, math.inf))
                v = min(max(v, lo), hi)
                vals[k] = int(round(v)) if k in INTEGER_SLIDERS or k == "Temperature" else round(v, 2)
        out[i] = vals
    return out


def crop_settings(df, images, emb, cm):
    """Crops for non-key, not-yet-cropped, upright photos: {row index: {Crop*: value}}."""
    todo = df.index[~df.key & ~df.cropped & df.upright].tolist()
    if not todo:
        return {}
    poses = analyse(load_pose_model(), [images[i] for i in todo])
    rows, boxes, idx = [], [], []
    for i, pose in zip(todo, poses):
        W, H = images[i].size
        if not pose["persons"]:
            continue
        pf = person_features(pose["persons"], W, H, np.asarray(images[i].convert("L")))
        probs = cm["selector"].predict_proba(pf[cm["sel_cols"]].astype(float))[:, 1]
        box, f = photo_features(pose["persons"], W, H, probs)
        if box is None:  # nobody detected: leave this photo uncropped
            continue
        f.update(tilt_deg=pose["tilt_deg"], tilt_n=pose["tilt_n"], focal_length=df.focal_length[i])
        rows.append(f)
        boxes.append(box)
        idx.append(i)
    if not rows:
        return {}
    X = pd.DataFrame(rows).reindex(columns=cm["feat_cols"])
    M = np.hstack([X.to_numpy(float), cm["pca"].transform(emb[idx])])
    pred = pd.DataFrame({t: cm["models"][t].predict(M) for t in CROP_TARGETS})
    pred["t_angle"] = 0.0  # straightening was not learnable (evaluation: no better than 0)
    out = {}
    for j, i in enumerate(idx):
        W, H = images[i].size
        c = targets_to_crop(pred.iloc[j], boxes[j], W, H)
        if rows[j]["subject_p"] >= SUBJECT_P:  # confident it's the couple: never cut into them
            c = contain_subject(c, boxes[j], W, H)
        out[i] = {k: round(float(v), 5) for k, v in c.items()}
        out[i]["CropConstrainAspectRatio"] = False
    return out


def main(req_path, resp_path, models_dir, do_crop):
    photos = json.load(open(req_path, encoding="utf-8"))["photos"]
    keys = [p for p in photos if p.get("key")]
    if not keys:
        sys.exit("No key photos: give 1-3 edited photos in the selection a green label.")
    with open(os.path.join(models_dir, "autoedit.pkl"), "rb") as f:
        tone_model = pickle.load(f)

    df, images, emb = build_frame(photos)
    settings = tone_settings(df, keys, emb, tone_model)
    n_crops = 0
    if do_crop:
        with open(os.path.join(models_dir, "crop.pkl"), "rb") as f:
            crops = crop_settings(df, images, emb, pickle.load(f))
        for i, c in crops.items():
            settings[i].update(c)
        n_crops = len(crops)

    with open(resp_path, "w", encoding="utf-8") as f:
        for i, vals in settings.items():
            for k, v in vals.items():
                f.write(f"{df.id[i]}\t{k}\t{str(v).lower() if isinstance(v, bool) else v}\n")
    print(f"{len(keys)} key photos -> settings for {len(settings)} photos, crops for {n_crops}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("request")
    ap.add_argument("response")
    ap.add_argument("--no-crop", action="store_true")
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    a = ap.parse_args()
    main(a.request, a.response, a.models, not a.no_crop)
