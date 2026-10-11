"""Compute per-photo features from the "before" previews.

Usage: python src/features.py <data_dir>

Reads <data_dir>/edits.parquet, previews.parquet and previews/*.jpg. Writes
  features.parquet  - image statistics + EXIF + as-shot WB, one row per photo
  emb.npy, emb_ids.npy - CLIP image embeddings (512-d) for the same photos
"""
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

N_BINS = 16


def image_stats(path):
    im = Image.open(path).convert("RGB")
    im.thumbnail((256, 256))
    a = np.asarray(im, dtype=np.float32) / 255.0
    lum = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
    hist = np.histogram(lum, bins=N_BINS, range=(0, 1))[0] / lum.size
    mx, mn = a.max(-1), a.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    h, w = lum.shape
    centre = lum[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    mid = (lum > 0.1) & (lum < 0.9)  # colour cast from well-exposed pixels only
    rgb_mid = a[mid].mean(0) if mid.any() else a.reshape(-1, 3).mean(0)
    f = {f"hist{i}": v for i, v in enumerate(hist)}
    f.update({f"lum_p{p}": np.percentile(lum, p) for p in (1, 5, 25, 50, 75, 95, 99)})
    f.update(
        lum_mean=lum.mean(), lum_std=lum.std(), centre_mean=centre.mean(),
        centre_minus_edge=centre.mean() - lum.mean(),
        clip_hi=(a.max(-1) > 0.98).mean(), clip_lo=(lum < 0.02).mean(),
        sat_mean=sat.mean(), sat_p90=np.percentile(sat, 90),
        cast_rg=math.log(rgb_mid[0] / max(rgb_mid[1], 1e-4) + 1e-4),
        cast_bg=math.log(rgb_mid[2] / max(rgb_mid[1], 1e-4) + 1e-4),
    )
    return f


def exif_features(df):
    return pd.DataFrame({
        "image_id": df.image_id,
        "log_iso": np.log2(df.iso.clip(lower=50)),
        "shutter_apex": df.shutter_apex,          # Tv = -log2(seconds)
        "aperture_apex": df.aperture_apex,        # Av = 2*log2(f-number)
        "exposure_value": df.shutter_apex + df.aperture_apex - np.log2(df.iso.clip(lower=50) / 100),
        "focal_length": df.focalLength,
        "flash": df.flashFired.fillna(0).astype(float),
        "log_wb_r": np.log(df.wb_r), "log_wb_b": np.log(df.wb_b),
        "hour": pd.to_datetime(df.captureTime, errors="coerce").dt.hour,
    })


def embed(paths, ids, data_dir):
    import open_clip
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model = model.eval().to(dev)
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), 128):
            batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[i:i + 128]]).to(dev)
            e = model.encode_image(batch)
            out.append((e / e.norm(dim=-1, keepdim=True)).float().cpu().numpy())
            print(f"  embedded {min(i + 128, len(paths))}/{len(paths)}", flush=True)
    np.save(os.path.join(data_dir, "emb.npy"), np.concatenate(out))
    np.save(os.path.join(data_dir, "emb_ids.npy"), np.asarray(ids))


def main(data_dir):
    edits = pd.read_parquet(os.path.join(data_dir, "edits.parquet"))
    prev = pd.read_parquet(os.path.join(data_dir, "previews.parquet"))
    df = edits.merge(prev[prev.ok], on="image_id")
    paths = [os.path.join(data_dir, "previews", f"{i}.jpg") for i in df.image_id]

    print(f"image stats for {len(df)} photos...", flush=True)
    with ThreadPoolExecutor(8) as pool:
        stats = pd.DataFrame(list(pool.map(image_stats, paths)))
    feats = pd.concat([exif_features(df).reset_index(drop=True), stats], axis=1)
    feats.to_parquet(os.path.join(data_dir, "features.parquet"), index=False)

    print("CLIP embeddings...", flush=True)
    embed(paths, df.image_id.tolist(), data_dir)
    print("done")


if __name__ == "__main__":
    main(sys.argv[1])
