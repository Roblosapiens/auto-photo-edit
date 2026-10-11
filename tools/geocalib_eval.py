"""Compare GeoCalib's camera-roll estimate with the photographer's CropAngle.

Usage: python tools/geocalib_eval.py <data_dir> [n_photos]
Writes <data_dir>/geocalib_eval.parquet.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from geocalib import GeoCalib

SENSOR_WIDTH_MM = 35.9  # Sony A7 III

data_dir = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
d = pd.read_parquet(os.path.join(data_dir, "crop_cv_predictions.parquet"))
feats = pd.read_parquet(os.path.join(data_dir, "features.parquet"))[["image_id", "focal_length"]]
s = d.sample(n, random_state=0).merge(feats, on="image_id").reset_index(drop=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

results = {}
for weights, camera_model in [("pinhole", "pinhole"), ("distorted", "simple_radial")]:
    model = GeoCalib(weights=weights).to(dev)
    rolls, unc, t0 = [], [], time.time()
    for k, r in enumerate(s.itertuples()):
        img = model.load_image(os.path.join(data_dir, "previews", f"{r.image_id}.jpg")).to(dev)
        focal_px = r.focal_length / SENSOR_WIDTH_MM * img.shape[-1] if r.focal_length == r.focal_length else None
        priors = {"focal": torch.tensor(focal_px, device=dev, dtype=torch.float32)} if focal_px else None
        out = model.calibrate(img, camera_model=camera_model, priors=priors)
        rolls.append(float(torch.rad2deg(out["gravity"].rp[..., 0]).squeeze()))
        unc.append(float(out["roll_uncertainty"].squeeze()) if "roll_uncertainty" in out else np.nan)
        if (k + 1) % 250 == 0:
            print(f"  {weights}: {k + 1}/{len(s)}  {time.time() - t0:.0f}s", flush=True)
    results[weights] = (np.array(rolls), np.array(unc))
    s[f"roll_{weights}"], s[f"unc_{weights}"] = results[weights]

s.to_parquet(os.path.join(data_dir, "geocalib_eval.parquet"), index=False)
truth = s.CropAngle.to_numpy()
rot = np.abs(truth) >= 0.5
for w, (roll, u) in results.items():
    c = np.corrcoef(roll, truth)[0, 1]
    sign = np.sign(c)
    pred = sign * roll
    print(f"\n{w}: corr(roll, CropAngle) = {c:.3f} -> CropAngle ~ {sign:+.0f} x roll; "
          f"slope {np.polyfit(roll, truth, 1)[0]:.2f}")
    err = np.abs(pred - truth)
    print(f"  all photos: error {err.mean():.2f} deg vs no rotation {np.abs(truth).mean():.2f}; within 0.5 deg {(err < 0.5).mean():.0%}")
    print(f"  photos you rotated: error {err[rot].mean():.2f} vs {np.abs(truth[rot]).mean():.2f}; "
          f"wrong direction {(np.sign(pred[rot]) != np.sign(truth[rot])).mean():.0%}")
    if np.isfinite(u).any():
        for q in (0.25, 0.5):
            m = u <= np.nanquantile(u, q)
            print(f"  most confident {q:.0%}: error {err[m].mean():.2f} vs {np.abs(truth[m]).mean():.2f}; "
                  f"wrong direction on rotated {(np.sign(pred[m & rot]) != np.sign(truth[m & rot])).mean():.0%}")
