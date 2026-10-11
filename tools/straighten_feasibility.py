"""Does the photographer's CropAngle match a near-vertical line near the centre of their crop?

Usage: python tools/straighten_feasibility.py <data_dir> [n_photos]
"""
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from crop_model import unpack  # noqa: E402
from straighten import detect_lines, line_features  # noqa: E402

data_dir = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
d = pd.read_parquet(os.path.join(data_dir, "crop_cv_predictions.parquet"))
pose = pd.read_parquet(os.path.join(data_dir, "pose.parquet"))[["image_id", "persons"]]
s = d.sample(n, random_state=0).merge(pose, on="image_id")

rows, t0 = [], time.time()
for k, r in enumerate(s.itertuples()):
    g = cv2.imread(os.path.join(data_dir, "previews", f"{r.image_id}.jpg"), cv2.IMREAD_GRAYSCALE)
    lines = detect_lines(g)
    crop = dict(CropLeft=r.CropLeft, CropTop=r.CropTop, CropRight=r.CropRight, CropBottom=r.CropBottom)
    f = line_features(lines, crop, r.preview_w, r.preview_h, unpack(r.persons)[0]) if len(lines) else pd.DataFrame()
    if len(f):
        f = f[(f.inside > 0) & (f.on_person == 0)]
    best, match = np.nan, np.inf
    if len(f):
        score = f.len_rel * np.exp(-(f.abs_dx / 0.2) ** 2)  # long line near the horizontal centre
        best = f.angle[score.idxmax()]
        match = (f.angle - r.CropAngle).abs().min()
    rows.append(dict(ang=r.CropAngle, best=best, match=match, n=len(f), n_all=len(lines)))
    if (k + 1) % 100 == 0:
        print(f"  {k + 1}/{len(s)}  {time.time() - t0:.0f}s", flush=True)

x = pd.DataFrame(rows)
rot = x.ang.abs() >= 0.3
print(f"near-vertical lines per photo: median {x.n_all.median():.0f}; with a candidate inside the crop: {(x.n > 0).mean():.0%}")
print(f"rotated photos: {rot.mean():.0%}. A candidate matches your angle within 0.2 deg: {(x.match[rot] < 0.2).mean():.0%}, "
      f"within 0.4 deg: {(x.match[rot] < 0.4).mean():.0%}")
b = rot & x.best.notna()
err = (x.best[b] - x.ang[b]).abs()
print(f"heuristic (long + central line): mean error {err.mean():.2f} deg, within 0.4 deg: {(err < 0.4).mean():.0%}, "
      f"sign agreement {(np.sign(x.best[b]) == np.sign(x.ang[b])).mean():.0%}")
print(f"no-rotation baseline error on rotated photos: {x.ang[rot].abs().mean():.2f} deg")
nr = ~rot & x.best.notna()
print(f"photos you left straight: heuristic would rotate by {x.best[nr].abs().median():.2f} deg (median)")
