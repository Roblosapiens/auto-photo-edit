"""Straightening: find near-vertical line segments and describe them relative to a crop.

The photographer straightens by picking a real-world vertical line near the centre of the crop and
rotating until it is vertical. Lightroom's CropAngle has the same sign as the line's lean
(positive = top of the line leans right = clockwise).

Usage: python src/straighten.py <data_dir>   -> <data_dir>/lines.parquet (segments for every preview)
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd

MAX_ANGLE = 12.0  # degrees from vertical considered "near vertical"
_LSD = None


def detect_lines(gray):
    """Near-vertical segments in a grayscale image -> DataFrame (x_bot, y_bot, x_top, y_top, length, angle, width, nfa)."""
    global _LSD
    if _LSD is None:
        _LSD = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)  # ADV is ~40x slower for no practical gain
    return _detect(_LSD, gray)


def _detect(lsd, gray):
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    lines, width, _, nfa = lsd.detect(g)
    cols = ["x_bot", "y_bot", "x_top", "y_top", "length", "angle", "width", "nfa"]
    if lines is None:
        return pd.DataFrame(columns=cols)
    x1, y1, x2, y2 = np.asarray(lines).reshape(-1, 4).T
    flip = y2 > y1  # orient bottom -> top
    xb, yb = np.where(flip, x2, x1), np.where(flip, y2, y1)
    xt, yt = np.where(flip, x1, x2), np.where(flip, y1, y2)
    length = np.hypot(xt - xb, yt - yb)
    angle = np.degrees(np.arctan2(xt - xb, yb - yt))
    d = pd.DataFrame({"x_bot": xb, "y_bot": yb, "x_top": xt, "y_top": yt, "length": length, "angle": angle,
                      "width": np.asarray(width).ravel(), "nfa": np.asarray(nfa).ravel() if nfa is not None else np.nan})
    return d[(d.angle.abs() < MAX_ANGLE) & (d.length >= 0.04 * gray.shape[0])].reset_index(drop=True)


def line_features(lines, crop, W, H, person_boxes=()):
    """Features of each line relative to an unrotated crop (Lightroom normalised L, T, R, B)."""
    L, T, R, B = crop["CropLeft"] * W, crop["CropTop"] * H, crop["CropRight"] * W, crop["CropBottom"] * H
    cw, ch = R - L, B - T
    mx, my = (lines.x_bot + lines.x_top) / 2, (lines.y_bot + lines.y_top) / 2
    f = pd.DataFrame({
        "angle": lines.angle, "abs_angle": lines.angle.abs(),
        "len_rel": lines.length / ch, "len_px": lines.length,
        "dx_centre": (mx - (L + R) / 2) / cw, "dy_centre": (my - (T + B) / 2) / ch,
        "abs_dx": ((mx - (L + R) / 2) / cw).abs(),
        "inside": ((mx > L) & (mx < R) & (my > T) & (my < B)).astype(float),
        "width": lines.width, "nfa": lines.nfa,
        "x_img": mx / W, "y_img": my / H,
    })
    on_person = np.zeros(len(lines))
    for b in person_boxes:
        on_person = np.maximum(on_person, ((mx > b[0]) & (mx < b[2]) & (my > b[1]) & (my < b[3])).astype(float))
    f["on_person"] = on_person
    # how many other lines in the photo agree with this angle (a wall usually has several parallel edges)
    a = lines.angle.to_numpy()
    w = lines.length.to_numpy()
    f["support"] = [(w * (np.abs(a - ai) < 0.3)).sum() / w.sum() for ai in a] if len(a) else []
    return f


# GeoCalib camera roll -> CropAngle. Fitted on 1500 held-out cropped photos (tools/geocalib_eval.py):
# CropAngle ~ -roll; scaling by 0.7 and skipping rotations under 0.5 deg minimised the error against
# the photographer's own straightening (0.94 deg vs 1.29 for never rotating; wrong direction 5%).
GEOCALIB_SCALE = 0.7
DEAD_ZONE_DEG = 0.5
SENSOR_WIDTH_MM = 35.9  # Sony A7 III
_GEOCALIB = None


def straighten_angles(images, focal_mm):
    """CropAngle (degrees) for each PIL image from GeoCalib's gravity estimate; 0 for small tilts."""
    import torch
    from geocalib import GeoCalib
    from torchvision.transforms.functional import to_tensor

    global _GEOCALIB
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if _GEOCALIB is None:
        _GEOCALIB = GeoCalib(weights="pinhole").to(dev)
    angles = []
    for im, f in zip(images, focal_mm):
        img = to_tensor(im.convert("RGB")).to(dev)
        priors = None
        if f and f == f:
            priors = {"focal": torch.tensor(f / SENSOR_WIDTH_MM * img.shape[-1], device=dev, dtype=torch.float32)}
        out = _GEOCALIB.calibrate(img, camera_model="pinhole", priors=priors)
        a = -GEOCALIB_SCALE * float(torch.rad2deg(out["gravity"].rp[..., 0]).squeeze())
        angles.append(0.0 if abs(a) < DEAD_ZONE_DEG else float(np.clip(a, -8, 8)))
    return angles


def main(data_dir):
    prev = pd.read_parquet(os.path.join(data_dir, "previews.parquet"))
    ids = prev[prev.ok].image_id.tolist()

    def one(image_id):
        gray = cv2.imread(os.path.join(data_dir, "previews", f"{image_id}.jpg"), cv2.IMREAD_GRAYSCALE)
        cv2.setNumThreads(1)
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)  # one detector per call: thread safe
        d = _detect(lsd, gray)
        return d.assign(image_id=image_id)

    with ThreadPoolExecutor(6) as pool:
        parts = []
        for k, d in enumerate(pool.map(one, ids), 1):
            parts.append(d)
            if k % 1000 == 0:
                print(f"  {k}/{len(ids)}", flush=True)
    pd.concat(parts, ignore_index=True).to_parquet(os.path.join(data_dir, "lines.parquet"), index=False)
    print("done")


if __name__ == "__main__":
    main(sys.argv[1])
