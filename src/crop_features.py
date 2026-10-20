"""Crop-related features for each preview: people + body keypoints (YOLO pose) and camera tilt.

Usage: python src/crop_features.py <data_dir>

Writes <data_dir>/pose.parquet: one row per photo with
  persons  - flat float list, 56 values per person: x0, y0, x1, y1, conf, then 17 x (x, y, conf),
             all in preview pixels; persons sorted by box area, largest first (max 10)
  tilt_deg, tilt_n - estimated camera roll from near-vertical lines (degrees, clockwise positive)
"""
import offline  # noqa: E402  must run before torch/open_clip/ultralytics are imported

offline.enable()
import os
import sys

import cv2
import numpy as np
import pandas as pd

MAX_PERSONS = 10
BATCH = 16  # larger batches overflow a 10 GB GPU shared with Lightroom
POSE_WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "yolo11m-pose.pt")


def tilt(gray):
    """Length-weighted median deviation from vertical of near-vertical line segments.

    Positive = lines lean clockwise (top of the line to the right of its bottom)."""
    gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)  # dim venues: boost local contrast
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 30, 90)
    h = gray.shape[0]
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=40, minLineLength=h // 14, maxLineGap=4)
    if lines is None:
        return np.nan, 0
    x1, y1, x2, y2 = np.asarray(lines).reshape(-1, 4).T.astype(float)
    flip = y2 > y1  # orient every segment bottom -> top
    x1, x2 = np.where(flip, x2, x1), np.where(flip, x1, x2)
    y1, y2 = np.where(flip, y2, y1), np.where(flip, y1, y2)
    ang = np.degrees(np.arctan2(x2 - x1, y1 - y2))  # 0 = vertical
    length = np.hypot(x2 - x1, y2 - y1)
    keep = np.abs(ang) < 12
    if keep.sum() < 3:
        return np.nan, int(keep.sum())
    a, w = ang[keep], length[keep]
    order = np.argsort(a)
    cw = np.cumsum(w[order])
    return float(a[order][np.searchsorted(cw, cw[-1] / 2)]), int(keep.sum())


def load_pose_model():
    from ultralytics import YOLO
    return YOLO(POSE_WEIGHTS)


def analyse(model, images, verbose=False, on_progress=None):
    """Pose + tilt for a list of file paths or PIL images -> list of dicts (see module docstring)."""
    from PIL import Image

    rows = []
    for start in range(0, len(images), BATCH):
        batch = images[start:start + BATCH]
        results = model.predict(batch, imgsz=1024, conf=0.25, verbose=False)
        for im, r in zip(batch, results):
            boxes = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            kpts = r.keypoints.data.cpu().numpy() if r.keypoints is not None else np.zeros((0, 17, 3))
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            persons = [np.concatenate([boxes[i], [conf[i]], kpts[i].ravel()])
                       for i in np.argsort(-area)[:MAX_PERSONS]]
            if isinstance(im, Image.Image):
                gray = np.asarray(im.convert("L"))
            else:
                gray = cv2.imread(im, cv2.IMREAD_GRAYSCALE)
            t, n = tilt(gray)
            rows.append({"n_persons": len(boxes),
                         "persons": np.concatenate(persons).astype(np.float32).tolist() if persons else [],
                         "tilt_deg": t, "tilt_n": n})
        if on_progress:
            on_progress(min(start + BATCH, len(images)), len(images))
        if verbose and (start // BATCH) % 50 == 0:
            print(f"  {min(start + BATCH, len(images))}/{len(images)}", flush=True)
    return rows


def main(data_dir):
    prev = pd.read_parquet(os.path.join(data_dir, "previews.parquet"))
    ids = prev[prev.ok].image_id.tolist()
    paths = [os.path.join(data_dir, "previews", f"{i}.jpg") for i in ids]
    rows = analyse(load_pose_model(), paths, verbose=True)
    pd.DataFrame(rows).assign(image_id=ids).to_parquet(os.path.join(data_dir, "pose.parquet"), index=False)
    print("done")


if __name__ == "__main__":
    main(sys.argv[1])
