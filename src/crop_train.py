"""Train the crop model and evaluate it on held-out events.

Usage: python src/crop_train.py <data_dir> [--save models/crop.pkl]

Uses every cropped Sony A7 III photo (edited or not). Compares, by IoU with your real crop:
  none      - no crop (full frame)
  mean      - your average crop in image coordinates
  template  - your median crop relative to the couple (fixed framing around the dancers)
  model     - LightGBM prediction from pose / scene features
"""
import argparse
import os
import pickle
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

from crop_model import TARGETS, crop_polygon, crop_to_targets, photo_features, targets_to_crop

warnings.filterwarnings("ignore", message="X does not have valid feature names")
N_PCA = 16
PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1)
CROP = ["CropLeft", "CropTop", "CropRight", "CropBottom", "CropAngle"]


def load(data_dir):
    e = pd.read_parquet(os.path.join(data_dir, "edits.parquet"))
    p = pd.read_parquet(os.path.join(data_dir, "previews.parquet"))
    pose = pd.read_parquet(os.path.join(data_dir, "pose.parquet"))
    feats = pd.read_parquet(os.path.join(data_dir, "features.parquet"))[["image_id", "focal_length"]]
    df = e.merge(p[p.ok], on="image_id").merge(pose, on="image_id").merge(feats, on="image_id")
    cropped = ((df.CropTop > 0.001) | (df.CropLeft > 0.001) | (df.CropBottom < 0.999)
               | (df.CropRight < 0.999) | (df.CropAngle.abs() > 0.01))
    df = df[cropped & (df.camera == "ILCE-7M3") & (df.orientation == "AB")].reset_index(drop=True)

    rows, boxes = [], []
    for r in df.itertuples():
        box, f = photo_features(r.persons, r.preview_w, r.preview_h)
        if box is None:
            rows.append(None)
            boxes.append(None)
            continue
        f.update(tilt_deg=r.tilt_deg, tilt_n=r.tilt_n, focal_length=r.focal_length)
        f.update(crop_to_targets(r.CropLeft, r.CropTop, r.CropRight, r.CropBottom, r.CropAngle,
                                 box, r.preview_w, r.preview_h))
        rows.append(f)
        boxes.append(box)
    has = np.array([f is not None for f in rows])
    print(f"cropped photos: {len(df)}, with a person detected: {has.sum()}")
    df = df[has].reset_index(drop=True)
    X = pd.DataFrame([f for f in rows if f is not None])
    emb = pd.DataFrame(np.load(os.path.join(data_dir, "emb.npy")), index=np.load(os.path.join(data_dir, "emb_ids.npy")))
    return df, X, np.array([b for b in boxes if b is not None]), emb.loc[df.image_id].to_numpy()


def iou(a, b, W, H):
    pa, pb = Polygon(crop_polygon(a, W, H)), Polygon(crop_polygon(b, W, H))
    return pa.intersection(pb).area / pa.union(pb).area


def main(data_dir, save):
    df, X, boxes, emb = load(data_dir)
    feat_cols = [c for c in X.columns if c not in TARGETS]
    W, H = df.preview_w.to_numpy(), df.preview_h.to_numpy()
    pred = pd.DataFrame(index=df.index, columns=TARGETS, dtype=float)
    templ = pd.DataFrame(index=df.index, columns=TARGETS, dtype=float)
    mean_crop = pd.DataFrame(index=df.index, columns=CROP, dtype=float)

    for fold, (tr, te) in enumerate(GroupKFold(5).split(df, groups=df.folder)):
        pca = PCA(N_PCA, random_state=0).fit(emb[tr])
        M = np.hstack([X[feat_cols].to_numpy(float), pca.transform(emb)])
        for t in TARGETS:
            y = X[t].to_numpy(float)
            pred.loc[te, t] = lgb.LGBMRegressor(**PARAMS).fit(M[tr], y[tr]).predict(M[te])
            templ.loc[te, t] = np.median(y[tr]) if t != "t_angle" else 0.0
        for c in CROP:
            mean_crop.loc[te, c] = df.loc[tr, c].mean() if c != "CropAngle" else 0.0
        print(f"  fold {fold + 1}/5", flush=True)

    res = {"none": [], "mean": [], "template": [], "model": []}
    pred_crops = []
    for i in range(len(df)):
        truth = df.loc[i, CROP].to_dict()
        full = {"CropLeft": 0, "CropTop": 0, "CropRight": 1, "CropBottom": 1, "CropAngle": 0}
        pc = targets_to_crop(pred.loc[i], boxes[i], W[i], H[i])
        pred_crops.append(pc)
        res["none"].append(iou(full, truth, W[i], H[i]))
        res["mean"].append(iou(mean_crop.loc[i].to_dict(), truth, W[i], H[i]))
        res["template"].append(iou(targets_to_crop(templ.loc[i], boxes[i], W[i], H[i]), truth, W[i], H[i]))
        res["model"].append(iou(pc, truth, W[i], H[i]))
    res = pd.DataFrame(res)
    print(f"\nOverlap with your real crop (IoU, 1 = identical), {len(df)} photos in held-out events:")
    print(pd.DataFrame({"mean IoU": res.mean(), "median IoU": res.median(),
                        "IoU>0.7 (close)": (res > 0.7).mean(), "IoU>0.8": (res > 0.8).mean()}).round(3).to_string())
    ang_err = (pred.t_angle - X.t_angle).abs()
    print(f"\nangle error: model {ang_err.mean():.2f} deg, no rotation {X.t_angle.abs().mean():.2f} deg")
    out = df[["image_id", "folder", "preview_w", "preview_h"] + CROP].copy()
    out[[f"pred_{c}" for c in CROP]] = pd.DataFrame(pred_crops)[CROP].to_numpy()
    out["iou_model"], out["iou_template"] = res.model, res.template
    out.to_parquet(os.path.join(data_dir, "crop_cv_predictions.parquet"), index=False)

    if save:
        pca = PCA(N_PCA, random_state=0).fit(emb)
        M = np.hstack([X[feat_cols].to_numpy(float), pca.transform(emb)])
        models = {t: lgb.LGBMRegressor(**PARAMS).fit(M, X[t].to_numpy(float)) for t in TARGETS}
        with open(save, "wb") as f:
            pickle.dump({"pca": pca, "feat_cols": feat_cols, "models": models}, f)
        print(f"saved final model ({len(df)} photos) -> {save}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--save")
    a = ap.parse_args()
    main(a.data_dir, a.save)
