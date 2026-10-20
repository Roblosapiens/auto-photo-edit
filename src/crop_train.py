"""Train the subject classifier + crop model and evaluate them on held-out events.

Usage: python src/crop_train.py <data_dir> [--save models/crop.pkl]

Uses every cropped Sony A7 III photo (edited or not).
Stage 1 - subject classifier: for each detected person, was it part of the subject (inside your crop)?
Stage 2 - crop model: crop relative to the couple picked with stage-1 probabilities (out-of-fold).
Crops are compared by IoU with your real crop against simple baselines, and checked for cutting into
the couple.
"""
import offline  # noqa: E402  must run before torch/open_clip/ultralytics are imported

offline.enable()
import argparse
import os
import pickle
import warnings

import cv2
import lightgbm as lgb
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

from crop_model import (TARGETS, crop_polygon, crop_to_targets, cuts_subject, pick_couple, protect_subjects,
                        person_features, photo_features, targets_to_crop, unpack)

warnings.filterwarnings("ignore", message="X does not have valid feature names")
N_PCA = 16
PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1)
SEL_PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=20, verbose=-1)
CROP = ["CropLeft", "CropTop", "CropRight", "CropBottom", "CropAngle"]


def load_photos(data_dir):
    e = pd.read_parquet(os.path.join(data_dir, "edits.parquet"))
    p = pd.read_parquet(os.path.join(data_dir, "previews.parquet"))
    pose = pd.read_parquet(os.path.join(data_dir, "pose.parquet"))
    feats = pd.read_parquet(os.path.join(data_dir, "features.parquet"))[["image_id", "focal_length"]]
    df = e.merge(p[p.ok], on="image_id").merge(pose, on="image_id").merge(feats, on="image_id")
    cropped = ((df.CropTop > 0.001) | (df.CropLeft > 0.001) | (df.CropBottom < 0.999)
               | (df.CropRight < 0.999) | (df.CropAngle.abs() > 0.01))
    df = df[cropped & (df.camera == "ILCE-7M3") & (df.orientation == "AB") & (df.n_persons > 0)]
    return df.reset_index(drop=True)


def person_table(df, data_dir):
    """Per-person features + label 'subject' (mostly inside your crop and among the tallest inside it)."""
    tabs = []
    for i, r in enumerate(df.itertuples()):
        gray = cv2.imread(os.path.join(data_dir, "previews", f"{r.image_id}.jpg"), cv2.IMREAD_GRAYSCALE)
        W, H = r.preview_w, r.preview_h
        t = person_features(r.persons, W, H, gray)
        boxes, _, _ = unpack(r.persons)
        L, T, R, B = r.CropLeft * W, r.CropTop * H, r.CropRight * W, r.CropBottom * H
        ix = np.clip(np.minimum(boxes[:, 2], R) - np.maximum(boxes[:, 0], L), 0, None)
        iy = np.clip(np.minimum(boxes[:, 3], B) - np.maximum(boxes[:, 1], T), 0, None)
        inside = ix * iy / ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
        h = boxes[:, 3] - boxes[:, 1]
        tallest_inside = h[inside >= 0.7].max() if (inside >= 0.7).any() else np.inf
        t["subject"] = (inside >= 0.7) & (h >= 0.5 * tallest_inside)
        t["photo"], t["folder"] = i, r.folder
        tabs.append(t)
    return pd.concat(tabs, ignore_index=True)


def photo_rows(df, probs_by_photo):
    rows, boxes, keep = [], [], []
    for i, r in enumerate(df.itertuples()):
        box, f = photo_features(r.persons, r.preview_w, r.preview_h, probs_by_photo[i])
        if box is None:
            continue
        f.update(tilt_deg=r.tilt_deg, tilt_n=r.tilt_n, focal_length=r.focal_length)
        f.update(crop_to_targets(r.CropLeft, r.CropTop, r.CropRight, r.CropBottom, r.CropAngle,
                                 box, r.preview_w, r.preview_h))
        rows.append(f)
        boxes.append(box)
        keep.append(i)
    return pd.DataFrame(rows), np.array(boxes), keep


def iou(a, b, W, H):
    pa, pb = Polygon(crop_polygon(a, W, H)), Polygon(crop_polygon(b, W, H))
    return pa.intersection(pb).area / pa.union(pb).area


def main(data_dir, save):
    df = load_photos(data_dir)
    emb_all = pd.DataFrame(np.load(os.path.join(data_dir, "emb.npy")), index=np.load(os.path.join(data_dir, "emb_ids.npy")))
    print(f"cropped photos with a person detected: {len(df)}; building person table...", flush=True)

    # Stage 1: subject classifier, out-of-fold probabilities by event
    pt = person_table(df, data_dir)
    sel_cols = [c for c in pt.columns if c not in ("subject", "photo", "folder")]
    pt["p"] = np.nan
    for tr, te in GroupKFold(5).split(pt, groups=pt.folder):
        clf = lgb.LGBMClassifier(**SEL_PARAMS).fit(pt.loc[tr, sel_cols].astype(float), pt.subject[tr])
        pt.loc[te, "p"] = clf.predict_proba(pt.loc[te, sel_cols].astype(float))[:, 1]
    probs_by_photo = pt.groupby("photo").p.apply(list).to_dict()
    top = pt.loc[pt.groupby("photo").p.idxmax()]
    print(f"{len(pt)} persons, {pt.subject.mean():.0%} labelled subject. "
          f"Main dancer outside your crop: classifier {1 - top.subject.mean():.1%}")

    # Stage 2: crop model on the classifier's couple
    X, boxes, keep = photo_rows(df, probs_by_photo)
    probs_kept = [probs_by_photo[i] for i in keep]
    df = df.iloc[keep].reset_index(drop=True)
    emb = emb_all.loc[df.image_id].to_numpy()
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

    res = {"none": [], "mean": [], "template": [], "model": [], "model+contain": []}
    cuts = {"yours": [], "model": [], "model+contain": []}
    pred_crops = []
    for i in range(len(df)):
        truth = df.loc[i, CROP].to_dict()
        full = {"CropLeft": 0, "CropTop": 0, "CropRight": 1, "CropBottom": 1, "CropAngle": 0}
        # no straightening: the predicted angle was no better than 0 (see angle error below)
        pc = targets_to_crop({**pred.loc[i].to_dict(), "t_angle": 0.0}, boxes[i], W[i], H[i])
        # production no-cut rule: the couple plus everyone likely to be kept (crop_model.protect_subjects)
        pick = pick_couple(unpack(df.persons[i])[0], W[i], H[i], np.asarray(probs_kept[i]))
        pcc = protect_subjects(pc, boxes[i], df.persons[i], probs_kept[i], {p for p in pick if p is not None}, W[i], H[i])
        pred_crops.append(pcc)
        res["none"].append(iou(full, truth, W[i], H[i]))
        res["mean"].append(iou(mean_crop.loc[i].to_dict(), truth, W[i], H[i]))
        res["template"].append(iou(targets_to_crop(templ.loc[i], boxes[i], W[i], H[i]), truth, W[i], H[i]))
        res["model"].append(iou(pc, truth, W[i], H[i]))
        res["model+contain"].append(iou(pcc, truth, W[i], H[i]))
        cuts["yours"].append(cuts_subject({**truth, "CropAngle": 0}, boxes[i], W[i], H[i]))
        cuts["model"].append(cuts_subject(pc, boxes[i], W[i], H[i]))
        cuts["model+contain"].append(cuts_subject(pcc, boxes[i], W[i], H[i]))
    res = pd.DataFrame(res)
    print(f"\nOverlap with your real crop (IoU, 1 = identical), {len(df)} photos in held-out events:")
    print(pd.DataFrame({"mean IoU": res.mean(), "median IoU": res.median(),
                        "IoU>0.7 (close)": (res > 0.7).mean(), "IoU>0.8": (res > 0.8).mean()}).round(3).to_string())
    print("\nCrops cutting into the picked couple:", {k: f"{np.mean(v):.1%}" for k, v in cuts.items()})
    ang_err = (pred.t_angle - X.t_angle).abs()
    print(f"angle error: model {ang_err.mean():.2f} deg, no rotation {X.t_angle.abs().mean():.2f} deg")
    out = df[["image_id", "folder", "preview_w", "preview_h"] + CROP].copy()
    out[[f"pred_{c}" for c in CROP]] = pd.DataFrame(pred_crops)[CROP].to_numpy()
    out[["box_x0", "box_y0", "box_x1", "box_y1"]] = boxes
    out["subject_p"], out["iou_model"] = X.subject_p, res["model+contain"]
    out.to_parquet(os.path.join(data_dir, "crop_cv_predictions.parquet"), index=False)

    if save:
        selector = lgb.LGBMClassifier(**SEL_PARAMS).fit(pt[sel_cols].astype(float), pt.subject)
        pca = PCA(N_PCA, random_state=0).fit(emb)
        M = np.hstack([X[feat_cols].to_numpy(float), pca.transform(emb)])
        models = {t: lgb.LGBMRegressor(**PARAMS).fit(M, X[t].to_numpy(float)) for t in TARGETS}
        with open(save, "wb") as f:
            pickle.dump({"pca": pca, "feat_cols": feat_cols, "models": models,
                         "selector": selector, "sel_cols": sel_cols}, f)
        print(f"saved final model ({len(df)} photos) -> {save}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--save")
    a = ap.parse_args()
    main(a.data_dir, a.save)
