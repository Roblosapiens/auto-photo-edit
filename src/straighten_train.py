"""Learn which near-vertical line the photographer straightens on, evaluated on held-out events.

Usage: python src/straighten_train.py <data_dir> [--save models/straighten.pkl]

Candidates are the line segments around the model's *predicted* crop (as at inference time). A line is
labelled positive when its angle is within 0.25 deg of the photographer's CropAngle. Prediction: the
best-scoring line (refined by the parallel lines that agree with it), or no rotation if nothing
scores above a threshold tuned on held-out events.
"""
import offline  # noqa: E402  must run before torch/open_clip/ultralytics are imported

offline.enable()
import argparse
import os
import pickle
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from crop_model import unpack
from straighten import line_features

warnings.filterwarnings("ignore", message="X does not have valid feature names")
PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=30,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
MATCH_DEG = 0.25
NEAR_CROP = 0.1  # keep lines whose midpoint lies within the crop expanded by this fraction


def candidates(lines, crop, W, H, persons):
    """Line features for one photo (lines near the crop only), plus photo-level context."""
    f = line_features(lines.reset_index(drop=True), crop, W, H, unpack(persons)[0] if len(persons) else ())
    f = f[(f.dx_centre.abs() < 0.5 + NEAR_CROP) & (f.dy_centre.abs() < 0.5 + NEAR_CROP)].reset_index(drop=True)
    f["n_cand"] = len(f)
    f["len_rank"] = f.len_rel.rank(ascending=False)
    f["centre_rank"] = f.abs_dx.rank()
    return f


def choose(f, p, thr):
    """Rotation angle from candidate probabilities; 0 when no line is convincing."""
    if len(f) == 0 or p.max() < thr:
        return 0.0
    best = f.angle.iloc[int(np.argmax(p))]
    agree = (f.angle - best).abs() < 0.3
    w = p[agree] * f.len_rel[agree]
    return float(np.clip((f.angle[agree] * w).sum() / w.sum(), -8, 8))


def build(data_dir):
    d = pd.read_parquet(os.path.join(data_dir, "crop_cv_predictions.parquet"))
    pose = pd.read_parquet(os.path.join(data_dir, "pose.parquet"))[["image_id", "persons"]]
    lines = pd.read_parquet(os.path.join(data_dir, "lines.parquet"))
    d = d.merge(pose, on="image_id").reset_index(drop=True)
    by_img = dict(tuple(lines.groupby("image_id")))
    tabs = []
    for i, r in enumerate(d.itertuples()):
        if r.image_id not in by_img:
            continue
        crop = dict(CropLeft=r.pred_CropLeft, CropTop=r.pred_CropTop, CropRight=r.pred_CropRight, CropBottom=r.pred_CropBottom)
        f = candidates(by_img[r.image_id], crop, r.preview_w, r.preview_h, r.persons)
        f["label"] = (f.angle - r.CropAngle).abs() < MATCH_DEG
        f["photo"], f["folder"] = i, r.folder
        tabs.append(f)
    return d, pd.concat(tabs, ignore_index=True)


def main(data_dir, save):
    d, c = build(data_dir)
    cols = [x for x in c.columns if x not in ("label", "photo", "folder")]
    print(f"{len(d)} photos, {len(c)} candidate lines, {c.label.mean():.1%} positive; "
          f"photos with a matching line: {c.groupby('photo').label.any().mean():.0%}")
    c["p"] = np.nan
    for tr, te in GroupKFold(5).split(c, groups=c.folder):
        clf = lgb.LGBMClassifier(**PARAMS).fit(c.loc[tr, cols].astype(float), c.label[tr])
        c.loc[te, "p"] = clf.predict_proba(c.loc[te, cols].astype(float))[:, 1]

    groups = dict(tuple(c.groupby("photo")))
    truth = d.CropAngle.to_numpy()
    rotated = np.abs(truth) >= 0.3
    best_thr, best_err = None, np.inf
    print("\nthreshold  mean error  within 0.5deg  wrong direction  rotated(%)")
    for thr in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        pred = np.array([choose(groups[i], groups[i].p.to_numpy(), thr) if i in groups else 0.0 for i in range(len(d))])
        err = np.abs(pred - truth)
        wrong = ((np.sign(pred) != np.sign(truth)) & (pred != 0) & rotated).mean()
        print(f"  {thr:.1f}      {err.mean():.2f}       {(err < 0.5).mean():.0%}          {wrong:.1%}        {(pred != 0).mean():.0%}")
        if err.mean() < best_err:
            best_thr, best_err, best_pred = thr, err.mean(), pred
    err = np.abs(best_pred - truth)
    print(f"\nchosen threshold {best_thr}: mean error {best_err:.2f} deg vs no rotation {np.abs(truth).mean():.2f} deg")
    print(f"  photos you rotated: error {err[rotated].mean():.2f} (no rotation {np.abs(truth[rotated]).mean():.2f}), "
          f"within 0.5 deg {(err[rotated] < 0.5).mean():.0%}")
    print(f"  photos you left straight: model rotates {(best_pred[~rotated] != 0).mean():.0%} of them, "
          f"mean |angle| {np.abs(best_pred[~rotated]).mean():.2f} deg")
    d.assign(pred_angle=best_pred)[["image_id", "CropAngle", "pred_angle"]].to_parquet(
        os.path.join(data_dir, "straighten_cv_predictions.parquet"), index=False)

    if save:
        clf = lgb.LGBMClassifier(**PARAMS).fit(c[cols].astype(float), c.label)
        with open(save, "wb") as f:
            pickle.dump({"clf": clf, "cols": cols, "threshold": best_thr}, f)
        print(f"saved -> {save}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--save")
    a = ap.parse_args()
    main(a.data_dir, a.save)
