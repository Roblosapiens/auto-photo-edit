"""Train slider-prediction models and evaluate them on held-out events.

Usage: python src/train.py <data_dir> [--keys 3]

Evaluation is grouped by event (folder): every photo of a test event is unseen in training.
Methods compared, all scored on the same non-key photos:
  global   - your overall median settings (one fixed preset)
  auto     - model prediction, no input from you
  sync     - mean settings of K key photos you edited in that event (= Lightroom Sync)
  hybrid   - sync + model's per-photo difference from the key photos
"""
import argparse
import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

from model import CORE, LGB_PARAMS, N_PCA, TARGETS, add_derived, design, feature_columns, fit_asshot_coef

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def load(data_dir):
    e = pd.read_parquet(os.path.join(data_dir, "edits.parquet"))
    f = pd.read_parquet(os.path.join(data_dir, "features.parquet"))
    emb = pd.DataFrame(np.load(os.path.join(data_dir, "emb.npy")),
                       index=np.load(os.path.join(data_dir, "emb_ids.npy")))
    df = e.drop(columns=[c for c in f.columns if c in e.columns and c != "image_id"]).merge(f, on="image_id")
    df = df[(df.camera == "ILCE-7M3") & (df.ProcessVersion == "10.0") & df.Temperature.notna()]
    df = df[(df[CORE] != 0).any(axis=1) | (df.WhiteBalance != "As Shot")].reset_index(drop=True)
    coef = fit_asshot_coef(df)
    df = add_derived(df, coef)
    feat_cols = feature_columns(c for c in f.columns if c != "image_id")
    return df, emb.loc[df.image_id].to_numpy(), feat_cols, coef


def main(data_dir, k_keys):
    df, emb, feat_cols, _ = load(data_dir)
    print(f"{len(df)} edited photos, {df.folder.nunique()} events, {len(feat_cols)}+{N_PCA} features")

    rng = np.random.default_rng(0)
    is_key = np.zeros(len(df), bool)
    for _, idx in df.groupby("folder").indices.items():
        if len(idx) > k_keys + 3:
            is_key[rng.choice(idx, k_keys, replace=False)] = True

    preds = {m: pd.DataFrame(index=df.index, columns=TARGETS, dtype=float) for m in ("global", "auto", "sync", "hybrid")}
    for fold, (tr, te) in enumerate(GroupKFold(5).split(df, groups=df.folder)):
        pca = PCA(N_PCA, random_state=0).fit(emb[tr])
        X, Xd = design(df, feat_cols, pca, emb)
        for t in TARGETS:
            y = df[t].to_numpy(float)
            yd = y - df.groupby("folder")[t].transform("mean").to_numpy()
            p = lgb.LGBMRegressor(**LGB_PARAMS).fit(X[tr], y[tr]).predict(X[te])
            pd_ = lgb.LGBMRegressor(**LGB_PARAMS).fit(Xd[tr], yd[tr]).predict(Xd[te])
            preds["global"].loc[te, t] = np.median(y[tr])
            preds["auto"].loc[te, t] = p
            te_df = pd.DataFrame({"folder": df.folder.iloc[te].to_numpy(), "y": y[te], "p": pd_, "key": is_key[te]}, index=te)
            for _, g in te_df.groupby("folder"):
                keys = g[g.key]
                if keys.empty:
                    continue
                preds["sync"].loc[g.index, t] = keys.y.mean()
                preds["hybrid"].loc[g.index, t] = keys.y.mean() + (g.p - keys.p.mean())
        print(f"  fold {fold + 1}/5 done", flush=True)

    ev = ~is_key & preds["sync"][TARGETS[0]].notna().to_numpy()
    rows = []
    for t in TARGETS:
        y = df.loc[ev, t].to_numpy(float)
        rows.append({"slider": t, "spread(std)": y.std(),
                     **{m: np.abs(preds[m].loc[ev, t].to_numpy() - y).mean() for m in preds}})
    res = pd.DataFrame(rows).set_index("slider")
    norm = res[list(preds)].div(res["spread(std)"], axis=0).mean()
    print(f"\nMean absolute error per slider (lower is better), {ev.sum()} test photos, {k_keys} key photos/event:\n")
    print(res.round(2).to_string())
    print("\nAverage error relative to slider spread:", norm.round(3).to_dict())
    out = df.loc[ev, ["image_id", "folder"] + TARGETS].copy()
    for m in preds:
        out = out.join(preds[m].loc[ev].add_prefix(f"{m}_"))
    out.to_parquet(os.path.join(data_dir, "cv_predictions.parquet"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--keys", type=int, default=3)
    a = ap.parse_args()
    main(a.data_dir, a.keys)
