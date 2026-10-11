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

CORE = ["Exposure2012", "Contrast2012", "Highlights2012", "Shadows2012", "Whites2012", "Blacks2012",
        "Clarity2012", "Dehaze", "Vibrance", "Saturation"]
TARGETS = ["temp_mired_delta", "Tint"] + CORE + ["Sharpness", "LuminanceSmoothing", "PostCropVignetteAmount"]
EVENT_CTX = ["lum_mean", "lum_p50", "lum_p95", "exposure_value", "log_iso", "cast_rg", "cast_bg",
             "asshot_mired", "flash", "clip_hi"]
N_PCA = 32
warnings.filterwarnings("ignore", message="X does not have valid feature names")
LGB_PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def load(data_dir):
    e = pd.read_parquet(os.path.join(data_dir, "edits.parquet"))
    f = pd.read_parquet(os.path.join(data_dir, "features.parquet"))
    emb = pd.DataFrame(np.load(os.path.join(data_dir, "emb.npy")),
                       index=np.load(os.path.join(data_dir, "emb_ids.npy")))
    df = e.drop(columns=[c for c in f.columns if c in e.columns and c != "image_id"]).merge(f, on="image_id")
    df = df[(df.camera == "ILCE-7M3") & (df.ProcessVersion == "10.0") & df.Temperature.notna()]
    df = df[(df[CORE] != 0).any(axis=1) | (df.WhiteBalance != "As Shot")].reset_index(drop=True)

    # As-shot colour temperature from the camera's WB multipliers, calibrated on photos
    # where Lightroom's Temperature *is* the as-shot value.
    x = (df.log_wb_b - df.log_wb_r).to_numpy()
    a = (df.WhiteBalance == "As Shot").to_numpy()
    coef = np.polyfit(x[a], 1e6 / df.Temperature[a], 2)
    df["asshot_mired"] = np.polyval(coef, x)
    df["temp_mired_delta"] = 1e6 / df.Temperature - df.asshot_mired

    df["days"] = (pd.to_datetime(df.captureTime, errors="coerce") - pd.Timestamp("2019-01-01")).dt.days

    # event context: event mean of key stats + this photo's deviation from it
    for c in EVENT_CTX:
        m = df.groupby("folder")[c].transform("mean")
        df[f"ev_{c}"] = m
        df[f"dev_{c}"] = df[c] - m
    return df, emb.loc[df.image_id].to_numpy()


def main(data_dir, k_keys):
    df, emb = load(data_dir)
    base_cols = [c for c in pd.read_parquet(os.path.join(data_dir, "features.parquet")).columns if c != "image_id"]
    feat_cols = base_cols + ["asshot_mired"] + [f"{p}_{c}" for c in EVENT_CTX for p in ("ev", "dev")]
    print(f"{len(df)} edited photos, {df.folder.nunique()} events, {len(feat_cols)}+{N_PCA} features")

    rng = np.random.default_rng(0)
    is_key = np.zeros(len(df), bool)
    for _, idx in df.groupby("folder").indices.items():
        if len(idx) > k_keys + 3:
            is_key[rng.choice(idx, k_keys, replace=False)] = True

    preds = {m: pd.DataFrame(index=df.index, columns=TARGETS, dtype=float) for m in ("global", "auto", "sync", "hybrid")}
    for fold, (tr, te) in enumerate(GroupKFold(5).split(df, groups=df.folder)):
        pca = PCA(N_PCA, random_state=0).fit(emb[tr])
        X = np.hstack([df[feat_cols + ["days"]].to_numpy(float), pca.transform(emb)])
        # within-event view: every feature and target relative to its event mean
        Xd = X - pd.DataFrame(X).groupby(df.folder.to_numpy()).transform("mean").to_numpy()
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
