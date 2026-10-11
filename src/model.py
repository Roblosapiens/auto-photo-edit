"""Shared model definition: derived features and design matrices used by training and prediction."""
import numpy as np
import pandas as pd

CORE = ["Exposure2012", "Contrast2012", "Highlights2012", "Shadows2012", "Whites2012", "Blacks2012",
        "Clarity2012", "Dehaze", "Vibrance", "Saturation"]
TARGETS = ["temp_mired_delta", "Tint"] + CORE + ["Sharpness", "LuminanceSmoothing", "PostCropVignetteAmount"]

# Evaluation (src/train.py) showed the per-photo model beats plain syncing only for these;
# every other slider is synced as the mean of the key photos.
HYBRID_TARGETS = ["temp_mired_delta", "Exposure2012"]
SYNC_SLIDERS = ["Tint"] + [c for c in CORE if c != "Exposure2012"] + [
    "Sharpness", "SharpenRadius", "SharpenDetail", "SharpenEdgeMasking",
    "LuminanceSmoothing", "ColorNoiseReduction", "PostCropVignetteAmount"]

EVENT_CTX = ["lum_mean", "lum_p50", "lum_p95", "exposure_value", "log_iso", "cast_rg", "cast_bg",
             "asshot_mired", "flash", "clip_hi"]
N_PCA = 32
LGB_PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def fit_asshot_coef(df):
    """Map camera WB multipliers to Lightroom's as-shot temperature (mired), using photos left at As Shot."""
    a = (df.WhiteBalance == "As Shot") & df.Temperature.notna()
    return np.polyfit((df.log_wb_b - df.log_wb_r)[a], 1e6 / df.Temperature[a], 2)


def add_derived(df, coef):
    """Add as-shot temperature, date and event-context features. Event = df.folder."""
    df = df.copy()
    df["asshot_mired"] = np.polyval(coef, df.log_wb_b - df.log_wb_r)
    if "Temperature" in df:
        df["temp_mired_delta"] = 1e6 / df.Temperature - df.asshot_mired
    df["days"] = (pd.to_datetime(df.captureTime, errors="coerce") - pd.Timestamp("2019-01-01")).dt.days
    for c in EVENT_CTX:
        m = df.groupby("folder")[c].transform("mean")
        df[f"ev_{c}"] = m
        df[f"dev_{c}"] = df[c] - m
    return df


def feature_columns(base_cols):
    return list(base_cols) + ["asshot_mired", "days"] + [f"{p}_{c}" for c in EVENT_CTX for p in ("ev", "dev")]


def design(df, feat_cols, pca, emb):
    """Absolute feature matrix X and its within-event version Xd (minus the event mean)."""
    X = np.hstack([df[feat_cols].to_numpy(float), pca.transform(emb)])
    Xd = X - pd.DataFrame(X).groupby(df.folder.to_numpy()).transform("mean").to_numpy()
    return X, Xd
