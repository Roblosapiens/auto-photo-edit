"""Train the production within-event models on all events and save them for predict.py.

Usage: python src/train_final.py <data_dir> <model_out.pkl>
"""
import pickle
import sys
import warnings

import lightgbm as lgb
from sklearn.decomposition import PCA

from model import HYBRID_TARGETS, LGB_PARAMS, N_PCA, design
from train import load

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def main(data_dir, out):
    df, emb, feat_cols, coef = load(data_dir)
    pca = PCA(N_PCA, random_state=0).fit(emb)
    _, Xd = design(df, feat_cols, pca, emb)
    models = {}
    for t in HYBRID_TARGETS:
        yd = df[t] - df.groupby("folder")[t].transform("mean")
        models[t] = lgb.LGBMRegressor(**LGB_PARAMS).fit(Xd, yd.to_numpy(float))
    with open(out, "wb") as f:
        pickle.dump({"coef": coef, "pca": pca, "feat_cols": feat_cols, "models": models}, f)
    print(f"trained on {len(df)} photos / {df.folder.nunique()} events -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
