"""Compare image-embedding backbones for the tone and crop models, on identical held-out events.

Usage: python tools/compare_embeddings.py <data_dir> [--keys 3] [--boot 2000]

Variants: hand-engineered features only ("none"), CLIP ViT-B/32 (production), CLIP ViT-L/14, DINOv2
ViT-B/14 (embeddings from tools/embed_models.py).

Protocol (identical for every variant):
  * GroupKFold(5) by event folder: every photo of a test event is unseen in training
  * the same random key photos per event (seed 0), the same LightGBM parameters (no tuning at all)
  * PCA of the embedding fitted on the training folds only; the crop subject classifier uses the same
    out-of-fold probabilities for every variant (it uses no embeddings)
  * only the parts that use embeddings differ: per-photo exposure / white balance (tone) and the crop
    regression; everything else (synced sliders, subject choice, no-cut rule) is shared

Reports aggregate and per-event errors, paired per-event differences vs. "none" and vs. CLIP B/32 with
95% bootstrap confidence intervals over events, and crop failure rates (tools/crop_failures.py).
Writes <data_dir>/embedding_comparison.json and embedding_comparison_events.csv.
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import offline  # noqa: E402

offline.enable()
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

import crop_train  # noqa: E402
import train  # noqa: E402
from crop_failures import analyse  # noqa: E402
from crop_model import SUBJECT_P, TARGETS as CROP_TARGETS, contain_subject, cuts_subject, targets_to_crop  # noqa: E402
from model import HYBRID_TARGETS, LGB_PARAMS, N_PCA  # noqa: E402

warnings.filterwarnings("ignore", message="X does not have valid feature names")
VARIANTS = ["none", "clip_b32", "clip_l14", "dinov2_b14"]
CROP_PCA = crop_train.N_PCA


def embeddings(data_dir, name, image_ids):
    if name == "none":
        return None
    e = pd.DataFrame(np.load(os.path.join(data_dir, f"emb_{name}.npy")),
                     index=np.load(os.path.join(data_dir, f"emb_{name}_ids.npy")))
    return e.loc[image_ids].to_numpy()


def boot_ci(x, n, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------------------------------- tone
def tone(data_dir, k_keys):
    df, _, feat_cols, _ = train.load(data_dir)
    rng = np.random.default_rng(0)  # same key photos as src/train.py
    is_key = np.zeros(len(df), bool)
    for _, idx in df.groupby("folder").indices.items():
        if len(idx) > k_keys + 3:
            is_key[rng.choice(idx, k_keys, replace=False)] = True
    folds = list(GroupKFold(5).split(df, groups=df.folder))
    base = df[feat_cols].to_numpy(float)
    ev = ~is_key & df.groupby("folder").folder.transform(lambda s: is_key[s.index].any()).to_numpy()

    def sync_pred(t):
        y = df[t].to_numpy(float)
        key_mean = pd.Series(np.where(is_key, y, np.nan)).groupby(df.folder.to_numpy()).transform("mean").to_numpy()
        return key_mean

    preds = {}
    for v in VARIANTS:
        emb = embeddings(data_dir, v, df.image_id)
        p = {t: np.full(len(df), np.nan) for t in HYBRID_TARGETS}
        for tr, te in folds:
            E = PCA(N_PCA, random_state=0).fit(emb[tr]).transform(emb) if emb is not None else np.empty((len(df), 0))
            X = np.hstack([base, E])
            Xd = X - pd.DataFrame(X).groupby(df.folder.to_numpy()).transform("mean").to_numpy()
            for t in HYBRID_TARGETS:
                y = df[t].to_numpy(float)
                yd = y - df.groupby("folder")[t].transform("mean").to_numpy()
                pd_ = lgb.LGBMRegressor(**LGB_PARAMS).fit(Xd[tr], yd[tr]).predict(Xd[te])
                g = pd.DataFrame({"f": df.folder.iloc[te].to_numpy(), "p": pd_, "k": is_key[te]}, index=te)
                key_p = g.p.where(g.k).groupby(g.f).transform("mean")
                p[t][te] = sync_pred(t)[te] + (g.p - key_p).to_numpy()
        preds[v] = p
        print(f"  tone: {v} done", flush=True)

    sd = {t: df.loc[ev, t].std() for t in HYBRID_TARGETS}
    rows = {}
    out = {"sd": sd}
    for name, pr in [("global median preset", {t: np.full(len(df), df[t].median()) for t in HYBRID_TARGETS}),
                     ("sync key photos (no model)", {t: sync_pred(t) for t in HYBRID_TARGETS})] + list(preds.items()):
        err = {t: np.abs(pr[t] - df[t].to_numpy(float)) for t in HYBRID_TARGETS}
        norm = sum(err[t] / sd[t] for t in HYBRID_TARGETS) / len(HYBRID_TARGETS)
        out[name] = {"exposure_mae_stops": float(err["Exposure2012"][ev].mean()),
                     "wb_mae_mired": float(err["temp_mired_delta"][ev].mean()),
                     "normalised_error": float(norm[ev].mean())}
        rows[name] = pd.Series(norm[ev]).groupby(df.folder[ev].to_numpy()).mean()
    return out, pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------- crop
def crop(data_dir):
    df = crop_train.load_photos(data_dir)
    print("  crop: building person table (subject classifier, out-of-fold)...", flush=True)
    pt = crop_train.person_table(df, data_dir)
    sel_cols = [c for c in pt.columns if c not in ("subject", "photo", "folder")]
    pt["p"] = np.nan
    for tr, te in GroupKFold(5).split(pt, groups=pt.folder):
        clf = lgb.LGBMClassifier(**crop_train.SEL_PARAMS).fit(pt.loc[tr, sel_cols].astype(float), pt.subject[tr])
        pt.loc[te, "p"] = clf.predict_proba(pt.loc[te, sel_cols].astype(float))[:, 1]
    probs = pt.groupby("photo").p.apply(list).to_dict()
    labels = pt.groupby("photo").subject.apply(list).to_dict()
    X, boxes, keep = crop_train.photo_rows(df, probs)
    probs = [probs[i] for i in keep]
    labels = [labels[i] for i in keep]
    df = df.iloc[keep].reset_index(drop=True)
    feat_cols = [c for c in X.columns if c not in CROP_TARGETS]
    W, H = df.preview_w.to_numpy(), df.preview_h.to_numpy()
    folds = list(GroupKFold(5).split(df, groups=df.folder))
    truth = df[crop_train.CROP].to_dict("records")

    out, per_event, failures = {}, {}, {}
    for v in VARIANTS:
        emb = embeddings(data_dir, v, df.image_id)
        pred = pd.DataFrame(index=df.index, columns=CROP_TARGETS, dtype=float)
        for tr, te in folds:
            E = PCA(CROP_PCA, random_state=0).fit(emb[tr]).transform(emb) if emb is not None else np.empty((len(df), 0))
            M = np.hstack([X[feat_cols].to_numpy(float), E])
            for t in CROP_TARGETS:
                pred.loc[te, t] = lgb.LGBMRegressor(**crop_train.PARAMS).fit(M[tr], X[t].to_numpy(float)[tr]).predict(M[te])
        ious, cut, fails = [], [], []
        for i in range(len(df)):
            c = targets_to_crop({**pred.loc[i].to_dict(), "t_angle": 0.0}, boxes[i], W[i], H[i])
            if X.subject_p[i] >= SUBJECT_P:
                c = contain_subject(c, boxes[i], W[i], H[i])
            ious.append(crop_train.iou(c, truth[i], W[i], H[i]))
            cut.append(cuts_subject(c, boxes[i], W[i], H[i]))
            fails.append(analyse(c, {**truth[i], "CropAngle": 0}, df.persons[i], probs[i], labels[i], W[i], H[i]))
        ious = np.array(ious)
        f = pd.DataFrame(fails)
        failures[v] = f.mean().to_dict()
        out[v] = {"mean_iou": float(ious.mean()), "close_iou_gt_0.7": float((ious > 0.7).mean()),
                  "cuts_into_couple": float(np.mean(cut))}
        per_event[v] = pd.Series(ious).groupby(df.folder.to_numpy()).mean()
        print(f"  crop: {v} done (mean IoU {ious.mean():.3f})", flush=True)
    return out, pd.DataFrame(per_event), failures


def paired(per_event, higher_is_better, boot):
    res = {}
    for ref in ["none", "clip_b32"]:
        for v in VARIANTS:
            if v == ref:
                continue
            d = (per_event[v] - per_event[ref]) * (1 if higher_is_better else -1)  # > 0 means v is better
            m, lo, hi = boot_ci(d.to_numpy(), boot)
            res[f"{v} vs {ref}"] = {"mean_improvement": m, "ci95": [lo, hi], "events_better": int((d > 0).sum()),
                                    "events_worse": int((d < 0).sum()), "events": int(len(d))}
    return res


def main(data_dir, k_keys, boot):
    print("tone models...", flush=True)
    tone_out, tone_ev = tone(data_dir, k_keys)
    print("crop models...", flush=True)
    crop_out, crop_ev, fails = crop(data_dir)
    bench = [json.loads(l) for l in open(os.path.join(data_dir, "embed_benchmark.jsonl"))]
    result = {"keys_per_event": k_keys, "tone": tone_out, "crop": crop_out, "crop_failures": fails,
              "tone_paired": paired(tone_ev[VARIANTS], False, boot),
              "crop_paired": paired(crop_ev, True, boot), "benchmark": bench,
              "network_attempts": offline.attempts}
    json.dump(result, open(os.path.join(data_dir, "embedding_comparison.json"), "w"), indent=1)
    tone_ev.add_prefix("tone_err_").join(crop_ev.add_prefix("crop_iou_"), how="outer").to_csv(
        os.path.join(data_dir, "embedding_comparison_events.csv"))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--keys", type=int, default=3)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    main(a.data_dir, a.keys, a.boot)
