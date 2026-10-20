"""Experiment: partner selection and multi-person crop protection, on identical held-out events.

Usage: python tools/partner_experiment.py <data_dir> [--boot 2000]

Variants (subject classifier, crop regressor, CLIP B/32 embedding and LightGBM settings as production):
  A  current     main dancer + partner (subject classifier + gap rule); protect that pair
  B  pairwise    partner chosen by a pairwise classifier: "does this person belong with the main dancer?"
                 (distance, relative size, vertical overlap, hand-to-hand distance, sharpness...); the crop
                 regressor is retrained in-fold on the new couple boxes
  C  protect     A's crop, but every person with subject probability >= tau is also protected, unless that
                 would grow the crop beyond `cap` x its area (then only the couple is protected)
  D  B + C

C and D are evaluated on a grid (tau x cap) and reported as a trade-off curve. For the headline result
one setting is chosen per test fold using only the other folds' results, by a rule fixed in advance:
minimise couple truncation (head/hand/foot cut) subject to mean IoU no more than 0.01 below A's and
too-loose crops no more than 2 points above A's; otherwise fall back to protecting the couple only.

All GroupKFold(5) splits by event are identical across variants; the pairwise classifier is trained
out-of-fold on the same splits. Writes <data_dir>/partner_experiment.json.
"""
import argparse
import json
import os
import sys
import time
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
from compare_embeddings import boot_ci, embeddings  # noqa: E402
from crop_failures import analyse  # noqa: E402
from crop_model import (SUBJECT_P, TARGETS, contain_subject, crop_to_targets, photo_features, pick_couple,  # noqa: E402
                        targets_to_crop, unpack)

warnings.filterwarnings("ignore", message="X does not have valid feature names")
TAUS = [0.5, 0.7, 0.9]
CAPS = [1.25, 1.5, 2.0, float("inf")]
PAIR_PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=20, verbose=-1)
WRISTS, HIPS = [9, 10], [11, 12]


# --------------------------------------------------------------------------- pairwise partner model
def pair_rows(persons, W, H, main, sharp, probs):
    """One row per candidate partner j for the photo's main dancer."""
    boxes, conf, kpts = unpack(persons)
    mh = boxes[main, 3] - boxes[main, 1]
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    rows = []
    for j in range(len(boxes)):
        if j == main:
            continue
        gap = max(0.0, max(boxes[j, 0], boxes[main, 0]) - min(boxes[j, 2], boxes[main, 2]))
        vov = max(0.0, min(boxes[j, 3], boxes[main, 3]) - max(boxes[j, 1], boxes[main, 1])) / mh
        ix = max(0.0, min(boxes[j, 2], boxes[main, 2]) - max(boxes[j, 0], boxes[main, 0]))
        wm = [kpts[main, k, :2] for k in WRISTS if kpts[main, k, 2] >= 0.3]
        wj = [kpts[j, k, :2] for k in WRISTS if kpts[j, k, 2] >= 0.3]
        hand = min((np.hypot(*(a - b)) for a in wm for b in wj), default=np.nan) / mh
        hm = [kpts[main, k, :2] for k in HIPS if kpts[main, k, 2] >= 0.3]
        hj = [kpts[j, k, :2] for k in HIPS if kpts[j, k, 2] >= 0.3]
        hip = (np.hypot(*(np.mean(hm, 0) - np.mean(hj, 0))) / mh) if hm and hj else np.nan
        rows.append({
            "j": j, "rel_h": (boxes[j, 3] - boxes[j, 1]) / mh, "rel_area": area[j] / area[main],
            "gap": gap / mh, "dx": ((boxes[j, 0] + boxes[j, 2]) - (boxes[main, 0] + boxes[main, 2])) / 2 / mh,
            "dy_bottom": (boxes[j, 3] - boxes[main, 3]) / mh, "v_overlap": vov, "x_overlap": ix / mh,
            "hand_dist": hand, "hip_dist": hip, "sharp_diff": sharp[j] - sharp[main],
            "p_j": probs[j], "p_main": probs[main], "conf_j": conf[j], "kp_vis_j": (kpts[j, :, 2] >= 0.5).sum(),
            "n_persons": len(boxes), "j_edge": float(boxes[j, 0] < 3 or boxes[j, 2] > W - 3),
        })
    return rows


def pairwise_partners(df, pt, probs, folds_photo):
    """Out-of-fold partner choice per photo: index of the partner (or None)."""
    sharp = pt.groupby("photo").sharp.apply(list).to_dict()
    labels = pt.groupby("photo").subject.apply(list).to_dict()
    rows, mains = [], {}
    for i, r in enumerate(df.itertuples()):
        boxes, _, _ = unpack(r.persons)
        main, _ = pick_couple(boxes, r.preview_w, r.preview_h, np.asarray(probs[i]))
        mains[i] = main
        for row in pair_rows(r.persons, r.preview_w, r.preview_h, main, np.asarray(sharp[i]), np.asarray(probs[i])):
            row.update(photo=i, label=bool(labels[i][row["j"]]))
            rows.append(row)
    pairs = pd.DataFrame(rows)
    cols = [c for c in pairs.columns if c not in ("j", "photo", "label")]
    pairs["p"] = np.nan
    for tr_ph, te_ph in folds_photo:
        tr, te = pairs.photo.isin(tr_ph), pairs.photo.isin(te_ph)
        clf = lgb.LGBMClassifier(**PAIR_PARAMS).fit(pairs.loc[tr, cols].astype(float), pairs.label[tr])
        pairs.loc[te, "p"] = clf.predict_proba(pairs.loc[te, cols].astype(float))[:, 1]
    partner = {}
    for i, g in pairs.groupby("photo"):
        best = g.loc[g.p.idxmax()]
        partner[i] = int(best.j) if best.p >= 0.5 else None
    return mains, partner, pairs


# --------------------------------------------------------------------------- crops
def crop_predictions(df, picks, emb, folds_photo):
    """Train the crop regressor in-fold on the couple boxes given by `picks`; predicted crops per photo."""
    rows, boxes = [], []
    for i, r in enumerate(df.itertuples()):
        box, f = photo_features(r.persons, r.preview_w, r.preview_h, np.asarray(r.probs), pick=picks[i])
        f.update(tilt_deg=r.tilt_deg, tilt_n=r.tilt_n, focal_length=r.focal_length)
        f.update(crop_to_targets(r.CropLeft, r.CropTop, r.CropRight, r.CropBottom, r.CropAngle,
                                 box, r.preview_w, r.preview_h))
        rows.append(f)
        boxes.append(box)
    X = pd.DataFrame(rows)
    feat_cols = [c for c in X.columns if c not in TARGETS]
    pred = pd.DataFrame(index=df.index, columns=TARGETS, dtype=float)
    for tr, te in folds_photo:
        E = PCA(crop_train.N_PCA, random_state=0).fit(emb[tr]).transform(emb)
        M = np.hstack([X[feat_cols].to_numpy(float), E])
        for t in TARGETS:
            pred.loc[te, t] = lgb.LGBMRegressor(**crop_train.PARAMS).fit(M[tr], X[t].to_numpy(float)[tr]).predict(M[te])
    crops = [targets_to_crop({**pred.loc[i].to_dict(), "t_angle": 0.0}, boxes[i], r.preview_w, r.preview_h)
             for i, r in enumerate(df.itertuples())]
    return crops, boxes, X.subject_p.to_numpy()


def protect(crop, couple_box, persons, probs, picked, tau, cap, W, H):
    """No-cut rule over the couple plus every other person with probability >= tau, within an area cap."""
    c0 = contain_subject(crop, couple_box, W, H)
    if tau is None:
        return c0
    boxes, _, _ = unpack(persons)
    extra = [i for i in range(len(boxes)) if i not in picked and probs[i] >= tau]
    if not extra:
        return c0
    allb = np.vstack([couple_box[None]] + [boxes[i][None] for i in extra])
    union = np.array([allb[:, 0].min(), allb[:, 1].min(), allb[:, 2].max(), allb[:, 3].max()])
    c1 = contain_subject(crop, union, W, H)
    area = lambda c: (c["CropRight"] - c["CropLeft"]) * (c["CropBottom"] - c["CropTop"])
    return c1 if area(c1) <= cap * area(c0) else c0


def evaluate(df, crops, labels, probs):
    rows = []
    for i, r in enumerate(df.itertuples()):
        truth = {"CropLeft": r.CropLeft, "CropTop": r.CropTop, "CropRight": r.CropRight, "CropBottom": r.CropBottom,
                 "CropAngle": 0.0}
        f = analyse(crops[i], truth, r.persons, probs[i], labels[i], r.preview_w, r.preview_h)
        f["cut_any"] = f["cut_head"] or f["cut_hands"] or f["cut_feet"]
        f["iou"] = crop_train.iou(crops[i], {**truth, "CropAngle": r.CropAngle}, r.preview_w, r.preview_h)
        ar = lambda c: (c["CropRight"] - c["CropLeft"]) * r.preview_w / ((c["CropBottom"] - c["CropTop"]) * r.preview_h)
        f["aspect_log_err"] = abs(np.log(ar(crops[i]) / ar(truth)))
        rows.append(f)
    return pd.DataFrame(rows)


SUMMARY = ["cut_any", "cut_head", "cut_hands", "cut_feet", "missed_partner", "other_people_cut",
           "wrong_subject", "too_tight", "too_loose", "iou", "aspect_log_err"]


def main(data_dir, boot):
    df = crop_train.load_photos(data_dir)
    print("person table + subject classifier (out-of-fold)...", flush=True)
    pt = crop_train.person_table(df, data_dir)
    sel_cols = [c for c in pt.columns if c not in ("subject", "photo", "folder")]
    pt["p"] = np.nan
    for tr, te in GroupKFold(5).split(pt, groups=pt.folder):
        clf = lgb.LGBMClassifier(**crop_train.SEL_PARAMS).fit(pt.loc[tr, sel_cols].astype(float), pt.subject[tr])
        pt.loc[te, "p"] = clf.predict_proba(pt.loc[te, sel_cols].astype(float))[:, 1]
    df["probs"] = pd.Series(pt.groupby("photo").p.apply(list))
    labels = pt.groupby("photo").subject.apply(list).to_dict()
    folds_photo = list(GroupKFold(5).split(df, groups=df.folder))
    fold_of = np.empty(len(df), int)
    for k, (_, te) in enumerate(folds_photo):
        fold_of[te] = k
    emb = embeddings(data_dir, "clip_b32", df.image_id)
    probs = df.probs.tolist()

    picks_a = {}
    for i, r in enumerate(df.itertuples()):
        boxes, _, _ = unpack(r.persons)
        picks_a[i] = pick_couple(boxes, r.preview_w, r.preview_h, np.asarray(probs[i]))
    print("pairwise partner classifier (out-of-fold)...", flush=True)
    mains, partner_b, pairs = pairwise_partners(df, pt, probs, folds_photo)
    picks_b = {i: (mains[i], partner_b.get(i)) for i in range(len(df))}

    print("crop regressors (in-fold) for A and B couples...", flush=True)
    base = {}
    for name, picks in (("A", picks_a), ("B", picks_b)):
        base[name] = crop_predictions(df, picks, emb, folds_photo)

    results, timing = {}, {}
    W, H = df.preview_w.to_numpy(), df.preview_h.to_numpy()
    for name, picks, tau, cap in ([("A", picks_a, None, None), ("B", picks_b, None, None)]
                                  + [(f"C tau={t} cap={c}", picks_a, t, c) for t in TAUS for c in CAPS]
                                  + [(f"D tau={t} cap={c}", picks_b, t, c) for t in TAUS for c in CAPS]):
        crops0, boxes, _ = base["B" if name.startswith(("B", "D")) else "A"]
        t0 = time.perf_counter()
        crops = [protect(crops0[i], boxes[i], df.persons[i], probs[i], {p for p in picks[i] if p is not None},
                         tau, cap, W[i], H[i]) for i in range(len(df))]
        timing[name] = 1000 * (time.perf_counter() - t0) / len(df)
        results[name] = evaluate(df, crops, labels, probs).assign(event=df.folder.to_numpy(), fold=fold_of)
        print(f"  {name}: cut_any {results[name].cut_any.mean():.3f} iou {results[name].iou.mean():.4f} "
              f"too_loose {results[name].too_loose.mean():.3f}", flush=True)

    # headline C / D: per test fold, pick (tau, cap) on the OTHER folds with the pre-declared rule
    def chosen(prefix, ref):
        parts, picked = [], {}
        for k in range(5):
            other = lambda n: results[n][results[n].fold != k]
            ok = [n for n in results if n.startswith(prefix + " ")
                  and other(n).iou.mean() >= other(ref).iou.mean() - 0.01
                  and other(n).too_loose.mean() <= other(ref).too_loose.mean() + 0.02]
            best = min(ok, key=lambda n: other(n).cut_any.mean()) if ok else ref
            picked[k] = best
            parts.append(results[best][results[best].fold == k])
        return pd.concat(parts).sort_index(), picked

    results["C (chosen out-of-fold)"], pick_c = chosen("C", "A")
    results["D (chosen out-of-fold)"], pick_d = chosen("D", "B")

    summary = {n: {m: float(r[m].mean()) for m in SUMMARY} for n, r in results.items()}
    paired = {}
    ref_ev = results["A"].groupby("event")
    for n in ["B", "C (chosen out-of-fold)", "D (chosen out-of-fold)"]:
        ev = results[n].groupby("event")
        paired[n] = {}
        for m, better_if_lower in [("cut_any", True), ("iou", False), ("too_loose", True), ("too_tight", True),
                                   ("missed_partner", True), ("other_people_cut", True)]:
            d = (ref_ev[m].mean() - ev[m].mean()) if better_if_lower else (ev[m].mean() - ref_ev[m].mean())
            mean, lo, hi = boot_ci(d.to_numpy(), boot)
            paired[n][m] = {"improvement": mean, "ci95": [lo, hi], "events_better": int((d > 0).sum()),
                            "events_worse": int((d < 0).sum())}
    out = {"photos": len(df), "events": int(df.folder.nunique()), "pairs": len(pairs),
           "pair_label_rate": float(pairs.label.mean()), "summary": summary, "paired_vs_A": paired,
           "chosen_settings_C": pick_c, "chosen_settings_D": pick_d, "ms_per_photo_protection_step": timing,
           "network_attempts": offline.attempts}
    json.dump(out, open(os.path.join(data_dir, "partner_experiment.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("chosen_settings_C", "chosen_settings_D", "paired_vs_A")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    main(a.data_dir, a.boot)
