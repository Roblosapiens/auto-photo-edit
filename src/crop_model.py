"""Shared crop-model code: pick the couple, build features, convert crops to/from couple-relative targets.

Coordinates are preview pixels (sensor orientation, as stored by Lightroom). A Lightroom crop is
(CropLeft, CropTop, CropRight, CropBottom) normalised to the image plus CropAngle in degrees; it is the
axis-aligned box (L,T,R,B) rotated clockwise by CropAngle about its centre (verified visually).
"""
import numpy as np
import pandas as pd

KP = 17  # COCO keypoints
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH = 0, 1, 2, 3, 4, 5, 6
L_HIP, R_HIP, L_ANK, R_ANK = 11, 12, 15, 16
TARGETS = ["t_dx", "t_dy", "t_logw", "t_logh", "t_angle"]
PER = 5 + 3 * KP


def unpack(persons):
    """Flat list -> (boxes [n,4], conf [n], kpts [n,17,3])."""
    a = np.asarray(persons, dtype=float).reshape(-1, PER)
    return a[:, :4], a[:, 4], a[:, 5:].reshape(-1, KP, 3)


SUBJECT_P = 0.5  # a person counts as part of the subject couple above this classifier probability


def sharpness(gray, box):
    """log variance of the Laplacian inside a box minus that of the whole image (blurry foreground < 0)."""
    import cv2

    x0, y0, x1, y1 = [int(round(v)) for v in box]
    patch = gray[max(y0, 0):max(y1, y0 + 2), max(x0, 0):max(x1, x0 + 2)]
    lap = lambda a: np.log(cv2.Laplacian(a, cv2.CV_32F).var() + 1e-3)
    return float(lap(patch) - lap(gray)) if patch.size > 16 else np.nan


def person_features(persons, W, H, gray=None):
    """One row per detected person, used by the subject classifier."""
    boxes, conf, kpts = unpack(persons)
    n = len(boxes)
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    area = w * h
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2 / W, (boxes[:, 1] + boxes[:, 3]) / 2 / H
    vis = kpts[:, :, 2] >= 0.5
    rows = []
    for i in range(n):
        others = [j for j in range(n) if j != i]
        gaps = [max(0.0, max(boxes[j, 0], boxes[i, 0]) - min(boxes[j, 2], boxes[i, 2])) / h[i] for j in others]
        rows.append({
            "cx": cx[i], "cy": cy[i], "w": w[i] / W, "h": h[i] / H, "x0": boxes[i, 0] / W, "x1": boxes[i, 2] / W,
            "y0": boxes[i, 1] / H, "y1": boxes[i, 3] / H,
            "area_rel": area[i] / area.max(), "h_rel": h[i] / h.max(), "rank": int((area > area[i]).sum()),
            "n_persons": n, "centre_dist": np.hypot(cx[i] - 0.5, cy[i] - 0.5),
            "edge_l": boxes[i, 0] < 3, "edge_r": boxes[i, 2] > W - 3, "edge_t": boxes[i, 1] < 3, "edge_b": boxes[i, 3] > H - 3,
            "conf": conf[i], "kp_conf": kpts[i, :, 2].mean(), "kp_vis": vis[i].sum(),
            "face_vis": vis[i, :5].sum(), "ankles_vis": vis[i, 15:].sum(),
            "nearest_gap": min(gaps) if gaps else np.nan,
            "sharp": sharpness(gray, boxes[i]) if gray is not None else np.nan,
        })
    return pd.DataFrame(rows)


def pick_couple(boxes, W, H, probs=None):
    """Indices of the main dancer and partner (partner may be None).

    With `probs` (subject-classifier probabilities per person) the most likely subject is the main
    dancer; otherwise a size x centrality heuristic is used."""
    if len(boxes) == 0:
        return None, None
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    if probs is not None:
        score = np.asarray(probs, float)
    else:
        cx = (boxes[:, 0] + boxes[:, 2]) / 2 / W
        cy = (boxes[:, 1] + boxes[:, 3]) / 2 / H
        score = area * (0.3 + 1 - np.clip(np.hypot(cx - 0.5, (cy - 0.5) * 0.6) / 0.7, 0, 1))
    main = int(np.argmax(score))
    mh = boxes[main, 3] - boxes[main, 1]
    best, best_score = None, 0.0
    for j in range(len(boxes)):
        if j == main or area[j] < 0.2 * area[main]:
            continue
        if probs is not None and probs[j] < SUBJECT_P:
            continue
        gap = max(0.0, max(boxes[j, 0], boxes[main, 0]) - min(boxes[j, 2], boxes[main, 2]))
        if gap > 0.6 * mh:
            continue
        s = (probs[j] if probs is not None else area[j] / area[main]) * (1 - gap / (0.6 * mh + 1e-6))
        if s > best_score:
            best, best_score = j, s
    return main, best


def _lean(k):
    """Body-axis angle (deg, 0 = upright, + = top leaning right) from shoulder and hip midpoints."""
    sh, hp = k[[L_SH, R_SH]], k[[L_HIP, R_HIP]]
    if min(sh[:, 2].min(), hp[:, 2].min()) < 0.3:
        return np.nan
    (sx, sy), (hx, hy) = sh[:, :2].mean(0), hp[:, :2].mean(0)
    return np.degrees(np.arctan2(sx - hx, hy - sy))


def photo_features(persons, W, H, probs=None):
    """Couple box + feature dict for one photo, or (None, None) when no person is found.

    f["subject_p"] is the lowest subject probability among the couple (NaN without probs)."""
    boxes, conf, kpts = unpack(persons)
    main, partner = pick_couple(boxes, W, H, probs)
    if main is None:
        return None, None
    idx = [main] + ([partner] if partner is not None else [])
    box = np.array([boxes[idx, 0].min(), boxes[idx, 1].min(), boxes[idx, 2].max(), boxes[idx, 3].max()])
    bw, bh = box[2] - box[0], box[3] - box[1]
    f = {
        "subject_p": min(probs[i] for i in idx) if probs is not None else np.nan,
        "b_x0": box[0] / W, "b_y0": box[1] / H, "b_x1": box[2] / W, "b_y1": box[3] / H,
        "b_w": bw / W, "b_h": bh / H, "b_aspect": bw / bh, "b_cx": (box[0] + box[2]) / 2 / W,
        "room_left": box[0] / bh, "room_right": (W - box[2]) / bh,
        "room_top": box[1] / bh, "room_bottom": (H - box[3]) / bh,
        "n_couple": len(idx), "n_persons": len(boxes), "conf_main": conf[main],
        "conf_partner": conf[partner] if partner is not None else np.nan,
    }
    others = [i for i in range(len(boxes)) if i not in idx]
    if others:
        oa = (boxes[others, 2] - boxes[others, 0]) * (boxes[others, 3] - boxes[others, 1])
        f["other_max_area"] = oa.max() / (bw * bh)
        ocx = (boxes[others, 0] + boxes[others, 2]) / 2
        f["other_near_dx"] = np.min(np.abs(ocx - (box[0] + box[2]) / 2)) / bh
    # keypoints of the couple, left person first, relative to the couple box (units of box height)
    order = sorted(idx, key=lambda i: boxes[i, 0] + boxes[i, 2])
    for p, i in enumerate(order[:2]):
        k = kpts[i]
        vis = k[:, 2] >= 0.3
        for j in range(KP):
            f[f"p{p}_k{j}_x"] = (k[j, 0] - box[0]) / bh if vis[j] else np.nan
            f[f"p{p}_k{j}_y"] = (k[j, 1] - box[1]) / bh if vis[j] else np.nan
        head = k[[NOSE, L_EYE, R_EYE, L_EAR, R_EAR]]
        head = head[head[:, 2] >= 0.3]
        f[f"p{p}_head_y"] = (head[:, 1].min() - box[1]) / bh if len(head) else np.nan
        ank = k[[L_ANK, R_ANK]]
        ank = ank[ank[:, 2] >= 0.3]
        f[f"p{p}_ankle_y"] = (ank[:, 1].max() - box[1]) / bh if len(ank) else np.nan
        f[f"p{p}_lean"] = _lean(k)
        f[f"p{p}_h"] = (boxes[i, 3] - boxes[i, 1]) / bh
    return box, f


def crop_to_targets(L, T, R, B, angle, box, W, H):
    bh = box[3] - box[1]
    cx, cy = (L + R) / 2 * W, (T + B) / 2 * H
    w, h = (R - L) * W, (B - T) * H
    return {"t_dx": (cx - (box[0] + box[2]) / 2) / bh, "t_dy": (cy - (box[1] + box[3]) / 2) / bh,
            "t_logw": np.log(w / bh), "t_logh": np.log(h / bh), "t_angle": angle}


def targets_to_crop(t, box, W, H):
    """Couple-relative targets -> Lightroom crop dict, shrunk/shifted to stay inside the image."""
    bh = box[3] - box[1]
    cx = (box[0] + box[2]) / 2 + t["t_dx"] * bh
    cy = (box[1] + box[3]) / 2 + t["t_dy"] * bh
    w, h = np.exp(t["t_logw"]) * bh, np.exp(t["t_logh"]) * bh
    a = np.radians(abs(t["t_angle"]))
    # bounding box of the rotated crop must fit in the image: scale down if needed, keep aspect
    bw, bhh = w * np.cos(a) + h * np.sin(a), w * np.sin(a) + h * np.cos(a)
    s = min(1.0, W / bw, H / bhh)
    w, h, bw, bhh = w * s, h * s, bw * s, bhh * s
    cx = np.clip(cx, bw / 2, W - bw / 2)
    cy = np.clip(cy, bhh / 2, H - bhh / 2)
    return {"CropLeft": (cx - w / 2) / W, "CropRight": (cx + w / 2) / W,
            "CropTop": (cy - h / 2) / H, "CropBottom": (cy + h / 2) / H, "CropAngle": float(t["t_angle"])}


# Minimum space kept around the couple box, in units of couple height. Roughly the 10th percentile of
# the margins in the photographer's own crops (they rarely cut into the couple).
CONTAIN_MARGINS = {"top": 0.03, "bottom": 0.01, "left": 0.05, "right": 0.05}


def contain_subject(c, box, W, H, margins=CONTAIN_MARGINS):
    """Push out any edge of an unrotated crop that cuts into the couple box (+ margin), within the image."""
    bh = box[3] - box[1]
    c = dict(c)
    c["CropLeft"] = max(0.0, min(c["CropLeft"], (box[0] - margins["left"] * bh) / W))
    c["CropRight"] = min(1.0, max(c["CropRight"], (box[2] + margins["right"] * bh) / W))
    c["CropTop"] = max(0.0, min(c["CropTop"], (box[1] - margins["top"] * bh) / H))
    c["CropBottom"] = min(1.0, max(c["CropBottom"], (box[3] + margins["bottom"] * bh) / H))
    return c


def cuts_subject(c, box, W, H, tol=0.02):
    """True if an unrotated crop cuts more than tol x couple height into the couple box (image edges excused)."""
    bh, t = box[3] - box[1], tol * (box[3] - box[1])
    return bool((box[0] - c["CropLeft"] * W < -t and box[0] > 3) or (c["CropRight"] * W - box[2] < -t and box[2] < W - 3)
                or (box[1] - c["CropTop"] * H < -t and box[1] > 3) or (c["CropBottom"] * H - box[3] < -t and box[3] < H - 3))


def crop_polygon(c, W, H):
    """Corners (preview pixels) of a Lightroom crop dict."""
    cx, cy = (c["CropLeft"] + c["CropRight"]) / 2 * W, (c["CropTop"] + c["CropBottom"]) / 2 * H
    w, h = (c["CropRight"] - c["CropLeft"]) * W, (c["CropBottom"] - c["CropTop"]) * H
    a = np.radians(c["CropAngle"])
    return [(cx + x * np.cos(a) - y * np.sin(a), cy + x * np.sin(a) + y * np.cos(a))
            for x, y in [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]]
