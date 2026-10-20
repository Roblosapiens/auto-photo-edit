"""Crop failure analysis beyond IoU: subject choice, truncated limbs, margins and composition.

Used by tools/compare_embeddings.py. Every check works on numbers only (pose keypoints, person boxes,
crop rectangles) - no images. Crops are treated as axis-aligned (L, T, R, B); the small straightening
angles are ignored here.

Per photo, for a predicted crop vs. the photographer's crop:
  wrong_subject     predicted main dancer is not one of the photographer's subjects
  missed_partner    a subject the photographer kept (similar height, beside the main dancer) is mostly
                    outside the predicted crop
  cut_head / cut_hands / cut_feet
                    a visible head, wrist or ankle keypoint of the photographer's couple (the one or two
                    largest people they kept in frame) lies outside the crop; also computed for the
                    photographer's own crop as a reference rate (ref_*)
  other_people_cut  another person the photographer kept in frame is sliced by the crop (10-90% inside)
  tight_head / tight_feet
                    under 3% of couple height between the couple's highest head point / lowest ankle (+5%
                    for shoes) and the crop edge, while still inside
  orientation       portrait / square / landscape class differs from the photographer's
  orientation_flip  one is portrait (aspect < 0.9) and the other landscape (> 1.1)
  too_tight / too_loose
                    crop area under 2/3 or over 1.5x the photographer's
  off_centre        couple's horizontal position within the crop differs by over 15% of crop width
"""
import numpy as np

from crop_model import pick_couple, unpack

HEAD = [0, 1, 2, 3, 4]
WRISTS = [9, 10]
ANKLES = [15, 16]
VIS = 0.5  # keypoint confidence considered visible


def _inside(pt, c, W, H):
    return c["CropLeft"] * W <= pt[0] <= c["CropRight"] * W and c["CropTop"] * H <= pt[1] <= c["CropBottom"] * H


def _shape(c, W, H):
    ar = (c["CropRight"] - c["CropLeft"]) * W / max((c["CropBottom"] - c["CropTop"]) * H, 1e-6)
    return "portrait" if ar < 0.9 else "square" if ar <= 1.1 else "landscape"


def analyse(pred, truth, persons, probs, labels, W, H):
    """pred / truth: crop dicts; persons: flat pose list; probs: subject-classifier probabilities;
    labels: per person, True when the photographer kept them (their subjects). Returns {check: bool}."""
    boxes, _, kpts = unpack(persons)
    main, partner = pick_couple(boxes, W, H, np.asarray(probs))
    subjects = [i for i, l in enumerate(labels) if l]
    out = {"wrong_subject": not labels[main]}

    mh = boxes[main, 3] - boxes[main, 1]
    missed = False
    for i in subjects:
        if i == main or boxes[i, 3] - boxes[i, 1] < 0.6 * mh:
            continue
        gap = max(0.0, max(boxes[i, 0], boxes[main, 0]) - min(boxes[i, 2], boxes[main, 2]))
        if gap > 0.6 * mh:
            continue
        L, T, R, B = pred["CropLeft"] * W, pred["CropTop"] * H, pred["CropRight"] * W, pred["CropBottom"] * H
        ix = max(0, min(boxes[i, 2], R) - max(boxes[i, 0], L))
        iy = max(0, min(boxes[i, 3], B) - max(boxes[i, 1], T))
        if ix * iy / ((boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])) < 0.5:
            missed = True
    out["missed_partner"] = missed

    # the photographer's couple: the one or two largest people they kept (independent of the model's pick)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    couple = sorted(subjects, key=lambda i: -area[i])[:2] or [main]
    if len(couple) == 2 and area[couple[1]] < 0.3 * area[couple[0]]:
        couple = couple[:1]

    def cuts(c, idx):
        pts = [kpts[s, j] for s in couple for j in idx if kpts[s, j, 2] >= VIS]
        return any(not _inside(p, c, W, H) for p in pts)

    def frac_inside(i, c):
        L, T, R, B = c["CropLeft"] * W, c["CropTop"] * H, c["CropRight"] * W, c["CropBottom"] * H
        ix = max(0, min(boxes[i, 2], R) - max(boxes[i, 0], L))
        iy = max(0, min(boxes[i, 3], B) - max(boxes[i, 1], T))
        return ix * iy / area[i]

    sub = couple
    top = min(boxes[i, 1] for i in sub)
    height = max(boxes[i, 3] for i in sub) - top
    heads = [kpts[s, j] for s in couple for j in HEAD if kpts[s, j, 2] >= VIS]
    ankles = [kpts[s, j] for s in couple for j in ANKLES if kpts[s, j, 2] >= VIS]
    out["other_people_cut"] = any(0.1 < frac_inside(i, pred) < 0.9 for i in subjects if i not in couple)
    for prefix, c in (("", pred), ("ref_", truth)):
        out[prefix + "cut_head"] = cuts(c, HEAD)
        out[prefix + "cut_hands"] = cuts(c, WRISTS)
        out[prefix + "cut_feet"] = cuts(c, ANKLES)
        head_y = min((p[1] for p in heads), default=None)
        foot_y = max((p[1] for p in ankles), default=None)
        out[prefix + "tight_head"] = head_y is not None and 0 <= head_y - c["CropTop"] * H < 0.03 * height
        out[prefix + "tight_feet"] = (foot_y is not None
                                      and 0 <= c["CropBottom"] * H - (foot_y + 0.05 * height) < 0.03 * height)

    out["orientation"] = _shape(pred, W, H) != _shape(truth, W, H)
    out["orientation_flip"] = {_shape(pred, W, H), _shape(truth, W, H)} == {"portrait", "landscape"}
    area = lambda c: (c["CropRight"] - c["CropLeft"]) * (c["CropBottom"] - c["CropTop"])
    ratio = area(pred) / max(area(truth), 1e-6)
    out["too_tight"], out["too_loose"] = ratio < 2 / 3, ratio > 1.5
    cx = (min(boxes[i, 0] for i in sub) + max(boxes[i, 2] for i in sub)) / 2 / W
    pos = lambda c: (cx - c["CropLeft"]) / max(c["CropRight"] - c["CropLeft"], 1e-6)
    out["off_centre"] = abs(pos(pred) - pos(truth)) > 0.15
    return out
