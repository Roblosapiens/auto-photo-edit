"""Extract the embedded "before" preview and as-shot white balance from each raw/DNG file.

Usage: python src/previews.py <edits.parquet> <out_dir>

Writes <out_dir>/previews/<image_id>.jpg (the untouched converter preview, max 1024 px) and
<out_dir>/previews.parquet with the camera's as-shot WB multipliers. Resumable: existing previews
are skipped (their WB is re-read, which is cheap).
"""
import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import rawpy
from PIL import Image

from dng import read_dng

MAX_SIDE = 1024


def load_preview(path):
    """Return (PIL image <= MAX_SIDE, wb_r, wb_b) for a DNG or other raw file, without writing anything."""
    if path.lower().endswith(".dng"):
        jpeg, neutral = read_dng(path)
        if jpeg and neutral:
            im = Image.open(io.BytesIO(jpeg))
            im.thumbnail((MAX_SIDE, MAX_SIDE))
            return im, neutral[1] / neutral[0], neutral[1] / neutral[2]
    with rawpy.imread(path) as raw:
        wb = raw.camera_whitebalance
        thumb = raw.extract_thumb()
    im = Image.open(io.BytesIO(thumb.data)) if thumb.format == rawpy.ThumbFormat.JPEG else Image.fromarray(thumb.data)
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    return im, wb[0] / wb[1], wb[2] / wb[1]


def process(row, prev_dir):
    image_id, path = row
    out = os.path.join(prev_dir, f"{image_id}.jpg")
    rec = {"image_id": image_id, "ok": False, "error": None}
    if not os.path.isfile(path):
        rec["error"] = "missing"
        return rec
    try:
        if path.lower().endswith(".dng"):
            jpeg, neutral = read_dng(path)
            if jpeg and neutral:
                rec.update(wb_r=neutral[1] / neutral[0], wb_b=neutral[1] / neutral[2])
                if not os.path.isfile(out):
                    im = Image.open(io.BytesIO(jpeg))
                    if max(im.size) > MAX_SIDE:
                        im = im.convert("RGB")
                        im.thumbnail((MAX_SIDE, MAX_SIDE))
                        im.save(out, quality=92)
                    else:  # keep the converter's JPEG bytes untouched
                        with open(out, "wb") as f:
                            f.write(jpeg)
                with Image.open(out) as im:
                    im.load()  # verify it decodes
                    rec["preview_w"], rec["preview_h"] = im.size
                rec["ok"] = True
                return rec
        # .ARW, or a DNG without an embedded JPEG preview: let LibRaw read the whole file
        with rawpy.imread(path) as raw:
            wb = raw.camera_whitebalance
            rec.update(wb_r=wb[0] / wb[1], wb_b=wb[2] / wb[1])
            if not os.path.isfile(out):
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    im = Image.open(io.BytesIO(thumb.data))
                else:
                    im = Image.fromarray(thumb.data)
                if max(im.size) > MAX_SIDE or thumb.format != rawpy.ThumbFormat.JPEG:
                    im = im.convert("RGB")
                    im.thumbnail((MAX_SIDE, MAX_SIDE))
                    im.save(out, quality=92)
                else:  # keep the converter's JPEG bytes untouched
                    with open(out, "wb") as f:
                        f.write(thumb.data)
        with Image.open(out) as im:
            rec["preview_w"], rec["preview_h"] = im.size
        rec["ok"] = True
    except Exception as e:  # corrupt / unsupported file: record and move on
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def main(edits, out_dir):
    df = pd.read_parquet(edits)
    df = df[df.fileFormat.isin(["DNG", "RAW"])]
    prev_dir = os.path.join(out_dir, "previews")
    os.makedirs(prev_dir, exist_ok=True)
    rows = list(zip(df.image_id, df.folder + df.filename))
    recs = []
    with ThreadPoolExecutor(4) as pool:
        for i, rec in enumerate(pool.map(lambda r: process(r, prev_dir), rows), 1):
            recs.append(rec)
            if i % 500 == 0:
                print(f"{i}/{len(rows)}  ok={sum(r['ok'] for r in recs)}", flush=True)
    res = pd.DataFrame(recs)
    res.to_parquet(os.path.join(out_dir, "previews.parquet"), index=False)
    print(f"done: {res.ok.sum()} ok, {(~res.ok).sum()} failed")
    print(res.error.str.split(":").str[0].value_counts().to_string())


if __name__ == "__main__":
    main(*sys.argv[1:3])
