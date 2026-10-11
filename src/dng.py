"""Minimal DNG (TIFF) reader: largest embedded JPEG preview + AsShotNeutral, reading only those bytes.

rawpy reads the whole ~20 MB file; this seeks straight to the tags and the ~150 KB preview.
"""
import struct

TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4}
TYPE_FMT = {1: "B", 3: "H", 4: "I", 5: "II", 6: "b", 8: "h", 9: "i", 10: "ii", 13: "I"}


def _read_ifd(f, offset, bo):
    f.seek(offset)
    (n,) = struct.unpack(bo + "H", f.read(2))
    raw = f.read(12 * n)
    (next_ifd,) = struct.unpack(bo + "I", f.read(4))
    tags = {}
    for i in range(n):
        tag, typ, count = struct.unpack(bo + "HHI", raw[12 * i:12 * i + 8])
        if typ not in TYPE_FMT:
            continue
        size = TYPE_SIZE[typ] * count
        if size <= 4:
            data = raw[12 * i + 8:12 * i + 8 + size]
        else:
            (ptr,) = struct.unpack(bo + "I", raw[12 * i + 8:12 * i + 12])
            if size > 4096:  # skip big blobs (maker notes etc.)
                continue
            pos = f.tell()
            f.seek(ptr)
            data = f.read(size)
            f.seek(pos)
        vals = struct.unpack(bo + TYPE_FMT[typ] * count, data)
        if typ in (5, 10):
            vals = tuple(vals[j] / vals[j + 1] if vals[j + 1] else 0.0 for j in range(0, len(vals), 2))
        tags[tag] = vals
    return tags, next_ifd


def read_dng(path, min_width=1024):
    """Return (jpeg_bytes or None, as_shot_neutral tuple or None).

    Picks the smallest embedded JPEG preview at least `min_width` wide (else the largest one)."""
    with open(path, "rb") as f:
        bo = {b"II": "<", b"MM": ">"}[f.read(2)]
        _, ifd0_off = struct.unpack(bo + "HI", f.read(6))
        ifd0, _ = _read_ifd(f, ifd0_off, bo)
        ifds = [ifd0] + [_read_ifd(f, off, bo)[0] for off in ifd0.get(330, ())]
        best = None
        for t in ifds:
            if t.get(259, (0,))[0] not in (6, 7):  # JPEG compression only
                continue
            if 273 in t and len(t[273]) == 1:
                off, size = t[273][0], t[279][0]
            elif 513 in t:
                off, size = t[513][0], t[514][0]
            else:
                continue
            width = t.get(256, (0,))[0]
            if t.get(254, (1,))[0] != 1:
                continue
            # rank: big-enough previews first, then the smallest of those; otherwise the largest
            rank = (width >= min_width, -width if width >= min_width else width)
            if best is None or rank > best[0]:
                best = (rank, off, size)
        jpeg = None
        if best:
            f.seek(best[1])
            jpeg = f.read(best[2])
        return jpeg, ifd0.get(50728)
