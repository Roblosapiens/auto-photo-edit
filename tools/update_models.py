"""Explicitly download or refresh the pretrained models. The ONLY part of the project that uses the network.

Usage: python tools/update_models.py [--clip ViT-B-32:laion2b_s34b_b79k ...] [--yolo yolo11m-pose.pt] [--geocalib]
       python tools/update_models.py --all

Everything else runs offline (src/offline.py). This script sets AUTOPHOTOEDIT_ALLOW_NETWORK=1 for itself
only and downloads weights into the usual local caches:
  CLIP     -> %USERPROFILE%\\.cache\\huggingface\\hub   (open_clip / Hugging Face Hub)
  YOLO     -> models\\                                 (Ultralytics GitHub release assets)
  GeoCalib -> %USERPROFILE%\\.cache\\torch\\hub\\geocalib (GitHub release)
No photos or other user data are sent; these are plain downloads.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ["AUTOPHOTOEDIT_ALLOW_NETWORK"] = "1"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

PRODUCTION_CLIP = "ViT-B-32:laion2b_s34b_b79k"


def update_clip(spec):
    import open_clip
    arch, tag = spec.split(":")
    open_clip.create_model_and_transforms(arch, pretrained=tag)
    print(f"CLIP {arch} ({tag}): cached")


def update_yolo(name):
    from ultralytics.utils.downloads import attempt_download_asset
    dest = ROOT / "models" / name
    dest.parent.mkdir(exist_ok=True)
    cwd = os.getcwd()
    os.chdir(dest.parent)  # attempt_download_asset saves into the working directory
    try:
        if dest.exists():
            dest.unlink()  # force a fresh copy
        attempt_download_asset(name)
    finally:
        os.chdir(cwd)
    print(f"YOLO {name}: {dest} ({dest.stat().st_size / 1e6:.0f} MB)")


def update_geocalib():
    import torch
    from geocalib import GeoCalib
    cached = Path(torch.hub.get_dir()) / "geocalib" / "pinhole.tar"
    if cached.exists():
        cached.unlink()  # force a fresh copy
    GeoCalib(weights="pinhole")
    print(f"GeoCalib pinhole: {cached}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", nargs="*", help="open_clip 'arch:pretrained' specs")
    ap.add_argument("--yolo", help="Ultralytics asset name, e.g. yolo11m-pose.pt")
    ap.add_argument("--geocalib", action="store_true")
    ap.add_argument("--all", action="store_true", help="the production models")
    a = ap.parse_args()
    for spec in (a.clip or []) + ([PRODUCTION_CLIP] if a.all else []):
        update_clip(spec)
    if a.yolo or a.all:
        update_yolo(a.yolo or "yolo11m-pose.pt")
    if a.geocalib or a.all:
        update_geocalib()
