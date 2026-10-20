"""Compute image embeddings of every preview with several backbones, with timing and GPU memory.

Usage: python tools/embed_models.py <data_dir> [model ...]     (models: clip_b32 clip_l14 dinov2_b14)

Writes <data_dir>/emb_<model>.npy (+ emb_<model>_ids.npy) and appends a line per model to
<data_dir>/embed_benchmark.jsonl. Runs offline (models must be cached: tools/update_models.py).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import offline  # noqa: E402

offline.enable()
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

BATCH = 32
MODELS = {
    "clip_b32": ("open_clip", "ViT-B-32", "laion2b_s34b_b79k"),
    "clip_l14": ("open_clip", "ViT-L-14", "laion2b_s32b_b82k"),
    "dinov2_b14": ("timm", "vit_base_patch14_dinov2.lvd142m", None),
}


def load(name):
    kind, arch, tag = MODELS[name]
    if kind == "open_clip":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=tag)
        return model, preprocess, model.encode_image
    import timm
    model = timm.create_model(arch, pretrained=True, num_classes=0)  # pooled (CLS) features
    cfg = timm.data.resolve_data_config({}, model=model)
    return model, timm.data.create_transform(**cfg), model


def main(data_dir, names):
    prev = pd.read_parquet(os.path.join(data_dir, "previews.parquet"))
    ids = prev[prev.ok].image_id.to_numpy()
    paths = [os.path.join(data_dir, "previews", f"{i}.jpg") for i in ids]
    dev = "cuda"
    for name in names:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        model, preprocess, forward = load(name)
        model = model.eval().to(dev)
        load_s = time.perf_counter() - t0
        out, gpu_s = [], 0.0
        t_total = time.perf_counter()
        with torch.no_grad(), ThreadPoolExecutor(8) as pool:
            prep = lambda p: preprocess(Image.open(p).convert("RGB"))
            for i in range(0, len(paths), BATCH):
                x = torch.stack(list(pool.map(prep, paths[i:i + BATCH]))).to(dev)
                torch.cuda.synchronize()
                g0 = time.perf_counter()
                e = forward(x)
                torch.cuda.synchronize()
                gpu_s += time.perf_counter() - g0
                out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
        total_s = time.perf_counter() - t_total
        emb = np.concatenate(out)
        np.save(os.path.join(data_dir, f"emb_{name}.npy"), emb)
        np.save(os.path.join(data_dir, f"emb_{name}_ids.npy"), ids)
        rec = {"model": name, "dim": int(emb.shape[1]), "images": len(ids),
               "input_px": int(x.shape[-1]), "load_s": round(load_s, 1),
               "gpu_ms_per_image": round(1000 * gpu_s / len(ids), 2),
               "total_ms_per_image": round(1000 * total_s / len(ids), 2),
               "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20),
               "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1)}
        with open(os.path.join(data_dir, "embed_benchmark.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(rec, flush=True)
        del model
    print("offline attempts:", offline.attempts)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or list(MODELS))
