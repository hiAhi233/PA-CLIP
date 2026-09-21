"""Data loading: read ThymomaCT jsonl, split by case, build feature caches."""
import json
import os
from pathlib import Path
import torch
from PIL import Image


def load_entries(meta_path):
    """Read full-shot.jsonl -> list of dicts with absolute image/mask paths."""
    entries = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            entries.append(e)
    return entries


def case_id(entry):
    """BMAD Brain 文件名 `{case}_{slice}.png` 的病例号。"""
    return Path(entry["image_path"]).name.split("_")[0]


def split_by_case(entries, val_cases, test_cases, brain_split=None):
    """Split entries into (prototype-pool, val, test).

    If entries carry an explicit 'split' field (train/proto_anom/valid/test),
    use it directly; otherwise fall back to patient-case matching ('003_65').

    brain_split='support44': 官方 valid 的 44 张异常全部进 pool(支持集),
    val 只留 39 张 valid/good。超参改由症例 CV 在支持集上选,不再用 22/22 切片拆分。
    """
    if entries and "split" in entries[0]:
        if brain_split == "support44":
            train = [e for e in entries if e["split"] == "train"]
            support_anom = [e for e in entries
                            if e["split"] in ("proto_anom", "valid") and float(e["label"]) > 0]
            val = [e for e in entries if e["split"] == "valid" and float(e["label"]) == 0]
            test = [e for e in entries if e["split"] == "test"]
            return train + support_anom, val, test
        proto = [e for e in entries if e["split"] in ("train", "proto_anom")]
        val = [e for e in entries if e["split"] == "valid"]
        test = [e for e in entries if e["split"] == "test"]
        return proto, val, test

    val, test, proto = [], [], []
    for e in entries:
        case = os.path.basename(e["image_path"]).split("_")[0] + "_" + os.path.basename(e["image_path"]).split("_")[1]
        if case in val_cases:
            val.append(e)
        elif case in test_cases:
            test.append(e)
        else:
            proto.append(e)
    return proto, val, test


def select_few_shot(entries, n_normal, n_anomaly, seed=111):
    """Randomly pick n_normal normal + n_anomaly anomaly entries (for prototype building)."""
    import random
    random.seed(seed)
    normals = [e for e in entries if e["label"] == 0]
    anoms = [e for e in entries if e["label"] == 1]
    picked = random.sample(normals, min(n_normal, len(normals)))
    picked += random.sample(anoms, min(n_anomaly, len(anoms)))
    return picked


def img_paths(entries, data_root):
    return [os.path.join(data_root, e["image_path"]) for e in entries]


def load_image(path):
    return Image.open(path).convert("RGB")


def mask_grid_14(mask_path, image_size=224):
    """ROI mask -> 14x14 binary grid aligned with ViT-B/16 patches (224/16=14)."""
    from PIL import Image
    import numpy as np
    m = np.array(Image.open(mask_path).convert("L").resize((14, 14), Image.NEAREST))
    return (m > 0).astype(np.float32)
