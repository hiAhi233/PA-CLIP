"""特征预计算与缓存。

为什么需要:PA-CLIP 的训练只用 (D,N) 的小矩阵,梯度不流经编码器。
所以编码是一次性成本,付一次就能反复训练。

为什么不直接调 pa_clip.model.encode_patch_features 存全量:
  它把所有结果留在内存里再 cat。pool 的 4 层 = 9139×196×512×2B ≈ 3.66 GiB/层,
  4 层 ≈ 14.6 GiB 峰值,在 23.6 GB 的机器上有 OOM 风险。
  这里改为分块编码 + 逐层落盘,峰值降到单块的 1/18。

磁盘布局(相对 cache_dir/<fingerprint>/):
  manifest.json          元信息 + 指纹
  <split>/label.pt       (N,)    int64
  <split>/has_mask.pt    (N,)    bool  —— meta 里是否有 mask_path
  <split>/mask14.pt      (N,196) bool  —— 14x14 网格;无 mask 的样本为全零
  <split>/cls.pt         (N,512) fp16
  <split>/patch_l{3,6,9,11}.pt  (N,196,512) fp16
"""
import hashlib
import pathlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from paclipf import data as pdata
from paclipf import model as biomed

CACHE_VERSION = 2
CHUNK = 512


# --------------------------------------------------------------------------- #
# 指纹:任何影响特征数值的东西变化 → 缓存失效
# --------------------------------------------------------------------------- #
def fingerprint(cfg):
    h = hashlib.sha1()
    h.update(f"v{CACHE_VERSION}".encode())
    h.update(str(cfg.image_size).encode())
    h.update(str(sorted(cfg.patch_layers)).encode())
    h.update(str(sorted(cfg.val_cases)).encode())
    h.update(str(sorted(cfg.test_cases)).encode())
    h.update(str(getattr(cfg, "brain_split", "") or "").encode())

    # 元数据内容
    with open(cfg.meta_path, "rb") as f:
        h.update(hashlib.sha1(f.read()).hexdigest().encode())

    # 特征提取源码(改这四个模块必须让缓存失效)
    pkg_dir = pathlib.Path(__file__).resolve().parent
    for name in ("model.py", "data.py", "prototypes.py", "localization.py"):
        p = pkg_dir / name
        h.update(hashlib.sha1(p.read_bytes()).hexdigest().encode())

    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# 构建
# --------------------------------------------------------------------------- #
def build(cfg, splits, device="cuda", force=False):
    """splits: {name: entries}。返回 cache 根目录。"""
    fp = fingerprint(cfg)
    root = Path(cfg.cache_dir) / fp
    manifest_p = root / "manifest.json"

    if manifest_p.is_file() and not force:
        man = json.loads(manifest_p.read_text(encoding="utf-8"))
        if all(s in man["splits"] for s in splits):
            print(f"缓存已存在且完整: {root}")
            return root
        print(f"缓存部分缺失,补建: {sorted(set(splits) - set(man['splits']))}")

    root.mkdir(parents=True, exist_ok=True)
    manifest = {"version": CACHE_VERSION, "fingerprint": fp, "splits": {}}
    if manifest_p.is_file():
        manifest = json.loads(manifest_p.read_text(encoding="utf-8"))

    model, preprocess, _ = biomed.load_model(device)
    layers = tuple(cfg.patch_layers)
    t_all = time.time()

    for name, entries in splits.items():
        if name in manifest["splits"] and not force:
            continue
        print(f"[cache] {name}: {len(entries)} 张")
        t0 = time.time()
        sdir = root / name
        sdir.mkdir(parents=True, exist_ok=True)
        paths = pdata.img_paths(entries, cfg.data_root)

        # ---- 元信息(不依赖编码器,可随时重建) ----
        labels, has_mask, masks14 = [], [], []
        for e in entries:
            labels.append(int(e["label"]))
            hm = bool(e.get("mask_path"))
            has_mask.append(hm)
            g = (
                pdata.mask_grid_14(str(Path(cfg.data_root) / e["mask_path"]))
                if hm
                else np.zeros((cfg._grid, cfg._grid), np.float32)
            )
            masks14.append(g)
        torch.save(torch.tensor(labels, dtype=torch.long), sdir / "label.pt")
        torch.save(torch.tensor(has_mask, dtype=torch.bool), sdir / "has_mask.pt")
        torch.save(torch.from_numpy(np.stack(masks14)).bool(), sdir / "mask14.pt")

        # ---- 分块编码 ----
        cdir = sdir / "_chunks"
        if cdir.exists():
            shutil.rmtree(cdir)
        cdir.mkdir()

        for ci, i in enumerate(tqdm(range(0, len(entries), CHUNK), desc=f"  encode", ncols=80)):
            bp = paths[i : i + CHUNK]
            cls = biomed.encode_images(model, preprocess, bp, batch_size=64, device=device)
            pf = biomed.encode_patch_features(
                model, preprocess, bp, layers=layers, batch_size=32, device=device
            )
            torch.save(cls.half(), cdir / f"cls_{ci:05d}.pt")
            for l in layers:
                torch.save(pf[l].half(), cdir / f"l{l}_{ci:05d}.pt")

        # ---- 逐层合并(峰值内存 = 单层全量 ≈ 3.7 GiB fp16 → 安全) ----
        def _merge(pattern, out):
            parts = sorted(cdir.glob(pattern))
            torch.save(torch.cat([torch.load(p, weights_only=True) for p in parts], 0), sdir / out)

        _merge("cls_*.pt", "cls.pt")
        for l in layers:
            _merge(f"l{l}_*.pt", f"patch_l{l}.pt")
        shutil.rmtree(cdir)

        manifest["splits"][name] = {
            "n": len(entries),
            "n_anomaly": int(sum(labels)),
            "n_with_mask": int(sum(has_mask)),
            "n_mask_nonempty": int(sum(1 for g in masks14 if g.any())),
            "layers": list(layers),
            "image_paths": [e["image_path"] for e in entries],
        }
        manifest_p.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        print(f"[cache] {name} 完成,用时 {(time.time()-t0)/60:.1f} 分钟")
        del paths

    print(f"[cache] 全部完成,总用时 {(time.time()-t_all)/60:.1f} 分钟 → {root}")
    return root


# --------------------------------------------------------------------------- #
# 读取
# --------------------------------------------------------------------------- #
def load(cfg, split, layers=None, dtype=torch.float32):
    """读取缓存。layers=None 时不载入 patch 特征(Stage 1 只需要 cls)。

    返回 dict: label, has_mask, mask14, cls, patch{layer: tensor}, manifest
    """
    fp = fingerprint(cfg)
    root = Path(cfg.cache_dir) / fp
    manifest_p = root / "manifest.json"
    if not manifest_p.is_file():
        raise FileNotFoundError(f"缓存不存在: {root}\n请先运行 --stage cache 构建。")
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    if split not in manifest["splits"]:
        raise KeyError(f"缓存里没有 split '{split}',现有: {list(manifest['splits'])}")

    sdir = root / split
    out = {
        "label": torch.load(sdir / "label.pt", weights_only=True),
        "has_mask": torch.load(sdir / "has_mask.pt", weights_only=True),
        "mask14": torch.load(sdir / "mask14.pt", weights_only=True),
        "cls": torch.load(sdir / "cls.pt", weights_only=True).to(dtype),
        "manifest": manifest["splits"][split],
    }
    if layers:
        out["patch"] = {
            l: torch.load(sdir / f"patch_l{l}.pt", weights_only=True).to(dtype) for l in layers
        }
    return out


# --------------------------------------------------------------------------- #
# 大特征的零常驻访问
# --------------------------------------------------------------------------- #
def pool_patch_mmap(cfg, layers, verbose=True):
    """把 pool 的 patch 特征以 **内存映射** 方式打开,而不是整块载入内存。

    为什么需要:pool 的 4 层 patch 特征 = 9139×196×512×4B = 3.67 GiB/层,
    4 层 fp32 共 14.7 GiB,fp16 也有 7.3 GiB。而 Stage 2 每步只用到 batch 大小的
    数据(bs=256 时约 200 MiB),整块常驻纯属浪费 —— 实测在 8.5 GB 可用内存的
    机器上直接 OOM。

    memmap 让特征留在磁盘,按需分页;内存由操作系统管理,工作集自然有界。
    首次调用会把 .pt 转成 .npy(逐层流式,峰值内存 = 单层 fp16),
    之后直接复用。
    """
    fp = fingerprint(cfg)
    sdir = Path(cfg.cache_dir) / fp / "pool"
    if not (sdir / "label.pt").is_file():
        raise FileNotFoundError(
            f"pool 缓存不存在: {sdir} —— 请先运行 --stage cache")

    out = {}
    for l in layers:
        npy = sdir / f"patch_l{l}_fp16.npy"
        if not npy.is_file():
            if verbose:
                print(f"[cache] 首次转换 layer {l} → npy(逐层流式,峰值内存 = 单层 fp16)")
            t = torch.load(sdir / f"patch_l{l}.pt", weights_only=True)   # fp16
            arr = t.numpy().astype(np.float16, copy=False)
            del t
            np.save(npy, arr)
            del arr
        out[l] = np.load(npy, mmap_mode="r")            # 零常驻
        if verbose:
            print(f"[cache] layer {l} memmap {out[l].shape} {out[l].dtype} → {npy.name}")
    return out


def exists(cfg, split):
    root = Path(cfg.cache_dir) / fingerprint(cfg)
    p = root / "manifest.json"
    if not p.is_file():
        return False
    return split in json.loads(p.read_text(encoding="utf-8"))["splits"]
