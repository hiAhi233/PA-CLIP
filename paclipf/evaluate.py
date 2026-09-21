"""评估 —— 指标一律走本项目 paclipf.metrics,不重写,保证与免训练通路逐位可比。

一个必须说清的差异:命中率这类**图内**指标不受逐图 z-score 影响(正仿射不变),
但**像素 AUC 把所有切片的像素池化成一个集合**,所以逐图标准化会改变它。
因此本模块同时报两版像素指标:
  paclip —— z-score+sigmoid 后再上采样(PA-CLIP 的原始管线,用于对齐基线)
  ours   —— 直接用原始 s(平滑后)上采样(与 S_map 口径一致)
"""
import numpy as np
import torch
from PIL import Image

from paclipf import localization as loc
from paclipf import metrics as pmetrics

from . import diag
from .fusion import to_np as fusion_tonp


def pixel_map(hm, size, mode="ours", tau=2.0):
    """(N,14,14) → (N,size,size) numpy。

    mode='paclip' : 先逐图 z-score + sigmoid,再上采样(PA-CLIP 原样)
    mode='ours'   : 直接上采样(与部署图口径一致)
    """
    if mode == "paclip":
        z = (hm - hm.mean(dim=(1, 2), keepdim=True)) / (hm.std(dim=(1, 2), keepdim=True) + 1e-8)
        hm = torch.sigmoid(tau * z)
    return loc.to_pixel_map(hm.cpu(), size=size)


def load_masks_px(entries, cfg):
    """把 mask 放到 eval_size。只对有 mask_path 的条目有效。"""
    out = []
    for e in entries:
        p = e.get("mask_path")
        if not p:
            out.append(None)
            continue
        m = np.array(
            Image.open(str(cfg.data_root) + "/" + p)
            .convert("L")
            .resize((cfg.eval_size, cfg.eval_size), Image.NEAREST)
        )
        out.append(m)
    return out


def pixel_metrics_fixed(pred, mask, max_pixels=8_000_000, seed=111):
    """修正版:无放回抽样。

    pa_clip.metrics._subsample 用 rng.randint(0, len(y), max_pixels) —— **有放回**,
    抽出的 800 万像素含大量重复,浪费有效样本、低估方差。
    实测该分支每次运行都会触发(test 集 555 张 × 512² = 1.45 亿像素)。
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(mask).ravel()
    p = np.asarray(pred).ravel()
    if len(y) > max_pixels:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(y), max_pixels, replace=False)
        y, p = y[idx], p[idx]
    y = (y > 0).astype(np.int32)
    if y.max() == y.min():
        return float("nan"), float("nan")
    return roc_auc_score(y, p), average_precision_score(y, p)


def full_report(cfg, sig_test, labels_test, entries_test, hm_for_pixel=None, tau=2.0):
    """test 上的完整指标表。返回 dict。"""
    out = {}
    labels_np = fusion_tonp(labels_test)
    anom = labels_np > 0

    # ---- 像素级:只在异常切片上(与 PA-CLIP 一致) ----
    hm = hm_for_pixel if hm_for_pixel is not None else sig_test["hm"]
    idx_anom = np.where(anom)[0]
    masks = load_masks_px([entries_test[i] for i in idx_anom], cfg)
    keep = [j for j, m in enumerate(masks) if m is not None]
    if keep:
        mk = np.stack([masks[j] for j in keep])
        for mode in ("paclip", "ours"):
            px = pixel_map(hm[idx_anom][keep], cfg.eval_size, mode=mode, tau=tau)
            a1, p1 = pmetrics.pixel_metrics(px, mk)
            a2, p2 = pixel_metrics_fixed(px, mk)
            out[f"pixel_auc_{mode}"] = a1
            out[f"pixel_ap_{mode}"] = p1
            out[f"pixel_auc_{mode}_fixed"] = a2
            out[f"pixel_ap_{mode}_fixed"] = p2
            if mode == "ours":
                out["pixel_aupro"] = pmetrics.aupro(px, mk)
        out["n_pixel_slices"] = len(keep)
    else:
        for k in ("paclip", "ours"):
            out[f"pixel_auc_{k}"] = out[f"pixel_ap_{k}"] = float("nan")
        out["pixel_aupro"] = float("nan")

    # ---- 定位诊断(全部基于部署图;图内指标用 14x14 mask,与部署图同分辨率) ----
    from paclipf import data as pdata

    s_flat = sig_test["s_flat"]
    m14 = []
    for i in idx_anom:
        e = entries_test[i]
        if e.get("mask_path"):
            m14.append(torch.tensor(pdata.mask_grid_14(str(cfg.data_root) + "/" + e["mask_path"])))
        else:
            m14.append(torch.zeros(cfg._grid, cfg._grid))
    # mask 要与分数同设备:diag 里会用分数的 topk 索引去 gather mask
    m14 = torch.stack(m14).reshape(len(idx_anom), -1).bool().to(s_flat.device)
    out.update(diag.hit_rate(s_flat[idx_anom], m14))
    out.update(diag.contrast(s_flat[idx_anom], m14))
    out["peak_gap_z"] = diag.peak_gap(s_flat[idx_anom], m14)
    out["_m14"] = m14  # 供 paired bootstrap 复用
    out["_idx_anom"] = idx_anom
    return out


_PCT_PREFIX = ("pixel_auc", "pixel_ap", "pixel_aupro",
               "image_auc", "image_ap", "oof_img_auc", "hit@", "acc")


def print_table(title, rows):
    """rows: list of (label, dict)

    列名取**所有行的并集**,不是第一行的键 —— 同一张表里会混入
    `_append_map_rows` 追加的定位对照行(来自 full_report,没有 tag 等键),
    只按第一行取列名会对不上而 KeyError。缺失的键按空值渲染。
    """
    keys = []
    for _, d in rows:
        for k in d:
            if not k.startswith("_") and k not in keys:
                keys.append(k)
    w = max(len(r[0]) for r in rows) + 2
    print(f"\n{'='*100}\n{title}\n{'='*100}")
    print(f"{'':<{w}}" + "".join(f"{k:>26}" for k in keys))
    for label, d in rows:
        cells = []
        for k in keys:
            v = d.get(k, "")
            if isinstance(v, float):
                v = v * 100 if k.startswith(_PCT_PREFIX) else v
                cells.append(f"{v:>26.4f}")
            else:
                cells.append(f"{str(v):>26}")
        print(f"{label:<{w}}" + "".join(cells))
