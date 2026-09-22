"""热力图可视化 —— 输出可直接对照的拼图。

布局(固定 4 列,便于逐行比较):
    原图 | GT mask | 左图(通常是 PA-CLIP 原口径) | 右图(通常是改进口径)

两列的标题由调用方指定。切片选择用**固定种子随机抽样**并在图上标注 hit@1 命中与否,
避免"挑好看的"造成的选择性展示 —— 每一行的命中情况都写在标签里,包括漏检的。
"""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import cv2

from paclipf import data as pdata
from paclipf import localization as loc

def heat_to_jet(hm_px, size=512):
    """(H,W) float score map -> RGB heatmap.

    Percentile clipping (10th-90th) rather than min-max: raw prototype scores of
    small lesions differ from surrounding tissue by only ~0.1, which min-max washes
    out once the dark background dominates the range.
    """
    lo, hi = np.percentile(hm_px, 10), np.percentile(hm_px, 90)
    v = np.clip((hm_px - lo) / (hi - lo + 1e-8), 0, 1)
    jet = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)          # cv2 emits BGR


def overlay(img_rgb, heat_rgb, alpha=0.45):
    """Alpha-blend a heatmap over the (RGB) image."""
    return (alpha * img_rgb.astype(np.float32) + (1 - alpha) * heat_rgb).astype(np.uint8)


def _px(hm, size):
    """(14,14) → (size,size) numpy,带高斯平滑(纯观感,不影响指标)。"""
    t = hm.unsqueeze(0) if hm.dim() == 2 else hm
    return loc.to_pixel_map(t.cpu(), size=size)[0]


def _load_img(cfg, entry, size):
    p = Path(cfg.data_root) / entry["image_path"]
    return np.array(Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR))


def _load_mask(cfg, entry, size):
    p = entry.get("mask_path")
    if not p:
        return None
    m = np.array(Image.open(Path(cfg.data_root) / p).convert("L").resize((size, size), Image.NEAREST))
    return m


def _mask_panel(cfg, entry, size):
    """GT mask 列 —— 永远显示**原始分辨率**的掩膜。

    不要在 14x14 网格为空时改显示"no lesion":病灶是存在的,
    只是 14x14 的粒度把它丢掉了(实测 15.7% 的异常切片如此)。
    用原始掩膜显示、在行标签里注明"mask empty at 14x14",才是如实的表达。
    """
    img = _load_img(cfg, entry, size)
    m = _load_mask(cfg, entry, size)
    if m is None or m.max() == 0:                       # 正常切片:确实没有病灶
        canvas = np.full_like(img, 245)
        cv2.putText(canvas, "no lesion", (10, size // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (120, 120, 120), 2)
        return canvas
    rgb = cv2.cvtColor((m > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)
    rgb[..., 1:] = 0                                   # 只留红色通道
    return overlay(img, rgb, alpha=0.45)


def _panel(cfg, entry, hm, size):
    img = _load_img(cfg, entry, size)
    return overlay(img, heat_to_jet(_px(hm, size), size), alpha=0.45)


def _label_bar(width, height, text, color=(0, 0, 0)):
    bar = np.full((height, width, 3), 255, np.uint8)
    cv2.putText(bar, text, (8, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
    return bar


def render(cfg, entries, labels, hm_left, hm_right, out_path,
           left_title="PA-CLIP", right_title="ours",
           n_anom=6, n_norm=2, seed=0, s_flat=None, s_flat_left=None):
    """拼图输出。

    hm_left / hm_right : (N, 14, 14) 与 entries 等长
    s_flat             : (N, 196) 可选。给了就在标签里标注该切片 hit@1 是否命中
    """
    size = cfg.eval_size
    labels = np.asarray(labels if not torch.is_tensor(labels) else labels.cpu().numpy())
    n = len(entries)

    rng = random.Random(seed)
    idx_anom = [i for i in range(n) if labels[i] > 0]
    idx_norm = [i for i in range(n) if labels[i] == 0]
    pick = rng.sample(idx_anom, min(n_anom, len(idx_anom))) + \
        rng.sample(idx_norm, min(n_norm, len(idx_norm)))

    def _np(x):
        return None if x is None else (x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x))

    s_np, s_np_l = _np(s_flat), _np(s_flat_left)

    rows = []
    for i in pick:
        e = entries[i]
        has_lesion = labels[i] > 0
        # hit@1 标注 —— 用 14x14 mask,与指标口径一致。
        # 左右两列都标:只标一边会让"对照"失去意义(读者看不到左列的漏检)
        note = ""
        if has_lesion and e.get("mask_path"):
            g = pdata.mask_grid_14(str(Path(cfg.data_root) / e["mask_path"])).reshape(-1).astype(bool)
            if g.any():
                def _hit(arr):
                    if arr is None:
                        return "?"
                    return "YES" if g[int(arr[i].argmax())] else "NO"
                note = f"  hit@1  L:{_hit(s_np_l)}  R:{_hit(s_np)}"
                note += f"  (lesion {int(g.sum())} patch)"
            else:
                note = "  (mask empty at 14x14)"
        tag = "ANOMALY" if has_lesion else "NORMAL"
        rows.append((
            f"{tag} #{i} {Path(e['image_path']).name}{note}",
            [_load_img(cfg, e, size),
             _mask_panel(cfg, e, size),
             _panel(cfg, e, hm_left[i], size),
             _panel(cfg, e, hm_right[i], size)],
            (200, 0, 0) if has_lesion else (0, 120, 0),
        ))

    bar_h, head_h = 30, 34
    W = size * 4
    canvas = np.full((head_h + len(rows) * (size + bar_h), W, 3), 255, np.uint8)
    for c, title in enumerate(["original", "GT mask (original resolution)",
                               left_title, right_title]):
        canvas[:head_h, c * size:(c + 1) * size] = _label_bar(size, head_h, title)
    for r, (tag, panels, color) in enumerate(rows):
        y = head_h + r * (size + bar_h)
        canvas[y:y + bar_h] = _label_bar(W, bar_h, tag, color)   # 标签在**行上方**
        for c, p in enumerate(panels):
            canvas[y + bar_h:y + bar_h + size, c * size:(c + 1) * size] = p

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_path)
    return out_path, canvas.shape


def render_maps(cfg, entries, labels, maps, out_path,
                n_anom=6, n_norm=2, seed=0, s_flat=None):
    """流水线拼图: original | GT | proto | text | mem | fused | mask。

    maps: dict, 键为列名,值为 (N,14,14) 或 (N,H,W) numpy/tensor。
    `mask` 若已是像素二值图 (N,eval_size,eval_size),直接贴。
    """
    size = cfg.eval_size
    labels = np.asarray(labels if not torch.is_tensor(labels) else labels.cpu().numpy())
    n = len(entries)
    rng = random.Random(seed)
    idx_anom = [i for i in range(n) if labels[i] > 0]
    idx_norm = [i for i in range(n) if labels[i] == 0]
    pick = rng.sample(idx_anom, min(n_anom, len(idx_anom))) + \
        rng.sample(idx_norm, min(n_norm, len(idx_norm)))
    s_np = None if s_flat is None else (s_flat.cpu().numpy() if torch.is_tensor(s_flat) else np.asarray(s_flat))

    order = [k for k in ("proto", "text", "mem", "fused", "mask") if k in maps]
    titles = ["original", "GT mask"] + order
    ncols = 2 + len(order)

    def _as(hm, i):
        x = maps[hm]
        if torch.is_tensor(x):
            return x[i].detach().cpu()
        return np.asarray(x)[i]

    rows = []
    for i in pick:
        e = entries[i]
        has_lesion = labels[i] > 0
        note = ""
        if has_lesion and e.get("mask_path"):
            g = pdata.mask_grid_14(str(Path(cfg.data_root) / e["mask_path"])).reshape(-1).astype(bool)
            if g.any() and s_np is not None:
                hit = "YES" if g[int(s_np[i].argmax())] else "NO"
                note = f"  hit@1 {hit}  (lesion {int(g.sum())} patch)"
            elif not g.any():
                note = "  (mask empty at 14x14)"
        tag = "ANOMALY" if has_lesion else "NORMAL"
        panels = [_load_img(cfg, e, size), _mask_panel(cfg, e, size)]
        for k in order:
            x = _as(k, i)
            if k == "mask":
                arr = x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
                if arr.ndim == 2 and arr.shape[0] == size:
                    rgb = cv2.cvtColor((arr > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)
                    rgb[..., 1:] = 0
                    panels.append(overlay(_load_img(cfg, e, size), rgb, alpha=0.45))
                else:
                    panels.append(_panel(cfg, e, torch.as_tensor(arr).float() if not torch.is_tensor(x) else x, size))
            else:
                t = x if torch.is_tensor(x) else torch.as_tensor(x)
                panels.append(_panel(cfg, e, t.float(), size))
        rows.append((f"{tag} #{i} {Path(e['image_path']).name}{note}",
                     panels, (200, 0, 0) if has_lesion else (0, 120, 0)))

    bar_h, head_h = 30, 34
    W = size * ncols
    canvas = np.full((head_h + len(rows) * (size + bar_h), W, 3), 255, np.uint8)
    for c, t in enumerate(titles):
        canvas[:head_h, c * size:(c + 1) * size] = _label_bar(size, head_h, t)
    for r, (tg, panels, color) in enumerate(rows):
        y = head_h + r * (size + bar_h)
        canvas[y:y + bar_h] = _label_bar(W, bar_h, tg, color)
        for c, p in enumerate(panels):
            canvas[y + bar_h:y + bar_h + size, c * size:(c + 1) * size] = p

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_path)
    return out_path, canvas.shape


def render_single(cfg, entries, labels, hm, out_path, title="PA-CLIP",
                  n_anom=6, n_norm=2, seed=0, s_flat=None):
    """三列拼图:原图 | GT mask(原始分辨率) | 预测热图.

    供 PA-CLIP 单独出图(不与其他方法并排)。切片为固定种子随机抽样,
    行标签标注 hit@1 命中与否(含漏检),避免选择性展示。
    """
    size = cfg.eval_size
    labels = np.asarray(labels if not torch.is_tensor(labels) else labels.cpu().numpy())
    n = len(entries)
    rng = random.Random(seed)
    idx_anom = [i for i in range(n) if labels[i] > 0]
    idx_norm = [i for i in range(n) if labels[i] == 0]
    pick = rng.sample(idx_anom, min(n_anom, len(idx_anom))) + \
        rng.sample(idx_norm, min(n_norm, len(idx_norm)))
    s_np = None if s_flat is None else (s_flat.cpu().numpy() if torch.is_tensor(s_flat) else np.asarray(s_flat))

    rows = []
    for i in pick:
        e = entries[i]
        has_lesion = labels[i] > 0
        note = ""
        if has_lesion and e.get("mask_path"):
            g = pdata.mask_grid_14(str(Path(cfg.data_root) / e["mask_path"])).reshape(-1).astype(bool)
            if g.any():
                if s_np is not None:
                    hit = "YES" if g[int(s_np[i].argmax())] else "NO"
                    note = f"  hit@1 {hit}  (lesion {int(g.sum())} patch)"
            else:
                note = "  (mask empty at 14x14)"
        tag = "ANOMALY" if has_lesion else "NORMAL"
        rows.append((
            f"{tag} #{i} {Path(e['image_path']).name}{note}",
            [_load_img(cfg, e, size), _mask_panel(cfg, e, size), _panel(cfg, e, hm[i], size)],
            (200, 0, 0) if has_lesion else (0, 120, 0),
        ))

    bar_h, head_h = 30, 34
    W = size * 3
    canvas = np.full((head_h + len(rows) * (size + bar_h), W, 3), 255, np.uint8)
    for c, t in enumerate(["original", "GT mask (original resolution)", title]):
        canvas[:head_h, c * size:(c + 1) * size] = _label_bar(size, head_h, t)
    for r, (tg, panels, color) in enumerate(rows):
        y = head_h + r * (size + bar_h)
        canvas[y:y + bar_h] = _label_bar(W, bar_h, tg, color)
        for c, p in enumerate(panels):
            canvas[y + bar_h:y + bar_h + size, c * size:(c + 1) * size] = p

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_path)
    return out_path, canvas.shape
