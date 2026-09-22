"""高置信热图 → 边界细化 → Top-k 连通域 → 二值 mask。

顺序与图一致：双线性上采样 → 原图引导滤波贴边 → 阈值 τ → 闭运算 → Top-k。
τ 只在 OOF 上选（最大化 Dice），不在 test 上调。不上可训 CRF。
"""
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path
from PIL import Image

from .grid import nested


def load_guides(entries, cfg):
    """原图灰度 [0,1]，(N, eval_size, eval_size)，给引导滤波当边界。"""
    size = int(getattr(cfg, "eval_size", 240))
    root = Path(cfg.data_root)
    out = np.zeros((len(entries), size, size), dtype=np.float32)
    for i, e in enumerate(entries):
        p = e.get("image_path") if isinstance(e, dict) else None
        if not p:
            continue
        img = np.array(
            Image.open(root / p).convert("L").resize((size, size), Image.BILINEAR),
            dtype=np.float32,
        )
        out[i] = img / 255.0
    return out


def _box(src, r):
    k = max(1, int(2 * r + 1))
    return cv2.blur(src, (k, k))


def guided_filter(guide, src, radius=8, eps=1e-4):
    """He et al. 引导滤波。guide/src: (H,W) float32。把粗热图贴到解剖边界上。"""
    I = np.asarray(guide, dtype=np.float32)
    p = np.asarray(src, dtype=np.float32)
    mean_I = _box(I, radius)
    mean_p = _box(p, radius)
    corr_I = _box(I * I, radius)
    corr_Ip = _box(I * p, radius)
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + float(eps))
    b = mean_p - a * mean_I
    return _box(a, radius) * I + _box(b, radius)


def refine_maps(hm_px, guides, radius=8, eps=1e-4):
    """逐图引导滤波。guides 缺失或全 0 时跳过该张。"""
    hm_px = np.asarray(hm_px, dtype=np.float32)
    guides = np.asarray(guides, dtype=np.float32)
    out = hm_px.copy()
    n = min(len(hm_px), len(guides))
    for i in range(n):
        g = guides[i]
        if float(g.max()) <= 0:
            continue
        out[i] = guided_filter(g, hm_px[i], radius=radius, eps=eps)
    return out


def _to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def upsample(hm, size):
    """(N,H,H) → (N,size,size) float32 numpy，双线性。"""
    if torch.is_tensor(hm):
        x = hm.float().unsqueeze(1)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        return x.squeeze(1).detach().cpu().numpy()
    t = torch.from_numpy(np.asarray(hm, dtype=np.float32)).unsqueeze(1)
    x = F.interpolate(t, size=size, mode="bilinear", align_corners=True)
    return x.squeeze(1).numpy()


def _norm01(hm):
    """逐图 min-max 到 [0,1]。hm: (N,H,W) numpy。"""
    out = np.empty_like(hm, dtype=np.float32)
    for i in range(len(hm)):
        v = hm[i]
        lo, hi = float(v.min()), float(v.max())
        out[i] = (v - lo) / (hi - lo + 1e-8)
    return out


def close_binary(mask, k=5):
    """形态学闭运算。mask (H,W) uint8 {0,1}。"""
    k = int(k)
    if k <= 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)


def dice(pred, gt):
    p = pred.astype(bool).ravel()
    g = gt.astype(bool).ravel()
    inter = np.logical_and(p, g).sum()
    den = p.sum() + g.sum()
    if den == 0:
        return 1.0
    return float(2.0 * inter / den)


def iou(pred, gt):
    p = pred.astype(bool).ravel()
    g = gt.astype(bool).ravel()
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def topk_components(binary, score, k=1):
    """连通域按区域内平均分数取 Top-k。binary/score: (H,W)。"""
    k = max(1, int(k))
    m = (binary > 0).astype(np.uint8)
    nlab, labels, _, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if nlab <= 1:
        return m
    scored = []
    for lab in range(1, nlab):
        sel = labels == lab
        scored.append((float(score[sel].mean()) if sel.any() else -1e9, lab))
    scored.sort(reverse=True)
    keep = {lab for _, lab in scored[:k]}
    out = np.zeros_like(m)
    for lab in keep:
        out[labels == lab] = 1
    return out


def binarize(hm_px, tau, close_k=5, topk=1):
    """hm_px: (N,H,W) 已 min-max。返回 (N,H,W) uint8。"""
    hm_px = _norm01(np.asarray(hm_px, dtype=np.float32))
    out = np.zeros(hm_px.shape, dtype=np.uint8)
    for i in range(len(hm_px)):
        raw = (hm_px[i] >= float(tau)).astype(np.uint8)
        raw = close_binary(raw, close_k)
        out[i] = topk_components(raw, hm_px[i], k=topk)
    return out


def search_tau(hm, mask, grid=None, close_k=5, topk=1):
    """OOF 上最大化平均 Dice。hm/mask 必须已是同一分辨率（与 apply_maps 一致=eval_size）。"""
    if grid is None:
        grid = [i / 20.0 for i in range(2, 19)]
    hm = _norm01(_to_np(hm).astype(np.float32))
    mask = (_to_np(mask) > 0).astype(np.uint8)
    if mask.ndim == 2:
        g = int(mask.shape[-1] ** 0.5)
        mask = mask.reshape(len(hm), g, g)
    best_t, best_d = 0.5, -1.0
    for t in grid:
        pred = binarize(hm, t, close_k=close_k, topk=topk)
        ds = [dice(pred[i], mask[i]) for i in range(len(hm)) if mask[i].any()]
        d = float(np.mean(ds)) if ds else 0.0
        if d > best_d:
            best_d, best_t = d, float(t)
    return best_t, best_d


def _to_eval_maps(hm14, cfg, guides=None, entries=None):
    """14×14 → eval_size，可选原图引导滤波。"""
    size = int(getattr(cfg, "eval_size", 240))
    px = upsample(hm14, size)
    do = bool(nested(cfg, "postprocess", "refine", True))
    if do:
        if guides is None and entries is not None:
            guides = load_guides(entries, cfg)
        if guides is not None:
            px = refine_maps(
                px, guides,
                radius=int(nested(cfg, "postprocess", "refine_radius", 8)),
                eps=float(nested(cfg, "postprocess", "refine_eps", 1e-4)),
            )
    return px


def search_tau_eval(hm14, mask_px, cfg, grid=None, guides=None, entries=None):
    """与 apply_maps 同一套：上采样 → 边界细化 → 闭运算 / Top-k / 扫 τ。"""
    close_k = int(nested(cfg, "postprocess", "close_k", 5))
    topk = int(nested(cfg, "postprocess", "topk", 1))
    px = _to_eval_maps(hm14, cfg, guides=guides, entries=entries)
    mask_px = np.stack([_to_np(m) for m in mask_px], 0)
    return search_tau(px, mask_px, grid=grid, close_k=close_k, topk=topk)


def apply_maps(hm14, cfg, tau=0.5, guides=None, entries=None):
    """14×14 热图 → 细化像素热图 + 二值 mask。"""
    close_k = int(nested(cfg, "postprocess", "close_k", 5))
    topk = int(nested(cfg, "postprocess", "topk", 1))
    px = _to_eval_maps(hm14, cfg, guides=guides, entries=entries)
    n01 = _norm01(px)
    mask = binarize(n01, tau, close_k=close_k, topk=topk)
    return {"hm_px": px, "hm_norm": n01, "mask": mask, "tau": float(tau)}


def enabled(cfg):
    return bool(nested(cfg, "postprocess", "enable", True))
