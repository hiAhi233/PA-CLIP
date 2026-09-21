"""Evaluation metrics: pixel-level (subsampled, OOM-safe) and image-level."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def _subsample(y, p, max_pixels=8_000_000, seed=111):
    y = np.asarray(y).ravel()
    p = np.asarray(p).ravel()
    if len(y) > max_pixels:
        rng = np.random.RandomState(seed)
        idx = rng.randint(0, len(y), max_pixels)
        y, p = y[idx], p[idx]
    return y, p


def pixel_metrics(pred_map, mask_map, max_pixels=8_000_000):
    """pred_map / mask_map: (N, H, W) numpy. Returns (AUC, AP)."""
    y, p = _subsample(mask_map, pred_map, max_pixels)
    y = (y > 0).astype(np.int32)
    if y.max() == y.min():
        return float("nan"), float("nan")
    return roc_auc_score(y, p), average_precision_score(y, p)


def image_metrics(scores, labels):
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels)
    if labels.max() == labels.min():
        return float("nan"), float("nan")
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def top1_acc(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item() * 100


def aupro(pred_map, mask_map, max_fpr=0.3, n_th=200):
    """Per-Region Overlap AUC, FPR 截到 max_fpr 后再除以 max_fpr(满分 1)。

    pred_map / mask_map: (N, H, W)。只应传入异常切片。无病灶或无背景则 nan。
    """
    from scipy.ndimage import label as cc_label

    pred = np.asarray(pred_map, dtype=np.float64)
    mask = np.asarray(mask_map) > 0
    if pred.shape != mask.shape:
        raise ValueError(f"aupro 形状不符: pred {pred.shape} mask {mask.shape}")
    bg = ~mask
    if not mask.any() or not bg.any():
        return float("nan")

    regions = []
    for i in range(pred.shape[0]):
        lab, n = cc_label(mask[i])
        for k in range(1, n + 1):
            regions.append(pred[i][lab == k])
    if not regions:
        return float("nan")

    scores_bg = pred[bg]
    n_bg = float(bg.sum())
    lo, hi = float(pred.min()), float(pred.max())
    if hi <= lo:
        return float("nan")
    ths = np.linspace(hi, lo, int(n_th))
    fprs, pros = [], []
    for th in ths:
        fprs.append(float((scores_bg >= th).sum()) / n_bg)
        pros.append(float(np.mean([(r >= th).mean() for r in regions])))
    fprs = np.clip(np.asarray(fprs, dtype=np.float64), 0.0, 1.0)
    pros = np.asarray(pros, dtype=np.float64)
    order = np.argsort(fprs)
    fprs, pros = fprs[order], pros[order]
    # 单调:FPR 相同处保留较大 PRO
    uniq_f, uniq_p = [fprs[0]], [pros[0]]
    for f, p in zip(fprs[1:], pros[1:]):
        if f == uniq_f[-1]:
            uniq_p[-1] = max(uniq_p[-1], p)
        else:
            uniq_f.append(f)
            uniq_p.append(p)
    fprs, pros = np.asarray(uniq_f), np.asarray(uniq_p)
    if fprs[0] > 0:
        fprs = np.r_[0.0, fprs]
        pros = np.r_[pros[0], pros]
    keep = fprs <= max_fpr + 1e-12
    if keep.sum() < 1:
        return float("nan")
    f_k, p_k = fprs[keep], pros[keep]
    if f_k[-1] < max_fpr and (~keep).any():
        i = int(np.argmax(~keep))
        t = (max_fpr - fprs[i - 1]) / (fprs[i] - fprs[i - 1] + 1e-12)
        f_k = np.r_[f_k, max_fpr]
        p_k = np.r_[p_k, p_k[-1] + t * (pros[i] - p_k[-1])]
    elif f_k[-1] < max_fpr:
        f_k = np.r_[f_k, max_fpr]
        p_k = np.r_[p_k, p_k[-1]]
    trapz = getattr(np, "trapezoid", None)
    if trapz is None:
        trapz = getattr(np, "trapz", None)
    if trapz is not None:
        return float(trapz(p_k, f_k) / max_fpr)
    return float(np.sum((p_k[1:] + p_k[:-1]) * 0.5 * np.diff(f_k)) / max_fpr)

