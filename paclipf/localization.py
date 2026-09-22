"""Anomaly localization: patch x prototype-anchor similarity -> heatmap."""
import torch
import torch.nn.functional as F
import numpy as np
import cv2


@torch.no_grad()
def patch_anomaly_scores(patch_feats, c_anom, c_norm, tau=10.0):
    """s(p) = max_j sim(p, c_anom^j) − max_k sim(p, c_norm^k)。

    patch_feats: (N, L, D)；c_anom: (D,) 或 (D, Ka)；c_norm: (D, Kn)。
    """
    if c_anom.dim() == 1:
        sim_anom = patch_feats @ c_anom
    else:
        sim_anom = (patch_feats @ c_anom).max(dim=-1).values
    sim_norm = (patch_feats @ c_norm).max(dim=-1).values
    return tau * (sim_anom - sim_norm)


@torch.no_grad()
def multi_layer_heatmap(patch_feats_dict, c_anom, c_norm, layers, image_size=224, tau=1.0):
    """分层原型：第 i 层用第 i 套 (c_norm, c_anom)；否则四层共用一对。"""
    from .prototypes import layer_pair

    H = image_size // 16
    maps = []
    for i, l in enumerate(layers):
        cn, ca = layer_pair(c_norm, c_anom, i)
        s = patch_anomaly_scores(patch_feats_dict[l], ca, cn, tau=tau)
        maps.append(s.view(-1, H, H))
    return torch.stack(maps, dim=0).mean(dim=0)


@torch.no_grad()
def zscore_sigmoid(hm, tau=2.0):
    """Per-image z-score then sigmoid (whole-image statistics).

    NOTE: for medical images with large dark backgrounds this is suboptimal --
    the background (very low scores) biases mean/std, compressing the clinically
    meaningful lesion-vs-tissue margin. Prefer zscore_sigmoid_roi.
    """
    z = (hm - hm.mean(dim=(1, 2), keepdim=True)) / (hm.std(dim=(1, 2), keepdim=True) + 1e-8)
    return torch.sigmoid(tau * z)


@torch.no_grad()
def zscore_sigmoid_roi(hm, roi_mask, tau=2.0):
    """Per-image z-score computed INSIDE an anatomical ROI (brain/body), sigmoid,
    and patches outside the ROI are pushed to the minimum.

    hm: (N, 14, 14) raw scores; roi_mask: (N, 14, 14) bool (True = tissue).
    Rationale: background pixels are trivially separable and otherwise dominate
    the statistics, flattening the lesion-vs-normal-tissue contrast.
    """
    out = torch.zeros_like(hm)
    for i in range(hm.shape[0]):
        m = roi_mask[i]
        if int(m.sum()) < 5:            # degenerate ROI -> fall back to whole image
            m = torch.ones_like(m)
        v = hm[i][m]
        z = (hm[i] - v.mean()) / (v.std() + 1e-8)
        s = torch.sigmoid(tau * z)
        s[~m] = 0.0
        out[i] = s
    return out


def tissue_mask_from_image(img_path, size14=14, thr=20):
    """Coarse anatomical ROI from the grayscale image: pixels brighter than thr
    (brain/body tissue vs dark background). Returns bool tensor (14, 14)."""
    from PIL import Image
    import numpy as np
    img = np.array(Image.open(img_path).convert("L").resize((size14, size14), Image.NEAREST))
    return torch.from_numpy(img > thr)


def to_pixel_map(heatmaps, size=512, sigma=1.5, kernel=9):
    """(N, 14, 14) -> (N, size, size) float32 numpy, bilinear upsample + gaussian blur."""
    hm = F.interpolate(heatmaps.unsqueeze(1), size=size, mode="bilinear", align_corners=True)
    hm = hm.squeeze(1).cpu().numpy()
    blurred = np.stack([cv2.GaussianBlur(m, (kernel, kernel), sigma) for m in hm])
    return blurred


def attention_weights(heatmaps, temperature=1.0):
    """Heatmap -> softmax attention over 196 patches. (N, 196)."""
    flat = heatmaps.view(heatmaps.shape[0], -1) / temperature
    return torch.softmax(flat, dim=-1)
