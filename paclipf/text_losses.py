"""文本侧损失(文档公式)。

P(·|z) = softmax(scale * [cos(z, t_N), cos(z, t_A)])
CLS 用全局层;patch 用病灶层;属性层只进 L_div / L_dis。
"""
import torch
import torch.nn.functional as F


IDX_N, IDX_A = 0, 1


def _pair_logits(feat, t_n, t_a, scale):
    """feat (N, D) 或 (N, P, D); t_n/t_a (D,)。返回 (..., 2) logits。"""
    f = F.normalize(feat.float(), dim=-1)
    tn = F.normalize(t_n.float(), dim=-1)
    ta = F.normalize(t_a.float(), dim=-1)
    return scale * torch.stack([f @ tn, f @ ta], dim=-1)


def _ce_nll(logits, target):
    """logits (..., 2), target 整数 0/1 或同形 long。"""
    logp = F.log_softmax(logits, dim=-1)
    return -logp.gather(-1, target.long().unsqueeze(-1)).squeeze(-1)


def text_losses(cfg, pack, pack0, f_cls, patch, mask_flat, label):
    """返回 (L_T, stats)。

    pack / pack0 : apply_adapter 前后的分层锚点
    f_cls        : (N, D)
    patch        : (N, 196, D)  layer 11
    mask_flat    : (N, 196) bool
    label        : (N,) 0/1
    """
    t = getattr(cfg, "text", cfg)
    scale = float(getattr(t, "scale", 20.0))
    m_dis = float(getattr(t, "m_dis", 0.3))
    delta = float(getattr(t, "delta_div", 0.5))
    lam_p = float(getattr(t, "lam_p", 1.0))
    lam_d = float(getattr(t, "lam_d", 0.5))
    lam_r = float(getattr(t, "lam_r", 0.5))
    lam_v = float(getattr(t, "lam_v", 0.2))

    y = label.long()
    is_anom = y == 1
    is_norm = y == 0
    empty = ~mask_flat.any(dim=1)
    has_target = is_anom & ~empty

    t_g_n, t_g_a = pack["t_global"][0], pack["t_global"][1]
    t_l_n, t_l_a = pack["t_lesion"][0], pack["t_lesion"][1]

    # ---- L_global: 全部切片的 CLS ----
    lg = _pair_logits(f_cls, t_g_n, t_g_a, scale)
    L_global = _ce_nll(lg, y).mean()

    # ---- 正常图 patch → N ----
    parts_n = []
    if is_norm.any():
        lp = _pair_logits(patch[is_norm], t_l_n, t_l_a, scale)   # (Nn, 196, 2)
        nll = _ce_nll(lp, torch.zeros_like(lp[..., 0], dtype=torch.long))
        parts_n.append(nll.mean())
    # ---- 异常非空 mask: 内→A, 外→N(各自集合内平均,避免面积碾压) ----
    L_patch = f_cls.new_zeros(())
    if has_target.any():
        pa = patch[has_target]
        ma = mask_flat[has_target]
        lp = _pair_logits(pa, t_l_n, t_l_a, scale)
        nll_a = _ce_nll(lp, torch.ones_like(lp[..., 0], dtype=torch.long))
        nll_n = _ce_nll(lp, torch.zeros_like(lp[..., 0], dtype=torch.long))
        inside = (nll_a * ma).sum(1) / ma.sum(1).clamp(min=1)
        outside = (nll_n * ~ma).sum(1) / (~ma).sum(1).clamp(min=1)
        L_patch = inside.mean() + outside.mean()
    if parts_n:
        L_patch = L_patch + parts_n[0]

    # ---- L_dis: 全部正常锚点均值 vs 全部异常锚点均值 ----
    t_n_all = F.normalize(torch.cat(
        [pack["t_global"][0:1], pack["t_attr_n"], pack["t_lesion"][0:1]], 0).mean(0), dim=0)
    t_a_all = F.normalize(torch.cat(
        [pack["t_global"][1:2], pack["t_attr_a"], pack["t_lesion"][1:2]], 0).mean(0), dim=0)
    L_dis = F.relu((t_n_all * t_a_all).sum() - m_dis)

    # ---- L_preserve ----
    from .text_adapter import stack_all
    t1, t0 = F.normalize(stack_all(pack), dim=-1), F.normalize(stack_all(pack0), dim=-1)
    L_preserve = (1.0 - (t1 * t0).sum(-1)).mean()

    # ---- L_div: 属性层异常锚点两两 ----
    ta = F.normalize(pack["t_attr_a"], dim=-1)
    k = ta.shape[0]
    if k >= 2:
        sim = ta @ ta.t()
        eye = torch.eye(k, device=ta.device, dtype=torch.bool)
        L_div = F.relu(sim.masked_fill(eye, 0) - delta).sum() / (k * (k - 1))
    else:
        L_div = f_cls.new_zeros(())

    L_T = L_global + lam_p * L_patch + lam_d * L_dis + lam_r * L_preserve + lam_v * L_div
    stats = {
        "L_global": float(L_global.detach().item()),
        "L_patch": float(L_patch.detach().item()),
        "L_dis": float(L_dis.detach().item()),
        "L_preserve": float(L_preserve.detach().item()),
        "L_div": float(L_div.detach().item()),
        "n_target": int(has_target.sum()),
        "n_empty": int((is_anom & empty).sum()),
    }
    return L_T, stats
