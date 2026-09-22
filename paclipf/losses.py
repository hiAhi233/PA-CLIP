"""Stage 2 的定位损失 + 图上的 L_visual 其余四项。

直接优化 PA-CLIP 的打分公式 `s(p) = sim(p,c_anom) − max_k sim(p,c_norm^k)`,
而不是优化 sigmoid 之后的图(后者不可导且不稳;实测逐图 z-score 会把
图像级 AUC 从 0.754 削到 0.638,在损失里它同样有害 —— 逐图标准化允许模型
靠"缩小图内方差"降损失,从而破坏跨图标定)。

三个必须处理的实测约束(训练池 5279 张异常切片):
  · mask 面积:均值 1.39%,中位 1.02%  → 病斑中位只有 2.0 个 patch
  · 15.7% 的 mask 在 14x14 下完全为空
  · 非病灶 patch 193.3 个 → 不做各自均值归一化的话权重比达 71:1

设计:
  · 用 **top-q**(q 按 mask 面积自适应)而不是 max:每张图有 q_i 个 patch 同时
    被抬,而不是只抬一个点去追噪声;且 t_in 就是 hit@3 的连续松弛。
  · 两项**各自在集合内取均值**,避免 71:1 的碾压。
  · 绝对项锚在 s=0(s=0 恰好是"到病灶锚点与到最近正常锚点等距"),
    这是整个设计里唯一优化**跨图标定**的机制 —— 而跨图标定正是图像级 AUC 的来源。
"""
import torch
import torch.nn.functional as F


def _adaptive_topq(s, mask, q_max):
    """在 mask 内取 top-q_i 的均值,q_i = min(|M_i|, q_max) 按面积自适应。

    s: (N,196) 可微;mask: (N,196) bool
    返回 (N,)
    """
    neg = torch.finfo(s.dtype).min
    v, _ = s.masked_fill(~mask, neg).topk(q_max, dim=1)      # (N,q_max) 降序
    size = mask.sum(1).clamp(min=1)
    qi = size.clamp(max=q_max)                                # q_i ≤ |M_i|,安全
    cum = v.cumsum(1)
    return cum.gather(1, (qi - 1)[:, None]).squeeze(1) / qi


def _topq_all(s, q):
    return s.topk(q, dim=1).values.mean(1)


def localize_loss(s_flat, mask_flat, is_anomaly, cfg):
    """返回 (L_normal, L_abnormal, stats)。梯度只经 s 流向 c_norm / c_anom。

    s_flat     : (N,196) 可微分数
    mask_flat  : (N,196) bool
    is_anomaly : (N,) bool
    """
    b = cfg.loc_b
    w_abs = cfg.loc_w_abs
    empty = ~mask_flat.any(1)

    # 空 mask 的异常切片:在 14x14 上没有可定位目标,强行要求 mask 内出峰等于注入标签噪声。
    # 移到 L_normal(要求"任何地方都不该像病灶"),但它们在 L_cls 里仍是异常类。
    to_normal = is_anomaly & empty if cfg.mask_empty_to_normal else torch.zeros_like(is_anomaly)
    has_target = is_anomaly & ~empty
    normal_side = (~is_anomaly) | to_normal

    stats = {
        "n_normal_side": int(normal_side.sum()),
        "n_abnormal_target": int(has_target.sum()),
        "n_moved_to_normal": int(to_normal.sum()),
    }

    # ---- L_normal:这些切片上"任何地方"都不该像病灶 ----
    if normal_side.any():
        t_all = _topq_all(s_flat, cfg.q_bar)
        L_normal = F.softplus(t_all[normal_side] - b).mean()
    else:
        L_normal = s_flat.new_zeros(())

    # ---- L_abnormal:图内排序 + 绝对标定 ----
    if has_target.any():
        s_a = s_flat[has_target]
        m_a = mask_flat[has_target]
        t_in = _adaptive_topq(s_a, m_a, cfg.q_max)
        t_out = _adaptive_topq(s_a, ~m_a, cfg.q_max)
        sigma = s_a.std(1).detach() + 1e-6          # 只做尺度参照,不进梯度
        # (a) 图内排序:病灶内要比病灶外高出 δ_r 个 σ —— 这一项直接对应命中率
        L_rank = F.softplus(cfg.loc_delta_r + (t_out - t_in) / sigma).mean()
        # (b) 绝对标定:病灶内要高于绝对零点 b + δ_a —— 这一项对应图像级 AUC
        L_abs = F.softplus(b + cfg.loc_delta_a - t_in).mean()
        L_abnormal = L_rank + w_abs * L_abs
        stats.update({
            "t_in": t_in.mean().item(), "t_out": t_out.mean().item(),
            "rank_term": L_rank.item(), "abs_term": L_abs.item(),
            "margin_over_sigma": ((t_in - t_out) / sigma).mean().item(),
        })
    else:
        L_abnormal = s_flat.new_zeros(())

    return L_normal, L_abnormal, stats


def _sigmoid_score(s_flat):
    """原始 cos 差约 ±0.2，直接 sigmoid 会挤在 0.5 附近。×10 拉开。"""
    return torch.sigmoid(10.0 * s_flat)


def focal_loss(s_flat, mask_flat, is_anomaly, gamma=2.0):
    """逐 patch 二分类 Focal。mask 内为异常、外为正常；正常图全部为负。

    s_flat     : (N,196)
    mask_flat  : (N,196) bool
    is_anomaly : (N,) bool
    """
    p = _sigmoid_score(s_flat).clamp(1e-6, 1.0 - 1e-6)
    target = torch.zeros_like(p)
    has = is_anomaly & mask_flat.any(1)
    if has.any():
        target[has] = mask_flat[has].float()
    pos = target * ((1.0 - p) ** gamma) * (-torch.log(p))
    neg = (1.0 - target) * (p ** gamma) * (-torch.log(1.0 - p))
    n_pos = target.sum().clamp(min=1.0)
    n_neg = (1.0 - target).sum().clamp(min=1.0)
    return pos.sum() / n_pos + neg.sum() / n_neg


def tversky_loss(s_flat, mask_flat, is_anomaly, alpha=0.3, beta=0.7):
    """软 mask 对 GT 的 Tversky（α=0.3, β=0.7 偏向召回）。只在有 mask 的异常上算。"""
    has = is_anomaly & mask_flat.any(1)
    if not has.any():
        return s_flat.new_zeros(())
    p = _sigmoid_score(s_flat[has])
    m = mask_flat[has].float()
    tp = (p * m).sum(1)
    fp = (p * (1.0 - m)).sum(1)
    fn = ((1.0 - p) * m).sum(1)
    tversky = tp / (tp + alpha * fp + beta * fn + 1e-6)
    return (1.0 - tversky).mean()


def contrast_loss(z, pack, c_anom, c_norm, mask_flat, is_anomaly, scale=20.0):
    """病灶内 z' 拉近异常文本锚点与 c_anom，拉远正常锚点；病灶外相反。"""
    if pack is None or "t_lesion" not in pack:
        return z.new_zeros(())
    f = F.normalize(z.float(), dim=-1)
    t_n = F.normalize(pack["t_lesion"][0].float(), dim=-1)
    t_a = F.normalize(pack["t_lesion"][1].float(), dim=-1)
    from .prototypes import as_dk

    c_a = as_dk(c_anom).float()
    c_n = as_dk(c_norm).float()
    c_a = F.normalize(c_a, dim=0)
    c_n = F.normalize(c_n, dim=0)
    logits_t = scale * torch.stack([f @ t_n, f @ t_a], dim=-1)
    sim_a = (f @ c_a).max(dim=-1).values
    sim_n = (f @ c_n).max(dim=-1).values
    logits_p = scale * torch.stack([sim_n, sim_a], dim=-1)

    def _nll(logits, target):
        logp = F.log_softmax(logits, dim=-1)
        return -logp.gather(-1, target.long().unsqueeze(-1)).squeeze(-1)

    parts = []
    is_norm = ~is_anomaly
    if is_norm.any():
        z0 = torch.zeros_like(logits_t[is_norm, ..., 0], dtype=torch.long)
        parts.append(0.5 * (_nll(logits_t[is_norm], z0).mean() + _nll(logits_p[is_norm], z0).mean()))
    has = is_anomaly & mask_flat.any(1)
    if has.any():
        lt, lp, ma = logits_t[has], logits_p[has], mask_flat[has]
        one = torch.ones_like(lt[..., 0], dtype=torch.long)
        zero = torch.zeros_like(lt[..., 0], dtype=torch.long)
        nll_t_a = _nll(lt, one)
        nll_t_n = _nll(lt, zero)
        nll_p_a = _nll(lp, one)
        nll_p_n = _nll(lp, zero)
        inside = ((nll_t_a + nll_p_a) * ma).sum(1) / ma.sum(1).clamp(min=1)
        outside = ((nll_t_n + nll_p_n) * ~ma).sum(1) / (~ma).sum(1).clamp(min=1)
        parts.append(0.5 * (inside.mean() + outside.mean()))
    if not parts:
        return z.new_zeros(())
    return sum(parts) / len(parts)


def _zscore_map(hm):
    return (hm - hm.mean(dim=(1, 2), keepdim=True)) / (hm.std(dim=(1, 2), keepdim=True) + 1e-8)


def consistency_loss(layer_maps, hm_proto, hm_text=None):
    """四层热图两两一致 + A_proto 与冻结 A_text 一致。layer_maps: (L,N,H,H)。"""
    terms = []
    n_l = int(layer_maps.shape[0])
    for i in range(n_l):
        zi = _zscore_map(layer_maps[i])
        for j in range(i + 1, n_l):
            terms.append(F.mse_loss(zi, _zscore_map(layer_maps[j])))
    if hm_text is not None:
        terms.append(F.mse_loss(_zscore_map(hm_proto), _zscore_map(hm_text.detach())))
    if not terms:
        return hm_proto.new_zeros(())
    return sum(terms) / len(terms)

