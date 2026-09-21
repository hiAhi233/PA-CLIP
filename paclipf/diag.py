"""定位诊断指标 —— 判断 Stage 2 是否有效的关键判据。

背景:像素 AUC/AP 可以在"峰值完全没落在病灶上"时依然很高
(把整片软组织都打高分即可)。所以需要一个直接度量"病灶有没有被排到最前"的指标。

一个容易搞混的点:逐图 z-score 是**正仿射变换** (s-μ)/σ,σ>0,
所以它**不改变图内的任何排序** —— 命中率、病灶在软组织中的百分位在 s 和 z 上完全相同。
只有"以 σ 为单位的对比度"必须用 z 表达。本模块据此选择:命中率用原始 s,
对比度用 z。
"""
import numpy as np
import torch


def _align(s_flat, mask_flat):
    """把 mask 对齐到分数的设备/类型。

    diag 的调用方来源多样(缓存在 CPU、训练在 GPU、有时是 numpy),
    在这一处统一,好过在每个调用点各修一次 —— 已经漏过两次。
    """
    if not torch.is_tensor(s_flat):
        s_flat = torch.as_tensor(s_flat)
    m = mask_flat if torch.is_tensor(mask_flat) else torch.as_tensor(mask_flat)
    return s_flat, m.to(device=s_flat.device, dtype=torch.bool)


def _zscore(s):
    """(N,196) → 逐图标准化。仅用于需要 σ 单位的量。"""
    return (s - s.mean(-1, keepdim=True)) / (s.std(-1, keepdim=True) + 1e-8)


def hit_rate(s_flat, mask_flat, ks=(1, 3, 10)):
    """峰值命中率。

    s_flat    : (Na, 196) 异常切片的分数
    mask_flat : (Na, 196) bool

    返回 dict,每个 k 给两个口径:
      valid  —— 分母只算 14x14 下 mask 非空的切片(定位问题的有效子集)
      strict —— 分母算全部异常切片,mask 为空者记为 miss
    两个口径都必须报:实测 15.7% 的异常切片 mask 在 14x14 下会完全消失,
    只报一个口径会让数字无法与其他实现对齐。
    """
    s_flat, mask_flat = _align(s_flat, mask_flat)
    ok = mask_flat.any(-1)
    out = {}
    for k in ks:
        idx = s_flat.topk(k, dim=-1).indices
        hit = mask_flat.gather(1, idx).any(-1)          # (Na,)
        out[f"hit@{k}_valid"] = hit[ok].float().mean().item() if ok.any() else float("nan")
        out[f"hit@{k}_strict"] = hit.float().mean().item()
        # 随机基线:病灶面积占比(理论上限的参照)
        out[f"hit@{k}_chance"] = (k * mask_flat.float().mean(-1)).clamp(max=1).mean().item()
    out["n_valid"] = int(ok.sum())
    out["n_total"] = int(len(ok))
    out["lesion_area_mean"] = mask_flat.float().mean(-1)[ok].mean().item() if ok.any() else float("nan")
    return out


def contrast(s_flat, mask_flat):
    """病灶 vs 背景的对比度。σ 单位,便于跨切片比较。"""
    s_flat, mask_flat = _align(s_flat, mask_flat)
    ok = mask_flat.any(-1)
    if not ok.any():
        return {"lesion_z": float("nan"), "background_z": float("nan"), "gap_sigma": float("nan")}
    z = _zscore(s_flat)
    zi = (z * mask_flat).sum(-1) / mask_flat.sum(-1).clamp(min=1)
    zo = (z * ~mask_flat).sum(-1) / (~mask_flat).sum(-1).clamp(min=1)
    return {
        "lesion_z": zi[ok].mean().item(),
        "background_z": zo[ok].mean().item(),
        "gap_sigma": (zi - zo)[ok].mean().item(),
    }


def peak_gap(s_flat, mask_flat):
    """max(病灶内) − max(病灶外)。比均值对比度更能预测命中率。"""
    s_flat, mask_flat = _align(s_flat, mask_flat)
    ok = mask_flat.any(-1)
    if not ok.any():
        return float("nan")
    z = _zscore(s_flat)
    zmax_out = z.masked_fill(mask_flat, float("-inf")).max(-1).values
    zmax_in = z.masked_fill(~mask_flat, float("-inf")).max(-1).values
    return (zmax_in - zmax_out)[ok].mean().item()


def proto_usage(patch_l11, c_norm):
    """每个正常原型在 max_k 里"抢到"多少 patch。检测死原型。

    若某原型占比接近 0,说明它在 max_k 里从不胜出 → 它的梯度也接近 0,
    此时应把 reduce 从 'max' 换成 'lse'(软化),或减少 k。
    """
    with torch.no_grad():
        k = c_norm.shape[1]
        sim = patch_l11 @ c_norm                                  # (N,196,K)
        win = sim.argmax(-1).reshape(-1)
        cnt = torch.bincount(win, minlength=k).float()
        return (cnt / cnt.sum()).tolist()


def summarize(name, s_flat, mask_flat=None):
    """打印一段诊断。mask_flat=None 时只报分数分布。"""
    lines = [f"[diag] {name}"]
    if mask_flat is not None:
        for k, v in hit_rate(s_flat, mask_flat).items():
            lines.append(f"  {k:<20} {v:.4f}" if isinstance(v, float) else f"  {k:<20} {v}")
        c = contrast(s_flat, mask_flat)
        lines.append(f"  lesion_z             {c['lesion_z']:+.3f}")
        lines.append(f"  background_z         {c['background_z']:+.3f}")
        lines.append(f"  gap_sigma            {c['gap_sigma']:+.3f}")
        lines.append(f"  peak_gap_z           {peak_gap(s_flat, mask_flat):+.3f}")
    return "\n".join(lines)


def bootstrap_diff(a, b, n_boot=2000, seed=111):
    """paired bootstrap:在同一批样本上比较两个指标序列的差值。

    比"比较两个独立 CI"灵敏得多 —— 独立比较需要差值 >0.04 才显著,
    配对差值通常小 3-5 倍。输入是两个等长的逐样本指标数组。
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, "paired bootstrap 要求两个数组等长"
    rng = np.random.RandomState(seed)
    n = len(a)
    d0 = a.mean() - b.mean()
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        boot[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"diff": d0, "lo": lo, "hi": hi, "significant": (lo > 0) or (hi < 0)}


def per_sample_hit(s_flat, mask_flat, k=1):
    """逐样本的 hit 指示(0/1),供 paired bootstrap 使用。"""
    s_flat, mask_flat = _align(s_flat, mask_flat)
    idx = s_flat.topk(k, dim=-1).indices
    return mask_flat.gather(1, idx).any(-1).float().cpu().numpy()
