"""超参搜索 —— support44 下在症例 OOF 上做,test 只在最终评估时用一次。
(val 同时含两类的旧切分仍可走本模块的 val 搜索。)

两条搜索线:
  图像级异常分  alpha_fuse :  α·z(S_heat) + (1−α)·z(S_text)   目标 图像 AUC
  分类 logits   alpha_pt / lam_pt : clip + α·(proto_img + λ·proto_les)  目标 准确率

**每次搜索都返回整条曲线**,而不是只返回最优值。原因:每折只有约 4 张异常,
搜 2-4 个系数会有可观的过拟合,必须看得见"最优点是不是一根尖峰"。
"""
import numpy as np
from sklearn.metrics import roc_auc_score

def zscore_fit(values):
    """values: (N,) numpy -> (mean, std). Statistics come from the VALIDATION split."""
    v = np.asarray(values, dtype=np.float32)
    return float(v.mean()), float(v.std()) + 1e-8


def zscore_apply(values, mean, std):
    """Apply validation-set z-score statistics to any split (no test leakage)."""
    return (np.asarray(values, dtype=np.float32) - mean) / std


def to_np(x):
    """统一的张量→numpy。调用方常持有 CUDA 张量,np.asarray 会直接报错。"""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def search_alpha_fuse(s_heat, s_text, labels, grid=np.arange(0.0, 1.05, 0.05)):
    """返回 (best_alpha, best_auc, curve)。与 pa_clip.fusion.search_alpha 同构。"""
    s_heat, s_text = to_np(s_heat), to_np(s_text)
    h = zscore_apply(s_heat, *zscore_fit(s_heat))
    t = zscore_apply(s_text, *zscore_fit(s_text))
    curve = []
    best_a, best_auc = 0.0, -1.0
    for a in grid:
        s = a * h + (1.0 - a) * t
        try:
            auc = roc_auc_score(labels, s)
        except ValueError:
            continue
        curve.append((float(a), float(auc)))
        if auc > best_auc:
            best_auc, best_a = auc, float(a)
    return best_a, best_auc, curve


def search_fusion(cfg, sig, labels, acc_fn):
    """在 val 上搜 alpha_pt × lam_pt,目标为分类准确率。

    返回 (best_alpha, best_lam, best_acc, grid_rows, edge_warn)
    edge_warn 在最优值落在网格边界时为 True —— 说明该扩网格。
    """
    from . import signals as SG

    a_grid = list(cfg.alpha_pt_grid)
    l_grid = list(cfg.lam_pt_grid)
    rows = []
    best = (-1.0, a_grid[0], l_grid[0])
    for a in a_grid:
        for l in l_grid:
            logits = SG.fused_logits(sig, a, l, adapter_scale=float(
                getattr(cfg, "adapter_logit_scale", 100.0)))
            acc = acc_fn(logits, labels)
            rows.append((float(a), float(l), float(acc)))
            if acc > best[0]:
                best = (float(acc), float(a), float(l))

    acc, a, l = best
    edge = a in (a_grid[0], a_grid[-1]) or l in (l_grid[0], l_grid[-1])
    return a, l, acc, rows, edge


def best_row(rows, key):
    return max(rows, key=lambda r: r[key])


def search_lambda(logits_img, logits_lesion, labels, grid=np.arange(0.0, 2.05, 0.25)):
    """Grid-search lambda for the classification fusion (image + lesion logits),
    maximizing accuracy on the VALIDATION split. Returns (best_lambda, best_acc)."""
    import torch
    acc_best, lam_best = -1.0, 0.0
    for lam in grid:
        logits = logits_img + lam * logits_lesion
        acc = (logits.argmax(1) == labels).float().mean().item()
        if acc > acc_best:
            acc_best, lam_best = acc, lam
    return lam_best, acc_best


def search_map_weights(maps, mask_flat, case_ids, grid=None, metric="hit@1_valid",
                       min_w=None):
    """OOF 异常切片上按症例平均命中率搜热力图融合权重。

    默认 hit@1:support 上 hit@3 容易饱和,选不出权重。
    min_w: {name: 下限},例如 mem≥0.25,避免 Stage2 把 Memory 搜成 0。
    """
    from . import diag
    from . import signals as SG

    if grid is None:
        grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    min_w = dict(min_w or {})
    names = [k for k in ("proto", "text", "mem") if k in maps]
    zs = {}
    for k in names:
        x = maps[k].float()
        if x.dim() == 2:
            g = int(x.shape[-1] ** 0.5)
            x = x.view(x.shape[0], g, g)
        zs[k] = SG._zscore(x)

    cases = sorted(set(case_ids))
    best = (-1.0, {n: 0.0 for n in names}, [])
    table = []

    def _combos():
        if len(names) == 1:
            yield (1.0,)
            return
        if len(names) == 2:
            for a in grid:
                yield (a, 1.0 - a)
            return
        for a in grid:
            for b in grid:
                c = 1.0 - a - b
                if c < -1e-9:
                    continue
                yield (a, b, float(max(c, 0.0)))

    def _ok(w):
        for k, lo in min_w.items():
            if k in w and w[k] < float(lo) - 1e-9:
                return False
        return True

    def _eval_combos(enforce_min):
        local_best = (-1.0, {n: 0.0 for n in names}, [])
        local_table = []
        for combo in _combos():
            w = {n: float(c) for n, c in zip(names, combo)}
            if enforce_min and not _ok(w):
                continue
            acc = None
            for n, wv in w.items():
                acc = zs[n] * wv if acc is None else acc + zs[n] * wv
            s_flat = acc.reshape(acc.shape[0], -1)
            hits = []
            for cid in cases:
                idx = [i for i, c in enumerate(case_ids) if c == cid]
                if not idx:
                    continue
                h = diag.hit_rate(s_flat[idx], mask_flat[idx])
                hits.append(h[metric])
            mean_h = float(sum(hits) / max(1, len(hits)))
            row = {**w, metric: mean_h}
            local_table.append(row)
            if mean_h > local_best[0]:
                local_best = (mean_h, w, local_table)
        return local_best, local_table

    best, table = _eval_combos(bool(min_w))
    if best[0] < 0 and min_w:
        best, table = _eval_combos(False)
    return best[1], best[0], table
