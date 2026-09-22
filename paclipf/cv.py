"""症例留一:支持集超参不得看过该病例的原型 / Memory / 适配器。

val 在 support44 下只有正常切片,不能当验证集。每一折 hold 一个 Ungood 病例:
  1. 用其余支持集重算原型和 Memory(40 正常 + 非 held-out 异常)
  2. 只在 held-out 异常 + leftover valid 正常(约 39 张)上打分
  3. 11 折汇总后再定 epoch / 融合权重 / α

leftover valid 正常只打分,不进原型、Memory、训练。few-shot 那 40 张正常
是建库样本,不能再当 CV 负例(会把 max_k sim(p, c_norm) 抬高,AUC 偏乐观)。
"""
from pathlib import Path

import numpy as np
import torch

from .fusion import to_np


def case_id_of(path):
    return Path(path).name.split("_")[0]


def uses_case_cv(ctx):
    """val 缺一类时必须走症例 CV,不能拿支持集自己评自己。"""
    y = ctx["val"]["label"]
    return int((y == 0).sum()) == 0 or int((y == 1).sum()) == 0


def few_shot_idx(cfg, ctx, device=None):
    from . import pipeline

    idx = pipeline.few_shot_indices(cfg, ctx["entries"][0], ctx["pool"])
    t = torch.as_tensor(idx, dtype=torch.long)
    return t.to(device) if device is not None else t


def train_idx_for_scope(cfg, ctx, device, scope=None):
    """训练下标:few_shot → 40+44;full → 整个 pool。"""
    scope = scope or cfg.train_scope
    if scope == "few_shot":
        return few_shot_idx(cfg, ctx, device=device)
    n = len(ctx["pool"]["label"])
    return torch.arange(n, device=device)


def _pool_case_ids(ctx, indices):
    paths = ctx["pool"]["manifest"]["image_paths"]
    if torch.is_tensor(indices):
        indices = indices.detach().cpu().tolist()
    return [case_id_of(paths[int(i)]) for i in indices]


def iter_folds(cfg, ctx):
    """yield (held_case, train_idx_cpu, held_idx_cpu)。只折 Ungood 病例。

    train_idx = 40 正常 + 其余异常;held_idx = 该病例全部切片。
    """
    idx = few_shot_idx(cfg, ctx).cpu()
    y = ctx["pool"]["label"]
    cids = _pool_case_ids(ctx, idx)
    labels = [int(y[int(i)]) for i in idx.tolist()]
    anom_cases = sorted({c for c, lab in zip(cids, labels) if lab == 1})
    for held in anom_cases:
        train, held_i = [], []
        for i, c, lab in zip(idx.tolist(), cids, labels):
            if lab == 1 and c == held:
                held_i.append(i)
            else:
                train.append(i)
        yield (
            held,
            torch.tensor(train, dtype=torch.long),
            torch.tensor(held_i, dtype=torch.long),
        )


def pool_mmap(cfg, ctx):
    """折内复用四层 memmap,避免每次 pool_split / 重建都重新 np.load。"""
    need = tuple(cfg.patch_layers)
    mm = ctx.get("_mmap")
    if mm is None or any(l not in mm for l in need):
        from . import cache

        extra = cache.pool_patch_mmap(cfg, need, verbose=False)
        mm = dict(mm or {})
        mm.update(extra)
        ctx["_mmap"] = mm
    return mm


def pool_split(cfg, ctx, indices, device=None):
    """从 pool 按下标切出一个 compute() 能吃的 split(含四层 patch)。"""
    device = device or ctx["device"]
    layers = tuple(cfg.patch_layers)
    mmap = pool_mmap(cfg, ctx)
    bi = indices.detach().cpu().numpy() if torch.is_tensor(indices) else np.asarray(indices)
    pool = ctx["pool"]
    paths = pool["manifest"]["image_paths"]
    return {
        "cls": pool["cls"][bi].to(device),
        "patch": {
            l: torch.as_tensor(np.asarray(mmap[l][bi])).to(device, torch.float32)
            for l in layers
        },
        "label": pool["label"][bi].to(device),
        "mask14": pool["mask14"][bi].to(device),
        "has_mask": pool["has_mask"][bi].to(device),
        "case_ids": [case_id_of(paths[int(i)]) for i in bi],
        "pool_idx": np.asarray(bi, dtype=np.int64),
    }


def leftover_norm_split(ctx, device=None):
    """官方 leftover valid 正常(support44 下 ctx['val'] 的 39 张 good)。

    只打分,不进原型 / Memory / 训练。val 里若混有异常,只取 label=0。
    """
    device = device or ctx["device"]
    val = ctx["val"]
    y = val["label"]
    sel = (y == 0).nonzero(as_tuple=True)[0]
    if torch.is_tensor(sel):
        sel = sel.detach().cpu()
    if len(sel) == 0:
        return None
    paths = (val.get("manifest") or {}).get("image_paths") or []
    case_ids = [
        case_id_of(paths[int(i)]) if int(i) < len(paths) else "val_good"
        for i in sel.tolist()
    ]
    patch = val.get("patch") or {}
    return {
        "cls": val["cls"][sel].to(device),
        "patch": {l: t[sel].to(device) for l, t in patch.items()},
        "label": val["label"][sel].to(device),
        "mask14": val["mask14"][sel].to(device),
        "has_mask": val["has_mask"][sel].to(device),
        "case_ids": case_ids,
        "pool_idx": np.full(len(sel), -1, dtype=np.int64),
    }


def concat_splits(parts):
    """按样本维拼接 compute() 用的 split。"""
    parts = [p for p in parts if p is not None]
    if not parts:
        raise RuntimeError("concat_splits: 没有可拼的 split")
    if len(parts) == 1:
        return parts[0]
    out = {}
    for k in ("cls", "label", "mask14", "has_mask"):
        out[k] = torch.cat([p[k] for p in parts], 0)
    layers = parts[0]["patch"].keys()
    out["patch"] = {l: torch.cat([p["patch"][l] for p in parts], 0) for l in layers}
    out["case_ids"] = sum((list(p.get("case_ids") or ["?"] * len(p["label"])) for p in parts), [])
    out["pool_idx"] = np.concatenate([
        np.asarray(p.get("pool_idx", np.full(len(p["label"]), -1)), dtype=np.int64)
        for p in parts
    ])
    return out


def fold_eval_split(cfg, ctx, held_idx, device=None):
    """held-out 异常(pool) + leftover valid 正常。返回 (split, n_held)。

    leftover 缺失时退回 few-shot 正常,并打警告(旧口径,AUC 会偏乐观)。
    """
    device = device or ctx["device"]
    held = pool_split(cfg, ctx, held_idx, device)
    n_held = int(held_idx.numel() if torch.is_tensor(held_idx) else len(held_idx))
    leftover = leftover_norm_split(ctx, device)
    if leftover is None:
        fs = few_shot_idx(cfg, ctx)
        y = ctx["pool"]["label"]
        norms = fs[y[fs] == 0]
        print("[cv] 警告: val 没有 leftover 正常,CV 负例退回 few-shot 正常(AUC 会偏乐观)")
        leftover = pool_split(cfg, ctx, norms, device)
    else:
        missing = [l for l in tuple(cfg.patch_layers) if l not in leftover["patch"]]
        if missing:
            raise RuntimeError(
                f"leftover val 缺少 patch 层 {missing},pipeline.prepare() 必须加载全部 patch_layers")
        if not ctx.get("_cv_neg_logged"):
            print(f"[cv] OOF 负例: leftover valid 正常 {len(leftover['label'])} 张 "
                  f"(不进原型 / Memory / 训练)")
            ctx["_cv_neg_logged"] = True
    return concat_splits([held, leftover]), n_held


def fold_eval_idx(ctx, train_idx, held_idx):
    """已废弃:负例改在 leftover val 上,不再是 pool 下标。请用 fold_eval_split。"""
    raise RuntimeError(
        "fold_eval_idx 已废弃:CV 负例是 leftover valid 正常,请改用 fold_eval_split"
    )


def _idx_key(idx):
    a = idx.detach().cpu().tolist() if torch.is_tensor(idx) else list(idx)
    return tuple(int(i) for i in a)


def rebuild_proto(cfg, ctx, train_idx, device=None):
    """折内重算原型。train_idx 不得含 held-out 病例。同 train_idx 命中缓存。"""
    from . import prototypes as PR

    device = device or ctx["device"]
    key = _idx_key(train_idx)
    cached = (ctx.setdefault("_proto_by_train", {})).get(key)
    if cached is not None:
        return {
            "c_norm": cached["c_norm"].to(device),
            "c_anom": cached["c_anom"].to(device),
            "pi": cached["pi"].to(device),
            "meta": cached.get("meta"),
        }
    sel = train_idx.detach().cpu() if torch.is_tensor(train_idx) else torch.as_tensor(train_idx)
    pool = ctx["pool"]
    layers = tuple(cfg.patch_layers)
    per_layer = bool(getattr(cfg, "proto_per_layer", True))
    mmap = pool_mmap(cfg, ctx) if per_layer or layers[-1] not in (pool.get("patch") or {}) else None
    bi = sel.numpy() if hasattr(sel, "numpy") else np.asarray(sel)
    if per_layer:
        patch_dict = {l: torch.as_tensor(np.asarray(mmap[l][bi])) for l in layers}
    elif layers[-1] in (pool.get("patch") or {}):
        patch_dict = {layers[-1]: pool["patch"][layers[-1]][sel]}
    else:
        patch_dict = {layers[-1]: torch.as_tensor(np.asarray(mmap[layers[-1]][bi]))}
    common = dict(
        mask14=pool["mask14"][sel],
        label=pool["label"][sel],
        has_mask=pool["has_mask"][sel],
        k=cfg.normal_proto_k,
        k_anom=int(getattr(cfg, "anomaly_proto_k", 6)),
        double_norm=cfg.double_norm,
        seed=getattr(cfg, "seed", 111),
        n_init=getattr(cfg, "kmeans_n_init", 10),
        include_lesion_outside=getattr(cfg, "normal_pool_include_lesion_outside", False),
    )
    if per_layer:
        pr = PR.build_layered(patch_dict, layers, **common)
    else:
        pr = PR.build_from_features(patch_dict[layers[-1]], **common)
    ctx["_proto_by_train"][key] = {
        "c_norm": pr["c_norm"].detach().cpu(),
        "c_anom": pr["c_anom"].detach().cpu(),
        "pi": pr["pi"].detach().cpu(),
        "meta": pr.get("meta"),
    }
    pr["c_norm"] = pr["c_norm"].to(device)
    pr["c_anom"] = pr["c_anom"].to(device)
    pr["pi"] = pr["pi"].to(device)
    return pr


def rebuild_memory(cfg, ctx, train_idx, device=None):
    """折内 Memory:不得含 held-out 病灶 patch。同 train_idx 命中缓存。带网格坐标。"""
    import torch.nn.functional as F
    from .grid import patch_xy

    device = device or ctx["device"]
    key = _idx_key(train_idx)
    cached = (ctx.setdefault("_mem_by_train", {})).get(key)
    if cached is not None:
        out = {k: cached[k].to(device) for k in cached}
        return out
    sel = train_idx.detach().cpu() if torch.is_tensor(train_idx) else torch.as_tensor(train_idx)
    pool = ctx["pool"]
    l11 = tuple(cfg.patch_layers)[-1]
    if l11 in (pool.get("patch") or {}):
        patch = pool["patch"][l11][sel]
    else:
        mmap = pool_mmap(cfg, ctx)
        patch = torch.as_tensor(np.asarray(mmap[l11][sel.numpy()]))
    n, l, d = patch.shape
    H = int(l ** 0.5)
    xy = patch_xy(H, device=patch.device, dtype=torch.float32)
    mask = pool["mask14"][sel].reshape(len(sel), -1)
    y = pool["label"][sel]
    has = pool["has_mask"][sel]
    valid = (y == 1) & has & mask.any(1)
    if not valid.any():
        raise RuntimeError("折内 Memory Bank 没有非空 mask 的异常切片")
    a_feat = patch[valid][mask[valid].bool()].float()
    a_pos = xy.unsqueeze(0).expand(int(valid.sum()), -1, -1)[mask[valid].bool()]
    a_mem = F.normalize(a_feat, dim=-1)
    n_sel = y == 0
    n_feat = patch[n_sel].reshape(-1, d).float()
    n_pos = xy.unsqueeze(0).expand(int(n_sel.sum()), -1, -1).reshape(-1, 2)
    n_mem = F.normalize(n_feat, dim=-1)
    packed = {
        "a": a_mem.detach().cpu(), "n": n_mem.detach().cpu(),
        "a_pos": a_pos.detach().cpu(), "n_pos": n_pos.detach().cpu(),
    }
    ctx["_mem_by_train"][key] = packed
    return {k: v.to(device) for k, v in packed.items()}


@torch.no_grad()
def fold_signals(cfg, ctx, train_idx, held_idx, adapter_img=None, adapter_lesion=None,
                 text_pack=None, c_norm=None, c_anom=None, memory=None,
                 patch_adapter=None, spatial_gate=None):
    """折内:重算(或传入)原型/Memory,在 held-out+正常 上算信号。fuse_w 不进,留给后续搜。"""
    from . import signals as SG

    device = ctx["device"]
    pr = None
    if c_norm is None or c_anom is None:
        pr = rebuild_proto(cfg, ctx, train_idx, device)
        c_norm, c_anom = pr["c_norm"], pr["c_anom"]
    if memory is None and getattr(cfg, "use_memory_bank", False):
        memory = rebuild_memory(cfg, ctx, train_idx, device)
    split, n_held = fold_eval_split(cfg, ctx, held_idx, device)
    pack = ctx.get("text_pack") if text_pack is None else text_pack
    sig = SG.compute(
        cfg, split, c_norm, c_anom, ctx["w_text"] if pack is None else
        _w_text(pack, ctx),
        adapter_img=adapter_img, adapter_lesion=adapter_lesion,
        text_pack=pack, memory=memory, fuse_w=None,
        text_sign=ctx.get("text_sign", 1.0),
        patch_adapter=patch_adapter if patch_adapter is not None else ctx.get("patch_adapter"),
        spatial_gate=spatial_gate,
    )
    return {
        "sig": sig, "split": split, "n_held": n_held,
        "c_norm": c_norm, "c_anom": c_anom, "memory": memory, "pr": pr,
    }


def _w_text(pack, ctx):
    from . import text_adapter as TA

    if pack is None:
        return ctx["w_text"]
    return TA.w_text_from_pack(pack)


def collect_oof(cfg, ctx, adapter_img=None, adapter_lesion=None, text_pack=None,
                per_fold=None):
    """11 折 OOF 信号。per_fold(held, train_idx, held_idx) -> 覆盖 dict(可选)。"""
    rows = []
    for held, tr, ho in iter_folds(cfg, ctx):
        extra = {}
        if per_fold is not None:
            extra = per_fold(held, tr, ho) or {}
        rec = fold_signals(
            cfg, ctx, tr.to(ctx["device"]), ho.to(ctx["device"]),
            adapter_img=extra.get("adapter_img", adapter_img),
            adapter_lesion=extra.get("adapter_lesion", adapter_lesion),
            text_pack=extra.get("text_pack", text_pack),
            c_norm=extra.get("c_norm"),
            c_anom=extra.get("c_anom"),
            memory=extra.get("memory"),
            patch_adapter=extra.get("patch_adapter"),
        )
        rec["held"] = held
        rec["train_idx"] = tr
        rec["held_idx"] = ho
        rows.append(rec)
    return rows


def oof_anomaly_maps(records):
    """拼 OOF 异常切片的三路热力图 + mask + case_id(每张只出现一次)。"""
    maps, masks, cases = {}, [], []
    for rec in records:
        n = rec["n_held"]
        sig, split = rec["sig"], rec["split"]
        item = {"proto": sig["hm"][:n]}
        if "hm_text" in sig:
            item["text"] = sig["hm_text"][:n]
        if "hm_mem" in sig:
            item["mem"] = sig["hm_mem"][:n]
        for k, v in item.items():
            maps.setdefault(k, []).append(v)
        masks.append(split["mask14"][:n].reshape(n, -1))
        cases.extend(split["case_ids"][:n])
    maps = {k: torch.cat(vs, 0) for k, vs in maps.items()}
    mask = torch.cat(masks, 0)
    return maps, mask, cases


def fused_oof_hit(records, metric="hit@1_valid"):
    """部署图（hm_fused）上的症例平均 hit，与 search_map_weights 同一口径。"""
    from . import diag

    flats, masks, cases_all = [], [], []
    for rec in records:
        n = rec["n_held"]
        sig, split = rec["sig"], rec["split"]
        hm = sig.get("hm_fused", sig["hm"])[:n]
        flats.append(hm.reshape(n, -1))
        masks.append(split["mask14"][:n].reshape(n, -1))
        cases_all.extend(split["case_ids"][:n])
    if not flats:
        return float("nan")
    s = torch.cat(flats, 0)
    m = torch.cat(masks, 0)
    vals = []
    for cid in sorted(set(cases_all)):
        idx = [i for i, c in enumerate(cases_all) if c == cid]
        vals.append(diag.hit_rate(s[idx], m[idx])[metric])
    return float(sum(vals) / max(1, len(vals)))


def oof_held_heatmaps(records, key="hm_fused"):
    parts = []
    for rec in records:
        n = rec["n_held"]
        sig = rec["sig"]
        parts.append(sig.get(key, sig["hm"])[:n])
    return torch.cat(parts, 0) if parts else None


def oof_held_entries(ctx, records):
    """OOF held-out 切片对应的 pool entries（给像素 GT）。"""
    by_path = {e["image_path"]: e for e in ctx["entries"][0]}
    paths = ctx["pool"]["manifest"]["image_paths"]
    out = []
    for rec in records:
        n = rec["n_held"]
        for i in rec["split"]["pool_idx"][:n]:
            i = int(i)
            if i < 0:
                out.append({"mask_path": None})
                continue
            out.append(by_path.get(paths[i], {"mask_path": None}))
    return out


def search_map_oof(cfg, records, metric="hit@1_valid"):
    from . import fusion as FU
    from .grid import nested

    maps, mask, cases = oof_anomaly_maps(records)
    grid = list(getattr(cfg, "heatmap_fuse_grid", [0.0, 0.25, 0.5, 0.75, 1.0]))
    min_w = dict(getattr(cfg, "heatmap_min_w", None) or {})
    objective = str(nested(cfg, "heatmap", "fuse_objective",
                           getattr(cfg, "heatmap_fuse_objective", "hit_ap")))
    w, score, table = FU.search_map_weights(
        maps, mask, cases, grid=grid, metric=metric, min_w=min_w, objective=objective)
    return w, score, table


def search_alpha_oof(records, heat_key, text_key="S_text", grid=None):
    """逐折图像 AUC 再平均。leftover 负例每折相同,拼接 AUC 会被负例重复主导。"""
    from . import fusion as FU
    from paclipf import metrics as pmetrics

    if grid is None:
        grid = np.arange(0.0, 1.05, 0.05)
    best_a, best = 0.0, -1.0
    curve = []
    for a in grid:
        aucs = []
        for rec in records:
            sig, y = rec["sig"], rec["split"]["label"]
            if heat_key not in sig:
                continue
            h = FU.zscore_apply(to_np(sig[heat_key]), *FU.zscore_fit(to_np(sig[heat_key])))
            t = FU.zscore_apply(to_np(sig[text_key]), *FU.zscore_fit(to_np(sig[text_key])))
            s = a * h + (1.0 - a) * t
            try:
                aucs.append(pmetrics.image_metrics(s, to_np(y))[0])
            except Exception:
                continue
        if not aucs:
            continue
        m = float(np.nanmean(aucs))
        curve.append((float(a), m))
        if m > best:
            best, best_a = m, float(a)
    return best_a, best, curve


def search_three_oof(records, heat_key, text_key="S_text", adapt_key="S_adapt_margin",
                     step=0.1):
    """OOF 上搜 z(热力图)+z(文本头)+z(适配器) 的图像 AUC。逐折平均,避免 leftover 负例重复。"""
    from . import fusion as FU
    from paclipf import metrics as pmetrics

    grid = np.arange(0.0, 1.0001, step)
    best_w, best, curve = (0.0, 1.0, 0.0), -1.0, []
    for wh in grid:
        for wt in grid:
            wa = 1.0 - float(wh) - float(wt)
            if wa < -1e-9:
                continue
            aucs = []
            for rec in records:
                sig, y = rec["sig"], rec["split"]["label"]
                if heat_key not in sig or text_key not in sig or adapt_key not in sig:
                    continue
                h = FU.zscore_apply(to_np(sig[heat_key]), *FU.zscore_fit(to_np(sig[heat_key])))
                t = FU.zscore_apply(to_np(sig[text_key]), *FU.zscore_fit(to_np(sig[text_key])))
                a = FU.zscore_apply(to_np(sig[adapt_key]), *FU.zscore_fit(to_np(sig[adapt_key])))
                s = float(wh) * h + float(wt) * t + float(wa) * a
                try:
                    aucs.append(pmetrics.image_metrics(s, to_np(y))[0])
                except Exception:
                    continue
            if not aucs:
                continue
            m = float(np.nanmean(aucs))
            curve.append((float(wh), float(wt), float(wa), m))
            if m > best:
                best, best_w = m, (float(wh), float(wt), float(wa))
    return best_w, best, curve


def search_fusion_oof(cfg, records, acc_fn):
    """OOF 上搜 alpha_pt × lam_pt,目标分类准确率(折平均)。"""
    from . import signals as SG

    a_grid = list(cfg.alpha_pt_grid)
    l_grid = list(cfg.lam_pt_grid)
    scale = float(getattr(cfg, "adapter_logit_scale", 100.0))
    best = (-1.0, a_grid[0], l_grid[0])
    rows = []
    for a in a_grid:
        for l in l_grid:
            accs = []
            for rec in records:
                sig, y = rec["sig"], rec["split"]["label"]
                if "proto_img" not in sig:
                    continue
                logits = SG.fused_logits(sig, a, l, adapter_scale=scale)
                accs.append(acc_fn(logits, y))
            if not accs:
                continue
            m = float(np.mean(accs))
            rows.append((float(a), float(l), m))
            if m > best[0]:
                best = (m, float(a), float(l))
    acc, a, l = best
    edge = a in (a_grid[0], a_grid[-1]) or l in (l_grid[0], l_grid[-1])
    return a, l, acc, rows, edge


def add_fused_scores(cfg, records, fuse_w):
    from . import signals as SG

    q = max(1, int(getattr(cfg, "q_img", 3)))
    for rec in records:
        sig = rec["sig"]
        maps = {"proto": sig["hm"]}
        if "hm_text" in sig:
            maps["text"] = sig["hm_text"]
        if "hm_mem" in sig:
            maps["mem"] = sig["hm_mem"]
        hm_f = SG.fuse_maps(maps, fuse_w)
        sig["hm_fused"] = hm_f
        flat = hm_f.reshape(hm_f.shape[0], -1)
        sig["s_flat_fused"] = flat
        sig["S_heat_fused"] = flat.topk(q, dim=-1).values.mean(-1)


def search_hparams_oof(cfg, ctx, adapter_img=None, adapter_lesion=None, text_pack=None,
                       per_fold=None, verbose=True):
    """一次 OOF:融合权重 + alpha_fuse(+ 若有适配器则 alpha_pt)。"""
    records = collect_oof(
        cfg, ctx, adapter_img=adapter_img, adapter_lesion=adapter_lesion,
        text_pack=text_pack, per_fold=per_fold,
    )
    fuse_w, hit, table = search_map_oof(cfg, records)
    add_fused_scores(cfg, records, fuse_w)
    heat_key = "S_heat_fused" if "S_heat_fused" in records[0]["sig"] else "S_heat_raw"
    a_f, auc_f, _ = search_alpha_oof(records, heat_key)
    hp = {"fuse_w": fuse_w, "fuse_alpha": float(a_f), "oof_hit": hit,
          "oof_img_auc": auc_f, "table": table}
    if adapter_img is not None or any("proto_img" in r["sig"] for r in records):
        from paclipf import metrics as pmetrics

        a_pt, l_pt, acc, _, edge = search_fusion_oof(cfg, records, pmetrics.top1_acc)
        hp.update({"alpha_pt": a_pt, "lam_pt": l_pt, "oof_acc": acc, "edge_warning": edge})
        if any("S_adapt_margin" in r["sig"] for r in records):
            w3, auc3, _ = search_three_oof(records, heat_key)
            hp.update({"w_heat": w3[0], "w_text": w3[1], "w_adapt": w3[2],
                       "oof_img_auc": float(auc3)})
    if verbose:
        print(f"[cv] OOF fuse_w {fuse_w}  hit@1={hit:.4f}  alpha_fuse={a_f:.3f}  AUC={hp['oof_img_auc']:.4f}")
        if "alpha_pt" in hp:
            print(f"[cv] OOF α_pt={hp['alpha_pt']} λ_pt={hp['lam_pt']}  acc={hp['oof_acc']:.2f}")
        if "w_adapt" in hp:
            print(f"[cv] OOF 图像分 w_heat={hp['w_heat']:.2f} w_text={hp['w_text']:.2f} "
                  f"w_adapt={hp['w_adapt']:.2f}")
    return hp, records
