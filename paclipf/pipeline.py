"""胶水层:把缓存、原型、文本锚点、信号组装成一次可复现的运行。

刻意与 cache.py / prototypes.py 分开:那两个只做纯计算,
这里负责"按配置决定用哪些样本、怎么组合",便于在报告里如实描述每一行。
"""
import json
from pathlib import Path

import torch

from paclipf import data as pdata
from paclipf import model as biomed
from paclipf import prototypes as pproto

from . import cache, prototypes as PR, signals as SG


def load_entries(cfg):
    """返回 (proto_entries, val_entries, test_entries),顺序与缓存一致。"""
    ent = pdata.load_entries(cfg.meta_path)
    return pdata.split_by_case(
        ent, cfg.val_cases, cfg.test_cases,
        brain_split=getattr(cfg, "brain_split", None),
    )


def few_shot_indices(cfg, proto_entries, subset_cache):
    """PA-CLIP 的 few-shot 子集在 pool 里的下标。

    复用 pa_clip.data.select_few_shot(seed=111)以与 PA-CLIP 逐位一致;
    用 image_path 对齐而不是假设顺序,避免缓存与源列表顺序不一致时静默取错样本。
    """
    fs = pdata.select_few_shot(
        proto_entries, cfg.few_shot_normal, cfg.few_shot_anomaly,
        seed=int(getattr(cfg, "seed", 111)),
    )
    want = {e["image_path"] for e in fs}
    paths = subset_cache["manifest"]["image_paths"]
    idx = [i for i, p in enumerate(paths) if p in want]
    if len(idx) != len(want):
        raise RuntimeError(f"few-shot 子集对齐失败:期望 {len(want)} 张,在 pool 里找到 {len(idx)} 张")
    return idx


def build_prototypes(cfg, pool, proto_entries, device="cpu"):
    """按 cfg.proto_source 构建原型。pool 必须是含 layer11 patch 的 pool 缓存。"""
    layers = tuple(cfg.patch_layers)
    per_layer = bool(getattr(cfg, "proto_per_layer", True))
    if cfg.proto_source == "few_shot":
        sel = torch.as_tensor(few_shot_indices(cfg, proto_entries, pool), dtype=torch.long)
        patch = {l: t[sel] for l, t in (pool.get("patch") or {}).items()}
        if per_layer and any(l not in patch for l in layers):
            import numpy as np
            mmap = cache.pool_patch_mmap(cfg, layers, verbose=False)
            bi = sel.detach().cpu().numpy()
            patch = {l: torch.as_tensor(np.asarray(mmap[l][bi])) for l in layers}
        sub = {
            "patch": patch,
            "mask14": pool["mask14"][sel],
            "label": pool["label"][sel],
            "has_mask": pool["has_mask"][sel],
        }
        scope = f"few_shot({cfg.few_shot_normal}+{cfg.few_shot_anomaly}, {len(sel)} 张)"
    else:
        sub = pool
        if per_layer and any(l not in (sub.get("patch") or {}) for l in layers):
            import numpy as np
            mmap = cache.pool_patch_mmap(cfg, layers, verbose=False)
            sub = dict(sub)
            sub["patch"] = {l: torch.as_tensor(np.asarray(mmap[l])) for l in layers}
        scope = f"full_pool({len(pool['label'])} 张)"

    layers = tuple(cfg.patch_layers)
    k_anom = int(getattr(cfg, "anomaly_proto_k", 6))
    per_layer = bool(getattr(cfg, "proto_per_layer", True))
    common = dict(
        mask14=sub["mask14"],
        label=sub["label"],
        has_mask=sub["has_mask"],
        k=cfg.normal_proto_k,
        k_anom=k_anom,
        double_norm=cfg.double_norm,
        seed=getattr(cfg, "seed", 111),
        n_init=getattr(cfg, "kmeans_n_init", 10),
        include_lesion_outside=getattr(cfg, "normal_pool_include_lesion_outside", False),
    )
    if per_layer:
        need = [l for l in layers if l not in sub["patch"]]
        if need:
            raise RuntimeError(f"分层原型需要 patch 层 {need}，prepare/rebuild 必须加载全部 patch_layers")
        res = PR.build_layered(sub["patch"], layers, **common)
    else:
        res = PR.build_from_features(sub["patch"][layers[-1]], **common)
    res["meta"]["proto_source"] = cfg.proto_source
    res["meta"]["scope"] = scope
    ka = res["meta"].get("k_anom", 1)
    lay = res["meta"].get("layers", [layers[-1]])
    print(f"[proto] 来源 {scope} | Kn={res['meta']['k']} Ka={ka} | "
          f"layers={lay} | double_norm={res['meta']['double_norm']}")
    print(
        f"[proto] 正常 patch {res['meta']['n_normal_patches']:,} | "
        f"异常 patch {res['meta']['n_anomaly_patches']:,} "
        f"(来自 {res['meta']['n_anomaly_slices_used']}/{res['meta']['n_anomaly_slices_total']} 张异常切片)"
    )
    if res["meta"]["n_anomaly_slices_mask_empty"]:
        pct = 100 * res["meta"]["n_anomaly_slices_mask_empty"] / max(1, res["meta"]["n_anomaly_slices_total"])
        print(f"[proto] 注意: {res['meta']['n_anomaly_slices_mask_empty']} 张异常切片的 14x14 mask 为空 ({pct:.1f}%),已排除")
    return res


def build_text_anchors(cfg, model, tokenizer, device):
    """返回 (w_text, text_pack0)。text_pack0 是冻结 t0;无分层提示词时 pack 为 None。"""
    from . import text_adapter as TA

    w_text, names = pproto.build_text_weights(tokenizer, model, cfg.class_templates, device=device)
    assert names.index("normal") == SG.IDX_NORMAL, f"类别顺序不符: {names}"
    print(f"[text] 基线锚点 {names} -> W_text {tuple(w_text.shape)}")

    prompts = getattr(cfg, "text_prompts", None)
    pack0 = None
    if prompts:
        pack0 = TA.encode_hierarchy(tokenizer, model, prompts, device=device)
        print(
            f"[text] 分层 t0 global {tuple(pack0['t_global'].shape)} "
            f"attr {pack0['t_attr_a'].shape[0]} lesion {tuple(pack0['t_lesion'].shape)}"
        )
    return w_text, pack0


@torch.no_grad()
def infer_text_sign(cfg, pool, pack, proto_entries, device="cpu", verbose=True):
    """支持集上 A_text 的 peak_gap<0 则翻转符号。冻结 lesion 提示词在脑 MRI 上常是反的。"""
    if pack is None or "t_lesion" not in pack:
        return 1.0
    from . import diag

    idx = few_shot_indices(cfg, proto_entries, pool)
    y = pool["label"]
    anom = [i for i in idx if int(y[i]) == 1]
    if not anom:
        return 1.0
    l11 = tuple(cfg.patch_layers)[-1]
    patch = pool["patch"][l11][anom].to(device)
    hm = SG.text_heatmap({l11: patch}, pack, (l11,), cfg.image_size,
                         attr_w=float(getattr(getattr(cfg, "text", None), "attr_w", 0.5)))
    mask = pool["mask14"][anom].reshape(len(anom), -1).to(hm.device)
    gap = diag.peak_gap(hm.reshape(len(anom), -1), mask)
    if gap != gap:
        return 1.0
    sign = -1.0 if float(gap) < 0.0 else 1.0
    if verbose:
        print(f"[text] 支持集 A_text peak_gap_z={float(gap):+.3f} → sign={int(sign)}")
    return sign


def prepare(cfg, device="cuda", need_pool_layers=False):
    """加载缓存 + 模型 + 原型 + 文本锚点。返回一个 context dict。"""
    model, preprocess, tokenizer = biomed.load_model(device)
    proto_entries, val_entries, test_entries = load_entries(cfg)

    val = cache.load(cfg, "val", layers=tuple(cfg.patch_layers))
    test = cache.load(cfg, "test", layers=tuple(cfg.patch_layers))
    # pool 默认只留最后一层给 Memory；分层原型在 build_prototypes 里用 memmap 抽 few-shot
    pool = cache.load(cfg, "pool", layers=((tuple(cfg.patch_layers)[-1],) if not need_pool_layers
                                          else tuple(cfg.patch_layers)))

    pr = build_prototypes(cfg, pool, proto_entries, device)
    c_norm, c_anom = pr["c_norm"].to(device), pr["c_anom"].to(device)
    w_text, pack0 = build_text_anchors(cfg, model, tokenizer, device)

    # 把张量搬到 GPU(特征整体不大;pool 只在需要时用)
    # label/has_mask/mask14 一并搬 —— 否则每个 top1_acc/image_metrics 调用点
    # 都要单独处理设备,极易漏(已经踩过两次)
    for d in (val, test):
        d["cls"] = d["cls"].to(device)
        d["patch"] = {k: v.to(device) for k, v in d["patch"].items()}
        for k in ("label", "has_mask", "mask14"):
            d[k] = d[k].to(device)

    adapter, pack = None, pack0
    from . import train as ptrn
    from . import text_adapter as TA
    tcfg = getattr(cfg, "text", None)
    st_text = ptrn.load_state(cfg, "text") if tcfg is not None and getattr(tcfg, "enable", False) else None
    if st_text and pack0 is not None and "adapter" in st_text:
        adapter = TA.TextResidualAdapter(
            dim=pack0["t_global"].shape[-1],
            bottleneck=int(getattr(tcfg, "bottleneck", 256)),
            lambda_t=float(getattr(tcfg, "lambda_t", 0.08)),
        ).to(device)
        adapter.load_state_dict(st_text["adapter"])
        adapter.eval()
        # 必须 no_grad:否则 pack 里几个张量会带着挂在文本适配器上的计算图。
        # train_text 的 backward 会把那张图释放,而 Stage 2 的 contrast_loss /
        # hm_text 会把 pack 当冻结锚点用 —— 那时 backward 走到已被释放的图上,
        # 报 "Trying to backward through the graph a second time"。
        # 架构上文本锚点在阶段二本来就是冻结的,不该带梯度。
        with torch.no_grad():
            pack = TA.apply_adapter(adapter, pack0)
        w_text = TA.w_text_from_pack(pack)
        print("[text] 已加载 runs/.../text.pt 适配器")

    mem = None
    if getattr(cfg, "use_memory_bank", False):
        mem = build_memory_bank(cfg, pool, proto_entries, device)

    text_sign = infer_text_sign(cfg, pool, pack, proto_entries, device)

    return {
        "cfg": cfg,
        "device": device,
        "model": model,
        "pr": pr,
        "c_norm": c_norm,
        "c_anom": c_anom,
        "w_text": w_text,
        "text_pack0": pack0,
        "text_pack": pack,
        "text_adapter": adapter,
        "text_sign": text_sign,
        "memory": mem,
        "fuse_w": (st_text or {}).get("fuse_w"),
        "fuse_alpha": (st_text or {}).get("fuse_alpha"),
        "hp": ({
            "fuse_w": (st_text or {}).get("fuse_w"),
            "fuse_alpha": (st_text or {}).get("fuse_alpha"),
        } if st_text else {}),
        "val": val,
        "test": test,
        "pool": pool,
        "entries": (proto_entries, val_entries, test_entries),
    }


def build_memory_bank(cfg, pool, proto_entries, device="cpu"):
    """支持集病灶内 patch + few-shot 正常 patch。每条记忆带 14×14 网格坐标。"""
    import torch.nn.functional as F
    from .grid import nested, patch_xy

    l11 = tuple(cfg.patch_layers)[-1]
    patch = pool["patch"][l11]
    n, l, d = patch.shape
    H = int(l ** 0.5)
    mask = pool["mask14"].reshape(n, -1)
    y = pool["label"]
    xy = patch_xy(H, device=patch.device, dtype=torch.float32)
    valid = (y == 1) & pool["has_mask"] & mask.any(1)
    if not valid.any():
        raise RuntimeError("Memory Bank: 支持集没有非空 mask 的异常切片")
    a_feat = patch[valid][mask[valid].bool()].float()
    a_pos = xy.unsqueeze(0).expand(int(valid.sum()), -1, -1)[mask[valid].bool()]
    a_mem = F.normalize(a_feat, dim=-1)
    fs = pdata.select_few_shot(
        proto_entries, cfg.few_shot_normal, 0,
        seed=int(getattr(cfg, "seed", 111)),
    )
    want = {e["image_path"] for e in fs}
    paths = pool["manifest"]["image_paths"]
    n_idx = [i for i, p in enumerate(paths) if p in want]
    if not n_idx:
        n_idx = (y == 0).nonzero(as_tuple=True)[0][: int(cfg.few_shot_normal)].tolist()
    n_feat = patch[n_idx].reshape(-1, d).float()
    n_pos = xy.unsqueeze(0).expand(len(n_idx), -1, -1).reshape(-1, 2)
    n_mem = F.normalize(n_feat, dim=-1)
    pos_on = bool(nested(cfg, "memory", "pos_aware", True))
    print(f"[mem] 异常 patch {a_mem.shape[0]} | 正常 patch {n_mem.shape[0]} "
          f"(from {len(n_idx)} 张) | 位置感知={'开' if pos_on else '关'}")
    return {
        "a": a_mem.to(device), "n": n_mem.to(device),
        "a_pos": a_pos.to(device), "n_pos": n_pos.to(device),
    }
