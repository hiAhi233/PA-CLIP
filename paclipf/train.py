"""Stage 1 / Stage 2 训练循环。

成本说明:两个适配器共 (2,512)×2 = 2,048 个参数,训练数据是**预计算好的特征**。
把病灶特征先池化成 (N,512) 之后,一个 epoch 只是 9139×512 的矩阵乘 ——
20 个 epoch 在秒级完成,所以可以放心地跑 5 个 seed 报 mean±std。
"""
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from paclipf import metrics as pmetrics

from .fusion import to_np
from . import adapter as AD
from . import lesion as LSN
from . import signals as SG


def save_state(cfg, which, **tensors):
    """保存训练好的权重。

    之前只存指标不存权重,导致"训练完想回头看热力图"必须重训 —— 
    Stage 2 要 11 分钟,而权重只有几 MB。
    """
    from pathlib import Path

    d = Path(cfg.results_dir).parent / "runs" / getattr(cfg, "_config_stem", "run")
    d.mkdir(parents=True, exist_ok=True)
    tag = getattr(cfg, "_tag", "") or ""
    name = f"{which}__{tag}.pt" if tag else f"{which}.pt"
    p = d / name
    torch.save({k: (v.state_dict() if hasattr(v, "state_dict") else v)
                for k, v in tensors.items()}, p)
    return p


def load_state(cfg, which):
    from pathlib import Path

    d = Path(cfg.results_dir).parent / "runs" / getattr(cfg, "_config_stem", "run")
    tag = getattr(cfg, "_tag", "") or ""
    candidates = []
    if tag:
        candidates.append(d / f"{which}__{tag}.pt")
    candidates.append(d / f"{which}.pt")
    seed = getattr(cfg, "seed", None)
    if seed is not None:
        candidates.append(d / f"{which}__seed{int(seed)}.pt")
    for p in candidates:
        if p.is_file():
            return torch.load(p, weights_only=False)
    return None


def _gen(device, seed):
    """与目标设备匹配的 generator —— CPU generator 不能驱动 CUDA 张量。"""
    dev = "cuda" if str(device).startswith("cuda") else "cpu"
    return torch.Generator(device=dev).manual_seed(int(seed))


def _train_batch(n, requested, min_steps=4):
    """few-shot 下 yaml 的 batch 常 ≥ 支持集,一个 epoch 只剩 1 次更新。

    n≤256 时把 batch 压到最多 n/min_steps,保证每轮至少约 min_steps 步。
    全量监督(几千张)不改动 requested。
    """
    n = max(2, int(n))
    bs = max(2, min(int(requested), n))
    if n <= 256 and n >= 2 * min_steps:
        cap = max(2, n // int(min_steps))
        bs = min(bs, cap)
    return bs, max(1, n // bs)


def _fold_early_stop(cfg, ctx):
    """症例 CV 的 val 只有约 4 张异常,patience 会在 epoch 0 停住。

    yaml `cv_early_stop` 显式覆盖;未写时 few-shot 症例 CV 默认关掉早停,
    折内训满 epoch、用最后一步做 OOF,最终也训满 yaml 里的 epoch。
    """
    v = getattr(cfg, "cv_early_stop", None)
    if v is not None:
        return bool(v)
    from . import cv
    return not (cv.uses_case_cv(ctx) and getattr(cfg, "train_scope", "") == "few_shot")


# --------------------------------------------------------------------------- #
def make_lesion_features(patch_l11, mask14, label, q, generator=None):
    """把逐 patch 特征池化成 (N,512) 的病灶特征,只做一次。

    尺寸匹配是必须的:异常切片用 mask 内 patch(q≈3 个),
    正常切片若用全部 196 个 patch 的均值,两者方差差一个量级,
    适配器会走"看方差"的捷径而不是学语义。所以正常侧也用 q 个随机 patch。

    mask 为空的异常切片(实测 15.7%)同样退化到随机子集:
    在 14x14 网格上它们本就无可定位的目标,强行用 mask 只会注入噪声。
    """
    n, l, _ = patch_l11.shape
    mask_flat = mask14.reshape(n, l)
    f = LSN.from_mask(patch_l11, mask_flat, fallback_random=q, generator=generator)
    use_random = (label == 0) | ~mask_flat.any(-1)
    if use_random.any():
        f[use_random] = LSN.random_subset(patch_l11[use_random], q, generator=generator)
    return f


# --------------------------------------------------------------------------- #
def _init_adapters(cfg, ctx, f_cls, f_les, y, c_norm=None, c_anom=None, pi=None):
    """两个适配器的初始化。传入的三个张量必须**已经**是训练子集(等长)。

    病灶头与原型同空间(patch 池化特征),直接用原型初始化是对的。
    整图头是 CLS 空间,与 patch 空间不同构,跨空间初始化是隐性损失 ——
    所以用**训练集里两类的 CLS 类均值**初始化(等价于 Proto-Adapter 的
    "原型 = 类均值"思想,只是换到 CLS 空间)。
    """
    c_norm = ctx["c_norm"] if c_norm is None else c_norm
    c_anom = ctx["c_anom"] if c_anom is None else c_anom
    pi = ctx["pr"]["pi"] if pi is None else pi

    if getattr(cfg, "adapter_init_img", "headspace_mean") == "patch_proto":
        W_img = AD.init_weights(c_norm, c_anom, pi)
    else:
        m_n = F.normalize(f_cls[y == 0].mean(0), dim=0)
        m_a = F.normalize(f_cls[y == 1].mean(0), dim=0)
        W_img = torch.stack([m_n, m_a], 0)

    W_les = AD.init_weights(c_norm, c_anom, pi)
    return AD.AngularAdapter(W_img), AD.AngularAdapter(W_les)


@torch.no_grad()
def _eval_adapters(cfg, ctx, ad_img, ad_les, tgt, c_norm=None, c_anom=None):
    """在指定 split 上算分类准确率与适配器图像级 AUC。

    c_norm/c_anom 必须显式传入当前值:Stage 2 里原型在学习,
    若沿用 ctx 里的冻结副本,评估的将不是正在训练的那个模型。
    """
    c_norm = ctx["c_norm"] if c_norm is None else c_norm
    c_anom = ctx["c_anom"] if c_anom is None else c_anom
    s = SG.compute_ctx(cfg, tgt, ctx, c_norm=c_norm, c_anom=c_anom,
                       adapter_img=ad_img, adapter_lesion=ad_les)
    y = tgt["label"]
    acc_img = pmetrics.top1_acc(s["proto_img"], y)
    acc_les = pmetrics.top1_acc(s["proto_les_uniform"], y)
    auc_adapt = pmetrics.image_metrics(to_np(s["S_adapt_margin"]), to_np(y))[0]
    return {"acc_img": acc_img, "acc_les": acc_les, "auc_adapt": auc_adapt}, s


def _adapter_from_state(state, device):
    ad = AD.AngularAdapter(state["weight"])
    ad.load_state_dict(state)
    return ad.to(device)


def _norm_anom_idx(idx, y):
    return idx[y[idx] == 0], idx[y[idx] == 1]


def _run_stage1(cfg, ctx, device, f_cls_all, f_les_all, y_all, train_idx,
                c_norm, c_anom, pi, eval_tgt=None, n_epochs=None, patience=None,
                rng=None, verbose=False, tag=""):
    """跑一轮 Stage1。eval_tgt 为 None 时训满 n_epochs,取最后一步。"""
    n_epochs = int(n_epochs if n_epochs is not None else cfg.stage1_epochs)
    patience = int(patience if patience is not None else (getattr(cfg, "patience", 0) or 0))
    ad_img, ad_les = _init_adapters(
        cfg, ctx, f_cls_all[train_idx], f_les_all[train_idx], y_all[train_idx],
        c_norm=c_norm, c_anom=c_anom, pi=pi)
    ad_img, ad_les = ad_img.to(device), ad_les.to(device)
    params = list(ad_img.parameters()) + list(ad_les.parameters())
    opt = torch.optim.Adam(params, lr=cfg.stage1_lr, eps=1e-4)
    bs, per_epoch = _train_batch(len(train_idx), cfg.stage1_batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, n_epochs * per_epoch), eta_min=cfg.stage1_lr_min)
    n_idx, a_idx = _norm_anom_idx(train_idx, y_all)
    best = {"auc_adapt": -1.0, "epoch": -1, "state": None, "wait": 0}
    hist = []
    for ep in range(n_epochs):
        ad_img.train(); ad_les.train()
        tot, nb = 0.0, 0
        for _ in range(per_epoch):
            if len(a_idx) and len(n_idx):
                b = _oversample_idx(n_idx, a_idx, bs, device, rng)
            else:
                b = train_idx[torch.randperm(len(train_idx), generator=rng, device=device)[:bs]]
            if len(b) < 2:
                continue
            loss = ad_img.loss(f_cls_all[b], y_all[b], cfg.margin, cfg.arcface_scale)
            loss = loss + cfg.beta_lesion * ad_les.loss(
                f_les_all[b], y_all[b], cfg.margin, cfg.arcface_scale)
            if getattr(cfg, "ortho_weight", 0.0) > 0:
                loss = loss + cfg.ortho_weight * sum(
                    (a.unit_weight[0] * a.unit_weight[1]).sum() ** 2 for a in (ad_img, ad_les))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item(); nb += 1
        ad_img.eval(); ad_les.eval()
        row = {"epoch": ep, "loss": tot / max(1, nb),
               "cos_img": ad_img.class_cos(), "cos_les": ad_les.class_cos()}
        if eval_tgt is not None:
            met, _ = _eval_adapters(cfg, ctx, ad_img, ad_les, eval_tgt, c_norm, c_anom)
            row.update(met)
            improved = met["auc_adapt"] > best["auc_adapt"]
            if improved:
                best = {"auc_adapt": met["auc_adapt"], "epoch": ep, "wait": 0,
                        "state": copy.deepcopy({"img": ad_img.state_dict(), "les": ad_les.state_dict()})}
            else:
                best["wait"] = best.get("wait", 0) + 1
                if patience and best["wait"] >= patience:
                    hist.append(row)
                    if verbose:
                        print(f"  {tag}ep{ep:>3} early stop (patience={patience})")
                    break
        else:
            best = {"auc_adapt": float("nan"), "epoch": ep, "wait": 0,
                    "state": copy.deepcopy({"img": ad_img.state_dict(), "les": ad_les.state_dict()})}
        hist.append(row)
        if verbose and (ep % 4 == 0 or ep == n_epochs - 1):
            extra = (f" val AUC {row.get('auc_adapt', float('nan')):.4f}"
                     if "auc_adapt" in row else "")
            print(f"  {tag}ep{ep:>3} loss {row['loss']:.4f}{extra} | cos {ad_img.class_cos():.3f}")
    # 有 patience 才回滚到 val 最优;关掉早停时用最后一步,避免 4 张异常把 epoch 0 当最优。
    if patience and best["state"] is not None:
        ad_img.load_state_dict(best["state"]["img"])
        ad_les.load_state_dict(best["state"]["les"])
    return ad_img, ad_les, hist, best


def train_stage1(cfg, ctx, device="cuda", seed=None, verbose=True):
    """冻结原型,只训练两个角空间适配器。support44 下走症例留一选 epoch。"""
    from . import cache, cv

    seed = int(seed if seed is not None else getattr(cfg, "seed", 111))
    torch.manual_seed(seed)
    rng = _gen(device, seed)

    pool = cache.load(cfg, "pool", layers=(tuple(cfg.patch_layers)[-1],))
    f_cls_all = pool["cls"].to(device)
    patch_all = pool["patch"][tuple(cfg.patch_layers)[-1]].to(device)
    y_all = pool["label"].to(device)

    idx = cv.train_idx_for_scope(cfg, ctx, device)
    early = _fold_early_stop(cfg, ctx)
    if verbose:
        bs0, st0 = _train_batch(len(idx), cfg.stage1_batch)
        print(f"[stage1] 训练样本 {len(idx)} 张 (train_scope={cfg.train_scope}) "
              f"| 异常 {int(y_all[idx].sum())} 正常 {int((y_all[idx]==0).sum())}"
              f" | batch {bs0} × {st0} step/epoch"
              f" | 折内早停={'开' if early else '关(训满 epoch)'}")

    q = max(1, int(getattr(cfg, "q_sel", 3)))
    t0 = time.time()
    f_les_all = make_lesion_features(patch_all, pool["mask14"].to(device), y_all, q, rng)
    del patch_all
    if verbose:
        print(f"[stage1] 病灶特征池化完成 {tuple(f_les_all.shape)} ({time.time()-t0:.1f}s)")

    fold_states = []
    if cv.uses_case_cv(ctx) and cfg.train_scope == "few_shot":
        fold_best = []
        for held, tr, ho in cv.iter_folds(cfg, ctx):
            tr, ho = tr.to(device), ho.to(device)
            pr = cv.rebuild_proto(cfg, ctx, tr, device)
            tgt, _ = cv.fold_eval_split(cfg, ctx, ho, device)
            if verbose:
                print(f"[stage1] fold hold {held}  train {len(tr)} / val {len(ho)}")
            ad_img, ad_les, hist, best = _run_stage1(
                cfg, ctx, device, f_cls_all, f_les_all, y_all, tr,
                pr["c_norm"], pr["c_anom"], pr["pi"], eval_tgt=tgt,
                patience=(cfg.patience if early else 0),
                rng=rng, verbose=verbose, tag=f"{held} ")
            fold_best.append(best["epoch"] if best["epoch"] >= 0 else cfg.stage1_epochs - 1)
            fold_states.append({
                "held": held, "epoch": best["epoch"],
                "img": copy.deepcopy(ad_img.state_dict()),
                "les": copy.deepcopy(ad_les.state_dict()),
                "c_norm": pr["c_norm"].detach().cpu(),
                "c_anom": pr["c_anom"].detach().cpu(),
                "pi": pr["pi"].detach().cpu(),
                "train_idx": tr.detach().cpu(), "held_idx": ho.detach().cpu(),
            })
        if early:
            best_ep = int(round(float(np.mean(fold_best)))) + 1
            best_ep = max(1, min(int(cfg.stage1_epochs), best_ep))
        else:
            best_ep = int(cfg.stage1_epochs)
        if verbose:
            print(f"[stage1] 症例 CV {'选' if early else '固定'} epoch={best_ep} (折内最优 {fold_best})")
        ad_img, ad_les, hist, best = _run_stage1(
            cfg, ctx, device, f_cls_all, f_les_all, y_all, idx,
            ctx["c_norm"], ctx["c_anom"], ctx["pr"]["pi"],
            eval_tgt=None, n_epochs=best_ep, patience=0,
            rng=rng, verbose=verbose, tag="final ")
        best["epoch"] = best_ep - 1
        by_held = {fs["held"]: fs for fs in fold_states}

        def _pf(held, tr, ho):
            st = by_held[held]
            return {
                "adapter_img": _adapter_from_state(st["img"], device),
                "adapter_lesion": _adapter_from_state(st["les"], device),
                "c_norm": st["c_norm"].to(device), "c_anom": st["c_anom"].to(device),
            }

        hp, _ = cv.search_hparams_oof(
            cfg, ctx, per_fold=_pf, text_pack=ctx.get("text_pack"), verbose=verbose)
        ctx["hp"] = hp
        ctx["fuse_w"] = hp.get("fuse_w", ctx.get("fuse_w"))
        ctx["fuse_alpha"] = hp.get("fuse_alpha", ctx.get("fuse_alpha"))
    else:
        tgt_sel = ctx["val"] if not cv.uses_case_cv(ctx) else None
        ad_img, ad_les, hist, best = _run_stage1(
            cfg, ctx, device, f_cls_all, f_les_all, y_all, idx,
            ctx["c_norm"], ctx["c_anom"], ctx["pr"]["pi"],
            eval_tgt=tgt_sel, rng=rng, verbose=verbose)
        hp = {}

    if verbose:
        print(f"[stage1] 最优 epoch {best['epoch']} (val AUC {best.get('auc_adapt', float('nan')):.4f}),"
              f" 最终 cos img {ad_img.class_cos():.4f} / les {ad_les.class_cos():.4f}")
    save_state(cfg, "stage1", ad_img=ad_img, ad_les=ad_les, fold_states=fold_states, hp=hp)
    return {"ad_img": ad_img, "ad_les": ad_les, "hist": hist, "best": best,
            "seed": seed, "fold_states": fold_states, "hp": hp}


# --------------------------------------------------------------------------- #
def _stage2_select_score(sv, m14, labels):
    """折内选模:hit@1 与图像 AUC 平均;hit@3 只作平局。hit@3 在脑上会饱和。"""
    from . import diag

    y = to_np(labels)
    sel = y > 0
    auc = pmetrics.image_metrics(to_np(sv["S_adapt_margin"]), y)[0]
    if not sel.any():
        return {"score": float(auc) if np.isfinite(auc) else float("nan"),
                "hit1": float("nan"), "hit3": float("nan"), "auc": float(auc)}
    hits = diag.hit_rate(sv["s_flat"][sel], m14[sel])
    h1, h3 = float(hits["hit@1_valid"]), float(hits["hit@3_valid"])
    if np.isfinite(auc):
        score = 0.5 * h1 + 0.5 * float(auc)
    else:
        score = h1
    return {"score": float(score), "hit1": h1, "hit3": h3, "auc": float(auc)}


def _copy_adapter(src, device):
    ad = AD.AngularAdapter(src.weight.detach())
    ad.load_state_dict(src.state_dict())
    return ad.to(device)


def _run_stage2(cfg, ctx, device, patch_cpu, f_cls_all, y_dev, mask_dev, is_anom_dev,
                train_idx, ad_img, ad_les, c_norm_init, c_anom_init,
                eval_tgt=None, m14v=None, n_epochs=None, patience=None,
                rng=None, verbose=False, tag=""):
    from . import heatmap as HM, losses as LS, prototypes as PR

    n_epochs = int(n_epochs if n_epochs is not None else cfg.stage2_epochs)
    patience = int(patience if patience is not None else (getattr(cfg, "patience", 0) or 0))
    layers = tuple(cfg.patch_layers)
    proto = PR.LearnablePrototypes(c_norm_init, c_anom_init, cfg.double_norm).to(device)
    c_norm0 = c_norm_init.detach().clone()
    c_anom0 = c_anom_init.detach().clone()
    ad_img, ad_les = _copy_adapter(ad_img, device), _copy_adapter(ad_les, device)
    for p in list(ad_img.parameters()) + list(ad_les.parameters()):
        p.requires_grad_(True)
    opt = torch.optim.Adam([
        {"params": proto.parameters(), "lr": cfg.stage2_lr_proto},
        {"params": list(ad_img.parameters()) + list(ad_les.parameters()),
         "lr": cfg.stage2_lr_adapter},
    ], eps=1e-4)
    bs, per_epoch = _train_batch(len(train_idx), cfg.stage2_batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, n_epochs * per_epoch), eta_min=1e-6)
    q = max(1, int(getattr(cfg, "q_sel", 3)))
    n_idx, a_idx = _norm_anom_idx(train_idx, y_dev)
    best = {"score": -1.0, "hit1": -1.0, "hit3": -1.0, "epoch": -1, "state": None,
            "auc": -1.0, "wait": 0, "acc0": None}
    hist = []
    for ep in range(n_epochs):
        proto.train(); ad_img.train(); ad_les.train()
        tot, nb, acc_stats = 0.0, 0, {}
        for _ in range(per_epoch):
            if len(a_idx) and len(n_idx):
                b = _oversample_idx(n_idx, a_idx, bs, device, rng)
            else:
                b = train_idx[torch.randperm(len(train_idx), generator=rng, device=device)[:bs]]
            if len(b) < 2:
                continue
            c_norm, c_anom = proto.normalized()
            bi = b.detach().cpu().numpy()
            pf = {l: torch.as_tensor(np.asarray(patch_cpu[l][bi])).to(device, torch.float32)
                  for l in layers}
            s_flat = HM.heatmap_diff(
                pf, c_anom, c_norm, layers, image_size=cfg.image_size,
                smooth_kernel=getattr(cfg, "smooth_kernel", 0),
                reduce=cfg.reduce, lse_tau=cfg.lse_tau,
            ).reshape(len(b), -1)
            yb = y_dev[b]
            Ln, La, stats = LS.localize_loss(s_flat, mask_dev[b], is_anom_dev[b], cfg)
            L_loc = Ln + cfg.lam_loc * La
            m_b = mask_dev[b].reshape(-1, cfg._grid, cfg._grid)
            if cfg.lesion_source == "gt_mask":
                f_les = make_lesion_features(pf[layers[-1]], m_b, yb, q, rng)
            else:
                with torch.no_grad():
                    f_les = LSN.topq_from_score(
                        pf[layers[-1]], s_flat.reshape(-1, cfg._grid, cfg._grid), q)
            L_cls = ad_img.loss(f_cls_all[b], yb, cfg.margin, cfg.arcface_scale)
            L_cls = L_cls + cfg.beta_lesion * ad_les.loss(f_les, yb, cfg.margin, cfg.arcface_scale)
            loss = L_loc + cfg.alpha_cls * L_cls
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item(); nb += 1
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    acc_stats[k] = acc_stats.get(k, 0.0) + v / max(1, per_epoch)

        proto.eval(); ad_img.eval(); ad_les.eval()
        c_norm, c_anom = proto.normalized()
        row = {"epoch": ep, "loss": tot / max(1, nb), **proto.drift(c_norm0, c_anom0)}
        if eval_tgt is not None:
            met, sv = _eval_adapters(cfg, ctx, ad_img, ad_les, eval_tgt, c_norm, c_anom)
            sel = _stage2_select_score(sv, m14v, eval_tgt["label"])
            row.update(met)
            row.update({"val_hit1": sel["hit1"], "val_hit3": sel["hit3"],
                        "val_score": sel["score"]})
            if best["acc0"] is None:
                best["acc0"] = met["acc_img"]
            better = (sel["score"] > best["score"] + 1e-12 or
                      (abs(sel["score"] - best["score"]) <= 1e-12 and sel["hit3"] > best["hit3"]))
            guard = met["acc_img"] >= best["acc0"] - 1.0
            if better and guard:
                best.update({"score": sel["score"], "hit1": sel["hit1"], "hit3": sel["hit3"],
                             "epoch": ep, "auc": sel["auc"], "wait": 0,
                             "state": {"proto": {k: v.clone() for k, v in proto.state_dict().items()},
                                       "img": copy.deepcopy(ad_img.state_dict()),
                                       "les": copy.deepcopy(ad_les.state_dict())}})
            else:
                best["wait"] = best.get("wait", 0) + 1
                if patience and best["wait"] >= patience:
                    hist.append(row)
                    if verbose:
                        print(f"  {tag}ep{ep:>3} early stop (patience={patience})")
                    break
        else:
            best = {"score": float("nan"), "hit1": float("nan"), "hit3": float("nan"),
                    "epoch": ep, "auc": float("nan"), "wait": 0,
                    "state": {"proto": {k: v.clone() for k, v in proto.state_dict().items()},
                              "img": copy.deepcopy(ad_img.state_dict()),
                              "les": copy.deepcopy(ad_les.state_dict())}}
        hist.append(row)
        if verbose and (ep % 5 == 0 or ep == n_epochs - 1):
            extra = (f" score {row.get('val_score', float('nan')):.4f} "
                     f"hit@1 {row.get('val_hit1', float('nan')):.4f}"
                     if "val_score" in row else "")
            print(f"  {tag}ep{ep:>3} loss {row['loss']:.4f}{extra} "
                  f"| drift n {row['drift_norm']:.2e} a {row['drift_anom']:.2e}")
    if patience and best["state"]:
        proto.load_state_dict(best["state"]["proto"])
        ad_img.load_state_dict(best["state"]["img"])
        ad_les.load_state_dict(best["state"]["les"])
    return proto, ad_img, ad_les, hist, best


def train_stage2(cfg, ctx, stage1, device="cuda", seed=None, verbose=True):
    """解冻原型 + 定位损失。support44 下走症例留一,选 hit@1+图像 AUC。"""
    from . import cache, cv

    seed = int(seed if seed is not None else getattr(cfg, "seed", 111))
    rng = _gen(device, seed)
    layers = tuple(cfg.patch_layers)
    patch_cpu = cache.pool_patch_mmap(cfg, layers)
    meta = cache.load(cfg, "pool")
    y_all = meta["label"]
    mask_all = meta["mask14"].reshape(len(y_all), -1)
    n_all = len(y_all)
    f_cls_all = meta["cls"].to(device)
    y_dev = y_all.to(device)
    mask_dev = mask_all.to(device)
    is_anom_dev = (y_dev == 1)
    del meta

    idx = cv.train_idx_for_scope(cfg, ctx, device)
    early = _fold_early_stop(cfg, ctx)
    t0 = time.time()
    fold_states = stage1.get("fold_states") or []
    if verbose:
        bs0, st0 = _train_batch(len(idx), cfg.stage2_batch)
        print(f"[stage2] 训练样本 {len(idx)} 张 | batch {bs0} × {st0} step/epoch"
              f" | 折内早停={'开' if early else '关(训满 epoch)'}")
    if cv.uses_case_cv(ctx) and cfg.train_scope == "few_shot" and not fold_states:
        print("[stage2] 警告: 没有 Stage1 折内权重,无法做症例留一,改为全支持集训满 epoch")

    if cv.uses_case_cv(ctx) and cfg.train_scope == "few_shot" and fold_states:
        fold_best, oof_records = [], []
        by_held = {fs["held"]: fs for fs in fold_states}
        fold_pat = cfg.patience if early else 0
        for held, tr, ho in cv.iter_folds(cfg, ctx):
            st = by_held[held]
            tr, ho = st["train_idx"].to(device), st["held_idx"].to(device)
            tgt, _ = cv.fold_eval_split(cfg, ctx, ho, device)
            m14v = tgt["mask14"].reshape(len(tgt["label"]), -1).bool()
            ad_i = _adapter_from_state(st["img"], device)
            ad_l = _adapter_from_state(st["les"], device)
            if verbose:
                print(f"[stage2] fold hold {held}  train {len(tr)} / val {len(ho)}")
            proto_f, ad_if, ad_lf, _, best = _run_stage2(
                cfg, ctx, device, patch_cpu, f_cls_all, y_dev, mask_dev, is_anom_dev,
                tr, ad_i, ad_l, st["c_norm"].to(device), st["c_anom"].to(device),
                eval_tgt=tgt, m14v=m14v, patience=fold_pat,
                rng=rng, verbose=verbose, tag=f"{held} ")
            fold_best.append(best["epoch"] if best["epoch"] >= 0 else cfg.stage2_epochs - 1)
            cn, ca = proto_f.normalized()
            rec = cv.fold_signals(
                cfg, ctx, tr, ho, adapter_img=ad_if, adapter_lesion=ad_lf,
                text_pack=ctx.get("text_pack"), c_norm=cn, c_anom=ca)
            rec["held"] = held
            oof_records.append(rec)
        if early:
            best_ep = int(round(float(np.mean(fold_best)))) + 1
            best_ep = max(1, min(int(cfg.stage2_epochs), best_ep))
        else:
            best_ep = int(cfg.stage2_epochs)
        if verbose:
            print(f"[stage2] 症例 CV {'选' if early else '固定'} epoch={best_ep} (折内最优 {fold_best})")
        proto, ad_img, ad_les, hist, best = _run_stage2(
            cfg, ctx, device, patch_cpu, f_cls_all, y_dev, mask_dev, is_anom_dev,
            idx, stage1["ad_img"], stage1["ad_les"], ctx["c_norm"], ctx["c_anom"],
            eval_tgt=None, n_epochs=best_ep, patience=0,
            rng=rng, verbose=verbose, tag="final ")
        best["epoch"] = best_ep - 1

        fuse_w, hit, table = cv.search_map_oof(cfg, oof_records)
        cv.add_fused_scores(cfg, oof_records, fuse_w)
        a_f, auc_f, _ = cv.search_alpha_oof(oof_records, "S_heat_fused")
        a_pt, l_pt, acc, _, edge = cv.search_fusion_oof(cfg, oof_records, pmetrics.top1_acc)
        hp = {"fuse_w": fuse_w, "fuse_alpha": float(a_f), "oof_hit": hit,
              "oof_img_auc": auc_f, "alpha_pt": a_pt, "lam_pt": l_pt,
              "oof_acc": acc, "edge_warning": edge, "table": table}
        if any("S_adapt_margin" in r["sig"] for r in oof_records):
            w3, auc3, _ = cv.search_three_oof(oof_records, "S_heat_fused")
            hp.update({"w_heat": w3[0], "w_text": w3[1], "w_adapt": w3[2],
                       "oof_img_auc": float(auc3)})
        if verbose:
            print(f"[stage2] OOF fuse_w {fuse_w}  hit@1={hit:.4f}  "
                  f"alpha_fuse={a_f:.3f}  AUC={hp['oof_img_auc']:.4f}  α_pt={a_pt} λ_pt={l_pt}")
            if "w_adapt" in hp:
                print(f"[stage2] OOF 图像分 w_heat={hp['w_heat']:.2f} "
                      f"w_text={hp['w_text']:.2f} w_adapt={hp['w_adapt']:.2f}")
        ctx["hp"] = hp
        ctx["fuse_w"] = fuse_w
        ctx["fuse_alpha"] = float(a_f)
    else:
        tgt = ctx["val"] if not cv.uses_case_cv(ctx) else None
        m14v = _mask14_of(cfg, ctx, tgt) if tgt is not None else None
        proto, ad_img, ad_les, hist, best = _run_stage2(
            cfg, ctx, device, patch_cpu, f_cls_all, y_dev, mask_dev, is_anom_dev,
            idx, stage1["ad_img"], stage1["ad_les"], ctx["c_norm"], ctx["c_anom"],
            eval_tgt=tgt, m14v=m14v, rng=rng, verbose=verbose)

    if verbose:
        print(f"[stage2] 最优 epoch {best['epoch']} "
              f"(score {best.get('score', float('nan')):.4f} hit@1 {best.get('hit1', float('nan')):.4f}),"
              f" 用时 {(time.time()-t0)/60:.1f} 分钟")
    save_state(cfg, "stage2", proto=proto, ad_img=ad_img, ad_les=ad_les,
               hp=ctx.get("hp", stage1.get("hp", {})))
    return {"proto": proto, "ad_img": ad_img, "ad_les": ad_les, "hist": hist,
            "best": best, "seed": seed, "hp": ctx.get("hp", {})}


def _val_mask14(cfg, ctx):
    from paclipf import data as pdata

    ms = []
    for e, y in zip(ctx["entries"][1], ctx["val"]["label"]):
        ms.append(torch.tensor(pdata.mask_grid_14(str(cfg.data_root) + "/" + e["mask_path"]))
                  if (y == 1 and e.get("mask_path")) else torch.zeros(cfg._grid, cfg._grid))
    return torch.stack(ms).reshape(len(ms), -1).bool()


def _diag_hit3(sv, m14, labels):
    """val 上的 hit@3(定位质量本身),Stage 2 的主选择指标。"""
    from . import diag

    sel = to_np(labels) > 0
    if not sel.any():
        return float("nan")
    return float(diag.hit_rate(sv["s_flat"][sel], m14[sel])["hit@3_valid"])


def _select_split(cfg, ctx):
    """val 同时含两类时用 val;support44 下 val 无异常,改用 pool 异常 + val 正常。"""
    yv = ctx["val"]["label"]
    if int((yv == 0).sum()) and int((yv == 1).sum()):
        return ctx["val"]
    return _support_split(cfg, ctx)


def _mask14_of(cfg, ctx, split):
    if split is ctx["val"]:
        return _val_mask14(cfg, ctx)
    return split["mask14"].reshape(len(split["label"]), -1).bool()


def _support_split(cfg, ctx):
    """pool 中全部异常切片 + ctx['val'](support44 下为 39 张正常)。"""
    from . import cache

    device = ctx["device"]
    pool, val = ctx["pool"], ctx["val"]
    y_p = pool["label"]
    ai = (y_p == 1).nonzero(as_tuple=True)[0]
    layers = tuple(cfg.patch_layers)
    mmap = cache.pool_patch_mmap(cfg, layers, verbose=False)
    bi = ai.detach().cpu().numpy() if torch.is_tensor(ai) else np.asarray(ai)
    patch = {
        l: torch.cat([
            torch.as_tensor(np.asarray(mmap[l][bi])).to(device, torch.float32),
            val["patch"][l],
        ], 0) for l in layers
    }
    label = torch.cat([y_p.to(device)[ai], val["label"]], 0)
    cls = torch.cat([pool["cls"].to(device)[ai], val["cls"]], 0)
    mask14 = torch.cat([pool["mask14"].to(device)[ai], val["mask14"]], 0)
    has_mask = torch.cat([pool["has_mask"].to(device)[ai], val["has_mask"]], 0)
    paths = list(pool["manifest"]["image_paths"])
    from pathlib import Path as _P
    case_ids = [_P(paths[int(i)]).name.split("_")[0] for i in bi]
    case_ids += ["val_good"] * len(val["label"])
    return {
        "cls": cls, "patch": patch, "label": label, "mask14": mask14,
        "has_mask": has_mask, "case_ids": case_ids,
    }


def _oversample_idx(norm_idx, anom_idx, batch, device, rng):
    """每个 batch 吃进全部异常,其余用正常填;支持集大小等于 batch 时不放回。"""
    n_a = int(len(anom_idx))
    n_n = int(len(norm_idx))
    if n_a >= batch:
        perm = torch.randperm(n_a, generator=rng, device=device)
        return anom_idx[perm[:batch]]
    need_n = batch - n_a
    if need_n <= n_n:
        perm = torch.randperm(n_n, generator=rng, device=device)
        return torch.cat([anom_idx, norm_idx[perm[:need_n]]])
    extra = need_n - n_n
    more = norm_idx[torch.randint(0, n_n, (extra,), generator=rng, device=device)]
    return torch.cat([anom_idx, norm_idx, more])


def train_text(cfg, ctx, device="cuda", verbose=True):
    """Stage text:只训 TextResidualAdapter。11 症例留一选 epoch,OOF 搜融合,再全支持集重训。"""
    from . import cv
    from . import pipeline as PL
    from . import text_adapter as TA
    from . import text_losses as TL

    pack0 = ctx.get("text_pack0")
    if pack0 is None:
        raise RuntimeError("train_text 需要 cfg.text_prompts 分层提示词")
    pack0 = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in pack0.items()}
    tcfg = cfg.text
    seed = int(getattr(cfg, "seed", 111))
    torch.manual_seed(seed)
    rng = _gen(device, seed)

    pool = ctx["pool"]
    y_all = pool["label"].to(device)
    l11 = tuple(cfg.patch_layers)[-1]
    f_cls = pool["cls"].to(device)
    patch_l11 = pool["patch"][l11]
    mask = pool["mask14"].to(device).reshape(len(y_all), -1)

    scope = getattr(tcfg, "train_scope", None) or cfg.train_scope
    if scope == "few_shot":
        fs = cv.few_shot_idx(cfg, ctx, device)
        anom_idx = fs[y_all[fs] == 1]
        norm_idx = fs[y_all[fs] == 0]
    else:
        anom_idx = (y_all == 1).nonzero(as_tuple=True)[0]
        norm_idx = (y_all == 0).nonzero(as_tuple=True)[0]

    empty = (y_all[anom_idx] == 1) & ~mask[anom_idx].any(1)
    if int(empty.sum()):
        print(f"[text] 警告: {int(empty.sum())} 张异常 14x14 mask 为空,L_patch 将跳过它们")
    else:
        print(f"[text] 支持集异常 {int(len(anom_idx))} 张 / 正常 {int(len(norm_idx))} 张"
              f" (train_scope={scope}),14x14 mask 全非空")

    dim = pack0["t_global"].shape[-1]
    epochs = int(getattr(tcfg, "epochs", 80))
    req_bs = int(getattr(tcfg, "batch", 16))
    lr = float(getattr(tcfg, "lr", 1e-3))
    patience = int(getattr(cfg, "patience", 0) or 0)
    early = _fold_early_stop(cfg, ctx)
    bs_show, st_show = _train_batch(int(len(norm_idx) + len(anom_idx)), req_bs)
    print(f"[text] batch {bs_show} × {st_show} step/epoch | 折内早停="
          f"{'开' if early else '关(训满 epoch)'}")

    def _run(train_anom, train_norm, n_epochs, val_anom=None, tag="", use_patience=True):
        ad = TA.TextResidualAdapter(
            dim=dim,
            bottleneck=int(getattr(tcfg, "bottleneck", 256)),
            lambda_t=float(getattr(tcfg, "lambda_t", 0.08)),
        ).to(device)
        opt = torch.optim.Adam(ad.parameters(), lr=lr, eps=1e-4)
        hist, best, wait = [], {"score": -1.0, "epoch": -1, "state": None}, 0
        bs, n_steps = _train_batch(int(len(train_norm) + len(train_anom)), req_bs)
        for ep in range(n_epochs):
            ad.train()
            tot, nb = 0.0, 0
            for _ in range(n_steps):
                b = _oversample_idx(train_norm, train_anom, bs, device, rng)
                pack = TA.apply_adapter(ad, pack0)
                pb = patch_l11[b.detach().cpu()].to(device)
                loss, _ = TL.text_losses(
                    cfg, pack, pack0, f_cls[b], pb, mask[b], y_all[b])
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.detach()); nb += 1
            row = {"epoch": ep, "loss": tot / max(1, nb)}
            if val_anom is not None and len(val_anom):
                ad.eval()
                with torch.no_grad():
                    pack = TA.apply_adapter(ad, pack0)
                    score = _text_val_score(cfg, ctx, pack, val_anom, mask, device)
                row["val_score"] = score
                if score > best["score"]:
                    best = {"score": score, "epoch": ep, "state": copy.deepcopy(ad.state_dict())}
                    wait = 0
                else:
                    wait += 1
                    if use_patience and patience and wait >= patience:
                        hist.append(row)
                        if verbose:
                            print(f"  {tag}ep{ep:>3} early stop (patience={patience})")
                        break
            else:
                best = {"score": float("nan"), "epoch": ep,
                        "state": copy.deepcopy(ad.state_dict())}
            hist.append(row)
            if verbose and (ep % 10 == 0 or ep == n_epochs - 1):
                extra = f" val {row.get('val_score', float('nan')):.4f}" if "val_score" in row else ""
                print(f"  {tag}ep{ep:>3} loss {row['loss']:.4f}{extra}")
        if use_patience and best["state"] is not None:
            ad.load_state_dict(best["state"])
        return ad, hist, best

    fold_best, oof_records = [], []
    if cv.uses_case_cv(ctx):
        for held, tr, ho in cv.iter_folds(cfg, ctx):
            tr, ho = tr.to(device), ho.to(device)
            y_tr = y_all[tr]
            train_a, train_n = tr[y_tr == 1], tr[y_tr == 0]
            if verbose:
                print(f"[text] fold hold {held}  train {len(train_a)} / val {len(ho)}")
            ad_f, hist, best = _run(
                train_a, train_n, epochs, val_anom=ho, tag=f"{held} ",
                use_patience=early)
            fold_best.append(best["epoch"] if best["epoch"] >= 0 else epochs - 1)
            ad_f.eval()
            with torch.no_grad():
                pack_f = TA.apply_adapter(ad_f, pack0)
            ctx["text_sign"] = PL.infer_text_sign(
                cfg, ctx["pool"], pack_f, ctx["entries"][0], device, verbose=False)
            rec = cv.fold_signals(cfg, ctx, tr, ho, text_pack=pack_f)
            rec["held"] = held
            oof_records.append(rec)
        if early:
            best_ep = int(round(float(np.mean(fold_best)))) + 1
            best_ep = max(1, min(epochs, best_ep))
        else:
            best_ep = epochs
        print(f"[text] 症例 CV {'选' if early else '固定'} epoch={best_ep} (折内最优 {fold_best})")
        fuse_w, hit, table = cv.search_map_oof(cfg, oof_records)
        cv.add_fused_scores(cfg, oof_records, fuse_w)
        a_f, auc_f, _ = cv.search_alpha_oof(oof_records, "S_heat_fused")
        print(f"[text] OOF fuse_w {fuse_w}  hit@1={hit:.4f}  alpha_fuse={a_f:.3f}  AUC={auc_f:.4f}")
        fuse_meta = {"hit@1": hit, "table": table, "oof_img_auc": auc_f}
    else:
        best_ep = epochs
        fuse_w, a_f, fuse_meta = {"proto": 1.0}, 1.0, {}

    ad, hist, _ = _run(anom_idx, norm_idx, best_ep, val_anom=None, tag="final ",
                       use_patience=False)
    ad.eval()
    with torch.no_grad():
        pack = TA.apply_adapter(ad, pack0)
    ctx["text_sign"] = PL.infer_text_sign(
        cfg, ctx["pool"], pack, ctx["entries"][0], device)
    ctx["text_adapter"] = ad
    ctx["text_pack"] = pack
    ctx["w_text"] = TA.w_text_from_pack(pack)
    ctx["fuse_w"] = fuse_w
    ctx["fuse_alpha"] = float(a_f)
    ctx["hp"] = {"fuse_w": fuse_w, "fuse_alpha": float(a_f), **{
        k: v for k, v in fuse_meta.items() if k != "table"}}
    p = save_state(cfg, "text", adapter=ad, fuse_w=fuse_w, fuse_alpha=float(a_f),
                   best_ep=best_ep, fold_best=fold_best)
    print(f"[text] 权重 {p} | fuse_w {fuse_w} | alpha {a_f:.3f}")
    return {"adapter": ad, "pack": pack, "hist": hist, "best_ep": best_ep,
            "fuse_w": fuse_w, "fuse_alpha": float(a_f), "fuse_meta": fuse_meta}


@torch.no_grad()
def _text_val_score(cfg, ctx, pack, val_anom, mask, device):
    """留一病例:hit@1(A_text) 与 该病例异常+leftover valid 正常 的文本头 AUC 平均。"""
    from . import diag
    from . import text_adapter as TA
    from paclipf import classify

    l11 = tuple(cfg.patch_layers)[-1]
    va = val_anom
    p_va = ctx["pool"]["patch"][l11][va.detach().cpu()].to(device)
    hm = SG.text_heatmap({l11: p_va}, pack, (l11,), cfg.image_size)
    sign = float(ctx.get("text_sign", 1.0))
    if sign != 1.0:
        hm = sign * hm
    hit = diag.hit_rate(hm.reshape(len(va), -1), mask[va])["hit@1_valid"]
    leftover = ctx["val"]
    nsel = leftover["label"] == 0
    f_cls = torch.cat([
        ctx["pool"]["cls"].to(device)[va],
        leftover["cls"].to(device)[nsel],
    ], 0)
    y = torch.cat([
        torch.ones(len(va), device=device, dtype=torch.long),
        torch.zeros(int(nsel.sum()), device=device, dtype=torch.long),
    ], 0)
    w = TA.w_text_from_pack(pack)
    logits = classify.classify(f_cls, w)
    s_text = 1.0 - torch.softmax(logits, dim=-1)[:, 0]
    try:
        auc = pmetrics.image_metrics(to_np(s_text), to_np(y))[0]
    except Exception:
        auc = float("nan")
    if not np.isfinite(auc):
        return float(hit)
    return 0.5 * float(hit) + 0.5 * float(auc)
