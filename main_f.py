"""PA-CLIP-F: Prototype-Anchored Localization with a Fine-tuned Angular-Space Adapter.

Pipeline
  1) few-shot normal/anomaly slices -> normal prototypes (K-means) + anomaly prototype
     (mean of ROI-inside patches)
  2) the prototypes initialize a single (D, 2) adapter in angular space: its two
     L2-normalized weight columns are the normal / anomaly anchors, following the
     Proto-Adapter-F recipe (Kato et al., Sensors 2024)
  3) a patch-level training set is built from the same few-shot slices
     (ROI-inside patches -> anomaly; ROI-outside and normal slices -> normal)
  4) the adapter is fine-tuned with the Additive Angular Margin penalty (ArcFace) +
     cross-entropy, so lesion and normal patches separate even with very few samples
  5) patch x fine-tuned adapter -> anomaly heatmap (localization)
  6) heatmap attention pooling -> lesion feature; dual-path classification (image + lesion)
  7) z-score fusion of max-heatmap and 1-P(normal) -> image-level anomaly score
  8) alpha / lambda / fusion weights are selected by case-level OOF on the support
     set (leftover valid normals as CV negatives); test is scored once

Stages (every stage writes the same test-set metric table, for row-by-row attribution)
  cache       pre-compute BiomedCLIP features (once, ~5 min)
  baseline    training-free prototype baseline + localization diagnostics
  text        residual text adapter + hierarchical prompts (support-set CV)
  stage1      freeze the prototypes, train the classification adapter
  stage2      unfreeze the prototypes and add the localization loss
  visualize   heatmap figures for qualitative comparison
  all         text -> stage1 -> stage2

Usage: python main_f.py --config configs/brain_f.yaml --stage all
       python main_f.py --config configs/brain_f.yaml --stage stage2 --seed 111
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paclipf  # noqa: F401  必须最先,注册 sys.path
from paclipf import fusion as pfusion
from paclipf import metrics as pmetrics
from paclipf import (cache, config as C, cv, diag, evaluate, fusion, heatmap as HM,
                     pipeline, report, signals as SG, train)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--stage", required=True,
                   choices=["cache", "baseline", "text", "stage1", "stage2",
                            "visualize", "visualize_ft", "all"])
    p.add_argument("--device", default=None)
    p.add_argument("--tag", default="", help="结果文件名后缀")
    p.add_argument("--seed", type=int, default=None,
                   help="只跑这一个种子(覆盖 yaml 的 seeds 列表,便于调试)")
    return p.parse_args()


def resolve_device(cfg, arg):
    if arg:
        return arg
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
def stage_cache(cfg, device):
    proto_e, val_e, test_e = pipeline.load_entries(cfg)
    return cache.build(cfg, {"pool": proto_e, "val": val_e, "test": test_e}, device=device)


# --------------------------------------------------------------------------- #
@torch.no_grad()
def stage_baseline(cfg, device, ctx=None):
    """Training-free prototype baseline (no adapter training).

    This is the reference row every fine-tuned stage is compared against; it also
    verifies that the differentiable heatmap path matches the numpy inference path
    exactly, so later grad-based stages cannot silently drift.
    """
    if ctx is None:
        ctx = pipeline.prepare(cfg, device)
    print("\n=== parity check: differentiable heatmap vs numpy inference path ===")
    d = HM.assert_parity(cfg, ctx["val"]["patch"], ctx["c_anom"], ctx["c_norm"])
    print(f"  max|diff| = {d:.2e}  (通过)")

    proto_e, val_e, test_e = ctx["entries"]

    if cv.uses_case_cv(ctx):
        hp, oof = cv.search_hparams_oof(cfg, ctx, text_pack=ctx.get("text_pack"))
        ctx["hp"] = hp
        ctx["fuse_w"] = hp.get("fuse_w")
        ctx["fuse_alpha"] = hp.get("fuse_alpha")
    else:
        hp, oof = {}, None

    sv = SG.compute_ctx(cfg, ctx["val"], ctx)
    st = SG.compute_ctx(cfg, ctx["test"], ctx)
    yv, yt = fusion.to_np(ctx["val"]["label"]), fusion.to_np(ctx["test"]["label"])
    sv_search, y_search, lab_search = sv, yv, ctx["val"]["label"]
    use_oof = oof is not None

    rows = []

    def _alpha_for(heat_key, s_heat_v):
        if use_oof and heat_key in oof[0]["sig"]:
            a, auc_v, _ = cv.search_alpha_oof(oof, heat_key)
            return a, auc_v
        a, auc_v, _ = fusion.search_alpha_fuse(
            s_heat_v, sv_search["S_text"], y_search, np.arange(0.0, 1.05, 0.05))
        return a, auc_v

    def _score_test(s_heat_t, a):
        h = fusion.zscore_apply(fusion.to_np(s_heat_t), *fusion.zscore_fit(fusion.to_np(s_heat_t)))
        t = fusion.zscore_apply(fusion.to_np(st["S_text"]), *fusion.zscore_fit(fusion.to_np(st["S_text"])))
        return a * h + (1.0 - a) * t

    def run(tag, s_heat_v, s_heat_t, les_key, hm_key, heat_key):
        """一套口径的完整评估:OOF/val 搜超参 → test 出指标。"""
        a, auc_v = _alpha_for(heat_key, s_heat_v)
        score_t = _score_test(s_heat_t, a)
        auc_t, ap_t = pmetrics.image_metrics(score_t, yt)

        scale = float(getattr(cfg, "adapter_logit_scale", 100.0))
        lv = SG.fused_logits(sv_search, 0.0, 0.0, adapter_scale=scale)
        if use_oof:
            lam = 1.0
        else:
            lam, _ = pfusion.search_lambda(lv, sv_search[les_key], lab_search.to(lv.device))
        lt = SG.fused_logits(st, 0.0, 0.0, adapter_scale=scale)

        # 每一行都要用自己的热力图:full_report 的像素指标走 hm_for_pixel,
        # 但 hit@k / gap_sigma / peak_gap_z / lesion_z 走 sig_test["s_flat"]。
        # 之前只在 hm_fused 时同步 s_flat,导致 text-only / mem-only / A 行的
        # 定位诊断列报的是**原型图**的数(实测 hit@1 与原型行逐位相同)。
        st_rep = dict(st)
        _skey = {"hm_fused": "s_flat_fused", "hm": "s_flat", "hm_paclip": "s_flat_paclip"}
        if hm_key in st and hm_key != "hm":
            st_rep["hm"] = st[hm_key]
            _sk = _skey.get(hm_key)
            st_rep["s_flat"] = st[_sk] if (_sk and _sk in st) else \
                st[hm_key].reshape(len(yt), -1)
        rep = evaluate.full_report(
            cfg, st_rep, ctx["test"]["label"], test_e,
            hm_for_pixel=st[hm_key], tau=cfg.anomaly_tau,
        )
        rep = {k: v for k, v in rep.items() if not k.startswith("_")}
        rep.update({
            "image_auc": auc_t, "image_ap": ap_t,
            "acc_image_only": pmetrics.top1_acc(lt, ctx["test"]["label"]),
            "acc_lesion_fused": pmetrics.top1_acc(lt + lam * st[les_key],
                                                           ctx["test"]["label"]),
            "alpha_fuse": a, "lam_fuse": float(lam), "oof_img_auc": auc_v,
        })
        rows.append((tag, rep))

    run("A paclip(原样)", sv_search["S_heat_paclip"], st["S_heat_paclip"],
        "clip_les_attn", "hm_paclip", "S_heat_paclip")
    run("B +raw_max图像分", sv_search["S_heat_raw"], st["S_heat_raw"],
        "clip_les_attn", "hm", "S_heat_raw")
    run("C +均匀池化", sv_search["S_heat_raw"], st["S_heat_raw"],
        "clip_les_uniform", "hm", "S_heat_raw")
    if "S_heat_text" in st:
        run("text-only", sv.get("S_heat_text", sv_search.get("S_heat_text")),
            st["S_heat_text"], "clip_les_uniform", "hm_text", "S_heat_text")
    if "S_heat_mem" in st:
        run("mem-only", sv.get("S_heat_mem", sv_search.get("S_heat_mem")),
            st["S_heat_mem"], "clip_les_uniform", "hm_mem", "S_heat_mem")
    if "S_heat_fused" in st:
        run("fused", sv.get("S_heat_fused", sv_search.get("S_heat_fused")),
            st["S_heat_fused"], "clip_les_uniform", "hm_fused", "S_heat_fused")

    evaluate.print_table("Stage 0: training-free baseline + cheap wins (test set)", rows)
    report.save(cfg, "stage0_baseline", rows, meta={
        "prototypes": ctx["pr"]["meta"],
        "smooth_kernel": cfg.smooth_kernel,
        "q_img": cfg.q_img,
        "note": "row A = training-free baseline; rows B/C add one cheap win each",
    })

    m14 = _mask14(cfg, test_e, yt)
    print("\n--- 定位诊断:ours 口径(原始分 + 评分前平滑)---")
    print(diag.summarize("ours", st["s_flat"][yt > 0], m14))
    print("\n--- 定位诊断:paclip 口径(作对照)---")
    print(diag.summarize("paclip", st["hm_paclip"].reshape(len(yt), -1)[yt > 0], m14))

    return {"rows": rows, "val": sv, "test": st, "ctx": ctx}


def _mask14(cfg, entries, labels):
    from paclipf import data as pdata

    ms = []
    for e, y in zip(entries, labels):
        if y > 0 and e.get("mask_path"):
            ms.append(torch.tensor(pdata.mask_grid_14(str(cfg.data_root) + "/" + e["mask_path"])))
    return torch.stack(ms).reshape(len(ms), -1).bool()


# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_model(cfg, ctx, ad_img, ad_les, tag, c_norm=None, c_anom=None, rows=None):
    """OOF/val 上的超参 → test 出指标。支持集自己评自己不再作为主路径。"""
    c_norm = ctx["c_norm"] if c_norm is None else c_norm
    c_anom = ctx["c_anom"] if c_anom is None else c_anom
    hp = dict(ctx.get("hp") or {})
    if hp.get("fuse_w") is None and ctx.get("fuse_w") is not None:
        hp["fuse_w"] = ctx["fuse_w"]
    if hp.get("fuse_alpha") is None and ctx.get("fuse_alpha") is not None:
        hp["fuse_alpha"] = ctx["fuse_alpha"]
    if hp.get("fuse_w") is not None:
        ctx["fuse_w"] = hp["fuse_w"]
    scale = float(getattr(cfg, "adapter_logit_scale", 100.0))

    missing_fuse = hp.get("fuse_w") is None or hp.get("fuse_alpha") is None
    missing_pt = ad_img is not None and "alpha_pt" not in hp
    oof = None
    if cv.uses_case_cv(ctx) and (missing_fuse or missing_pt):
        print("[eval] hp 不齐,补跑 OOF 搜索")
        hp2, oof = cv.search_hparams_oof(
            cfg, ctx, adapter_img=ad_img, adapter_lesion=ad_les,
            text_pack=ctx.get("text_pack"))
        hp.update({k: v for k, v in hp2.items() if hp.get(k) is None})
        ctx["hp"] = hp
        if hp.get("fuse_w") is not None:
            ctx["fuse_w"] = hp["fuse_w"]

    sv = SG.compute_ctx(cfg, ctx["val"], ctx, c_norm=c_norm, c_anom=c_anom,
                        adapter_img=ad_img, adapter_lesion=ad_les)
    st = SG.compute_ctx(cfg, ctx["test"], ctx, c_norm=c_norm, c_anom=c_anom,
                        adapter_img=ad_img, adapter_lesion=ad_les)
    yv, yt = fusion.to_np(ctx["val"]["label"]), fusion.to_np(ctx["test"]["label"])

    out = {"tag": tag}
    if ad_img is None:
        a_pt, l_pt, acc_v, edge = 0.0, 0.0, float("nan"), False
    elif "alpha_pt" in hp:
        a_pt, l_pt = float(hp["alpha_pt"]), float(hp.get("lam_pt", 1.0))
        acc_v, edge = hp.get("oof_acc", float("nan")), hp.get("edge_warning", False)
    elif not cv.uses_case_cv(ctx):
        a_pt, l_pt, acc_v, grid, edge = fusion.search_fusion(
            cfg, sv, ctx["val"]["label"], pmetrics.top1_acc)
    else:
        if oof is None:
            _, oof = cv.search_hparams_oof(
                cfg, ctx, adapter_img=ad_img, adapter_lesion=ad_les,
                text_pack=ctx.get("text_pack"))
        a_pt, l_pt, acc_v, _, edge = cv.search_fusion_oof(cfg, oof, pmetrics.top1_acc)
    lt = SG.fused_logits(st, a_pt, l_pt, adapter_scale=scale)
    out.update({
        "alpha_pt": a_pt, "lam_pt": l_pt, "val_acc": acc_v, "edge_warning": bool(edge),
        "acc_image_only": pmetrics.top1_acc(
            SG.fused_logits(st, 0.0, 0.0, adapter_scale=scale), ctx["test"]["label"]),
        "acc_lesion_only": pmetrics.top1_acc(st["proto_les_uniform"], ctx["test"]["label"])
        if "proto_les_uniform" in st else float("nan"),
        "acc_fused": pmetrics.top1_acc(lt, ctx["test"]["label"]),
        "acc_text_fused": pmetrics.top1_acc(
            SG.fused_logits(st, 0.0, 0.0, adapter_scale=scale) + st["clip_les_uniform"],
            ctx["test"]["label"]),
    })
    if edge:
        print(f"  [warn] {tag}: 最优 (α_pt,λ_pt)=({a_pt},{l_pt}) 落在网格边界,建议扩网格")

    heat_key = "S_heat_fused" if "S_heat_fused" in st else "S_heat_raw"
    if hp.get("fuse_alpha") is not None:
        a_f = float(hp["fuse_alpha"])
        auc_v = hp.get("oof_img_auc", float("nan"))
    elif oof is not None:
        a_f, auc_v, _ = cv.search_alpha_oof(oof, heat_key)
    else:
        a_f, auc_v, _ = fusion.search_alpha_fuse(sv[heat_key], sv["S_text"], yv)
    heat = st[heat_key] if heat_key in st else st["S_heat_raw"]
    h = fusion.zscore_apply(fusion.to_np(heat), *fusion.zscore_fit(fusion.to_np(heat)))
    t = fusion.zscore_apply(fusion.to_np(st["S_text"]), *fusion.zscore_fit(fusion.to_np(st["S_text"])))
    if "S_adapt_margin" in st:
        a = fusion.zscore_apply(
            fusion.to_np(st["S_adapt_margin"]),
            *fusion.zscore_fit(fusion.to_np(st["S_adapt_margin"])))
        if hp.get("w_adapt") is None and oof is not None:
            w3, auc3, _ = cv.search_three_oof(oof, heat_key)
            hp.update({"w_heat": w3[0], "w_text": w3[1], "w_adapt": w3[2],
                       "oof_img_auc": float(auc3)})
            auc_v = auc3
        if hp.get("w_adapt") is not None:
            wh, wt, wa = float(hp["w_heat"]), float(hp["w_text"]), float(hp["w_adapt"])
            score = wh * h + wt * t + wa * a
            out["w_heat"], out["w_text"], out["w_adapt"] = wh, wt, wa
        else:
            score = a_f * h + (1.0 - a_f) * t
            out["w_heat"], out["w_text"], out["w_adapt"] = float(a_f), float(1.0 - a_f), 0.0
    else:
        score = a_f * h + (1.0 - a_f) * t
        out["w_heat"], out["w_text"], out["w_adapt"] = float(a_f), float(1.0 - a_f), 0.0
    out["image_auc"], out["image_ap"] = pmetrics.image_metrics(score, yt)
    out["alpha_fuse"] = float(a_f)
    out["oof_img_auc"] = float(auc_v) if auc_v == auc_v else float("nan")
    out["auc_heat_alone"] = pmetrics.image_metrics(fusion.to_np(heat), yt)[0]
    out["auc_text_alone"] = pmetrics.image_metrics(fusion.to_np(st["S_text"]), yt)[0]
    if "S_adapt_margin" in st:
        out["auc_adapt_alone"] = pmetrics.image_metrics(fusion.to_np(st["S_adapt_margin"]), yt)[0]
        out["auc_3signal"] = out["image_auc"]

    hm_px = st.get("hm_fused", st["hm"])
    st_rep = dict(st)
    if "s_flat_fused" in st:
        st_rep["s_flat"] = st["s_flat_fused"]
        st_rep["hm"] = hm_px
    rep = evaluate.full_report(cfg, st_rep, ctx["test"]["label"], ctx["entries"][2],
                               hm_for_pixel=hm_px, tau=cfg.anomaly_tau)
    out.update({k: v for k, v in rep.items() if not k.startswith("_")})
    out["_sig"] = st
    if rows is not None:
        rows.append((tag, {k: v for k, v in out.items() if not k.startswith("_")}))
        _append_map_rows(cfg, ctx, st, yt, rows, prefix=tag)
    return out


def _append_map_rows(cfg, ctx, st, yt, rows, prefix=""):
    """proto / text / mem / fused 定位对照行(test)。"""
    variants = [("proto-only", "hm", "s_flat")]
    if "hm_text" in st:
        variants.append(("text-only", "hm_text", None))
    if "hm_mem" in st:
        variants.append(("mem-only", "hm_mem", None))
    if "hm_fused" in st:
        variants.append(("fused", "hm_fused", "s_flat_fused"))
    if len(variants) <= 1:
        return
    test_e = ctx["entries"][2]
    for name, hk, sk in variants:
        st_rep = dict(st)
        st_rep["hm"] = st[hk]
        st_rep["s_flat"] = st[sk] if sk else st[hk].reshape(len(yt), -1)
        rep = evaluate.full_report(cfg, st_rep, ctx["test"]["label"], test_e,
                                   hm_for_pixel=st[hk], tau=cfg.anomaly_tau)
        rows.append((f"{prefix} {name}".strip(),
                     {k: v for k, v in rep.items() if not k.startswith("_")}))


def _save_train_rows(cfg, stage, rows, s1=None, s2=None):
    """训练阶段的指标落盘,附带训练历史便于回看早停行为。"""
    meta = {
        "smooth_kernel": cfg.smooth_kernel,
        "proto_source": cfg.proto_source,
        "double_norm": cfg.double_norm,
        "train_scope": cfg.train_scope,
        "margin": cfg.margin,
        "arcface_scale": cfg.arcface_scale,
        "seed": getattr(cfg, "seed", None),
    }
    if s1 is not None:
        meta["stage1_hist"] = s1["hist"]
        meta["stage1_best_epoch"] = s1["best"]["epoch"]
    if s2 is not None:
        meta["stage2_hist"] = s2["hist"]
        meta["stage2_best_epoch"] = s2["best"]["epoch"]
    report.save(cfg, stage, rows, meta=meta)


def out_lam_text(cfg, sv, st, labels):
    """文本头 λ 融合(训练免费通路),作为对照行。"""
    lv = SG.fused_logits(sv, 0.0, 0.0)
    lam, _ = pfusion.search_lambda(lv, sv["clip_les_uniform"],
                                   labels.to(lv.device))
    return lam * st["clip_les_uniform"]


_THREE_KEYS = ("S_heat_raw", "S_text", "S_adapt_margin")


def _zfit_apply(src, dst):
    """按 src(val)拟合 z-score 统计量,应用到 dst(test)。返回三路 numpy 列表。"""
    out = []
    for k in _THREE_KEYS:
        m, s = fusion.zscore_fit(fusion.to_np(src[k]))
        out.append(fusion.zscore_apply(fusion.to_np(dst[k]), m, s))
    return out


def _search_three(sv, yv, step=0.1):
    """3 信号单纯形网格:z(S_heat), z(S_text), z(S_adapt)。只在 val 上搜。

    返回 ((w_heat, w_text, w_adapt), val_auc)。test 上的分数由 _apply_three
    用**同一组权重 + val 拟合的统计量**得到。
    """
    h, t, a = _zfit_apply(sv, sv)
    best = (-1.0, (1 / 3, 1 / 3, 1 / 3), h)
    grid = np.arange(0, 1.0001, step)
    for wh in grid:
        for wt in grid:
            wa = 1.0 - wh - wt
            if wa < -1e-9:
                continue
            sc = wh * h + wt * t + wa * a
            auc = pmetrics.image_metrics(sc, yv)[0]
            if auc > best[0]:
                best = (auc, (float(wh), float(wt), float(wa)), sc)
    return best[1], best[0]


def _apply_three(sv, st, w):
    h, t, a = _zfit_apply(sv, st)          # 统计量来自 val,应用到 test
    return w[0] * h + w[1] * t + w[2] * a


# --------------------------------------------------------------------------- #
@torch.no_grad()
def stage_visualize(cfg, device):
    """输出热力图对照。回答两个可自己核对的问题。

    注意一件容易搞错的事:**"raw_max 修正"不改变热力图**。
    逐图 z-score 是正仿射变换 (s-μ)/σ (σ>0),所以 sigmoid(τz) 与 s 的图内排序完全相同,
    渲染出来一模一样 —— 被修的是"取 max 还是 top-3 均值"这个标量汇总,不是图本身。
    因此可视化真正能展示的是另外两件事:
      (1) 平滑的破坏性 —— 反直觉,且视觉上一眼可见
      (2) 命中率的真实情况 —— 固定种子抽样,标签里标注每张命中与否(含漏检)
    """
    from pathlib import Path

    from paclipf import heatmap as HM2
    from paclipf import visualize as VZ

    ctx = pipeline.prepare(cfg, device)
    entries, y = ctx["entries"][2], ctx["test"]["label"]
    hm0 = HM2.heatmap_exact(ctx["test"]["patch"], ctx["c_anom"], ctx["c_norm"],
                            tuple(cfg.patch_layers), image_size=cfg.image_size, smooth_kernel=0)
    s_flat = hm0.reshape(len(y), -1)
    out = Path(cfg.results_dir).parent / "visualize"
    outs = []

    for tag, k, na, nn_ in (("compare_smoothing.png", 3, 6, 2),
                            ("compare_smoothing_k7.png", 7, 4, 1)):
        p, shape = VZ.render(
            cfg, entries, y, hm0, HM2.smooth_map(hm0, k), out / tag,
            left_title="no smoothing (k=0)", right_title=f"smoothed (k={k})",
            n_anom=na, n_norm=nn_, seed=0, s_flat=s_flat)
        outs.append((p, shape))
        print(f"[viz] {p.name}  {shape[1]}x{shape[0]}  (不平滑 vs 平滑 k={k})")

    # (3) Stage1 vs Stage2 —— 权重已落盘则直接读,不必重训(Stage 2 要 11 分钟)
    st1, st2 = train.load_state(cfg, "stage1"), train.load_state(cfg, "stage2")
    if st1 and st2:
        from paclipf import prototypes as PR

        proto = PR.LearnablePrototypes(ctx["c_norm"], ctx["c_anom"], cfg.double_norm)
        proto.load_state_dict(st2["proto"])
        c2n, c2a = proto.normalized()
        hm2 = HM2.heatmap_exact(ctx["test"]["patch"], c2a, c2n, tuple(cfg.patch_layers),
                                image_size=cfg.image_size, smooth_kernel=0)
        drift = 1.0 - float(c2a @ ctx["c_anom"])
        print(f"[viz] Stage2 原型相对初始的漂移 1-cos = {drift:.3f} "
              f"(0=没动, 1=正交)")
        p, shape = VZ.render(
            cfg, entries, y, hm0, hm2, out / "compare_stage1_vs_stage2.png",
            left_title="Stage1 (prototypes frozen)", right_title="Stage2 (prototypes learned)",
            n_anom=6, n_norm=2, seed=0,
            s_flat=hm2.reshape(len(y), -1), s_flat_left=hm0.reshape(len(y), -1))
        outs.append((p, shape))
        print(f"[viz] {p.name}  {shape[1]}x{shape[0]}  (Stage1 vs Stage2)")
    else:
        print("[viz] 未找到 stage1/stage2 权重,跳过对照(先跑 --stage stage2)")

    print(f"\n[viz] 输出目录 {out}")
    print("[viz] 切片为固定种子(seed=0)随机抽样,不是挑好看的;")
    print("      每行标签标注该切片 hit@1 命中与否,包括漏检的。")
    return outs


# --------------------------------------------------------------------------- #
def stage_visualize_ft(cfg, device):
    """PA-CLIP-F 单独出图:原图 | GT mask | 本方法的热图(Stage 2 学到的原型)。

    不与其他方法并排 —— 只呈现 PA-CLIP-F 自身的结果。切片固定种子随机抽样,
    行标签标注 hit@1 (含漏检)。
    """
    from pathlib import Path

    from paclipf import heatmap as HM2
    from paclipf import prototypes as PR
    from paclipf import visualize as VZ

    ctx = pipeline.prepare(cfg, device)
    entries, y = ctx["entries"][2], ctx["test"]["label"]

    st2 = train.load_state(cfg, "stage2")
    if not st2:
        print("[viz] 未找到 stage2 权重 —— 请先跑 --stage stage2")
        return []

    proto = PR.LearnablePrototypes(ctx["c_norm"], ctx["c_anom"], cfg.double_norm)
    proto.load_state_dict(st2["proto"])
    c_n, c_a = proto.normalized()
    hm = HM2.heatmap_exact(ctx["test"]["patch"], c_a, c_n, tuple(cfg.patch_layers),
                           image_size=cfg.image_size, smooth_kernel=cfg.smooth_kernel)
    s_flat = hm.reshape(len(y), -1)

    out = Path(cfg.results_dir).parent / "visualize"
    p, shape = VZ.render_single(cfg, entries, y, hm, out / "paclip_f_heatmaps.png",
                                title="PA-CLIP-F", n_anom=6, n_norm=2, seed=0,
                                s_flat=s_flat)
    print(f"[viz] {p}  {shape[1]}x{shape[0]}")
    return [(p, shape)]


def stage_text(cfg, device):
    ctx = pipeline.prepare(cfg, device)
    print("\n=== Stage text: Residual Adapter + 分层锚点 ===")
    out = train.train_text(cfg, ctx, device)
    rows = []
    evaluate_model(cfg, ctx, None, None, "text-adapter", rows=rows)
    evaluate.print_table("Stage text (test 集)", rows)
    report.save(cfg, "text", rows, meta={
        "best_ep": out["best_ep"], "fuse_w": out["fuse_w"],
        "fuse_alpha": out["fuse_alpha"], "fuse_meta": {
            k: v for k, v in out["fuse_meta"].items() if k != "table"
        },
        "seed": getattr(cfg, "seed", None),
    })
    return {"ctx": ctx, "rows": rows, **out}


def stage_train(cfg, device, which):
    ctx = pipeline.prepare(cfg, device)
    rows = []
    print("\n=== Stage 1: 冻结原型,训练分类适配器 ===")
    s1 = train.train_stage1(cfg, ctx, device)
    evaluate_model(cfg, ctx, s1["ad_img"], s1["ad_les"], "Stage1 (冻结原型)", rows=rows)

    if which == "stage2":
        print("\n=== Stage 2: 解冻原型 + 定位损失 ===")
        s2 = train.train_stage2(cfg, ctx, s1, device)
        c_norm, c_anom = s2["proto"].normalized()
        evaluate_model(cfg, ctx, s2["ad_img"], s2["ad_les"], "Stage2 (解冻原型)",
                       c_norm=c_norm, c_anom=c_anom, rows=rows)
        evaluate.print_table("Stage 1 vs Stage 2(test 集)", rows)
        _save_train_rows(cfg, "stage1_vs_stage2", rows, s1=s1, s2=s2)
        return {"ctx": ctx, "stage1": s1, "stage2": s2, "rows": rows}
    evaluate.print_table("Stage 1(test 集)", rows)
    _save_train_rows(cfg, "stage1", rows, s1=s1)
    return {"ctx": ctx, "stage1": s1, "rows": rows}


def _seeds(cfg, args):
    if getattr(args, "seed", None) is not None:
        return [int(args.seed)]
    seeds = getattr(cfg, "seeds", None)
    if seeds:
        return [int(s) for s in seeds]
    return [int(getattr(cfg, "seed", 111))]


def _apply_seed(cfg, seed, user_tag):
    cfg.seed = int(seed)
    tag = f"seed{int(seed)}"
    if user_tag:
        tag = f"{user_tag}_{tag}"
    cfg._tag = tag
    return tag


def _run_stage(cfg, device, stage):
    if stage == "baseline":
        return stage_baseline(cfg, device)
    if stage == "text":
        return stage_text(cfg, device)
    if stage in ("stage1", "stage2"):
        return stage_train(cfg, device, stage)
    if stage == "all":
        text_out = stage_text(cfg, device)
        train_out = stage_train(cfg, device, "stage2")
        return {"text": text_out, "train": train_out,
                "rows": (text_out or {}).get("rows"),
                "rows_stage2": (train_out or {}).get("rows")}
    raise ValueError(stage)


def _lite_out(out):
    """多种子循环里丢掉 ctx / 权重,避免把五份特征留在内存里。"""
    if not isinstance(out, dict):
        return out
    keep = {}
    for k, v in out.items():
        if k in ("ctx", "stage1", "stage2", "adapter", "pack", "ad_img", "ad_les", "proto"):
            continue
        if k in ("text", "train") and isinstance(v, dict):
            keep[k] = {"rows": v.get("rows")}
        else:
            keep[k] = v
    return keep


def _write_seed_summary(cfg, stage, collected, user_tag):
    tag = f"{user_tag}_meanstd" if user_tag else "meanstd"
    if stage == "all":
        text_rows = [(s, (o.get("text") or {}).get("rows") or []) for s, o in collected]
        train_rows = [(s, (o.get("train") or {}).get("rows") or []) for s, o in collected]
        report.save_mean_std(cfg, "text", text_rows, tag=tag)
        report.save_mean_std(cfg, "stage1_vs_stage2", train_rows, tag=tag)
        return
    name = {"baseline": "stage0_baseline", "text": "text",
            "stage1": "stage1", "stage2": "stage1_vs_stage2"}.get(stage, stage)
    seed_rows = [(s, (o or {}).get("rows") or []) for s, o in collected]
    report.save_mean_std(cfg, name, seed_rows, tag=tag)


# --------------------------------------------------------------------------- #
def main():
    args = get_args()
    cfg = C.load(args.config)
    cfg._tag = args.tag
    device = resolve_device(cfg, args.device)
    print(f"config: {args.config}\ndevice: {device}\ncache : {cfg.cache_dir}"
          + (f"\ntag   : {args.tag}" if args.tag else ""))

    t0 = time.time()
    if args.stage == "cache":
        stage_cache(cfg, device)
    elif args.stage == "visualize":
        stage_visualize(cfg, device)
    elif args.stage == "visualize_ft":
        stage_visualize_ft(cfg, device)
    else:
        seeds = _seeds(cfg, args)
        print(f"seeds : {seeds}" + ("  (--seed 覆盖 yaml)" if args.seed is not None else ""))
        collected = []
        for i, s in enumerate(seeds):
            _apply_seed(cfg, s, args.tag)
            if len(seeds) > 1:
                print(f"\n======== seed {s} ({i + 1}/{len(seeds)}) ========")
            out = _run_stage(cfg, device, args.stage)
            collected.append((s, _lite_out(out)))
        if len(seeds) > 1:
            _write_seed_summary(cfg, args.stage, collected, args.tag)
    print(f"\n总用时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
