"""结果落盘 —— 每个阶段的指标写 results/<stage>.{csv,json}。

在此之前所有指标只 print 到终端,管道一截断就没了。这里保证:
  · results/<stage>.csv   一张宽表,每行一个对照项,可直接拖进 Excel
  · results/<stage>.json  同样内容 + 完整元信息(配置、原型统计、耗时)
  · results/all_stages.csv  累加所有阶段,便于跨阶段横向对比
"""
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

# CSV 里的列顺序(只列关心的;其余仍会写入 json)
_PREFERRED = [
    "image_auc", "image_ap", "pixel_auc_paclip", "pixel_ap_paclip",
    "pixel_auc_ours", "pixel_ap_ours", "pixel_aupro",
    "hit@1_valid", "hit@1_strict", "hit@3_valid", "hit@3_strict",
    "hit@10_valid", "hit@10_strict",
    "gap_sigma", "peak_gap_z", "lesion_z", "background_z",
    "acc_image_only", "acc_lesion_only", "acc_fused", "acc_text_fused",
    "alpha_fuse", "alpha_pt", "lam_pt", "lam_fuse", "oof_img_auc", "val_acc",
    "auc_heat_alone", "auc_text_alone", "auc_adapt_alone", "auc_3signal",
    "w_heat", "w_text", "w_adapt", "n_valid", "n_total", "n_seeds",
]


def _clean(v):
    """转成可写 CSV 的值。"""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, torch.Tensor):
        v = v.item() if v.numel() == 1 else v.tolist()
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return v


def _meta(cfg, extra=None):
    m = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config_path": getattr(cfg, "_config_path", None),
        "split": {"val_cases": cfg.val_cases, "test_cases": cfg.test_cases},
        "image_size": cfg.image_size,
        "patch_layers": list(cfg.patch_layers),
    }
    # 把配置里所有可 JSON 化的键也带上,便于复现
    cfgdump = {}
    for k, v in vars(cfg).items():
        if k.startswith("_"):
            continue
        try:
            json.dumps(v)
            cfgdump[k] = v
        except (TypeError, ValueError):
            cfgdump[k] = str(v)
    m["config"] = cfgdump
    if extra:
        m.update(extra)
    return m


def save(cfg, stage, rows, meta=None, verbose=True):
    """rows: [(label, metrics_dict)]。写 csv + json,并累加到 all_stages.csv。

    meta 里放该阶段特有的信息(原型统计、训练历史、超参搜索结果等)。
    """
    out = Path(cfg.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    # 配置名并入文件名:同名 stage 用不同 config 跑时不再互相覆盖
    # (踩过一次:full_pool 的实验覆盖掉了 few_shot 的基线表)
    stem = getattr(cfg, "_config_stem", "")
    if stem:
        stage = f"{stage}__{stem}"
    # --tag 后缀:同一 config 跑多组(多种子/多超参)时靠它区分,否则互相覆盖
    tag = getattr(cfg, "_tag", "")
    if tag:
        stage = f"{stage}__{tag}"

    # ---- 列顺序:先 _PREFERRED,再补齐实际出现的其余列 ----
    seen = []
    for _, d in rows:
        for k in d:
            if not k.startswith("_") and k not in seen:
                seen.append(k)
    cols = [c for c in _PREFERRED if c in seen] + [c for c in seen if c not in _PREFERRED]

    csv_p = out / f"{stage}.csv"
    with open(csv_p, "w", newline="", encoding="utf-8-sig") as f:   # BOM 让 Excel 正确识别中文
        w = csv.writer(f)
        w.writerow(["variant"] + cols)
        for label, d in rows:
            w.writerow([label] + [_clean(d.get(c)) for c in cols])

    json_p = out / f"{stage}.json"
    with open(json_p, "w", encoding="utf-8") as f:
        json.dump({
            "stage": stage,
            "meta": _meta(cfg, meta),
            "rows": [{"variant": lb, **{k: _clean(v) for k, v in d.items()
                                        if not k.startswith("_")}} for lb, d in rows],
        }, f, ensure_ascii=False, indent=2)

    # ---- 累加到总表 ----
    all_p = out / "all_stages.csv"
    new_file = not all_p.exists()
    with open(all_p, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["stage", "variant"] + cols)
        for label, d in rows:
            w.writerow([stage, label] + [_clean(d.get(c)) for c in cols])

    if verbose:
        print(f"[report] 已写入 {csv_p.name} / {json_p.name} / all_stages.csv  ({len(rows)} 行)")
    return csv_p, json_p


def aggregate_seed_rows(seed_rows):
    """seed_rows: [(seed, [(variant, metrics)])] → [(variant, mean/std dict)]。

    这是 test 上多种子的 mean±std,不要拿 11 折 OOF 方差冒充。
    """
    variants = []
    for _, rows in seed_rows:
        if not rows:
            continue
        for lab, _ in rows:
            if lab not in variants:
                variants.append(lab)
    out = []
    for lab in variants:
        dicts = []
        for seed, rows in seed_rows:
            if not rows:
                continue
            for l, d in rows:
                if l == lab:
                    dicts.append(d)
                    break
        if not dicts:
            continue
        merged = {"n_seeds": len(dicts)}
        keys = []
        for d in dicts:
            for k in d:
                if not k.startswith("_") and k not in keys:
                    keys.append(k)
        for k in keys:
            vals = []
            for d in dicts:
                v = d.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v:
                    vals.append(float(v))
            if len(vals) >= 1:
                merged[k] = float(np.mean(vals))
                merged[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            else:
                merged[k] = dicts[0].get(k)
        out.append((lab, merged))
    return out


def save_mean_std(cfg, stage, seed_rows, verbose=True, tag="meanstd"):
    """多种子 test 指标的 mean±std 落盘。"""
    rows = aggregate_seed_rows(seed_rows)
    if not rows:
        return None
    seeds = [s for s, _ in seed_rows]
    prev = getattr(cfg, "_tag", "")
    cfg._tag = tag
    try:
        return save(cfg, stage, rows, meta={
            "seeds": seeds, "n_seeds": len(seeds),
            "note": "test metrics mean±std over seeds; not OOF-fold std",
        }, verbose=verbose)
    finally:
        cfg._tag = prev
