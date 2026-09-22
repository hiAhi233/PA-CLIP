"""配置加载 + 派生量校验。

约定:
  cache_dir / results_dir —— 相对 **本仓库根目录** 解析
  meta_path              —— 相对 **本仓库根目录** 解析(元数据随仓库提供)
  data_root              —— 相对 **本仓库的上一级目录** 解析(数据自备,通常放在仓库外)

不依赖当前工作目录 —— 这样从任何 cwd 调用、仓库改成任何文件夹名结果都一致。
"""
import os
from pathlib import Path
from types import SimpleNamespace

import yaml

PKG_ROOT = Path(__file__).resolve().parents[1]          # 本仓库根目录
PROJECT_ROOT = PKG_ROOT.parent                          # 上一级目录(放数据用)


def _abs(p, base=PKG_ROOT):
    """相对路径按 base 解析;绝对路径原样返回。"""
    p = Path(p)
    return p if p.is_absolute() else (base / p)


def load(config_path):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = dict(cfg)
    cfg["_config_path"] = str(Path(config_path).resolve())
    cfg["_config_stem"] = Path(config_path).stem

    cfg["cache_dir"] = str(_abs(cfg.get("cache_dir", "caches")))
    cfg["results_dir"] = str(_abs(cfg.get("results_dir", "results")))
    # meta_path 随仓库分发 → 按 PKG_ROOT 解析,仓库文件夹叫什么名都不影响。
    # data_root 是外部数据 → 按上一级目录解析。
    if cfg.get("meta_path"):
        cfg["meta_path"] = str(_abs(cfg["meta_path"], PKG_ROOT))
    if cfg.get("data_root"):
        cfg["data_root"] = str(_abs(cfg["data_root"], PROJECT_ROOT))

    if "text" in cfg and isinstance(cfg["text"], dict):
        cfg["text"] = SimpleNamespace(**cfg["text"])
    elif "text" not in cfg:
        cfg["text"] = SimpleNamespace(enable=False)
    for key in ("patch_adapter", "visual", "spatial_fuse", "postprocess", "memory"):
        if key in cfg and isinstance(cfg[key], dict):
            cfg[key] = SimpleNamespace(**cfg[key])

    _validate(cfg)
    return SimpleNamespace(**cfg)


def _validate(cfg):
    """校验派生量,提前暴露形状不匹配,而不是等到训练中途。"""
    ps, img = 16, cfg["image_size"]          # BiomedCLIP ViT-B/16
    if img % ps:
        raise ValueError(f"image_size={img} 不是 patch_size={ps} 的整数倍")
    cfg["_grid"] = img // ps
    cfg["_n_patch"] = cfg["_grid"] ** 2

    if cfg["_n_patch"] != 196:
        raise ValueError(
            f"当前实现假定 14x14=196 个 patch(与 PA-CLIP 的 224px 一致),"
            f"但 image_size={img} 给出 {cfg['_grid']}x{cfg['_grid']}={cfg['_n_patch']}。"
            f"若确要改分辨率,需同步复核 mask_grid_14 与所有硬编码的 196。"
        )

    if cfg["score_mode"] not in ("raw_max", "zscore_max"):
        raise ValueError(f"score_mode 只支持 raw_max / zscore_max,得到 {cfg['score_mode']}")
    if cfg["proto_source"] not in ("few_shot", "full_pool"):
        raise ValueError(f"proto_source 只支持 few_shot / full_pool,得到 {cfg['proto_source']}")
    if cfg["train_scope"] not in ("full", "few_shot"):
        raise ValueError(f"train_scope 只支持 full / few_shot,得到 {cfg['train_scope']}")
    tscope = getattr(cfg.get("text"), "train_scope", None) if cfg.get("text") is not None else None
    if tscope is not None and tscope not in ("full", "few_shot"):
        raise ValueError(f"text.train_scope 只支持 full / few_shot,得到 {tscope}")
    if cfg["lesion_source"] not in ("gt_mask", "heatmap_topq"):
        raise ValueError(f"lesion_source 只支持 gt_mask / heatmap_topq,得到 {cfg['lesion_source']}")
    if cfg["reduce"] not in ("max", "lse"):
        raise ValueError(f"reduce 只支持 max / lse,得到 {cfg['reduce']}")
    if cfg.get("seeds") is not None:
        cfg["seeds"] = [int(s) for s in cfg["seeds"]]

    overlap = set(cfg["val_cases"]) & set(cfg["test_cases"])
    if overlap:
        raise ValueError(f"val/test 病例重叠,会泄漏: {overlap}")

    bs = cfg.get("brain_split") or None
    if bs not in (None, "", "support44", "legacy_22"):
        raise ValueError(f"brain_split 只支持 support44 / legacy_22 / 空,得到 {bs}")
    cfg["brain_split"] = bs if bs else None

    t = cfg.get("text")
    enable = bool(getattr(t, "enable", False)) if t is not None else False
    if enable and not cfg.get("text_prompts"):
        raise ValueError("text.enable=true 时必须提供 text_prompts")
