# PA-CLIP-F

冻结 **BiomedCLIP** 的小样本医学异常检测。支持集只训很小的头（文本 Residual Adapter + 两个角空间适配器 + Stage 2 可学习原型），图像塔和文本塔全程冻结。

主数据是 **BMAD Brain**。主配置：`configs/brain_f.yaml`。

## 协议

身份只有一句：**冻结 BiomedCLIP，支持集 40 正常 + 44 异常，只训很小的头；test 3715 评一次。**

| 集合 | 内容 | 用途 |
|------|------|------|
| 支持集 | 40 正常（从 official train 按 `seed` 抽）+ 官方 valid 全部 44 异常（11 病例） | 原型、Memory、训练、选 epoch |
| leftover valid 正常 | 39 张 | **只作症例 CV 负例**（不进原型 / Memory / 训练） |
| official train 其余 | ~7460 张正常 | 只提供抽 40 张的池子，不训练、不选超参 |
| test | 3715 | 只评一次 |

超参（融合权重、`α`、epoch）由 **11 个 Ungood 病例留一**：折内重建原型和 Memory，在 held-out 异常 + leftover 正常上打分。主表报 test 上 `seeds: [111, 222, 333, 444, 555]` 的 **mean±std**。

本仓库自带 vendored `open_clip/`，不依赖外部 PA-CLIP 包。

## 环境与数据

```bash
pip install -r requirements.txt
```

`torch` / `torchvision` 按 CUDA 版本单独装：**Blackwell（RTX 50 系，sm_120）需要 torch ≥ 2.7 + cu128**；Ada / Ampere 用任意 torch ≥ 2.0 即可。

BiomedCLIP 权重首次运行时自动下载（约 748 MB）。国内建议先 `export HF_ENDPOINT=https://hf-mirror.com`，否则会很慢甚至卡住。

**数据需要自备**（BMAD Brain 不在本仓库内）。`configs/brain_f.yaml` 里两条路径按**本仓库的上一级目录**解析：

```yaml
data_root: 'AA-CLIP/data/MedAD/Brain_AD'              # 图像根目录,改成你自己的
meta_path: 'PA-CLIP-F/dataset/metadata/brain.jsonl'   # 元数据;本仓库若改名请同步改
```

元数据 `dataset/metadata/brain.jsonl` 随仓库提供（11298 行，字段 `image_path / label / class_name / split`），图像按 `split` 字段指向 `train/good/`、`valid/good/img/`、`valid/Ungood/img/`、`test/good/img/`、`test/Ungood/img/`。

## 运行

```bash
# 0. 特征缓存（一次性）
python main_f.py --config configs/brain_f.yaml --stage cache

# 1. 免训练原型基线
python main_f.py --config configs/brain_f.yaml --stage baseline

# 2. 文本 Residual Adapter
python main_f.py --config configs/brain_f.yaml --stage text

# 3. 文本 + Stage 1 + Stage 2（yaml 里 5 个 seed 依次跑，写出 mean±std 主表）
python main_f.py --config configs/brain_f.yaml --stage all

# 单次调试（覆盖 seeds 列表）
python main_f.py --config configs/brain_f.yaml --stage all --seed 111

# 只跑训练部分（跳过文本侧）
python main_f.py --config configs/brain_f.yaml --stage stage2

# 4. 热力图
python main_f.py --config configs/brain_f.yaml --stage visualize
```

`full` / `full_pool` 只许当消融，主表不出现。

## 产出物

```
results/
  stage0_baseline__brain_f__seed111.csv   单种子对照表
  stage0_baseline__brain_f__meanstd.csv   多种子 mean±std（主表）
  all_stages.csv
runs/brain_f/
  stage1__seed111.pt / stage2__seed111.pt
caches_brain/<指纹>/                      pool / val / test 特征
```

## 目录

```
main_f.py            入口 --stage {cache,baseline,text,stage1,stage2,visualize,all}
configs/brain_f.yaml 主配置
paclipf/
  cv.py              症例留一：折内重建原型/Memory，OOF 搜超参
  train.py           Stage 1 / Stage 2 / 文本适配器
  signals.py         热力图 / 文本 / Memory 信号与融合
  evaluate.py        test 指标（含 AUPRO）
  report.py          csv / json
```
