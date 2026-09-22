# PA-CLIP

冻结 **BiomedCLIP** 的小样本医学异常检测。图像塔与文本塔全程冻结，只训很小的头：
**文本 Residual Adapter + 多层 Patch Adapter + 角空间分类适配器 + 分层可学习原型**。

主数据是 **BMAD Brain**（BraTS2021 切片）。主配置：`configs/brain_f.yaml`。

## 架构

**阶段一 · 原型构建与文本适配**

少量正常/异常样本 → 冻结 BiomedCLIP → 两路：

- 视觉特征（全局 CLS + 多层 patch）→ **多正常/异常原型**（每层独立 K-means，正常 6 簇 + 异常 6 簇）
  → 位置感知 **Patch Memory Bank**（每条记忆带 14×14 网格坐标，匹配时乘高斯位置核）
- 文本提示（正常解剖描述 / 异常视觉属性描述，三层）→ **文本 Residual Adapter**（可训练，`W_up` 零初始）
  → 分层正常/异常文本锚点

$$L_{text} = L_{global} + \lambda_p L_{patch} + \lambda_d L_{dis} + \lambda_r L_{preserve} + \lambda_v L_{div}$$

**阶段二 · 视觉适配与局部对齐**

多层特征（layer 3/6/9/11）→ **多层 Patch Adapter**（共享一套残差瓶颈权重，`W_up` 零初始，
`z' = Norm(z + λ_v·W_up·GELU(W_down z))`）→ 与冻结文本锚点、可学习原型对齐。

$$L_{visual} = L_{loc} + \alpha\,L_{cls} + \lambda_f L_{focal} + \lambda_{tv} L_{tversky} + \lambda_c L_{contrast} + \lambda_s L_{consistency}$$

**推理 · 定位与后处理**

三路热图 $A_{text}$ / $A_{proto}$ / $A_{mem}$ → **校准与空间自适应融合**（逐位置 softmax 门控，
带 `heatmap_min_w` 下限）→ 高分辨率异常热力图 → Top-k 病灶池化 → 图像级判别；
另支 → **引导滤波边界细化**（He 2010，原图作引导）→ 形态学闭运算 + Top-k 连通域 → 二值掩膜。

阈值 $\tau$ 只在 OOF 上选（最大化 Dice），不在 test 上调。

## 协议

身份只有一句：**冻结 BiomedCLIP，支持集 40 正常 + 44 异常，只训很小的头；test 3715 评一次。**

| 集合 | 内容 | 用途 |
|------|------|------|
| 支持集 | 40 正常（从 official train 按 `seed` 抽）+ 官方 valid 全部 44 异常（11 病例） | 原型、Memory、训练、选 epoch |
| leftover valid 正常 | 39 张 | **只作症例 CV 负例**（不进原型 / Memory / 训练） |
| official train 其余 | ~7460 张正常 | 只提供抽 40 张的池子，不训练、不选超参 |
| test | 3715 | 只评一次 |

超参（融合权重、`α`、`τ`、epoch）由 **11 个 Ungood 病例留一**：折内重建原型和 Memory，
在 held-out 异常 + leftover 正常上打分。主表报 test 上 `seeds: [111, 222, 333, 444, 555]` 的 **mean±std**。

本仓库自带 vendored `open_clip/`，不依赖外部 PA-CLIP 包。

## 环境与数据

```bash
pip install -r requirements.txt
```

`torch` / `torchvision` 按 CUDA 版本单独装：**Blackwell（RTX 50 系，sm_120）需要 torch ≥ 2.7 + cu128**；Ada / Ampere 用任意 torch ≥ 2.0 即可。

BiomedCLIP 权重首次运行时自动下载（约 748 MB）。国内建议先 `export HF_ENDPOINT=https://hf-mirror.com`。

**数据需要自备**（BMAD Brain 不在本仓库内）。`configs/brain_f.yaml` 里两条路径：

```yaml
data_root: 'AA-CLIP/data/MedAD/Brain_AD'    # 相对本仓库的上一级目录;改成你自己的
meta_path: 'dataset/metadata/brain.jsonl'   # 相对本仓库根;随仓库提供
```

`data_root` 相对**本仓库的上一级目录**解析（数据通常放在仓库外），`meta_path` / `cache_dir` / `results_dir` 相对**本仓库根**解析 —— 所以仓库文件夹叫什么名字都不影响。

元数据 `dataset/metadata/brain.jsonl` 随仓库提供（11298 行，字段 `image_path / label / class_name / split`）。

## 运行

```bash
# 0. 特征缓存（一次性，约 15 GB）
python main_f.py --config configs/brain_f.yaml --stage cache

# 1. 免训练原型基线
python main_f.py --config configs/brain_f.yaml --stage baseline

# 2. 文本 Residual Adapter
python main_f.py --config configs/brain_f.yaml --stage text

# 3. 文本 + Stage1 + Stage2（yaml 里 5 个 seed 依次跑，写出 mean±std 主表）—— 约 131 分钟 / RTX 4090
python main_f.py --config configs/brain_f.yaml --stage all

# 4. 热力图
python main_f.py --config configs/brain_f.yaml --stage visualize
```

`full` / `full_pool` 只许当消融，主表不出现。

## 结果（BMAD Brain，test 3715，5 seed mean±std）

**Stage2（解冻原型）**

| 变体 | image_auc | pixel_auc | pixel_ap | AUPRO | hit@1 | gap_σ |
|------|-----------|-----------|----------|-------|-------|-------|
| **主行（门控融合）** | **0.9646** | 0.9879 | 0.7902 | 0.8903 | 0.9437 | 2.643 |
| proto-only | — | **0.9910** | **0.8401** | **0.8985** | **0.9584** | 2.668 |
| text-only | — | 0.9833 | 0.7602 | 0.8415 | 0.8890 | 3.057 |
| mem-only | — | 0.9476 | 0.4760 | 0.7337 | 0.6087 | 1.693 |

主行标准差：`hit@1` ±0.0090、`pixel_aupro` ±0.0029、`image_auc` ±0.0058。

**二值掩膜（引导滤波后处理，τ 由 OOF 选出）**：Dice **0.7091** ± 0.0059，IoU **0.5899** ± 0.0088，τ = 0.79 ± 0.022。

**Stage1 → Stage2**：`hit@1` 0.8970 → 0.9437，`AUPRO` 0.8713 → 0.8903，`image_auc` 0.9585 → 0.9646。

## 已知局限

- **图像级分类头退化。** `acc_image_only = acc_lesion_only = acc_fused = acc_text_fused = 82.77%`
  （= 3075/3715，即"全部判异常"的平凡基线），`S_text` 的 AUC 为 0.500。
  CLS 级文本判别在脑 MRI 上不起作用，图像分实际来自热图与适配器两路。
- **空间门控存在自选择。** 门控在 OOF 上以 Adam 训 40 epoch（163 参数），
  同一批切片又用于报告 OOF 指标，因此 OOF 的 `hit@1` 偏高（0.9303 vs 线性 0.8850）；
  test 上主行反而低于 proto-only。要作为结论需留出未训练的折。
- **`α_pt` 搜索撞网格边界**（Stage1 撞上界 50，Stage2 撞下界 0），该列不可用。

## 目录

```
main_f.py            入口 --stage {cache,baseline,text,stage1,stage2,visualize,visualize_ft,all}
configs/brain_f.yaml 主配置
paclipf/
  grid.py            14x14 patch 网格坐标 + 嵌套配置读取
  prototypes.py      分层 / 多异常原型构建(K-means + 双重 L2 归一化)
  patch_adapter.py   多层共享残差瓶颈(视觉侧)
  text_adapter.py    残差适配器 + 三层提示词编码
  signals.py         三路热图 / 文本 / Memory 信号
  spatial.py         逐位置空间自适应融合门控
  postprocess.py     上采样 / 引导滤波 / 闭运算 / Top-k 连通域
  cv.py              症例留一：折内重建原型/Memory，OOF 搜超参
  train.py           Stage1 / Stage2 / 文本适配器
  evaluate.py        test 指标（含 AUPRO）
  report.py          csv / json
```
