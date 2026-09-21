"""原型构建 —— 修 PA-CLIP 缺失的 channel-wise 归一化。

PA-CLIP 的 pa_clip/prototypes.py:21 只做了 class-wise 那一次:
    centers = l2norm(centers, dim=-1)          # (K,D) 沿特征轴
而 Proto-Adapter(Sensors 2024,表 3)的做法是两次都做,且顺序固定:
    C = F.normalize(C, dim=0)   # channel-wise:每个特征维在各类原型间单位化
    C = F.normalize(C, dim=1)   # class-wise:每个原型单位化

**为什么必须从原始中心重算而不是对已存的 c_norm.pt 补一次**:
两次归一化不可交换,且 PA-CLIP 是在归一化之后才转置的,顺序与论文不同。
对已归一化的结果补做 dim=0 与在原始中心上做 dim=0 不等价。

关于"7 个原型一起归一化还是只归一化 6 个正常原型":
本实现把 6 个正常中心 + 1 个异常均值堆成 (7,D) **一起**归一化。
理由:PA-CLIP 的打分是 sim(p,c_anom) − max_k sim(p,c_norm^k),两项必须同尺度,
否则差值有偏。Proto-Adapter 也是在全部类别原型上一起归一化。
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


def _kmeans_centers(x, k, seed, max_samples=200_000, n_init=3):
    """K-means 中心 + 簇占比。

    退化到 max_samples 抽样:pool 的 patch 可达 178 万,KMeans(n_init=10) 会跑很久,
    而只要放 6 个中心,20 万样本足够稳定。抽样只在超出 max_samples 时发生,
    所以 few_shot(40×196=7840)时**不会**抽样 —— 此时 n_init 是唯一影响复现的参数,
    baseline 关卡需要把它设成 10 以逐位对齐 PA-CLIP。
    """
    n = x.shape[0]
    if n < k:
        raise ValueError(f"样本数 {n} 少于簇数 {k}")
    rng = np.random.RandomState(seed)
    if n > max_samples:
        sub = x[rng.choice(n, max_samples, replace=False)]
    else:
        sub = x
    km = KMeans(n_clusters=k, random_state=seed, n_init=n_init).fit(sub)
    centers = torch.from_numpy(km.cluster_centers_).float()
    # 用全量样本重新分配,得到更准的簇占比(只用中心的最近邻,便宜)
    full = torch.from_numpy(x) if not torch.is_tensor(x) else x
    assign = torch.cdist(full, centers).argmin(dim=1)
    pi = torch.bincount(assign, minlength=k).float()
    pi = pi / pi.sum()
    return centers, pi


def build_from_features(
    patch_l11,
    mask14,
    label,
    has_mask,
    k=6,
    double_norm=True,
    seed=111,
    kmeans_max=200_000,
    n_init=3,
    include_lesion_outside=False,
):
    """从缓存的 (layer 11) patch 特征构建原型。

    patch_l11 : (N, 196, D) 已 L2 归一化的 patch 特征
    mask14    : (N, 14, 14) bool,病灶掩膜(无 mask 的样本为全零)
    label     : (N,) 0=正常 1=异常
    has_mask  : (N,) bool,meta 里是否有 mask_path

    返回 dict:
      c_norm       (D, K)  L2 归一化后的正常原型
      c_anom       (D,)    L2 归一化后的异常原型
      c_norm_raw   (K, D)  归一化前的中心(Stage 2 的参数初始化用)
      c_anom_raw   (D,)    归一化前的均值
      pi           (K,)    各正常簇的样本占比(适配器加权初始化用)
      meta         dict    样本量等,便于在日志里如实报告
    """
    n, n_patch, D = patch_l11.shape
    patch_l11 = patch_l11.float()
    mask_flat = mask14.reshape(n, n_patch)
    anom = label == 1
    norm_idx = label == 0

    # ---------------- 正常 patch 池 ----------------
    # PA-CLIP 的注释声称包含"异常切片 mask 外的 patch",但代码里没有。
    # 这里默认复现 PA-CLIP 的实际行为(只用正常切片),开关打开则补上注释承诺的部分。
    parts = [patch_l11[norm_idx].reshape(-1, D)]
    n_from_normal = int(norm_idx.sum()) * n_patch
    n_from_outside = 0
    if include_lesion_outside:
        les = anom & has_mask
        if les.any():
            outside = patch_l11[les][~mask_flat[les]]
            parts.append(outside)
            n_from_outside = outside.shape[0]
    pool = torch.cat(parts, 0).numpy().astype(np.float32)

    # ---------------- 异常 patch 池 ----------------
    valid = anom & has_mask & mask_flat.any(dim=1)      # 14x14 下 mask 为空的切片无监督信号
    if not valid.any():
        raise RuntimeError("没有任何非空 mask 的异常切片,无法构建异常原型")
    a_patches = patch_l11[valid][mask_flat[valid]]      # (P, D)
    c_anom_raw = a_patches.mean(0)

    # ---------------- K-means ----------------
    centers, pi = _kmeans_centers(pool, k, seed, kmeans_max, n_init)

    # ---------------- 双重 L2 归一化 ----------------
    stacked = torch.cat([centers, c_anom_raw[None]], 0)          # (K+1, D)
    if double_norm:
        stacked = F.normalize(stacked, dim=0)                    # channel-wise ← PA-CLIP 缺的
    stacked = F.normalize(stacked, dim=1)                        # class-wise

    c_norm = stacked[:k].t().contiguous()                        # (D, K)
    c_anom = stacked[k].contiguous()                             # (D,)

    return {
        "c_norm": c_norm,
        "c_anom": c_anom,
        "c_norm_raw": centers,
        "c_anom_raw": c_anom_raw,
        "pi": pi,
        "meta": {
            "k": k,
            "double_norm": double_norm,
            "n_normal_patches": int(pool.shape[0]),
            "n_from_normal_slices": n_from_normal,
            "n_from_lesion_outside": n_from_outside,
            "include_lesion_outside": include_lesion_outside,
            "n_anomaly_slices_used": int(valid.sum()),
            "n_anomaly_slices_total": int(anom.sum()),
            "n_anomaly_patches": int(a_patches.shape[0]),
            "n_anomaly_slices_mask_empty": int((anom & has_mask & ~mask_flat.any(dim=1)).sum()),
        },
    }


class LearnablePrototypes(torch.nn.Module):
    """Stage 2:把 c_norm / c_anom 变成可学习参数。

    参数保存为**未归一化**的 raw 形式,前向时现场归一化 —— 与构建期同一套公式
    ((K+1,D) 上先 dim=0 channel-wise 再 dim=1 class-wise),
    这样 Stage 2 的起点与 Stage 1 完全一致,任何变化都只能归因于定位损失。

    只存 raw 而不存归一化结果,是因为归一化会破坏梯度流(单位化投影),
    Adam 在 raw 上更新、前向投影,是这类"球面参数"的常规做法。
    """

    def __init__(self, c_norm, c_anom, double_norm=True):
        super().__init__()
        self.double_norm = double_norm
        self.c_norm_raw = torch.nn.Parameter(c_norm.t().float().clone().contiguous())  # (K,D)
        self.c_anom_raw = torch.nn.Parameter(c_anom.float().clone())                   # (D,)

    @property
    def n_normal(self):
        return self.c_norm_raw.shape[0]

    def normalized(self):
        """返回 (c_norm (D,K), c_anom (D,))。"""
        stacked = torch.cat([self.c_norm_raw, self.c_anom_raw[None]], 0)   # (K+1,D)
        if self.double_norm:
            stacked = torch.nn.functional.normalize(stacked, dim=0)        # channel-wise
        stacked = torch.nn.functional.normalize(stacked, dim=1)            # class-wise
        return stacked[: self.n_normal].t().contiguous(), stacked[self.n_normal].contiguous()

    @torch.no_grad()
    def drift(self, c_norm0, c_anom0):
        """相对初始原型的漂移量,用于判断 Stage 2 是否真的动了参数。"""
        c_norm, c_anom = self.normalized()
        dn = 1.0 - (c_norm.t() @ c_norm0).mean()
        da = 1.0 - (c_anom @ c_anom0).item()
        return {"drift_norm": dn.item(), "drift_anom": da}


def channel_wise_weights(pi, c_norm_raw):
    """K-means 簇质量加权的正常原型的均值方向(适配器初始化用)。

    不用简单平均:k 个簇的样本量往往很不均衡,大簇应当占更大权重。
    """
    w = (pi[:, None] * c_norm_raw).sum(0)
    return F.normalize(w, dim=0)


def build_text_weights(tokenizer, model, class_templates, device="cuda"):
    """Text anchors: each class name -> prompt templates -> text encoder (L2-normed).

    class_templates: {"normal": [...], "tumor": [...]} (dict order = class order).
    Returns W_text: (D, C) and the class-name list.
    """
    with torch.no_grad():
        weights, names = [], []
        for cls_name, templates in class_templates.items():
            texts = [t.format(cls=cls_name) for t in templates] if templates else [cls_name]
            tokens = tokenizer(texts).to(device)
            embs = model.encode_text(tokens, normalize=True)
            w = F.normalize(embs.mean(dim=0), dim=0)
            weights.append(w)
            names.append(cls_name)
    return torch.stack(weights, dim=1), names
