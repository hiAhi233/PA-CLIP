"""原型构建 —— 修 PA-CLIP 缺失的 channel-wise 归一化。

PA-CLIP 的 pa_clip/prototypes.py:21 只做了 class-wise 那一次:
    centers = l2norm(centers, dim=-1)          # (K,D) 沿特征轴
而 Proto-Adapter(Sensors 2024,表 3)的做法是两次都做,且顺序固定:
    C = F.normalize(C, dim=0)   # channel-wise:每个特征维在各类原型间单位化
    C = F.normalize(C, dim=1)   # class-wise:每个原型单位化

**为什么必须从原始中心重算而不是对已存的 c_norm.pt 补一次**:
两次归一化不可交换,且 PA-CLIP 是在归一化之后才转置的,顺序与论文不同。
对已归一化的结果补做 dim=0 与在原始中心上做 dim=0 不等价。

关于"正常+异常一起归一化":
本实现把 Kn 个正常中心 + Ka 个异常中心堆成 (Kn+Ka, D) **按层一起**归一化。
打分是 max_j sim(p,c_anom^j) − max_k sim(p,c_norm^k)，两项必须同尺度。
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


def feat_dim(c):
    """(D,) / (D,K) / (L,D,K) → D。"""
    if c.dim() == 1:
        return int(c.shape[0])
    if c.dim() == 2:
        return int(c.shape[0])
    return int(c.shape[-2])


def as_dk(c):
    """收成 (D, K)。1-D → (D,1)；3-D 取最后一层。"""
    if c.dim() == 1:
        return c.unsqueeze(-1)
    if c.dim() == 3:
        return c[-1]
    return c


def layer_pair(c_norm, c_anom, layer_i):
    """第 i 层的 (c_norm (D,Kn), c_anom (D,Ka))。层数不够则复用最后一套。"""
    if c_norm.dim() == 3:
        i = min(int(layer_i), c_norm.shape[0] - 1)
        cn = c_norm[i]
        if c_anom.dim() == 3:
            j = min(int(layer_i), c_anom.shape[0] - 1)
            ca = c_anom[j]
        else:
            ca = as_dk(c_anom)
    else:
        cn, ca = as_dk(c_norm), as_dk(c_anom)
    return cn, ca


def _proto_vecs(c):
    """任意形状 → (P, D) 单位化前的原型行。"""
    if c.dim() == 1:
        return c.unsqueeze(0)
    if c.dim() == 2:
        return c.t()
    return c.permute(0, 2, 1).reshape(-1, c.shape[1])


def build_from_features(
    patch_l11,
    mask14,
    label,
    has_mask,
    k=6,
    k_anom=6,
    double_norm=True,
    seed=111,
    kmeans_max=200_000,
    n_init=3,
    include_lesion_outside=False,
):
    """从一层 patch 特征构建多正常 / 多异常原型。

    patch_l11 : (N, 196, D) 已 L2 归一化的 patch 特征
    mask14    : (N, 14, 14) bool,病灶掩膜(无 mask 的样本为全零)
    label     : (N,) 0=正常 1=异常
    has_mask  : (N,) bool,meta 里是否有 mask_path

    返回 dict:
      c_norm       (D, Kn)
      c_anom       (D, Ka)  Ka≥1，不再是单个向量
      c_norm_raw   (Kn, D)
      c_anom_raw   (Ka, D)
      pi           (Kn,)
      meta         dict
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

    # ---------------- K-means 正常 ----------------
    kn = max(1, min(int(k), int(pool.shape[0])))
    centers, pi = _kmeans_centers(pool, kn, seed, kmeans_max, n_init)

    # ---------------- 异常：多个原型（K-means；patch 不够则退回均值） ----------------
    valid = anom & has_mask & mask_flat.any(dim=1)
    if not valid.any():
        raise RuntimeError("没有任何非空 mask 的异常切片,无法构建异常原型")
    a_patches = patch_l11[valid][mask_flat[valid]]
    ka = max(1, min(int(k_anom), int(a_patches.shape[0])))
    if ka <= 1:
        a_centers = a_patches.mean(0, keepdim=True)
        a_pi = torch.ones(1)
    else:
        a_centers, a_pi = _kmeans_centers(
            a_patches.detach().cpu().numpy().astype(np.float32), ka, seed, kmeans_max, n_init)

    stacked = torch.cat([centers, a_centers], 0)
    if double_norm:
        stacked = F.normalize(stacked, dim=0)
    stacked = F.normalize(stacked, dim=1)

    kn = int(centers.shape[0])
    c_norm = stacked[:kn].t().contiguous()
    c_anom = stacked[kn:].t().contiguous()

    return {
        "c_norm": c_norm,
        "c_anom": c_anom,
        "c_norm_raw": centers,
        "c_anom_raw": a_centers,
        "pi": pi,
        "meta": {
            "k": kn,
            "k_anom": int(c_anom.shape[1]),
            "double_norm": double_norm,
            "n_normal_patches": int(pool.shape[0]),
            "n_from_normal_slices": n_from_normal,
            "n_from_lesion_outside": n_from_outside,
            "include_lesion_outside": include_lesion_outside,
            "n_anomaly_slices_used": int(valid.sum()),
            "n_anomaly_slices_total": int(anom.sum()),
            "n_anomaly_patches": int(a_patches.shape[0]),
            "n_anomaly_slices_mask_empty": int((anom & has_mask & ~mask_flat.any(dim=1)).sum()),
            "anom_cluster_pi": a_pi.detach().cpu().tolist() if torch.is_tensor(a_pi) else list(a_pi),
        },
    }


def build_layered(patch_dict, layers, mask14, label, has_mask, k=6, k_anom=6, **kwargs):
    """每层独立 K-means：c_norm (L,D,Kn)，c_anom (L,D,Ka)。"""
    layers = tuple(layers)
    norms, anoms, pis, metas = [], [], [], []
    for l in layers:
        r = build_from_features(
            patch_dict[l], mask14, label, has_mask, k=k, k_anom=k_anom, **kwargs)
        norms.append(r["c_norm"])
        anoms.append(r["c_anom"])
        pis.append(r["pi"])
        metas.append(r["meta"])
    kn = min(t.shape[1] for t in norms)
    ka = min(t.shape[1] for t in anoms)
    c_norm = torch.stack([t[:, :kn] for t in norms], 0).contiguous()
    c_anom = torch.stack([t[:, :ka] for t in anoms], 0).contiguous()
    meta = dict(metas[-1])
    meta.update({
        "layers": list(layers),
        "per_layer": True,
        "k": kn,
        "k_anom": ka,
        "per_layer_meta": metas,
    })
    return {
        "c_norm": c_norm,
        "c_anom": c_anom,
        "c_norm_raw": c_norm.permute(0, 2, 1).contiguous(),
        "c_anom_raw": c_anom.permute(0, 2, 1).contiguous(),
        "pi": pis[-1],
        "meta": meta,
    }


class LearnablePrototypes(torch.nn.Module):
    """Stage 2:把 c_norm / c_anom 变成可学习参数。

    支持 (D,K) 或分层 (L,D,K)。内部一律存 (L, K, D) raw，前向时按层
    把正常+异常堆在一起做与构建期相同的双重归一化。
    """

    def __init__(self, c_norm, c_anom, double_norm=True):
        super().__init__()
        self.double_norm = double_norm
        cn, ca = c_norm.float(), as_dk(c_anom).float() if c_anom.dim() < 3 else c_anom.float()
        if cn.dim() == 2:
            cn = cn.unsqueeze(0)
        if ca.dim() == 2:
            ca = ca.unsqueeze(0)
        if ca.dim() == 1:
            ca = ca.view(1, -1, 1)
        self.c_norm_raw = torch.nn.Parameter(cn.permute(0, 2, 1).contiguous())  # (L,Kn,D)
        self.c_anom_raw = torch.nn.Parameter(ca.permute(0, 2, 1).contiguous())  # (L,Ka,D)

    @property
    def n_normal(self):
        return self.c_norm_raw.shape[1]

    @property
    def n_layers(self):
        return self.c_norm_raw.shape[0]

    def normalized(self):
        """(L,D,Kn), (L,D,Ka)；单层时仍返回 3-D，heatmap 按层索引。"""
        ns, a_s = [], []
        kn = self.c_norm_raw.shape[1]
        for i in range(self.n_layers):
            stacked = torch.cat([self.c_norm_raw[i], self.c_anom_raw[i]], 0)
            if self.double_norm:
                stacked = torch.nn.functional.normalize(stacked, dim=0)
            stacked = torch.nn.functional.normalize(stacked, dim=1)
            ns.append(stacked[:kn].t())
            a_s.append(stacked[kn:].t())
        return torch.stack(ns, 0).contiguous(), torch.stack(a_s, 0).contiguous()

    @torch.no_grad()
    def drift(self, c_norm0, c_anom0):
        c_norm, c_anom = self.normalized()
        v = F.normalize(_proto_vecs(c_norm), dim=-1)
        v0 = F.normalize(_proto_vecs(c_norm0.to(v.device)), dim=-1)
        a = F.normalize(_proto_vecs(c_anom), dim=-1)
        a0 = F.normalize(_proto_vecs(c_anom0.to(a.device)), dim=-1)
        n = min(len(v), len(v0))
        m = min(len(a), len(a0))
        dn = 1.0 - (v[:n] * v0[:n]).sum(-1).mean()
        da = 1.0 - (a[:m] * a0[:m]).sum(-1).mean()
        return {"drift_norm": float(dn), "drift_anom": float(da)}


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
