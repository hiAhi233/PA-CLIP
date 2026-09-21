"""病灶特征池化。

PA-CLIP 用热力图注意力池化,但实测它是负贡献:
    归一化注意力熵 0.994,cos(f_attn, f_uniform) = 0.9968
    分类准确率 CLS 单路 72.29% | +注意力池化 71.67% | +均匀池化 73.33%
所以这里用**均匀池化**。

选均匀池化还有一个结构性好处:病灶特征与 c_norm/c_anom 解耦。
Stage 2 训练原型时热力图会变,但病灶特征不变,于是
"分类适配器 W" 和 "定位原型 c" 的梯度互不污染,两个改动可以干净分离。

**尺寸匹配**:训练时正常切片和异常切片的病灶特征必须用**同样数量的 patch**。
否则正常侧是 196 个 patch 的均值、异常侧是约 3 个 patch 的均值(实测中位 2 个),
两者方差差一个量级,适配器会走"看方差"的捷径而不是学语义。
"""
import torch
import torch.nn.functional as F


def _randint(high, size, device, generator=None):
    """跨设备的 randint。

    torch.randint 默认在 CPU 上建张量,且 **CPU generator 无法驱动 CUDA 张量**
    (报 "Expected a 'cuda' device type for generator")。两处都要显式指定。
    """
    if generator is not None and generator.device.type != torch.device(device).type:
        generator = None          # 跨设备 generator 不被支持,退回全局 RNG
    return torch.randint(0, high, size, device=device, generator=generator)


def _pool(patch, weights):
    """weights: (N, 196) 非负、已按行归一化。返回 L2 归一化的 (N, D)。"""
    f = torch.einsum("nl,nld->nd", weights, patch)
    return F.normalize(f, dim=-1)


def from_mask(patch, mask_flat, fallback_random=None, generator=None):
    """GT mask 内均匀平均。mask 为空的样本用 fallback_random 个随机 patch 顶替。

    patch     : (N, 196, D)
    mask_flat : (N, 196) bool
    """
    cnt = mask_flat.sum(-1, keepdim=True).clamp(min=1)
    w = mask_flat.float() / cnt
    empty = ~mask_flat.any(-1)
    if empty.any() and fallback_random:
        n, l = mask_flat.shape
        idx = _randint(l, (int(empty.sum()), fallback_random), w.device, generator)
        w_empty = torch.zeros(int(empty.sum()), l, dtype=w.dtype, device=w.device)
        w_empty.scatter_(1, idx, 1.0 / fallback_random)
        w[empty] = w_empty
    return _pool(patch, w)


def random_subset(patch, q, generator=None):
    """随机 q 个 patch 均匀平均(正常切片 / 空 mask 切片用,保证尺寸匹配)。"""
    n, l, _ = patch.shape
    idx = _randint(l, (n, q), patch.device, generator)
    w = torch.zeros(n, l, dtype=patch.dtype, device=patch.device)
    w.scatter_(1, idx, 1.0 / q)
    return _pool(patch, w)


def topq_from_score(patch, score_map, q):
    """按热力图取 top-q 个 patch 均匀平均 —— 推理时的病灶特征。

    score_map: (N, 14, 14)。选取用 topk 的索引,权重是均匀的 1/q,
    且本函数在调用处应处于 no_grad 或被 .detach() 保护:
    这是"c 的梯度只来自定位损失"的实现基础。
    """
    n, l, _ = patch.shape
    idx = score_map.reshape(n, l).topk(q, dim=-1).indices
    w = torch.zeros(n, l, dtype=patch.dtype, device=patch.device)
    w.scatter_(1, idx, 1.0 / q)
    return _pool(patch, w)
