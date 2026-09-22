"""异常热力图 —— 精确版(推理)与可微版(训练)。

两个入口必须数值一致,否则"训练时优化的图"和"部署时打分的图"不是同一个东西。
启动时用 assert_parity() 强制校验,成本几乎为零。

同时这里是"评分前平滑"的落点。**但实测表明平滑应关闭(smooth_kernel=0)**。

test 集 555 张异常切片(k=0/3/5/7):
    hit@1  27.90% / 18.20% /  4.73% /  0.24%
    hit@10 92.43% / 85.82% / 47.04% / 23.40%
    AUC    0.7655 / 0.7522 / 0.6830 / 0.6657
机制:病灶中位只有 2.0 个 patch,3x3 平均池化里 8/9 是非病灶 patch,
峰值被稀释得比背景噪声还快。k=7 时病灶几乎被完全抹掉。
换句话说热力图的峰值本来就尖锐且正确(hit@10 已达 92%),失效模式不是
"单点噪声压住病灶"而是"病灶偶尔被更强的软组织峰盖过",平均化救不了后者。

保留该开关是为了让这个结论可复现,不是因为它有用。
"""
import torch
import torch.nn.functional as F

from paclipf import localization as loc


def smooth_map(hm, kernel):
    """在 14x14 上做 k×k 平均池化。kernel<=1 时原样返回。"""
    if not kernel or kernel <= 1:
        return hm
    x = hm.unsqueeze(1)
    x = F.avg_pool2d(x, kernel, stride=1, padding=kernel // 2, count_include_pad=False)
    return x.squeeze(1)


def heatmap_exact(patch_feats_dict, c_anom, c_norm, layers, image_size=224, smooth_kernel=0):
    """推理用。数值与 PA-CLIP 的 multi_layer_heatmap 完全一致(+ 可选评分前平滑)。"""
    with torch.no_grad():
        hm = loc.multi_layer_heatmap(
            patch_feats_dict, c_anom, c_norm, layers, image_size=image_size
        )
        return smooth_map(hm, smooth_kernel)


def heatmap_diff(patch_feats_dict, c_anom, c_norm, layers, image_size=224,
                 smooth_kernel=0, reduce="max", lse_tau=0.1, return_layers=False):
    """可微版。不经过 z-score / sigmoid(逐图标准化在损失里同样有害:
    它会让模型靠"缩小图内方差"降损失,从而破坏跨图标定)。

    reduce: 'max' 与部署一致;'lse' 软化,缓解"6 个正常原型里只有少数拿到梯度"。
    return_layers: True 时额外返回 (n_layers, N, H, H)，给 L_consistency。
    """
    H = image_size // 16
    maps = []
    from .prototypes import layer_pair
    for i, l in enumerate(layers):
        f = patch_feats_dict[l]
        cn, ca = layer_pair(c_norm, c_anom, i)
        sim_anom = (f @ ca).max(dim=-1).values if ca.dim() == 2 else f @ ca
        sim_norm = f @ cn
        if reduce == "max":
            base = sim_norm.max(dim=-1).values
        elif reduce == "lse":
            base = lse_tau * torch.logsumexp(sim_norm / lse_tau, dim=-1)
        else:
            raise ValueError(f"reduce 只支持 max / lse,得到 {reduce}")
        maps.append((sim_anom - base).view(-1, H, H))
    stacked = torch.stack(maps, 0)
    hm = smooth_map(stacked.mean(0), smooth_kernel)
    if return_layers:
        return hm, stacked
    return hm


@torch.no_grad()
def assert_parity(cfg, patch_feats_dict, c_anom, c_norm, atol=1e-6):
    """可微版(no_grad, reduce=max, 不平滑)必须与 pa_clip 的精确实现逐位一致。"""
    a = heatmap_diff(patch_feats_dict, c_anom, c_norm, tuple(cfg.patch_layers),
                     image_size=cfg.image_size, smooth_kernel=0, reduce="max")
    b = loc.multi_layer_heatmap(patch_feats_dict, c_anom, c_norm,
                                tuple(cfg.patch_layers), image_size=cfg.image_size)
    d = (a - b).abs().max().item()
    if d > atol:
        raise AssertionError(
            f"heatmap_diff 与 pa_clip 的精确实现不一致 (max|diff|={d:.2e} > {atol:.0e})。\n"
            f"说明两份实现已漂移,训练优化的将不是部署使用的那个图。"
        )
    return d
