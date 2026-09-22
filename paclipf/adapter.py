"""角空间原型适配器 + ArcFace 间隔损失。

来源:Proto-Adapter-F(Kato et al., Sensors 2024)。
整个可训练部分就是一个 (N_cls, D) 矩阵 —— 本任务 N_cls=2、D=512,
即 **1,024 个参数**。这也解释了为什么训练成本可以忽略:
图像特征在 no_grad 下预计算,梯度不流经编码器。

移植时修正了参考实现的两处缺陷(见 Forward 与 select_best 的注释)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def arcface_margin_loss(cos_logits, target, margin):
    """加性角度间隔(ArcFace,Proto-Adapter 论文式 8)。

    移植自 proto-Adapter/utils.py:139,数学完全一致:只把正类的 logit 从
    cos(θ_y) 改成 cos(θ_y + m),其余列不变。

    cos_logits : (B, N) 已 L2 归一化的特征与权重的余弦
    target     : (B,)   类别索引
    """
    cos_theta = cos_logits.clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    out = cos_theta.clone()
    idx = torch.arange(cos_logits.size(0), device=cos_logits.device)
    out[idx, target] = torch.cos(theta[idx, target] + margin)
    return out


class AngularAdapter(nn.Module):
    """(N_cls, D) 的角空间线性探针,权重单位化后与单位化特征做内积。

    与 Proto-Adapter-F 的关键差异(必须保留的修正):
      参考实现在训练时用 F.normalize(weight) 算 cos,但推理时直接
      adapter(features) 用**未归一化**的权重。由于损失对权重模长不变,
      模长在 Adam 下是随机游走 → 推理时 logits 尺度不可控、不可复现。
      本实现在 forward 里**始终**归一化,训练与推理一致。
    """

    def __init__(self, weight_init):
        super().__init__()
        w = weight_init.detach().float().clone()
        # 初始就归一化,让 step 0 的 cos 与后续可比
        self.weight = nn.Parameter(F.normalize(w, dim=1))

    @property
    def unit_weight(self):
        return F.normalize(self.weight, dim=1)

    def cos_logits(self, f):
        return F.normalize(f.float(), dim=1) @ self.unit_weight.t()

    def forward(self, f):
        """推理:返回裸余弦 logits(不乘 ArcFace scale,与 Proto-Adapter-F 一致)。"""
        return self.cos_logits(f)

    def loss(self, f, target, margin, scale):
        return F.cross_entropy(scale * arcface_margin_loss(self.cos_logits(f), target, margin), target)

    @torch.no_grad()
    def class_cos(self):
        """两类权重方向的余弦。初始实测 0.7773 —— 越高说明两类越难分。"""
        w = self.unit_weight
        return (w[0] * w[1]).sum().item() if w.shape[0] >= 2 else float("nan")


def init_weights(c_norm, c_anom, pi=None, weighted=True):
    """由原型构造适配器初始权重 (2, D)。分层时用最后一层；多异常取均值方向。"""
    from .prototypes import as_dk

    cn = as_dk(c_norm)
    ca = as_dk(c_anom)
    dev = cn.device
    if weighted and pi is not None:
        pi = pi.to(device=dev, dtype=cn.dtype)
        kn = min(int(pi.numel()), int(cn.shape[1]))
        w_norm = F.normalize((pi[:kn, None] * cn[:, :kn].t()).sum(0), dim=0)
    else:
        w_norm = F.normalize(cn.mean(1), dim=0)
    w_anom = F.normalize(ca.mean(1), dim=0)
    return torch.stack([w_norm, w_anom], 0)
