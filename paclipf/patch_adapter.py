"""多层 Patch Residual Adapter。

与文本侧同一形式: z' = Norm(z + λ_v W_up GELU(W_down z))
W_up 零初始 → step 0 等于冻结 BiomedCLIP patch。各层默认共享一套权重。
不解冻视觉塔；只训这一点加上已有原型 / 角空间头。
"""
import torch.nn as nn
import torch.nn.functional as F

from .grid import nested


class PatchResidualAdapter(nn.Module):
    def __init__(self, dim=512, bottleneck=256, lambda_v=0.08):
        super().__init__()
        self.lambda_v = float(lambda_v)
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, z):
        residual = self.up(F.gelu(self.down(z)))
        return F.normalize(z + self.lambda_v * residual, dim=-1)


def from_cfg(cfg, dim, device="cpu"):
    """yaml `patch_adapter.enable=false` 时返回 None（恒等，旧行为）。"""
    enable = bool(nested(cfg, "patch_adapter", "enable", True))
    if not enable:
        return None
    ad = PatchResidualAdapter(
        dim=int(dim),
        bottleneck=int(nested(cfg, "patch_adapter", "bottleneck", 256)),
        lambda_v=float(nested(cfg, "patch_adapter", "lambda_v", 0.08)),
    )
    return ad.to(device)


def apply(adapter, patch_dict):
    """adapter 为 None 时原样返回（不拷贝）。否则新 dict，不改缓存。"""
    if adapter is None:
        return patch_dict
    return {l: adapter(v) for l, v in patch_dict.items()}
