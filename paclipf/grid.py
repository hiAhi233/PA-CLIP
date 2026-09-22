"""14x14 patch 网格坐标。Memory 匹配与空间门控共用同一套 [0,1]^2 坐标。"""
import torch


def patch_xy(H, device=None, dtype=torch.float32):
    """patch 中心坐标 (H*H, 2)，列为 (x, y)，范围 [0, 1]。"""
    xs = torch.linspace(0.5 / H, 1.0 - 0.5 / H, H, device=device, dtype=dtype)
    ys = torch.linspace(0.5 / H, 1.0 - 0.5 / H, H, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def grid_hw(H, device=None, dtype=torch.float32):
    """(H, H, 2) 的 (x, y)，供逐位置门控拼接。"""
    return patch_xy(H, device=device, dtype=dtype).view(H, H, 2)


def nested(cfg, group, key, default):
    """读 yaml 嵌套段；缺省时回落到顶层同名键，再回落到 default。"""
    g = getattr(cfg, group, None)
    if g is not None:
        if isinstance(g, dict) and key in g:
            return g[key]
        if hasattr(g, key):
            return getattr(g, key)
    return getattr(cfg, key, default)
