"""逐位置空间自适应融合。

h(p) = Σ_s w_s(p) Ã_s(p)
w(p) = softmax(g([A_s(p)]_s, p))
g 是极小 MLP（3 路分数 + 2 维坐标 → hidden=16 → 3）。
只在 OOF 异常切片上训，不在 test 上调。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .grid import grid_hw, nested


class SpatialGate(nn.Module):
    def __init__(self, n_src=3, hidden=16, min_w=None, order=None):
        super().__init__()
        self.order = tuple(order) if order else ("proto", "text", "mem")
        # Linear 维数跟 order 走，忽略与架构不符的 n_src。
        self.n_src = len(self.order)
        if n_src is not None and int(n_src) != self.n_src:
            print(f"[spatial] n_src={n_src} 与 order={self.order} 不一致，"
                  f"按 {self.n_src} 路构建")
        self.net = nn.Sequential(
            nn.Linear(self.n_src + 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.n_src),
        )
        self.min_w = {k: float(v) for k, v in dict(min_w or {}).items()}

    def _stack(self, maps):
        """maps dict → (N, S, H, H) z-score，缺的路用 0。S = len(order)。"""
        from . import signals as SG

        if not any(k in maps for k in self.order):
            raise ValueError("SpatialGate: maps 为空")
        ref = next(iter(maps.values()))
        n, h, w = ref.shape
        zs = []
        for k in self.order:
            if k in maps:
                zs.append(SG._zscore(maps[k].float()))
            else:
                zs.append(ref.new_zeros(n, h, w))
        return torch.stack(zs, dim=1), self.order

    def _apply_min_w(self, weight):
        """softmax 后再垫 yaml heatmap_min_w，避免门控把 proto/mem 抹成 0。"""
        if not self.min_w:
            return weight
        floors = weight.new_tensor([self.min_w.get(k, 0.0) for k in self.order])
        floors = floors.clamp(min=0.0)
        s = float(floors.sum())
        if s <= 0:
            return weight
        if s >= 1.0 - 1e-6:
            floors = floors / s * 0.99
        remain = (1.0 - floors.sum()).clamp(min=0.0)
        return floors.view(1, 1, 1, -1) + remain * weight

    def forward(self, maps):
        stacked, _ = self._stack(maps)
        n, s, h, w = stacked.shape
        xy = grid_hw(h, device=stacked.device, dtype=stacked.dtype)
        xy = xy.view(1, h, w, 2).expand(n, -1, -1, -1)
        src = stacked.permute(0, 2, 3, 1)  # (N,H,W,S)
        inp = torch.cat([src, xy], dim=-1)
        logits = self.net(inp)
        weight = self._apply_min_w(torch.softmax(logits, dim=-1))
        fused = (weight * src).sum(-1)
        return fused, weight

    def fuse(self, maps):
        hm, _ = self.forward(maps)
        return hm


def from_cfg(cfg, device="cpu"):
    if not bool(nested(cfg, "spatial_fuse", "enable", True)):
        return None
    hidden = int(nested(cfg, "spatial_fuse", "hidden", 16))
    n_src = nested(cfg, "spatial_fuse", "n_src", None)
    min_w = dict(getattr(cfg, "heatmap_min_w", None) or {})
    return SpatialGate(n_src=n_src, hidden=hidden, min_w=min_w).to(device)


def train_gate(gate, maps, mask_flat, epochs=40, lr=1e-3, verbose=False):
    """OOF 异常切片上用 Tversky 训门控。maps: {name: (N,H,H)}；mask (N,196) bool。"""
    from . import losses as LS

    if gate is None:
        return gate
    n = next(iter(maps.values())).shape[0]
    if n == 0:
        return gate
    h = next(iter(maps.values())).shape[-1]
    mask = mask_flat.reshape(n, h, h).float().to(next(iter(maps.values())).device)
    is_anom = mask.reshape(n, -1).any(1)
    if not is_anom.any():
        return gate
    maps_t = {k: v.detach() for k, v in maps.items()}
    opt = torch.optim.Adam(gate.parameters(), lr=float(lr), eps=1e-4)
    gate.train()
    for ep in range(int(epochs)):
        hm, _ = gate(maps_t)
        s = hm.reshape(n, -1)
        loss = LS.tversky_loss(s, mask.reshape(n, -1).bool(), is_anom)
        if not torch.isfinite(loss):
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  [spatial] ep{ep:>3} tversky {float(loss.detach()):.4f}")
    gate.eval()
    return gate
