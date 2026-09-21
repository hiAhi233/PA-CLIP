"""文本 Residual Adapter + 三层提示词编码。

文档公式: t' = Norm(t + λ_t W_up GELU(W_down t))
加在 BiomedCLIP encode_text 的输出端(512 维 CLIP 空间),不解冻文本塔。
W_up 零初始 → step 0 时 t' = t,与冻结基线一致。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextResidualAdapter(nn.Module):
    def __init__(self, dim=512, bottleneck=256, lambda_t=0.08):
        super().__init__()
        self.lambda_t = float(lambda_t)
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, t):
        residual = self.up(F.gelu(self.down(t)))
        return F.normalize(t + self.lambda_t * residual, dim=-1)


def _mean_embed(tokenizer, model, templates, device):
    if not templates:
        raise ValueError("提示词列表为空")
    tokens = tokenizer(list(templates)).to(device)
    embs = model.encode_text(tokens, normalize=True)
    return F.normalize(embs.mean(dim=0), dim=0)


@torch.no_grad()
def encode_hierarchy(tokenizer, model, text_prompts, device="cuda"):
    """冻结编码器一次性编码三层提示词。返回 dict 的 t0,全部 (..., D) 且 L2 归一化。

    text_prompts:
      global:    {normal: [...], anomaly: [...]}
      attribute: [{name, normal: [...], anomaly: [...]}, ...]
      lesion:    {normal: [...], anomaly: [...]}
    """
    g = text_prompts["global"]
    les = text_prompts["lesion"]
    attrs = list(text_prompts.get("attribute") or [])

    t_global = torch.stack([
        _mean_embed(tokenizer, model, g["normal"], device),
        _mean_embed(tokenizer, model, g["anomaly"], device),
    ], 0)
    t_lesion = torch.stack([
        _mean_embed(tokenizer, model, les["normal"], device),
        _mean_embed(tokenizer, model, les["anomaly"], device),
    ], 0)

    if attrs:
        t_attr_n = torch.stack([_mean_embed(tokenizer, model, a["normal"], device) for a in attrs], 0)
        t_attr_a = torch.stack([_mean_embed(tokenizer, model, a["anomaly"], device) for a in attrs], 0)
    else:
        t_attr_n = t_global[0:1]
        t_attr_a = t_global[1:2]

    return {
        "t_global": t_global,
        "t_lesion": t_lesion,
        "t_attr_n": t_attr_n,
        "t_attr_a": t_attr_a,
        "attr_names": [a.get("name", f"a{i}") for i, a in enumerate(attrs)],
    }


def apply_adapter(adapter, pack):
    """把同一 adapter 作用到 pack 里所有锚点。返回新 dict(与 pack 同键)。"""
    out = dict(pack)
    for k in ("t_global", "t_lesion", "t_attr_n", "t_attr_a"):
        out[k] = adapter(pack[k])
    return out


def stack_all(pack):
    """全部锚点拼成 (M, D),顺序固定,供 L_preserve / L_dis。"""
    return torch.cat([pack["t_global"], pack["t_attr_n"], pack["t_attr_a"], pack["t_lesion"]], 0)


def w_text_from_pack(pack):
    """全局层 → 现有 classify 用的 (D, 2)。"""
    return pack["t_global"].t().contiguous()
