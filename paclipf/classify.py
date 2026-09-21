"""Classification: CLIP text head + lesion-focused (attention-pooled) head."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def lesion_features(patch_feats, attn_weights):
    """Attention-weighted aggregation of patch features -> lesion feature (N, 512), L2-normed.

    patch_feats: (N, 196, 512); attn_weights: (N, 196).
    512 = BiomedCLIP 共享空间维度(trunk 768 经 head.proj 投影后),不是裸 ViT 的 768。
    """
    f = torch.einsum("nl,nld->nd", attn_weights, patch_feats)
    return F.normalize(f, dim=-1)


def classify(feats, w_text, scale=100.0):
    """feats: (N, 512); w_text: (512, C). Returns logits (N, C)."""
    return scale * feats @ w_text


def fused_logits(logits_img, logits_lesion, lam=1.0):
    """Same-space logits (both from w_text) -> direct weighted sum."""
    return logits_img + lam * logits_lesion
