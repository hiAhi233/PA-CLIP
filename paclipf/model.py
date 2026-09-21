"""BiomedCLIP loading + patch-level feature extraction (training-free)."""
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer

MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def load_model(device="cuda"):
    """Load BiomedCLIP from local HF cache. Returns (model, preprocess, tokenizer)."""
    model, preprocess = create_model_from_pretrained(MODEL_NAME)
    tokenizer = get_tokenizer(MODEL_NAME)
    model = model.to(device).eval()
    return model, preprocess, tokenizer


class PatchHooker:
    """Register forward hooks on trunk ViT blocks to collect patch features.

    Output of block i: [B, N+1, C] (cls + patches). Patch layers selected by config.
    """

    def __init__(self, model, layers=(3, 6, 9, 11)):
        self.trunk = model.visual.trunk
        self.layers = list(layers)
        self.hooks = []
        self.outputs = {}
        for i in self.layers:
            self.hooks.append(
                self.trunk.blocks[i].register_forward_hook(self._make_hook(i))
            )

    def _make_hook(self, i):
        def fn(module, inp, out):
            self.outputs[i] = out
        return fn

    def clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.outputs = {}


@torch.no_grad()
def encode_images(model, preprocess, img_paths, batch_size=64, device="cuda"):
    """Image-level (CLS, L2-normalized) features for a list of image paths."""
    from PIL import Image
    feats = []
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i : i + batch_size]
        images = torch.stack(
            [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
        ).to(device)
        f = model.encode_image(images, normalize=True)
        feats.append(f.cpu())
    return torch.cat(feats, dim=0)


@torch.no_grad()
def encode_patch_features(model, preprocess, img_paths, layers=(3, 6, 9, 11),
                          batch_size=32, device="cuda"):
    """Patch features per layer: dict {layer: Tensor[N, 196, 768]} (L2-normalized)."""
    from PIL import Image
    hooker = PatchHooker(model, layers)
    out = {l: [] for l in layers}
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i : i + batch_size]
        images = torch.stack(
            [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
        ).to(device)
        model.encode_image(images, normalize=False)  # run trunk, hooks collect patches
        for l in layers:
            x = hooker.outputs.pop(l)          # [B, N+1, C]
            patches = x[:, 1:, :]              # drop CLS
            patches = model.visual.trunk.norm(patches)  # apply final LN (as in forward)
            patches = model.visual.head.proj(patches)   # 768 -> 512 (BiomedCLIP shared space)
            patches = nn.functional.normalize(patches, dim=-1)
            out[l].append(patches.cpu())
    hooker.clear()
    return {l: torch.cat(v, dim=0) for l, v in out.items()}
