"""PA-CLIP: Prototype-Anchored Localization with a Fine-tuned Angular-Space Adapter.

Self-contained package -- no dependency on any sibling project. BiomedCLIP is loaded
through the vendored `open_clip/` that ships with this repository.

Method sketch
  The few-shot prototypes of the training-free pipeline become the initialization of a
  single (D, 2) adapter in angular space (L2-normalized weight columns = anchors).
  The adapter is then fine-tuned on patch-level labels with the Additive Angular Margin
  penalty (ArcFace), following Proto-Adapter-F (Kato et al., Sensors 2024). Only this
  tiny matrix is trained; the image encoder runs in no_grad throughout.

Module map
  paths         sys.path bootstrap (import FIRST)
  config        config loading + derived-value validation
  data          jsonl loading, case-level splitting, 14x14 mask grids
  model         BiomedCLIP backend + multi-layer patch feature extraction
  cache         feature pre-computation and caching (fp16, content-addressed)
  prototypes    prototype construction (double L2 normalization, Proto-Adapter Table 3)
  adapter       angular-space adapter + ArcFace head
  heatmap       differentiable heatmap (parity-checked against the numpy inference path)
  lesion        lesion feature pooling (uniform top-q / GT mask)
  localization  14x14 -> pixel map, ROI-aware z-score, tissue mask
  losses        Stage-2 localization + L_visual (focal / tversky / contrast / consistency)
  patch_adapter residual bottleneck on frozen patch tokens (shared across layers)
  spatial       per-location fusion gate
  postprocess   upsample, morphological close, Top-k binary mask
  text_adapter  residual adapter on frozen text embeddings + hierarchical prompts
  text_losses   Stage-text global / patch / diversity / preserve losses
  classify      text head and same-space logit combination
  signals       per-split signal computation (heat / text / memory) + fusion
  metrics       pixel AUC/AP, image AUC/AP, top-1 accuracy
  evaluate      per-variant report assembly
    fusion        alpha / lambda grid search (validation split only)
    cv            症例留一:折内重建原型/Memory,OOF 搜超参
    train         Stage 1 / Stage 2 / Stage-text training loops
  diag          localization diagnostics (peak hit-rate, contrast, bootstrap)
  report        csv / json result writing
  pipeline      shared prepare() context builder
  visualize     qualitative heatmap figures
"""
from . import paths  # noqa: F401  must run first: registers the vendored open_clip

__all__ = ["paths"]
