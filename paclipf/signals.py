"""给定切分,产出全部信号与 logits。

一次前向同时产出**两套口径**,因为它们共享全部特征,多算一套几乎免费,
而分开算会强迫跑两遍:

  paclip 口径 —— 与 pa_clip/main.py 逐位一致(z-score + sigmoid 的图、注意力池化)
                用于复现基线,是所有 A/B 的参照点
  ours   口径 —— 原始分数图 + 评分前平滑 + 均匀池化
                修掉两个已定位的问题(见下)

两个问题:
 1. 图像级打分 `max(sigmoid(τz))` ≡ `sigmoid(τ·max(z))`(sigmoid 单调),
    所以 anomaly_tau 对图像级分数是**空操作**;而 max(z) = (max(s)−mean(s))/std(s)
    除以了每张图自己的 std,跨图标定被摧毁。实测 AUC 0.638 vs 原始 0.754。
 2. PA-CLIP 的平滑在 to_pixel_map,用 kernel=9 @512px,
    折算到 patch 尺度只有 9/(512/14) = 0.25 个 patch —— 等于没有平滑。

设计原则:ours 口径只用**一张**部署图,同时服务图像级打分、病灶特征选择、
像素指标与全部诊断指标。PA-CLIP 用了三张不同的图(sigmoid(2z) / max(sigmoid(2z))
/ to_pixel_map(sigmoid(2z))),导致"报告的像素 AUC 用的图"和"决定图像分的图"不是同一个。
"""
import torch
from paclipf import classify

from . import heatmap as HM
from . import lesion as LSN

# 类别顺序固定:0 = normal, 1 = anomaly(与 meta 的 label 一致)
IDX_NORMAL, IDX_ANOMALY = 0, 1


@torch.no_grad()
def compute(cfg, split_data, c_norm, c_anom, w_text, adapter_img=None, adapter_lesion=None,
            text_pack=None, memory=None, fuse_w=None, text_sign=1.0):
    """split_data: cache.load() 的返回值(需含 cls 与 4 层 patch)。"""
    layers = tuple(cfg.patch_layers)
    patch = split_data["patch"]
    f_cls = split_data["cls"]
    patch_l11 = patch[layers[-1]]

    # ---------- paclip 口径:与 pa_clip/main.py 一致 ----------
    hm_raw = HM.heatmap_exact(patch, c_anom, c_norm, layers,
                              image_size=cfg.image_size, smooth_kernel=0)
    hm_paclip = torch.sigmoid(cfg.anomaly_tau * _zscore(hm_raw))

    # ---------- ours 口径:评分前平滑 ----------
    hm_ours = HM.smooth_map(hm_raw, getattr(cfg, "smooth_kernel", 0))

    s_flat = hm_ours.reshape(hm_ours.shape[0], -1)
    q = max(1, int(getattr(cfg, "q_img", 3)))
    S_heat_raw = s_flat.topk(q, dim=-1).values.mean(-1)
    S_heat_paclip = hm_paclip.reshape(hm_paclip.shape[0], -1).max(dim=-1).values

    # ---------- 病灶特征两版 ----------
    attn = torch.softmax(hm_paclip.reshape(hm_paclip.shape[0], -1) / cfg.attn_temp, dim=-1)
    f_les_attn = classify.lesion_features(patch_l11, attn)          # PA-CLIP 原样
    f_les_uniform = LSN.topq_from_score(patch_l11, hm_ours, max(1, int(cfg.q_sel)))

    # ---------- 文本头 ----------
    clip_img = classify.classify(f_cls, w_text)
    out = {
        # 部署图(ours)
        "hm": hm_ours,
        "s_flat": s_flat,
        # paclip 口径的图与信号
        "hm_paclip": hm_paclip,
        "S_heat_paclip": S_heat_paclip,
        "f_lesion_attn": f_les_attn,
        # ours 口径的信号
        "S_heat_raw": S_heat_raw,
        "f_lesion_uniform": f_les_uniform,
        # 共用
        "f_cls": f_cls,
        "clip_img": clip_img,
        "clip_les_attn": classify.classify(f_les_attn, w_text),
        "clip_les_uniform": classify.classify(f_les_uniform, w_text),
        "S_text": 1.0 - torch.softmax(clip_img, dim=-1)[:, IDX_NORMAL],
        "S_text_margin": clip_img[:, IDX_ANOMALY] - clip_img[:, IDX_NORMAL],
    }

    # ---------- 原型适配器(Stage 1 之后才有)----------
    if adapter_img is not None:
        pi = adapter_img(f_cls)
        out["proto_img"] = pi
        scale = float(getattr(cfg, "adapter_logit_scale", 100.0))
        out["S_adapt"] = 1.0 - torch.softmax(scale * pi, dim=-1)[:, IDX_NORMAL]
        out["S_adapt_margin"] = pi[:, IDX_ANOMALY] - pi[:, IDX_NORMAL]
    if adapter_lesion is not None:
        out["proto_les_uniform"] = adapter_lesion(f_les_uniform)
        out["proto_les_attn"] = adapter_lesion(f_les_attn)

    # ---------- A_text / A_mem / 融合热力图 ----------
    if text_pack is not None:
        hm_text = float(text_sign) * text_heatmap(patch, text_pack, layers, cfg.image_size)
        out["hm_text"] = hm_text
        out["S_heat_text"] = hm_text.reshape(hm_text.shape[0], -1).topk(q, dim=-1).values.mean(-1)
    if memory is not None:
        last = (layers[-1],)
        hm_mem = memory_heatmap({layers[-1]: patch[layers[-1]]}, memory, last, cfg.image_size)
        out["hm_mem"] = hm_mem
        out["S_heat_mem"] = hm_mem.reshape(hm_mem.shape[0], -1).topk(q, dim=-1).values.mean(-1)

    maps = {"proto": hm_ours}
    if "hm_text" in out:
        maps["text"] = out["hm_text"]
    if "hm_mem" in out:
        maps["mem"] = out["hm_mem"]
    if fuse_w is not None and len(maps) > 1:
        hm_f = fuse_maps(maps, fuse_w)
        out["hm_fused"] = hm_f
        out["s_flat_fused"] = hm_f.reshape(hm_f.shape[0], -1)
        out["S_heat_fused"] = out["s_flat_fused"].topk(q, dim=-1).values.mean(-1)
    return out


def text_heatmap(patch_dict, pack, layers, image_size=224):
    """A_text(p) = sim(p, t_lesion^A) - sim(p, t_lesion^N), 多层平均。"""
    import torch.nn.functional as F
    t_n = F.normalize(pack["t_lesion"][0].float(), dim=-1)
    t_a = F.normalize(pack["t_lesion"][1].float(), dim=-1)
    H = image_size // 16
    maps = []
    for l in layers:
        f = F.normalize(patch_dict[l].float(), dim=-1)
        s = f @ t_a - f @ t_n
        maps.append(s.view(-1, H, H))
    return torch.stack(maps, 0).mean(0)


def memory_heatmap(patch_dict, memory, layers, image_size=224, chunk=256):
    """A_mem(p) = max_a sim(p, m_a) - max_n sim(p, m_n)。分块以免 (N,L,M) 爆显存。"""
    import torch.nn.functional as F
    a_mem = F.normalize(memory["a"].float(), dim=-1)
    n_mem = F.normalize(memory["n"].float(), dim=-1)
    H = image_size // 16

    def _maxsim(f, mem):
        n, l, d = f.shape
        flat = f.reshape(-1, d)
        best = None
        for i in range(0, mem.shape[0], chunk):
            s = flat @ mem[i:i + chunk].t()
            m = s.max(-1).values
            best = m if best is None else torch.maximum(best, m)
        return best.view(n, l)

    maps = []
    for l in layers:
        f = F.normalize(patch_dict[l].float(), dim=-1)
        maps.append((_maxsim(f, a_mem) - _maxsim(f, n_mem)).view(-1, H, H))
    return torch.stack(maps, 0).mean(0)


def fuse_maps(maps, weights):
    """maps: {name: (N,H,H)}; weights: {name: float} 或 (w_proto, w_text, w_mem)。"""
    order = ("proto", "text", "mem")
    if not isinstance(weights, dict):
        weights = {k: float(w) for k, w in zip(order, weights)}
    acc = None
    wsum = 0.0
    for k, hm in maps.items():
        w = float(weights.get(k, 0.0))
        if w == 0:
            continue
        z = _zscore(hm)
        acc = z * w if acc is None else acc + z * w
        wsum += w
    if acc is None:
        return next(iter(maps.values()))
    return acc if wsum == 0 else acc



def _zscore(hm):
    return (hm - hm.mean(dim=(1, 2), keepdim=True)) / (hm.std(dim=(1, 2), keepdim=True) + 1e-8)


def fused_logits(sig, alpha_pt, lam_pt, adapter_scale=100.0):
    """分类 logits = 文本头 + α·scale·(整图原型路 + λ·病灶原型路)。

    CLIP 头与适配器余弦不在同一量纲(约 ×100 vs ±1)。推理时给原型项乘
    `adapter_scale`(默认 100)后再融,避免靠 α 网格硬补、总撞上界。
    病灶路固定用 `proto_les_uniform`;注意力池化那一路只进诊断。
    """
    logits = sig["clip_img"]
    if "proto_img" in sig:
        proto = sig["proto_img"] + lam_pt * sig["proto_les_uniform"]
        logits = logits + alpha_pt * (adapter_scale * proto)
    return logits


def fuse_scores(z_heat, z_text, alpha_fuse):
    """图像级异常分:α·z(S_heat) + (1−α)·z(S_text),与 PA-CLIP 同形以便归因。"""
    return alpha_fuse * z_heat + (1.0 - alpha_fuse) * z_text


def compute_ctx(cfg, split_data, ctx, c_norm=None, c_anom=None,
                adapter_img=None, adapter_lesion=None, fuse_w="ctx"):
    """compute() 的 ctx 包装:自动带上分层文本 / Memory Bank / 融合权重。"""
    fw = ctx.get("fuse_w") if fuse_w == "ctx" else fuse_w
    return compute(
        cfg, split_data,
        ctx["c_norm"] if c_norm is None else c_norm,
        ctx["c_anom"] if c_anom is None else c_anom,
        ctx["w_text"],
        adapter_img=adapter_img, adapter_lesion=adapter_lesion,
        text_pack=ctx.get("text_pack"),
        memory=ctx.get("memory"),
        fuse_w=fw,
        text_sign=ctx.get("text_sign", 1.0),
    )
