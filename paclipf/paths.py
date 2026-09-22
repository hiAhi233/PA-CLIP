"""路径引导:注册本项目自带的 open_clip。

PA-CLIP 自带一份 vendored 的 open_clip(加载 BiomedCLIP 所需),不依赖任何
外部项目、也不回退到 site-packages 里的同名包 —— 与方法独立、可复现的定位一致。

本模块必须在任何 `open_clip` 的 import 之前执行,因此 paclipf/__init__.py
的第一件事就是 `from . import paths`。
"""
import sys
from pathlib import Path

# paclipf/paths.py -> paclipf -> PA-CLIP
PKG_ROOT = Path(__file__).resolve().parents[1]
OPEN_CLIP_DIR = PKG_ROOT / "open_clip"


def _bootstrap():
    if not (OPEN_CLIP_DIR / "src" / "open_clip" / "__init__.py").is_file():
        raise RuntimeError(
            f"找不到 vendored 的 open_clip: {OPEN_CLIP_DIR}\n"
            f"BiomedCLIP 依赖它加载,不能回退到 PyPI 版本。"
        )

    # 插到最前,保证优先于任何 site-packages 里的同名包
    root = str(PKG_ROOT)
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)

    # 断言 open_clip 确实解析到本项目内的副本(而非某个已安装的 PyPI 包)
    import open_clip  # noqa: E402

    got = Path(open_clip.__file__).resolve()
    if not str(got).startswith(str(PKG_ROOT)):
        raise RuntimeError(
            f"open_clip 解析到了 {got},不在 {PKG_ROOT} 下。\n"
            f"说明环境里存在另一个 open_clip,行为将不可复现。"
        )


_bootstrap()
