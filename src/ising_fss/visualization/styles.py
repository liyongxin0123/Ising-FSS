# -*- coding: utf-8 -*-
"""
Matplotlib 样式系统（支持 LaTeX 安全回退）

实现功能：
    - publication_style() / presentation_style() 上下文管理器
    - 自动检测 LaTeX 可用性 → 降级到 mathtext
    - 内置 Paul Tol 色盲友好配色
    - 统一保存接口 save_figure（多格式 + 日志）
"""

from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np

__all__ = [
    'apply_publication_style',
    'apply_presentation_style',
    'publication_style',
    'presentation_style',
    'get_color_palette',
    'COLORS',
    'set_latex_style',
    'style_phase_diagram',
    'style_fss_plot',
    'style_comparison_plot',
    'temperature_colormap',
    'phase_colormap',
    'optimize_legend',
    'save_figure',
]

# -----------------------------------------------------------------------------
# 颜色常量（中文注释）
# -----------------------------------------------------------------------------
# 常用颜色映射（用于快速调用）
COLORS: Dict[str, str] = {
    'primary': '#377eb8',    # 蓝
    'secondary': '#e41a1c',  # 红
    'accent': '#4daf4a',     # 绿
    'warning': '#ff7f00',    # 橙
    'info': '#984ea3',       # 紫

    'ferro': '#d73027',      # 铁磁 - 红
    'para': '#4575b4',       # 顺磁 - 蓝
    'critical': '#fee090',   # 临界 - 黄

    'cold': '#313695',
    'cool': '#4575b4',
    'warm': '#f46d43',
    'hot':  '#a50026',

    'gray': '#7f7f7f',
    'light_gray': '#d9d9d9',
    'dark_gray': '#525252',

    'highlight': '#ffff33',
    'background': '#f7f7f7',
}

# Paul Tol 色盲友好 10 色（常用）
_TOL_COLORBLIND_10 = [
    '#332288', '#88CCEE', '#44AA99', '#117733', '#999933',
    '#DDCC77', '#CC6677', '#882255', '#AA4499', '#6699CC'
]

# -----------------------------------------------------------------------------
# 内部工具：将 matplotlib colormap / array 转为 Python 列表（RGBA tuple 或 hex）
# -----------------------------------------------------------------------------
def _ensure_color_list(arr) -> List:
    """
    将 colormap(sampled) 或色列表标准化为 Python list。
    - 对于字符串色值，直接返回列表。
    - 对于 numpy 数组（Nx4 RGBA），返回 [(r,g,b,a), ...] 列表。
    """
    if arr is None:
        return []
    # 如果是 matplotlib colormap 输出（数组）
    try:
        a = np.asarray(arr)
    except Exception:
        # 直接当作可迭代对象
        return list(arr)
    if a.ndim == 1 and a.dtype.kind in {'U', 'S', 'O'}:
        # 一维字符串数组
        return list(a.tolist())
    if a.ndim == 2 and a.shape[1] in (3, 4):
        # RGBA 数组 -> 转成元组列表
        return [tuple(float(x) for x in row) for row in a.tolist()]
    # 兜底，尝试转成 list
    try:
        return list(arr)
    except Exception:
        return [arr]

# -----------------------------------------------------------------------------
# 调色板 API（公开）
# -----------------------------------------------------------------------------
def get_color_palette(name: str = 'default', n_colors: int = 10) -> List:
    """
    获取颜色调色板。
    参数:
      name: 'default'|'temperature'|'phase'|'vibrant'|'pastel'|'viridis'|'plasma'|'colorblind'|'mono'
      n_colors: 需要的颜色数量
    返回:
      list: 颜色列表（hex 或 RGBA tuples）
    """
    name = (name or 'default').lower()
    n_colors = int(max(0, n_colors))
    if n_colors == 0:
        return []

    if name == 'default':
        return _ensure_color_list(plt.cm.Set1(np.linspace(0, 1, n_colors)))
    elif name == 'temperature':
        return _ensure_color_list(plt.cm.RdYlBu_r(np.linspace(0, 1, n_colors)))
    elif name == 'phase':
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'phase', [COLORS['ferro'], COLORS['critical'], COLORS['para']]
        )
        return _ensure_color_list(cmap(np.linspace(0, 1, n_colors)))
    elif name == 'vibrant':
        return _ensure_color_list(plt.cm.tab10(np.linspace(0, 1, n_colors)))
    elif name == 'pastel':
        # Pastel1 颜色数有限，matplotlib 会循环，仍返回切片
        return _ensure_color_list(plt.cm.Pastel1(np.linspace(0, 1, n_colors)))
    elif name == 'viridis':
        return _ensure_color_list(plt.cm.viridis(np.linspace(0, 1, n_colors)))
    elif name == 'plasma':
        return _ensure_color_list(plt.cm.plasma(np.linspace(0, 1, n_colors)))
    elif name == 'colorblind':
        # 如果请求的数量 <= 10，直接返回 Paul Tol 色板的一部分（hex strings）
        if n_colors <= len(_TOL_COLORBLIND_10):
            return _TOL_COLORBLIND_10[:n_colors]
        # 否则在后面用 viridis 扩展
        extra = _ensure_color_list(plt.cm.viridis(np.linspace(0, 1, n_colors - len(_TOL_COLORBLIND_10))))
        return _TOL_COLORBLIND_10 + extra
    elif name == 'mono':
        # 灰度调色，便于黑白打印
        return _ensure_color_list(plt.cm.Greys(np.linspace(0.2, 0.85, n_colors)))
    else:
        raise ValueError(f"未知配色方案: {name}")

# -----------------------------------------------------------------------------
# 尝试应用样式（兼容 seaborn 版本差异）
# -----------------------------------------------------------------------------
def _try_use_styles(candidates: Iterable[str]) -> Optional[str]:
    """
    按顺序尝试样式名，成功则返回所用样式名；全部失败返回 None。
    这样可以兼容 seaborn 不同版本的样式名（例如 seaborn-v0_8-*）。
    """
    for s in candidates:
        try:
            plt.style.use(s)
            return s
        except Exception:
            continue
    return None

# -----------------------------------------------------------------------------
# 全局样式（适用于直接想改变全局 rc 的情形）
# -----------------------------------------------------------------------------
def apply_publication_style():
    """
    应用发表级论文样式（修改全局 mpl.rcParams）。
    - 尝试 seaborn-v0_8-paper，否则回退到 seaborn-paper（若存在）。
    - 设置字体、线宽、图例等常用 rc 参数，并明确设置 axes.prop_cycle。
    """
    _try_use_styles(['seaborn-v0_8-paper', 'seaborn-paper'])

    palette = get_color_palette('default', 10)
    # 确保 prop_cycle 中的 color 列表为 Python list（避免 generator 问题）
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,

        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'patch.linewidth': 0.5,

        'axes.linewidth': 1.0,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'grid.linestyle': ':',

        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,

        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.fancybox': False,
        'legend.edgecolor': '0.8',

        'figure.figsize': (8, 6),
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,

        # 明确使用 list(...) 保证兼容
        'axes.prop_cycle': cycler('color', list(palette)),
    })

def apply_presentation_style():
    """
    应用演示文稿（talk）样式（修改全局 mpl.rcParams）。
    """
    _try_use_styles(['seaborn-v0_8-talk', 'seaborn-talk'])
    palette = get_color_palette('vibrant', 10)
    mpl.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 20,

        'lines.linewidth': 2.5,
        'lines.markersize': 8,

        'axes.linewidth': 1.5,
        'grid.linewidth': 1.0,

        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.major.size': 6,
        'ytick.major.size': 6,

        'figure.figsize': (10, 7),
        'savefig.dpi': 150,

        'axes.prop_cycle': cycler('color', list(palette)),
    })

# -----------------------------------------------------------------------------
# 上下文管理器：在 with 块内临时应用样式（不会污染全局 rc）
# -----------------------------------------------------------------------------
@contextmanager
def publication_style():
    """论文样式上下文管理器：仅在 with 块内生效（使用 mpl.rc_context）"""
    with mpl.rc_context():
        apply_publication_style()
        yield

@contextmanager
def presentation_style():
    """演示样式上下文管理器：仅在 with 块内生效（使用 mpl.rc_context）"""
    with mpl.rc_context():
        apply_presentation_style()
        yield

# -----------------------------------------------------------------------------
# LaTeX 渲染安全设置
# -----------------------------------------------------------------------------
def set_latex_style(safe_fallback: bool = True, logger=None):
    """
    尝试启用 LaTeX 渲染（mpl.rcParams['text.usetex']=True）。
    - safe_fallback=True（默认）：若系统未安装 LaTeX，则回退到 mathtext（text.usetex=False）并用 logger.warning 报告。
    - safe_fallback=False：若未检测到 LaTeX，则抛出 RuntimeError。
    """
    has_tex = bool(shutil.which('latex') or shutil.which('pdflatex'))
    if has_tex:
        mpl.rcParams.update({
            'text.usetex': True,
            'text.latex.preamble': r'\usepackage{amsmath}',
            'font.family': 'serif',
            'font.serif': ['Computer Modern Roman'],
        })
        if logger is not None:
            try:
                logger.info("LaTeX 渲染已启用（系统检测到 latex/pdflatex）。")
            except Exception:
                pass
    else:
        if safe_fallback:
            mpl.rcParams.update({'text.usetex': False})
            if logger is not None:
                try:
                    logger.warning("未检测到系统 LaTeX，已回退到 matplotlib mathtext（text.usetex=False）。")
                except Exception:
                    pass
        else:
            raise RuntimeError("未检测到系统 LaTeX（latex/pdflatex）。如需回退请设置 safe_fallback=True。")

# -----------------------------------------------------------------------------
# 专用样式函数（可用于单个 Axes）
# -----------------------------------------------------------------------------
def style_phase_diagram(ax: mpl.axes.Axes):
    """相图专用样式：背景色、网格与刻度微调。"""
    ax.set_facecolor(COLORS['background'])
    ax.grid(True, alpha=0.2, color='white', linewidth=1.2)
    ax.tick_params(colors='black', which='both')

def style_fss_plot(ax: mpl.axes.Axes, show_critical_line: bool = True, Tc: float = 2.269185):
    """FSS 图专用样式：网格与临界线（若需要）。"""
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    if show_critical_line:
        ylim = ax.get_ylim()
        # 在绘制前检查 ylim 有效
        if ylim[0] < ylim[1]:
            ax.axvline(Tc, color=COLORS['secondary'], linestyle='--', linewidth=2, alpha=0.7, zorder=0)
            ax.text(
                Tc,
                ylim[1] - 0.05 * (ylim[1] - ylim[0]),
                f'$T_c={Tc:.4f}$',
                ha='center',
                va='top',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            )

def style_comparison_plot(ax: mpl.axes.Axes, n_groups: int):
    """
    对比图样式：返回 (colors, markers) 供外部循环使用。
    - colors: list
    - markers: list
    """
    colors = get_color_palette('default', n_groups)
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h'][:n_groups]
    return colors, markers

# -----------------------------------------------------------------------------
# 温度/相态 colormap 工具
# -----------------------------------------------------------------------------
def temperature_colormap(T_array: np.ndarray, T_min: Optional[float] = None, T_max: Optional[float] = None,
                         cmap: str = 'RdYlBu_r') -> Tuple[np.ndarray, mpl.colors.Normalize, mpl.colors.Colormap]:
    """
    根据温度数组生成 colors, norm, cmap（方便散点/曲线着色）。
    返回 (rgba_colors, Normalize, Colormap)
    """
    T_array = np.asarray(T_array, dtype=float)
    if T_array.size == 0:
        raise ValueError("T_array 为空")
    T_min = float(np.nanmin(T_array)) if T_min is None else float(T_min)
    T_max = float(np.nanmax(T_array)) if T_max is None else float(T_max)
    norm = mpl.colors.Normalize(vmin=T_min, vmax=T_max)
    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj(norm(T_array))
    return colors, norm, cmap_obj

def phase_colormap(phase_labels: Iterable[int]) -> List[str]:
    """相态颜色映射：0 (顺磁), 1 (铁磁), 2 (临界)。"""
    mapping = {0: COLORS['para'], 1: COLORS['ferro'], 2: COLORS['critical']}
    return [mapping.get(int(p), COLORS['gray']) for p in phase_labels]

# -----------------------------------------------------------------------------
# 图例优化：去重与美观
# -----------------------------------------------------------------------------
def optimize_legend(ax: mpl.axes.Axes, loc: str = 'best', ncol: int = 1, title: Optional[str] = None,
                    dedup: bool = True) -> mpl.legend.Legend:
    """
    优化图例显示：
      - dedup=True: 去重（保留最后出现的标签对应的句柄）
      - 返回 matplotlib.legend.Legend
    """
    handles, labels = ax.get_legend_handles_labels()
    if dedup and labels:
        # 使用有序字典语义（保留最后出现）
        uniq = {}
        for h, lab in zip(handles, labels):
            uniq[lab] = h
        labels = list(uniq.keys())
        handles = [uniq[lab] for lab in labels]
    legend = ax.legend(handles, labels, loc=loc, ncol=ncol, title=title,
                       frameon=True, framealpha=0.9, edgecolor='0.8', fancybox=False)
    if title:
        try:
            legend.get_title().set_fontweight('bold')
        except Exception:
            pass
    return legend

# -----------------------------------------------------------------------------
# 保存图形：统一接口（支持多格式）
# -----------------------------------------------------------------------------
def save_figure(fig: mpl.figure.Figure, filepath: Union[str, Path],
                formats: Iterable[str] = ('png', 'pdf'),
                logger=None, **kwargs) -> List[Path]:
    """
    保存图形为多种格式。
      - filepath: 基础路径（可以包含扩展名或不包含）
      - formats: 可迭代格式名，如 ('png','pdf','svg')
      - kwargs: 传递给 fig.savefig（覆盖默认 dpi/bbox/pad 等）
    返回保存的 Path 列表
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # 默认参数：可被 kwargs 覆盖
    default_kwargs = {'dpi': 300, 'bbox_inches': 'tight', 'pad_inches': 0.1}
    default_kwargs.update(kwargs)

    # 处理 formats 参数（如果 filepath 自带后缀，仍按 formats 保存以一致行为）
    if isinstance(formats, (str, Path)):
        formats = [str(formats)]

    saved_files: List[Path] = []
    stem = filepath.with_suffix('')  # 去掉已有后缀以便追加自定义后缀

    for fmt in formats:
        out = stem.with_suffix('.' + str(fmt))
        try:
            fig.savefig(out, format=str(fmt), **default_kwargs)
        except TypeError:
            # 某些后端/版本不接受 format kw，尝试去掉 format
            fig.savefig(out, **default_kwargs)
        saved_files.append(out)

    # 记录或打印保存信息
    message = "Saved figure(s): " + ", ".join(str(p) for p in saved_files)
    if logger is not None:
        try:
            logger.info(message)
        except Exception:
            pass
    else:
        # 以尽量简短的形式输出（避免在库中大量打印）
        print(message)
    return saved_files

# -----------------------------------------------------------------------------
# 模块演示（仅在直接运行时执行）
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # 简短演示 publish/presentation style 的作用
    print("style.py 演示：publication_style / save_figure")
    with publication_style():
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.linspace(0, 10, 200)
        for i, c in enumerate(get_color_palette('colorblind', 5)):
            ax.plot(x, np.sin(x + i * 0.5), label=f'系列 {i+1}', color=c)
        style_fss_plot(ax, show_critical_line=False)
        optimize_legend(ax, dedup=True)
        plt.title("示例图（publication_style）")
        save_figure(fig, Path('style_demo'), formats=['png'])
        try:
            plt.show()
        except Exception:
            pass

    print("演示完成。")

