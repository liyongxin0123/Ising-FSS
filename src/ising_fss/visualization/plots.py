# -*- coding: utf-8 -*-
"""
绘图封装（FSS 专用）

实现功能：
    - 一键生成完整 FSS 分析图（比热、磁化率、Binder 交叉、数据坍缩）
    - 自动误差条 + 降采样（避免过密）
    - 色盲友好配色 + LaTeX 安全
    - 支持多格式保存（png/pdf/svg）
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, Iterable, List
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import warnings

__all__ = [
    'plot_comprehensive_fss', 'plot_data_collapse', 'plot_binder_crossing',
    'plot_phase_diagram', 'plot_config_grid', 'plot_training_history',
    'plot_vae_reconstruction', 'plot_latent_space',
    'save_figure', 'savefig', 'export_figure'
]

# -----------------------------------------------------------------------------
# 保存工具（保持与旧接口兼容）
# -----------------------------------------------------------------------------

def save_figure(
    fig: Optional[plt.Figure] = None,
    path: str | os.PathLike | None = None,
    *,
    dpi: int = 300,
    transparent: bool = False,
    pad_inches: float = 0.02,
    tight: bool = True,
    create_dir: bool = True,
    formats: Optional[List[str] | Tuple[str, ...]] = None,
) -> str | List[str]:
    """
    通用图片保存助手（向后兼容）。
    若 path 包含后缀，则只按该后缀保存；否则按 formats 保存多个文件。
    """
    if path is None:
        raise ValueError("save_figure: 需要提供保存路径 path。")

    fig = fig if fig is not None else plt.gcf()
    path = Path(path)

    if create_dir and path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    # 推断输出格式
    if formats is None:
        if path.suffix:
            formats = [path.suffix.lstrip(".").lower()]
            stem = path.with_suffix("")
        else:
            formats = ["png"]
            stem = path
    else:
        formats = [f.lstrip(".").lower() for f in formats]
        stem = path.with_suffix("")

    # 布局收紧（容错）
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass

    saved: List[str] = []
    for ext in formats:
        out_path = stem.with_suffix("." + ext)
        fig.savefig(
            out_path,
            dpi=dpi,
            transparent=transparent,
            bbox_inches="tight" if tight else None,
            pad_inches=pad_inches,
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
        saved.append(str(out_path))

    return saved[0] if len(saved) == 1 else saved

# 兼容旧名
def savefig(*args, **kwargs):
    return save_figure(*args, **kwargs)

def export_figure(*args, **kwargs):
    return save_figure(*args, **kwargs)

# -----------------------------------------------------------------------------
# 工具函数（增强版）
# -----------------------------------------------------------------------------

def _maybe_log(logger, msg: str):
    if logger is not None:
        try:
            logger.info(msg)
        except Exception:
            pass

def _maybe_save(fig: plt.Figure, save_path: Optional[str], dpi: int = 300, logger=None):
    if not save_path:
        return None
    try:
        saved = save_figure(fig=fig, path=save_path, dpi=dpi)
        _maybe_log(logger, f"图像已保存: {saved}")
        return saved
    except Exception as e:
        if logger is not None:
            try:
                logger.exception(f"保存图像失败: {e}")
            except Exception:
                pass
        # 仍然将异常抛出，调用者可选择捕获
        raise

def _safe_get(d: Dict[str, Any], key: str, default=np.nan, warn_prefix: str = "", logger=None):
    """
    从字典安全取值：若缺键，返回 default 并警告（优先使用 logger.warning）。
    warn_prefix: 前缀信息（如 'L=16: '）
    logger: 可选 logger 对象，若提供则使用 logger.warning
    """
    if key in d:
        return d[key]
    msg = f"{warn_prefix}缺少键 '{key}'，已使用 NaN 代替。"
    if logger is not None:
        try:
            logger.warning(msg)
        except Exception:
            pass
    else:
        warnings.warn(msg)
    return default

def _downsample_xy(x: np.ndarray, y: np.ndarray, max_points: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    均匀下采样 x,y（保持对齐）。若 max_points 为 None 或长度足够小则返回原值。
    """
    if max_points is None or len(x) <= max_points:
        return x, y
    idx = np.linspace(0, len(x) - 1, max_points, dtype=int)
    return x[idx], y[idx]

def _downsample_xy_with_yerr(x: np.ndarray, y: np.ndarray, yerr: Optional[np.ndarray], max_points: Optional[int]):
    """
    对带误差棒的数据进行降采样，确保 yerr 与 y 对齐。
    返回 (x2, y2, yerr2)
    """
    if max_points is None or len(x) <= max_points:
        return x, y, yerr
    idx = np.linspace(0, len(x) - 1, max_points, dtype=int)
    yerr2 = None if yerr is None else yerr[idx]
    return x[idx], y[idx], yerr2

def _fit_powerlaw(x_pos: np.ndarray, y_pos: np.ndarray) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
    """
    在 log-log 空间拟合 y = a * x^b。
    返回 ((b, ln a), R2)；若点数不足或非法，返回 (None, None)。
    """
    # 过滤正、有限值
    mask = (x_pos > 0) & (y_pos > 0) & np.isfinite(x_pos) & np.isfinite(y_pos)
    if np.sum(mask) < 2:
        return None, None
    lx = np.log(x_pos[mask])
    ly = np.log(y_pos[mask])
    if lx.size < 2 or np.allclose(np.var(lx), 0.0) or np.allclose(np.var(ly), 0.0):
        return None, None
    try:
        p = np.polyfit(lx, ly, 1)
        ly_hat = p[0] * lx + p[1]
        ss_res = np.sum((ly - ly_hat) ** 2)
        ss_tot = np.sum((ly - np.mean(ly)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return (float(p[0]), float(p[1])), float(r2)
    except Exception:
        return None, None

# -----------------------------------------------------------------------------
# 1) FSS 综合图
# -----------------------------------------------------------------------------

def plot_comprehensive_fss(
    results_dict: Dict[int, Dict[float, Dict[str, Any]]],
    Tc_theory: float = 2.269185,
    save_path: Optional[str] = None,
    dpi: int = 300,
    rasterized: bool = True,
    max_points: Optional[int] = None,
    logger=None,
):
    """
    绘制完整的 FSS 分析图 (3×2 布局)

    参数:
        results_dict: {L: {T: {'E','M','C','chi','U','C_err','chi_err'}}}
        其余参数见 docstring。
    返回:
        fig: matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 17))
    axes = np.asarray(axes)

    L_list = sorted(results_dict.keys())
    if len(L_list) == 0:
        raise ValueError("results_dict 为空。")
    colors = plt.cm.viridis(np.linspace(0, 1, len(L_list)))

    # 曲线绘制
    for idx, L in enumerate(L_list):
        try:
            T_arr = np.array(sorted(results_dict[L].keys()), dtype=float)
            cur = results_dict[L]
            warnp = f"L={L}: "

            E_arr = np.array([_safe_get(cur[T], 'E', np.nan, warnp, logger=logger) for T in T_arr], dtype=float)
            M_arr = np.array([_safe_get(cur[T], 'M', np.nan, warnp, logger=logger) for T in T_arr], dtype=float)
            C_arr = np.array([_safe_get(cur[T], 'C', np.nan, warnp, logger=logger) for T in T_arr], dtype=float)
            chi_arr = np.array([_safe_get(cur[T], 'chi', np.nan, warnp, logger=logger) for T in T_arr], dtype=float)
            U_arr = np.array([_safe_get(cur[T], 'U', np.nan, warnp, logger=logger) for T in T_arr], dtype=float)
            C_err = np.array([_safe_get(cur[T], 'C_err', np.nan, warnp, logger=logger) for T in T_arr], dtype=float)
            chi_err = np.array([_safe_get(cur[T], 'chi_err', np.nan, warnp, logger=logger) for T in T_arr], dtype=float)

            # 若整条曲线全为 NaN，跳过
            if not np.any(np.isfinite(E_arr)) and not np.any(np.isfinite(M_arr)) and not np.any(np.isfinite(C_arr)) and not np.any(np.isfinite(chi_arr)):
                _maybe_log(logger, f"L={L}: 所有指标均为 NaN，跳过绘制。")
                continue

            # 可选降采样（保持 yerr 对齐）
            if max_points is not None and T_arr.size > max_points:
                idx_ds = np.linspace(0, T_arr.size - 1, max_points, dtype=int)
                T_arr = T_arr[idx_ds]
                E_arr, M_arr, C_arr, chi_arr, U_arr = E_arr[idx_ds], M_arr[idx_ds], C_arr[idx_ds], chi_arr[idx_ds], U_arr[idx_ds]
                C_err, chi_err = C_err[idx_ds], chi_err[idx_ds]

            # (0,0) 能量
            if np.any(np.isfinite(E_arr)):
                axes[0, 0].plot(T_arr, E_arr, 'o-', color=colors[idx], label=f'L={L}',
                                markersize=4.5, linewidth=1.25, rasterized=rasterized)

            # (0,1) 磁化强度
            if np.any(np.isfinite(M_arr)):
                axes[0, 1].plot(T_arr, M_arr, 's-', color=colors[idx], label=f'L={L}',
                                markersize=4.5, linewidth=1.25, rasterized=rasterized)

            # (1,0) 比热（带误差棒）
            if np.any(np.isfinite(C_arr)):
                axes[1, 0].errorbar(T_arr, C_arr, yerr=(C_err if np.any(np.isfinite(C_err)) else None), fmt='o-',
                                    color=colors[idx], label=f'L={L}', markersize=3.5, capsize=2.5, alpha=0.85,
                                    linewidth=1.1, rasterized=rasterized)

            # (1,1) 磁化率（带误差棒）
            if np.any(np.isfinite(chi_arr)):
                axes[1, 1].errorbar(T_arr, chi_arr, yerr=(chi_err if np.any(np.isfinite(chi_err)) else None),
                                    fmt='^-', color=colors[idx], label=f'L={L}', markersize=3.5, capsize=2.5, alpha=0.85,
                                    linewidth=1.1, rasterized=rasterized)

            # (2,0) Binder 累积量
            if np.any(np.isfinite(U_arr)):
                axes[2, 0].plot(T_arr, U_arr, 'd-', color=colors[idx], label=f'L={L}',
                                markersize=5.0, linewidth=1.25, rasterized=rasterized)
        except Exception as e:
            if logger is not None:
                try:
                    logger.exception(f"L={L} 绘制失败: {e}")
                except Exception:
                    pass
            else:
                warnings.warn(f"L={L} 绘制失败: {e}")

    # (2,1) 峰值幂律（双对数）
    L_arr = np.array(L_list, dtype=float)
    chi_max = []
    C_max = []
    for L in L_list:
        try:
            T_arr = np.array(sorted(results_dict[L].keys()), dtype=float)
            chi_v = np.array([_safe_get(results_dict[L][T], 'chi', np.nan, f'L={L}: ', logger=logger) for T in T_arr], dtype=float)
            C_v = np.array([_safe_get(results_dict[L][T], 'C', np.nan, f'L={L}: ', logger=logger) for T in T_arr], dtype=float)
            chi_max.append(np.nanmax(chi_v) if np.any(np.isfinite(chi_v)) else np.nan)
            C_max.append(np.nanmax(C_v) if np.any(np.isfinite(C_v)) else np.nan)
        except Exception as e:
            _maybe_log(logger, f"L={L}: 计算峰值失败: {e}")
            chi_max.append(np.nan)
            C_max.append(np.nan)
    chi_max = np.asarray(chi_max, dtype=float)
    C_max = np.asarray(C_max, dtype=float)

    # 若全部为 NaN，跳过该子图
    if np.all(~np.isfinite(chi_max)) and np.all(~np.isfinite(C_max)):
        _maybe_log(logger, "所有 L 的 chi_max/C_max 均为 NaN，跳过幂律绘制。")
    else:
        # 仅绘制有限点
        finite_idx_chi = np.isfinite(chi_max) & (chi_max > 0)
        finite_idx_C = np.isfinite(C_max) & (C_max > 0)

        if np.any(finite_idx_chi):
            axes[2, 1].loglog(L_arr[finite_idx_chi], chi_max[finite_idx_chi], 'o-', label='$\\chi_{\\mathrm{max}}$', markersize=7, linewidth=1.6)
        if np.any(finite_idx_C):
            axes[2, 1].loglog(L_arr[finite_idx_C], C_max[finite_idx_C], 's-', label='$C_{\\mathrm{max}}$', markersize=7, linewidth=1.6)

        # 拟合幂律并叠加（分别拟合 chi 与 C）
        if np.sum(finite_idx_chi) >= 2:
            p_chi, r2_chi = _fit_powerlaw(L_arr[finite_idx_chi], chi_max[finite_idx_chi])
        else:
            p_chi, r2_chi = None, None
        if np.sum(finite_idx_C) >= 2:
            p_C, r2_C = _fit_powerlaw(L_arr[finite_idx_C], C_max[finite_idx_C])
        else:
            p_C, r2_C = None, None

        if p_chi is not None:
            b, ln_a = p_chi
            L_fit = np.linspace(np.min(L_arr[finite_idx_chi]), np.max(L_arr[finite_idx_chi]), 200)
            axes[2, 1].loglog(L_fit, np.exp(ln_a) * (L_fit ** b), '--', alpha=0.6, linewidth=2,
                              label=f'$\\chi \\sim L^{{{b:.2f}}}$  (R$^2$={r2_chi:.3f})')
            _maybe_log(logger, f"幂律拟合 chi_max: 指数={b:.4f}, R2={r2_chi:.4f}")
        if p_C is not None:
            b, ln_a = p_C
            L_fit = np.linspace(np.min(L_arr[finite_idx_C]), np.max(L_arr[finite_idx_C]), 200)
            axes[2, 1].loglog(L_fit, np.exp(ln_a) * (L_fit ** b), '--', alpha=0.6, linewidth=2,
                              label=f'$C \\sim L^{{{b:.2f}}}$  (R$^2$={r2_C:.3f})')
            _maybe_log(logger, f"幂律拟合 C_max: 指数={b:.4f}, R2={r2_C:.4f}")

    # 标注理论临界温度
    for ax in axes.flat:
        try:
            ax.axvline(Tc_theory, color='crimson', linestyle='--', alpha=0.6, linewidth=2)
        except Exception:
            pass
    # 给 Binder 子图添加标签
    axes[2, 0].axvline(Tc_theory, color='crimson', linestyle='--', alpha=0.6, linewidth=2,
                       label=f'$T_c={Tc_theory:.4f}$')

    # 标签美化
    axes[0, 0].set_ylabel('能量/格点 $\\langle E \\rangle$', fontsize=13)
    axes[0, 1].set_ylabel('磁化强度 $\\langle |M| \\rangle$', fontsize=13)
    axes[1, 0].set_ylabel('比热 $C$', fontsize=13)
    axes[1, 1].set_ylabel('磁化率 $\\chi$', fontsize=13)
    axes[2, 0].set_ylabel('Binder累积量 $U_L$', fontsize=13)
    axes[2, 1].set_ylabel('峰值 (对数)', fontsize=13)

    axes[2, 0].set_xlabel('温度 $T$', fontsize=13)
    axes[2, 1].set_xlabel('晶格尺寸 $L$ (对数)', fontsize=13)

    for ax in axes.flat:
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
        ax.grid(alpha=0.3, linestyle=':')
        ax.tick_params(labelsize=11)

    axes[0, 0].set_title('(a) 能量', fontsize=14, fontweight='bold')
    axes[0, 1].set_title('(b) 磁化强度', fontsize=14, fontweight='bold')
    axes[1, 0].set_title('(c) 比热', fontsize=14, fontweight='bold')
    axes[1, 1].set_title('(d) 磁化率', fontsize=14, fontweight='bold')
    axes[2, 0].set_title('(e) Binder累积量', fontsize=14, fontweight='bold')
    axes[2, 1].set_title('(f) 临界指数', fontsize=14, fontweight='bold')

    plt.tight_layout()
    # 保存（若提供）
    _maybe_save(fig, save_path, dpi=dpi, logger=logger)
    return fig

# -----------------------------------------------------------------------------
# 2) 数据坍缩图
# -----------------------------------------------------------------------------

def plot_data_collapse(
    collapsed_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    observable: str = 'chi',
    Tc_est: Optional[float] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
    max_points: Optional[int] = None,
    rasterized: bool = True,
    logger=None,
):
    """
    绘制数据坍缩图。
    collapsed_data: {L: (x_scaled, y_scaled)}
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    L_list = sorted(collapsed_data.keys())
    if len(L_list) == 0:
        raise ValueError("collapsed_data 为空。")
    colors = plt.cm.plasma(np.linspace(0, 1, len(L_list)))

    for idx, L in enumerate(L_list):
        try:
            x, y = collapsed_data[L]
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            if x.shape != y.shape:
                raise ValueError(f"L={L} 的 x/y 形状不一致: {x.shape} vs {y.shape}")
            x, y = _downsample_xy(x, y, max_points)
            mask = np.isfinite(x) & np.isfinite(y)
            if not np.any(mask):
                _maybe_log(logger, f"L={L}: 数据全部非有限，跳过。")
                continue
            x, y = x[mask], y[mask]
            ax.plot(x, y, 'o', color=colors[idx], label=f'L={L}', markersize=5.5,
                    alpha=0.75, rasterized=rasterized)
        except Exception as e:
            if logger is not None:
                try:
                    logger.exception(f"L={L} 绘制折叠数据失败: {e}")
                except Exception:
                    pass
            else:
                warnings.warn(f"L={L} 绘制折叠数据失败: {e}")

    if observable == 'chi':
        ax.set_ylabel('$\\chi / L^{\\gamma/\\nu}$', fontsize=14)
        title = '磁化率数据坍缩'
    elif observable == 'C':
        ax.set_ylabel('$C / L^{\\alpha/\\nu}$', fontsize=14)
        title = '比热数据坍缩'
    elif observable == 'M':
        ax.set_ylabel('$M / L^{-\\beta/\\nu}$', fontsize=14)
        title = '磁化强度数据坍缩'
    else:
        ax.set_ylabel('Scaled Observable', fontsize=14)
        title = '数据坍缩'

    ax.set_xlabel('$(T - T_c) \\cdot L^{1/\\nu}$', fontsize=14)
    ax.set_title(title + (f' (T$_c$ = {Tc_est:.4f})' if Tc_est is not None else ''), fontsize=16)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=2)

    plt.tight_layout()
    _maybe_save(fig, save_path, dpi=dpi, logger=logger)
    return fig

# -----------------------------------------------------------------------------
# 3) Binder 交叉图
# -----------------------------------------------------------------------------

def plot_binder_crossing(
    results_dict: Dict[int, Dict[float, Dict[str, Any]]],
    Tc_theory: float = 2.269185,
    save_path: Optional[str] = None,
    dpi: int = 300,
    rasterized: bool = True,
    max_points: Optional[int] = None,
    logger=None,
):
    """
    专门绘制 Binder 累积量交叉图。
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    L_list = sorted(results_dict.keys())
    if len(L_list) == 0:
        raise ValueError("results_dict 为空。")
    colors = plt.cm.viridis(np.linspace(0, 1, len(L_list)))

    for idx, L in enumerate(L_list):
        try:
            T_all = np.array(sorted(results_dict[L].keys()), dtype=float)
            U_all = np.array([_safe_get(results_dict[L][T], 'U', np.nan, f'L={L}: ', logger=logger) for T in T_all], dtype=float)
            if max_points is not None and T_all.size > max_points:
                ids = np.linspace(0, T_all.size - 1, max_points, dtype=int)
                T_all, U_all = T_all[ids], U_all[ids]
            mask = np.isfinite(T_all) & np.isfinite(U_all)
            if not np.any(mask):
                _maybe_log(logger, f"L={L}: Binder 数据全部非有限，跳过。")
                continue
            T_arr, U_arr = T_all[mask], U_all[mask]
            ax.plot(T_arr, U_arr, 'o-', color=colors[idx], label=f'L={L}', markersize=6.5,
                    linewidth=1.6, rasterized=rasterized)
        except Exception as e:
            if logger is not None:
                try:
                    logger.exception(f"L={L} 绘制 Binder 失败: {e}")
                except Exception:
                    pass
            else:
                warnings.warn(f"L={L} 绘制 Binder 失败: {e}")

    # 标注理论临界温度与临界 U 值（2D Ising 常用近似）
    ax.axvline(Tc_theory, color='crimson', linestyle='--', alpha=0.7, linewidth=2.5,
               label=f'$T_c$ (理论) = {Tc_theory:.4f}')
    U_critical = 0.466
    ax.axhline(U_critical, color='gray', linestyle=':', alpha=0.6, linewidth=2,
               label=f'$U(T_c)$ ≈ {U_critical}')

    ax.set_xlabel('温度 $T$', fontsize=14)
    ax.set_ylabel('Binder累积量 $U_L$', fontsize=14)
    ax.set_title('Binder累积量交叉分析', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(alpha=0.3, linestyle=':')
    ax.tick_params(labelsize=12)

    textstr = '交叉点 → 临界温度\n不同 L 的曲线在 $T_c$ 处相交'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    _maybe_save(fig, save_path, dpi=dpi, logger=logger)
    return fig

# -----------------------------------------------------------------------------
# 4) (T,h) 相图与构型网格
# -----------------------------------------------------------------------------

def plot_phase_diagram(
    dataset: Dict[str, Any],
    save_path: Optional[str] = None,
    dpi: int = 300,
    logger=None,
):
    """
    绘制 (T, h) 相图（以平均 |M| 热图表示）。
    dataset 需包含 'configs' (n_h, n_T, n_cfg, L, L), 'temperatures', 'fields'
    """
    if not all(k in dataset for k in ('configs', 'temperatures', 'fields')):
        raise KeyError("dataset 需包含 'configs', 'temperatures', 'fields' 三个键。")

    configs = np.asarray(dataset['configs'])
    temps = np.asarray(dataset['temperatures'], dtype=float).ravel()
    fields = np.asarray(dataset['fields'], dtype=float).ravel()

    if configs.ndim != 5:
        raise ValueError("configs 形状应为 (n_h, n_T, n_cfg, L, L)")
    n_h, n_T, _, _, _ = configs.shape
    if temps.size != n_T or fields.size != n_h:
        warnings.warn("temperatures/fields 尺寸与 configs 不完全匹配，将尝试继续。")

    # 平均 |M|
    avg_mag = np.mean(np.abs(configs), axis=(2, 3, 4))  # -> (n_h, n_T)

    fig, ax = plt.subplots(figsize=(10, 8))
    extent = [np.nanmin(temps), np.nanmax(temps), np.nanmin(fields), np.nanmax(fields)]
    im = ax.imshow(
        avg_mag,
        extent=extent,
        aspect='auto', origin='lower', cmap='RdBu_r'
    )
    cbar = plt.colorbar(im, ax=ax, label=r'平均磁化强度 $\langle |M| \rangle$')
    cbar.ax.tick_params(labelsize=11)

    ax.set_xlabel('温度 $T$', fontsize=14)
    ax.set_ylabel('外部磁场 $h$', fontsize=14)
    ax.set_title('伊辛模型相图', fontsize=16, fontweight='bold')

    # 标注临界温度与 h=0
    Tc = 2.269185
    ax.axvline(Tc, color='white', linestyle='--', linewidth=2.0, label=f'$T_c$ = {Tc:.6f}')
    ax.axhline(0, color='white', linestyle=':', linewidth=2, alpha=0.7)
    ax.legend(fontsize=12, loc='upper right')

    ax.tick_params(labelsize=12)
    plt.tight_layout()
    _maybe_save(fig, save_path, dpi=dpi, logger=logger)
    return fig

def plot_config_grid(
    dataset: Dict[str, Any],
    n_temps: int = 5,
    n_fields: int = 5,
    config_idx: int = 0,
    save_path: Optional[str] = None,
    dpi: int = 300,
    logger=None,
):
    """
    绘制 (T,h) 网格的构型拼图。
    """
    if not all(k in dataset for k in ('configs', 'temperatures', 'fields')):
        raise KeyError("dataset 需包含 'configs', 'temperatures', 'fields' 三个键。")

    configs = np.asarray(dataset['configs'])
    temps = np.asarray(dataset['temperatures'], dtype=float).ravel()
    fields = np.asarray(dataset['fields'], dtype=float).ravel()

    if configs.ndim != 5:
        raise ValueError("configs 形状应为 (n_h, n_T, n_cfg, L, L)")

    n_h, n_T, n_cfg, L, _ = configs.shape
    if not (0 <= config_idx < n_cfg):
        raise ValueError(f"config_idx 超界: 0 <= {config_idx} < {n_cfg}")

    t_indices = np.linspace(0, n_T - 1, n_temps, dtype=int)
    h_indices = np.linspace(0, n_h - 1, n_fields, dtype=int)

    fig, axes = plt.subplots(n_fields, n_temps, figsize=(1.6 * n_temps + 3, 1.6 * n_fields + 2.5))
    axes = np.asarray(axes)

    for i, h_idx in enumerate(h_indices):
        for j, t_idx in enumerate(t_indices):
            config = configs[h_idx, t_idx, config_idx]
            axes[i, j].imshow(config, cmap='gray', vmin=-1, vmax=1)
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(f'T={temps[t_idx]:.2f}', fontsize=10)
            if j == 0:
                axes[i, j].set_ylabel(f'h={fields[h_idx]:.2f}', fontsize=10, rotation=0, labelpad=20)

    plt.suptitle(f'伊辛构型网格 (构型 #{config_idx})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _maybe_save(fig, save_path, dpi=dpi, logger=logger)
    return fig

# -----------------------------------------------------------------------------
# 5) 训练可视化
# -----------------------------------------------------------------------------

def plot_training_history(
    history: Dict[str, Iterable[float]],
    save_path: Optional[str] = None,
    dpi: int = 300,
    logger=None,
):
    """绘制训练历史：history={'train_loss': [...], 'val_loss': [...]}"""
    fig, ax = plt.subplots(figsize=(10, 6))
    train_loss = np.asarray(list(history.get('train_loss', [])), dtype=float)
    val_loss = np.asarray(list(history.get('val_loss', [])), dtype=float)
    N = int(max(len(train_loss), len(val_loss)))
    epochs = np.arange(1, N + 1)

    if len(train_loss):
        ax.plot(epochs[:len(train_loss)], train_loss, 'b-o', label='训练损失', linewidth=2, markersize=4.5)
    if len(val_loss):
        ax.plot(epochs[:len(val_loss)], val_loss, 'r-s', label='验证损失', linewidth=2, markersize=4.5)

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_title('训练历史', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    _maybe_save(fig, save_path, dpi=dpi, logger=logger)
    return fig

def plot_vae_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    sampled: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
    logger=None,
):
    """可视化 VAE 重构结果。原始/重构/采样 都假定为 (N, L, L)，取前 8 个。"""
    original = np.asarray(original)
    reconstructed = np.asarray(reconstructed)
    if original.ndim != 3 or reconstructed.ndim != 3:
        raise ValueError("original/reconstructed 需为 (N,L,L)。")
    n_samples = int(min(8, len(original)))
    n_rows = 3 if sampled is not None else 2
    fig, axes = plt.subplots(n_rows, n_samples, figsize=(2 * n_samples, 2.2 * n_rows))
    axes = np.asarray(axes)

    def _show(row, arr, label):
        arr = np.asarray(arr)
        vmin = np.nanmin(arr)
        vmax = np.nanmax(arr)
        for i in range(n_samples):
            axes[row, i].imshow(arr[i], cmap='gray', vmin=vmin, vmax=vmax)
            axes[row, i].axis('off')
        axes[row, 0].set_ylabel(label, fontsize=12, rotation=0, labelpad=24)

    _show(0, original, '原始')
    _show(1, reconstructed, '重构')
    if sampled is not None:
        sampled = np.asarray(sampled)
        if sampled.ndim != 3:
            raise ValueError("sampled 需为 (N,L,L)。")
        _show(2, sampled, '采样')

    plt.suptitle('VAE 重构结果', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _maybe_save(fig, save_path, dpi=dpi, logger=logger)
    return fig

def plot_latent_space(
    latent_codes: np.ndarray,
    labels: np.ndarray,
    label_type: str = 'temperature',
    save_path: Optional[str] = None,
    dpi: int = 300,
    logger=None,
):
    """
    可视化潜在空间（PCA 到 2D）。
    """
    try:
        from sklearn.decomposition import PCA
    except Exception as e:
        raise ImportError("plot_latent_space 需要 scikit-learn，请先安装 scikit-learn。") from e

    latent_codes = np.asarray(latent_codes, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if latent_codes.ndim != 2 or labels.ndim != 1 or labels.shape[0] != latent_codes.shape[0]:
        raise ValueError("latent_codes 需形状 (N, D)，labels 需形状 (N,) 且与之对齐。")

    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_codes)

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='coolwarm', alpha=0.55, s=8, rasterized=True)

    cbar = plt.colorbar(sc, ax=ax)
    if label_type == 'temperature':
        cbar.set_label('温度 $T$', fontsize=12)
    else:
        cbar.set_label('磁场 $h$', fontsize=12)

    ax.set_xlabel('第一主成分', fontsize=13)
    ax.set_ylabel('第二主成分', fontsize=13)
    ax.set_title(f'VAE潜在空间 (PCA投影，按{label_type}着色)', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=11)

    evr = pca.explained_variance_ratio_
    if evr.size >= 2:
        textstr = f'解释方差:\nPC1: {evr[0]:.1%}\nPC2: {evr[1]:.1%}'
    else:
        textstr = f'解释方差: {evr}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)

    plt.tight_layout()
    _maybe_save(fig, save_path, dpi=dpi, logger=logger)
    return fig

# -----------------------------------------------------------------------------
# 使用示例（可选）
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print("可视化模块（修补版）演示")
    print("=" * 70)

    # 创建简单示例 FSS 结果
    L_list = [8, 12, 16]
    T_range = np.linspace(2.0, 2.5, 20)
    results_dict: Dict[int, Dict[float, Dict[str, Any]]] = {}
    rng = np.random.default_rng(0)

    for L in L_list:
        results_dict[L] = {}
        for T in T_range:
            M = np.tanh((2.269185 - T) * L ** 0.5)
            C = L * (1 - M ** 2) + 0.02 * rng.normal()
            chi = L * (1 - M ** 2) + 0.02 * rng.normal()
            U = 1 - 1.0 / (3 * (1 + (T - 2.269185) ** 2 * L))
            results_dict[L][float(T)] = {
                'E': -1.5 + 0.5 * M,
                'M': abs(M),
                'C': float(C),
                'C_err': abs(float(C)) * 0.03 + 1e-6,
                'chi': float(chi),
                'chi_err': abs(float(chi)) * 0.03 + 1e-6,
                'U': float(U)
            }

    print("\n生成完整 FSS 分析图...")
    fig = plot_comprehensive_fss(results_dict, save_path='demo_fss.png', max_points=15)

    print("生成 Binder 交叉图...")
    fig2 = plot_binder_crossing(results_dict, save_path='demo_binder.png')

    print("完成。输出: demo_fss.png, demo_binder.png")

