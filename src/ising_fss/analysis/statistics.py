# -*- coding: utf-8 -*-
"""
时间序列统计误差分析工具包

实现功能：
    - 自动/手动估计积分自相关时间 τ_int（Sokal 窗口法 + Geyer IPS 兜底）
    - 有效样本数 ESS = n / (2 τ_int)
    - 支持多种误差估计方法：
        阻塞分析（blocking analysis）
        Jackknife
        IID Bootstrap
        Moving-block Bootstrap（推荐，处理长程相关）
    - 对短序列、常数序列、含 NaN 情况自动降级（返回 0 误差）
"""

from __future__ import annotations

import math
import numpy as np
from typing import Callable, Dict, Optional, Tuple, List

__all__ = [
    "autocorrelation_time",
    "effective_sample_size",
    "estimate_error_with_autocorr",
    "blocking_analysis",
    "jackknife_error",
    "bootstrap_error",
    "moving_block_bootstrap_error",
    "windowed_average",
    "estimate_block_len",
]

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _finite_clean(x: np.ndarray) -> np.ndarray:
    """返回仅包含有限值的一维 float 向量（丢弃 NaN/Inf），空输入返回空数组。"""
    y = np.asarray(x, dtype=float).ravel()
    if y.size == 0:
        return np.array([], dtype=float)
    mask = np.isfinite(y)
    return y[mask] if mask.any() else np.array([], dtype=float)


def _next_pow_two(n: int) -> int:
    """返回不小于 n 的 2 的幂（用于 FFT 长度）。"""
    n = int(max(1, n))
    return 1 << (n - 1).bit_length()


def _acf_fft(x: np.ndarray, max_lag: Optional[int] = None, demean: bool = True) -> np.ndarray:
    """
    使用 FFT 计算自协方差，并归一化得到自相关 rho[0..max_lag]（rho[0]=1）。
    - 采用“近似无偏”归一：Cov(k) / (n-k)
    - 对零方差/小样本做了稳健处理
    """
    x = _finite_clean(x)
    n = x.size
    if n == 0:
        return np.array([1.0], dtype=float)
    if demean:
        x = x - x.mean()

    if max_lag is None:
        max_lag = min(n - 1, 4096)
    else:
        max_lag = int(max(0, min(max_lag, n - 1)))

    m = _next_pow_two(2 * n)
    fx = np.fft.rfft(x, n=m)
    sxx = fx * np.conj(fx)
    # 只取前 n 项（k=0..n-1）
    acov_full = np.fft.irfft(sxx, n=m)[:n]

    # 无偏似归一：按配对数 (n-k) 归一
    denom = (n - np.arange(n)).astype(float)
    denom = np.maximum(denom, 1.0)
    acov = acov_full / denom

    v0 = acov[0]
    if not np.isfinite(v0) or v0 <= 0.0:
        # 方差退化（常数序列或几乎常数）：仅保留 rho[0]=1
        rho = np.zeros(max_lag + 1, dtype=float)
        rho[0] = 1.0
        return rho

    rho = (acov / v0)[: max_lag + 1]
    rho[0] = 1.0
    return rho


# ---------------------------------------------------------------------
# τ_int / ESS / 自相关修正误差
# ---------------------------------------------------------------------
def _tau_ips_from_rho(rho: np.ndarray) -> float:
    """
    Geyer 初始正序 (Initial Positive Sequence, IPS) 的 τ_int 兜底：
    - 在配对和 rho[2k] + rho[2k+1] 首次为非正时截断
    - τ_int = 0.5 + Σ rho[t]（t=1..T）
    - 若 rho 长度不足或全非正，退化为 1.0
    """
    if rho.size <= 1:
        return 1.0
    # 逐步累计，按 IPS 判据截断
    s = 0.0
    # 到达偶数长度边界
    max_t = rho.size - 1
    t = 1
    while t <= max_t:
        r1 = float(rho[t]) if np.isfinite(rho[t]) else 0.0
        r2 = float(rho[t + 1]) if (t + 1 <= max_t and np.isfinite(rho[t + 1])) else 0.0
        if (r1 + r2) <= 0.0:
            break
        s += r1
        if (t + 1) <= max_t:
            s += r2
        t += 2
    tau = 0.5 + s
    return float(max(1.0, tau))


def autocorrelation_time(
    data: np.ndarray,
    max_lag: Optional[int] = None,
    method: str = "sokal",
    c: float = 5.0,
) -> float:
    """
    估计积分自相关时间 τ_int。

    参数
    ----
    data : 1D 序列（NaN/Inf 会被剔除）
    max_lag : 最大滞后（默认 min(n-1,4096)）
    method : 'sokal' | 'threshold'
      - 'sokal'（默认）：Sokal 自洽窗口法（t <= c * tau）迭代
      - 'threshold'：累加到 rho[t] < 0.05 为止（简单粗暴）
    c : Sokal 窗口参数

    返回
    ----
    tau : float (>= 1.0)

    稳健性：
    - 对 n<4 或近似常数序列，退化返回 1.0
    - 当 Sokal 迭代不收敛 / 爆炸 / 非有限时，自动回退到 Geyer IPS 估计
    """
    x = _finite_clean(data)
    n = x.size
    if n < 4:
        return 1.0

    if max_lag is None:
        max_lag = min(n - 1, 4096)
    rho = _acf_fft(x, max_lag=max_lag, demean=True)

    if method == "threshold":
        tau = 0.5
        for t in range(1, rho.size):
            r = rho[t]
            if not np.isfinite(r) or r <= 0.05:
                break
            tau += float(r)
        return float(max(1.0, tau))

    # Sokal 自洽窗口法（推荐），失败则回退 IPS
    tau = 0.5
    sokal_ok = True
    for _ in range(50):
        W = int(min(rho.size - 1, max(1, math.floor(c * tau))))
        if W < 1:
            break
        r = rho[1 : W + 1]
        r = r[np.isfinite(r)]
        tau_new = 0.5 + (float(np.sum(r)) if r.size else 0.0)
        if not np.isfinite(tau_new) or tau_new <= 0.0 or tau_new > 1e12:
            sokal_ok = False
            break
        if math.isclose(tau, tau_new, rel_tol=1e-3, abs_tol=1e-6):
            tau = tau_new
            break
        tau = tau_new

    if (not sokal_ok) or (tau > (n * 0.5)) or (not np.isfinite(tau)):
        # 回退到 IPS（对重尾/强相关/短序列更稳健）
        tau = _tau_ips_from_rho(rho)

    return float(max(1.0, tau))


def effective_sample_size(data: np.ndarray, tau: Optional[float] = None) -> float:
    """
    有效样本数：Neff ≈ N / (2 τ_int)。空输入返回 0。
    """
    x = _finite_clean(data)
    N = x.size
    if N == 0:
        return 0.0
    if tau is None:
        tau = autocorrelation_time(x)
    tau = max(1.0, float(tau))
    return float(max(0.0, N / (2.0 * tau)))


def estimate_error_with_autocorr(data: np.ndarray, tau: Optional[float] = None) -> Tuple[float, float]:
    """
    均值的自相关修正标准误：
        err ≈ σ * sqrt(2 τ / N)
    返回 (err, tau)。空序列返回 (0.0, 1.0)。
    """
    x = _finite_clean(data)
    N = x.size
    if N == 0:
        return 0.0, 1.0
    if tau is None:
        tau = autocorrelation_time(x)
    sigma = float(np.std(x, ddof=1)) if N > 1 else 0.0
    tau = max(1.0, float(tau))
    err = sigma * math.sqrt(2.0 * tau / max(1, N))
    return float(err), float(tau)


# ---------------------------------------------------------------------
# 阻塞分析（Flyvbjerg–Petersen 风格）
# ---------------------------------------------------------------------
def blocking_analysis(
    data: np.ndarray,
    min_block_size: int = 1,
    platform_window: int = 3,
    slope_tol: float = 0.1,
    return_curve: bool = False,
) -> Tuple[float, Optional[Dict[str, np.ndarray]]]:
    """
    Level-doubling 阻塞分析。

    返回:
      err, curve(可选)
    其中 curve 含 'block_size','n_blocks','stderr' 三列。

    稳健性：
    - 若无法形成 >=2 个块，则退化为常规标准误 std(x)/sqrt(n)
    - 平台判定：最近 platform_window 个点的 log-log 斜率绝对值均值 < slope_tol 时，
      取这几段标准误的平均作为平台估计
    """
    x = _finite_clean(data)
    n = x.size
    if n == 0:
        curve = {"block_size": np.array([1], dtype=int),
                 "n_blocks": np.array([0], dtype=int),
                 "stderr": np.array([0.0], dtype=float)}
        return 0.0, (curve if return_curve else None)

    def _stderr_for_block(b: int) -> Tuple[float, int]:
        nb = n // b
        if nb < 2:
            # 退化为整体标准误
            if n <= 1:
                se = 0.0
            else:
                se = float(np.std(x, ddof=1) / math.sqrt(float(n)))
            return se, nb
        y = x[: nb * b].reshape(nb, b).mean(axis=1)
        se = float(np.std(y, ddof=1) / math.sqrt(float(max(1, nb))))
        return se, nb

    b = int(max(1, min_block_size))
    block_sizes: List[int] = []
    n_blocks_list: List[int] = []
    stderrs: List[float] = []

    while True:
        se, nb = _stderr_for_block(b)
        if nb < 2:
            break
        block_sizes.append(b)
        n_blocks_list.append(nb)
        stderrs.append(se)
        if (n // (b * 2)) < 2:
            break
        b *= 2

    if not block_sizes:
        # 无法阻塞：退化
        if n <= 1:
            err = 0.0
            nb_report = 0
        else:
            err = float(np.std(x, ddof=1) / math.sqrt(float(n)))
            nb_report = n
        curve = {"block_size": np.array([1], dtype=int),
                 "n_blocks": np.array([nb_report], dtype=int),
                 "stderr": np.array([err], dtype=float)}
        return err, (curve if return_curve else None)

    bs = np.asarray(block_sizes, dtype=int)
    nb_arr = np.asarray(n_blocks_list, dtype=int)
    se_arr = np.asarray(stderrs, dtype=float)

    # 平台检测
    err = float(se_arr[-1])
    if se_arr.size >= int(platform_window) + 1:
        lx = np.log(bs)
        ly = np.log(se_arr + 1e-30)
        slopes = np.diff(ly) / np.diff(lx)
        w = int(platform_window)
        mean_abs_slope = float(np.mean(np.abs(slopes[-w:])))
        if mean_abs_slope < slope_tol:
            err = float(np.mean(se_arr[-w:]))

    curve = {"block_size": bs, "n_blocks": nb_arr, "stderr": se_arr}
    return err, (curve if return_curve else None)


# ---------------------------------------------------------------------
# 阻塞长度启发式（与 MBB / Jackknife 一致）
# ---------------------------------------------------------------------
def estimate_block_len(data: np.ndarray, tau: Optional[float] = None, rule: str = "2tau") -> int:
    """
    选择阻塞长度的启发式：

    rule:
      - '2tau'（默认）：round(2 * τ_int)；并做如下稳健约束：
           * 至少 1，至多 N
           * 尝试保证块数不少于 8（若可能），否则适当减小 b
      - 'sqrt' : round(sqrt(N))
      - 'cubert': round(N ** (1/3))
    """
    x = _finite_clean(data)
    N = x.size
    if N == 0:
        return 1
    if rule == "sqrt":
        return int(max(1, round(math.sqrt(N))))
    if rule == "cubert":
        return int(max(1, round(N ** (1.0 / 3.0))))
    # 默认：基于 τ_int
    if tau is None:
        tau = autocorrelation_time(x)
    bl = int(max(1, round(2.0 * max(1.0, float(tau)))))
    bl = min(bl, N)  # 不超过 N
    # 尽量保证块数不少于 8（对误差更稳），若可能则收紧 b
    if N // bl < 8 and bl > 1:
        bl = max(1, min(bl, N // 8)) or 1
        bl = max(1, bl)
    return int(max(1, bl))


# ---------------------------------------------------------------------
# Jackknife / Bootstrap
# ---------------------------------------------------------------------
def jackknife_error(
    data: np.ndarray,
    func: Callable[[np.ndarray], float],
    block_len: Optional[int] = None,
    blocks: Optional[int] = None,
) -> Tuple[float, float]:
    """
    块 jackknife（相关数据）。
    返回 (theta_full, stderr_jk)。对极端短序列与退化块做降级处理。
    """
    x = _finite_clean(data)
    N = x.size
    if N == 0:
        return float("nan"), 0.0

    try:
        theta_full = float(func(x))
    except Exception:
        # 目标函数失效时返回 NaN + 0 误差（不中断上游）
        return float("nan"), 0.0

    # 块大小/块数的选择
    if blocks is not None and blocks > 1:
        B = int(min(blocks, N))
        b = max(1, N // B)
    else:
        if block_len is None:
            b = estimate_block_len(x)
        else:
            b = int(max(1, block_len))
        B = max(2, N // b)
        if B < 2:
            B = 2
            b = max(1, N // B)

    edges = np.linspace(0, N, B + 1, dtype=int)
    thetas = []
    for i in range(B):
        lo, hi = edges[i], edges[i + 1]
        if lo >= hi:
            continue
        # 留出第 i 块
        if lo == 0 and hi == N:
            xi = np.array([], dtype=float)
        else:
            xi = np.concatenate([x[:lo], x[hi:]]) if (lo > 0 or hi < N) else np.array([], dtype=float)
        if xi.size == 0:
            thetas.append(float("nan"))
        else:
            try:
                thetas.append(float(func(xi)))
            except Exception:
                thetas.append(float("nan"))

    thetas = np.asarray(thetas, dtype=float)
    m = np.isfinite(thetas)
    thetas = thetas[m]
    if thetas.size < 2:
        return theta_full, 0.0

    theta_bar = float(np.mean(thetas))
    var = float((thetas.size - 1) / thetas.size * np.sum((thetas - theta_bar) ** 2))
    stderr = math.sqrt(max(0.0, var))
    return theta_full, stderr


def bootstrap_error(
    data: np.ndarray,
    func: Callable[[np.ndarray], float],
    n_bootstrap: int = 1000,
    seed: Optional[int] = None,
) -> Tuple[float, float, Tuple[float, float]]:
    """
    IID bootstrap（近似独立样本）。
    返回 (theta_full, stderr, (ci_lo, ci_hi))；当采样失败或全 NaN 时给出退化区间。
    """
    x = _finite_clean(data)
    N = x.size
    if N == 0:
        return float("nan"), 0.0, (float("nan"), float("nan"))
    try:
        theta_full = float(func(x))
    except Exception:
        return float("nan"), 0.0, (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    boots = np.empty(int(n_bootstrap), dtype=float)
    for i in range(int(n_bootstrap)):
        idx = rng.integers(0, N, size=N, dtype=np.int64)
        try:
            boots[i] = float(func(x[idx]))
        except Exception:
            boots[i] = float("nan")
    boots = boots[np.isfinite(boots)]
    if boots.size < 2:
        return theta_full, 0.0, (theta_full, theta_full)
    stderr = float(np.std(boots, ddof=1))
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    return theta_full, stderr, ci


def moving_block_bootstrap_error(
    data: np.ndarray,
    func: Callable[[np.ndarray], float],
    block_len: Optional[int] = None,
    n_bootstrap: int = 1000,
    circular: bool = True,
    seed: Optional[int] = None,
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Moving-block bootstrap（相关序列）。
    返回 (theta_full, stderr, (ci_lo, ci_hi))。

    细节：
    - block_len 缺省时使用 estimate_block_len（目标：尽量 >=8 个块以稳定方差）
    - circular=True：块可越界并回绕；False：仅在 [0, N-b] 取起点
    - 极端/退化输入下保证不中断，返回退化区间
    """
    x = _finite_clean(data)
    N = x.size
    if N == 0:
        return float("nan"), 0.0, (float("nan"), float("nan"))

    try:
        theta_full = float(func(x))
    except Exception:
        return float("nan"), 0.0, (float("nan"), float("nan"))

    if block_len is None:
        block_len = estimate_block_len(x)
    b = int(max(1, min(block_len, N)))
    # 目标：尽量保证 B >= 8（如可能），以降低方差抖动
    B = int(max(1, math.ceil(N / b)))
    if B < 8 and b > 1:
        b = max(1, min(b, N // 8)) or 1
        B = int(max(1, math.ceil(N / b)))

    rng = np.random.default_rng(seed)
    boots = np.empty(int(n_bootstrap), dtype=float)

    if circular:
        for i in range(int(n_bootstrap)):
            starts = rng.integers(0, N, size=B, dtype=np.int64)
            parts = []
            for s in starts:
                if s + b <= N:
                    parts.append(x[s : s + b])
                else:
                    wrap = (s + b) - N
                    parts.append(np.concatenate([x[s:], x[:wrap]]))
            y = np.concatenate(parts, axis=0)[:N]
            try:
                boots[i] = float(func(y))
            except Exception:
                boots[i] = float("nan")
    else:
        max_start = max(0, N - b)
        for i in range(int(n_bootstrap)):
            starts = rng.integers(0, max_start + 1, size=B, dtype=np.int64)
            parts = [x[s : s + b] for s in starts]
            y = np.concatenate(parts, axis=0)[:N]
            try:
                boots[i] = float(func(y))
            except Exception:
                boots[i] = float("nan")

    boots = boots[np.isfinite(boots)]
    if boots.size < 2:
        # 仍然退化为点区间
        return theta_full, 0.0, (theta_full, theta_full)
    stderr = float(np.std(boots, ddof=1))
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    return theta_full, stderr, ci


# ---------------------------------------------------------------------
# 窗口平均（平滑辅助）
# ---------------------------------------------------------------------
def windowed_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    中心移动平均（same 卷积）。非有限值用有限值均值替换后再滤波。
    """
    x_raw = np.asarray(data, dtype=float).ravel()
    n = x_raw.size
    if n == 0:
        return x_raw
    xf = _finite_clean(x_raw)
    mean_f = float(np.mean(xf)) if xf.size else 0.0
    x = np.where(np.isfinite(x_raw), x_raw, mean_f)

    w = int(max(1, window_size))
    kernel = np.ones(w, dtype=float) / float(w)
    y = np.convolve(x, kernel, mode="same")
    return y


# ---------------------------------------------------------------------
# 轻量自检（不会抛异常）
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("statistics.py self-check...")

    # _finite_clean
    assert _finite_clean([1, 2, np.nan, 3]).size == 3

    # ACF 基本性质
    x = np.array([1.0, 0.0, -1.0, 0.0])
    rho = _acf_fft(x, max_lag=3)
    assert rho[0] == 1.0

    # τ_int：iid 噪声 ~ 1
    iid = np.random.default_rng(123).normal(size=1000)
    tau_iid = autocorrelation_time(iid)
    assert tau_iid >= 1.0

    # 阻塞分析：短序列与长序列
    short = np.ones(3)
    err_short, curve_short = blocking_analysis(short, return_curve=True)
    assert err_short >= 0.0

    long_series = np.random.default_rng(42).normal(size=1024)
    err_long, curve_long = blocking_analysis(long_series, return_curve=True)
    assert err_long >= 0.0

    # MBB 基本健壮性
    theta, se, ci = moving_block_bootstrap_error(
        long_series, func=lambda a: float(np.mean(a)), n_bootstrap=200, circular=True
    )
    assert np.isfinite(theta) and se >= 0.0

    print("statistics.py checks passed.")

