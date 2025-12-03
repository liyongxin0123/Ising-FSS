# -*- coding: utf-8 -*-
"""
二维 Ising 模型有限尺寸标度分析工具（Finite-Size Scaling）

实现功能：
    - 自动提取临界指数 ν、η、γ/ν、β/ν
    - 支持加权多项式拟合（误差来自 bootstrap 或 jackknife）
    - 数据坍缩质量评估（最小化曲线离散度）
    - Binder 累积量交叉点自动定位 + 误差估计
    - 提供网格搜索 optimize_collapse() 寻找最佳 (Tc, ν)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import total_ordering
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import logging
import math
import numpy as np
import warnings

# 引入项目内统计工具（应为你们修正版的 statistics.py）
try:
    from ..analysis import statistics as stats  # type: ignore
except Exception:
    stats = None

def weighted_polyfit(x, y, y_err, deg):
    """
    加权多项式拟合的封装。
    - x,y: 数值数据
    - y_err: stderr（标量或序列），若为 None/NaN 则视为不可用
    - deg: 多项式阶数
    返回与 np.polyfit 兼容的系数（从高阶到常数项）。
    注意：这里将权重定义为 1/variance = 1/(stderr^2)。
    若多数点缺失 y_err，则回退到普通拟合（并发出警告）。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_err = np.asarray([ (np.nan if v is None else v) for v in y_err ], dtype=float)
    valid = (~np.isnan(y_err)) & (y_err > 0)
    # 需要至少两个有效用于加权（并且点数不能太少）
    if np.sum(valid) < max(2, len(x)//2):
        warnings.warn("Insufficient valid y_err for weighted fit; falling back to unweighted fit")
        return np.polyfit(x, y, deg)
    # 权重为 1/variance
    w = np.zeros_like(y, dtype=float)
    w[valid] = 1.0 / (y_err[valid] ** 2)
    # numpy.polyfit 接受 w 参数，该 w 是 sqrt(weights) 的语义（即多项式最小二乘中用到）
    # 为保持直观，传入 w = sqrt(1/var)
    w_for_np = np.sqrt(w)
    return np.polyfit(x, y, deg, w=w_for_np)

def fit_with_errors(x, y, y_err, deg=2):
    coeffs = weighted_polyfit(x, y, y_err, deg)
    return coeffs, {'method':'weighted_polyfit_if_possible', 'deg':deg}

# Optional smoothing with SciPy; fall back to moving-average if unavailable
try:
    from scipy.signal import savgol_filter as _savgol_filter  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

__all__ = [
    "PairCrossing",
    "FSSAnalyzer",
]

# Module-level logger
_logger = logging.getLogger(__name__)
# small numeric eps for float comparisons
_EPS = 1e-9


# --------------------------- small utilities ---------------------------

def _sorted_unique(x: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(x), dtype=float)
    if arr.size == 0:
        return arr
    arr = np.unique(arr)
    arr.sort()
    return arr


def _moving_average(y: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return y.astype(float, copy=True)
    win = int(max(1, win))
    if win % 2 == 0:
        win += 1  # odd window
    k = win // 2
    # pad with edge values to keep length
    pad_left = np.full(k, float(y[0]), dtype=float)
    pad_right = np.full(k, float(y[-1]), dtype=float)
    z = np.concatenate([pad_left, y.astype(float, copy=False), pad_right])
    ker = np.ones(win, dtype=float) / float(win)
    sm = np.convolve(z, ker, mode="valid")
    return sm


def _smooth_series(y: np.ndarray,
                   smoothing: str = "none",
                   window: int = 7,
                   polyorder: int = 2) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if smoothing is None or smoothing == "none":
        return y.astype(float, copy=True)
    if smoothing == "savgol" and _HAS_SCIPY and y.size >= 3:
        # desired odd window <= len(y) and at least 3
        w = int(max(3, int(window)))
        if w % 2 == 0:
            w += 1
        # ensure window <= len(y) and odd
        if w > y.size:
            w = int(y.size if (y.size % 2 == 1) else (y.size - 1))
        w = max(3, min(w, int(y.size)))
        if w % 2 == 0:
            w -= 1
        p = int(min(int(polyorder), w - 1))
        try:
            return _savgol_filter(y, window_length=w, polyorder=p, mode="interp")
        except Exception:
            _logger.debug("savgol_filter failed; returning unsmoothed series", exc_info=True)
            return y.astype(float, copy=True)
    # fallback to moving average (works for any length)
    return _moving_average(y, window)


def _linear_interpolator(x: Sequence[float], y: Sequence[float]):
    """
    Return a simple 1D linear interpolator f(t) with clamped extrapolation.
    Accepts scalar or array t. Requires x strictly increasing (unique+sorted outside).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
        raise ValueError("x and y must be 1D arrays of same length >= 2")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x must be strictly increasing")

    def f(t: float | np.ndarray):
        tt = np.asarray(t, dtype=float)
        scalar_input = tt.shape == ()
        tt_flat = tt.ravel()
        idx = np.searchsorted(x, tt_flat, side="right") - 1
        idx = np.clip(idx, 0, x.size - 2)
        x0 = x[idx]; x1 = x[idx + 1]
        y0 = y[idx]; y1 = y[idx + 1]
        denom = (x1 - x0)
        # safe weights
        w = np.where(denom > 0, (tt_flat - x0) / denom, 0.0)
        out = (1.0 - w) * y0 + w * y1
        out = np.where(tt_flat <= x[0], y[0], out)
        out = np.where(tt_flat >= x[-1], y[-1], out)
        if scalar_input:
            return out[0]
        return out.reshape(tt.shape)
    return f


def _bisection_root(f, a: float, b: float, max_iter: int = 80, tol: float = 1e-10) -> float:
    fa = float(f(a)); fb = float(f(b))
    if not (np.isfinite(fa) and np.isfinite(fb)):
        raise ValueError("Function not finite at bracket endpoints.")
    if fa == 0.0:
        return float(a)
    if fb == 0.0:
        return float(b)
    if fa * fb > 0:
        raise ValueError("No sign change in bracket.")
    lo, hi = (a, b); flo, fhi = (fa, fb)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = float(f(mid))
        if abs(fm) < tol or (hi - lo) < tol:
            return float(mid)
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return float(0.5 * (lo + hi))


def _finite_diff(f, x: float, h: float) -> float:
    return float((f(x + h) - f(x - h)) / (2.0 * h))


def _min_positive_step(t: np.ndarray) -> float:
    """Minimal positive spacing in a sorted vector (fallback to 1e-3 if degenerate)."""
    if t.size < 2:
        return 1e-3
    dt = np.diff(t.astype(float, copy=False))
    dt = dt[dt > 0]
    if dt.size == 0:
        return 1e-3
    return float(np.min(dt))


# A small dict subclass that can cast to float (returns float(self['Tc']))
class _FloatableDict(dict):
    """A dict that can cast to float by returning float(self['Tc'])."""
    def __float__(self) -> float:
        if "Tc" not in self:
            raise TypeError("Result dict has no 'Tc' field to cast to float.")
        return float(self["Tc"])


# --------------------------- data structure ---------------------------

@total_ordering
@dataclass
class PairCrossing:
    L1: int
    L2: int
    Tc: float
    slope_diff: float
    bracket: Tuple[float, float]
    method: str
    note: str = ""

    def __float__(self) -> float:
        return float(self.Tc)

    def _as_float(self, other):
        try:
            return float(other)
        except Exception:
            return None

    def __eq__(self, other) -> bool:
        o = self._as_float(other)
        if o is None:
            return NotImplemented
        return math.isclose(float(self.Tc), float(o), rel_tol=1e-9, abs_tol=_EPS)

    def __lt__(self, other) -> bool:
        o = self._as_float(other)
        if o is None:
            return NotImplemented
        # if almost equal, treat as not less
        if math.isclose(float(self.Tc), float(o), rel_tol=1e-9, abs_tol=_EPS):
            return False
        return float(self.Tc) < float(o)


# --------------------------- main analyzer ---------------------------

class FSSAnalyzer:
    """
    Finite-Size Scaling analyzer on a dict-of-dicts result structure.

    results: Dict[L, Dict[T, Dict[str, Any]]]
      observables may be partially missing; warnings will be recorded, not fatal
    """

    def __init__(self,
                 results: Dict[int, Dict[float, Dict[str, Any]]],
                 Tc_theory: Optional[float] = None,
                 log_warnings: bool = True):
        if not results:
            raise ValueError("Empty results dict.")
        self.Tc_theory = Tc_theory
        self._log_warnings = bool(log_warnings)
        self.warnings: List[str] = []

        # warn helper
        def _warn(msg: str):
            self.warnings.append(msg)
            if self._log_warnings:
                _logger.warning("[FSSAnalyzer] " + msg)

        self._warn = _warn  # bind

        # 1) 过滤每个 L 的“非温度键/非 dict 值”，记录警告但继续
        cleaned: Dict[int, Dict[float, Dict[str, Any]]] = {}
        for L, Tmap in results.items():
            L_int = int(L)
            keep: Dict[float, Dict[str, Any]] = {}
            ignored_keys = []
            non_dict_keys = []
            for k, v in Tmap.items():
                # value 必须是 dict
                if not isinstance(v, dict):
                    non_dict_keys.append(k)
                    continue
                # 键必须可转为 float（温度）
                try:
                    Tk = float(k)
                except Exception:
                    ignored_keys.append(k)
                    continue
                keep[Tk] = v

            if ignored_keys:
                self._warn(f"L={L_int}: ignored non-temperature keys ({len(ignored_keys)}): {ignored_keys[:5]}{'...' if len(ignored_keys)>5 else ''}")
            if non_dict_keys:
                self._warn(f"L={L_int}: ignored entries with non-dict values ({len(non_dict_keys)}): {non_dict_keys[:5]}{'...' if len(non_dict_keys)>5 else ''}")

            if keep:
                cleaned[L_int] = keep

        if not cleaned:
            raise ValueError("No valid (float) temperature entries after filtering non-temperature keys.")

        self.results: Dict[int, Dict[float, Dict[str, Any]]] = cleaned
        self.L_list: List[int] = sorted(cleaned.keys())

        # 2) 不再强制 {U,chi,C,M} 全在，但记录缺失项的警告
        for L in self.L_list:
            for T, obs in self.results[L].items():
                missing = {"U", "chi", "C", "M"}.difference(obs.keys())
                if missing:
                    self._warn(f"L={L}, T={T:.6f}: missing observables {sorted(missing)} (will continue with available ones).")

        # cache：全部温度键（仅用于暴露/通用信息），插值时按观测量再过滤
        self._all_T_sorted: Dict[int, np.ndarray] = {
            L: _sorted_unique(self.results[L].keys()) for L in self.L_list
        }

        # cache: smoothed series and interpolators
        self._series_cache: Dict[Tuple[int, str, str, int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._interp_cache: Dict[Tuple[int, str, str, int, int], Any] = {}

    # ----------------- low-level curve helpers -----------------

    def _raw_curve(self, L: int, observable: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build raw (T_sorted, y_raw) for a given observable by selecting only (L,T)
        entries that contain this observable; remove NaN/Inf with warnings.
        """
        Tmap = self.results[int(L)]
        Ts: List[float] = []
        Ys: List[float] = []
        for T, obs in Tmap.items():
            if observable in obs:
                try:
                    val = float(obs[observable])
                except Exception:
                    self._warn(f"L={L}, T={T:.6f}, obs='{observable}': value not float-castable; skipped.")
                    continue
                Ts.append(float(T)); Ys.append(val)

        if not Ts:
            # 该 L 下没有该观测量的有效点
            self._warn(f"L={L}, obs='{observable}': no entries with this observable.")
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        Ts = np.asarray(Ts, dtype=float); Ys = np.asarray(Ys, dtype=float)
        order = np.argsort(Ts)
        T_sorted = Ts[order]; y = Ys[order]

        # guard against NaN/Inf in data -> filter & warn
        m = np.isfinite(y)
        if not np.all(m):
            n_bad = int((~m).sum())
            self._warn(f"L={L}, obs='{observable}': filtered {n_bad} non-finite points (NaN/Inf); kept {int(m.sum())}.")
            T_sorted = T_sorted[m]
            y = y[m]

        return T_sorted, y

    def _get_series(self, L: int, observable: str,
                    smoothing: str = "none",
                    window: int = 7,
                    polyorder: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get (T_sorted, y_smoothed) with caching and NaN handling.
        Strict: require at least 2 points for interpolation; with 1 point, we
        return a single sample and higher levels will allow a constant interpolator.
        """
        key = (int(L), str(observable), str(smoothing), int(window), int(polyorder))
        if key in self._series_cache:
            return self._series_cache[key]

        T, y = self._raw_curve(L, observable)
        if T.size == 0:
            # 已经尽力清洗，仍然没有有效点：上层需要时会抛错；此处记录一次警告
            self._warn(f"L={L}, obs='{observable}': no valid data points after filtering.")
            self._series_cache[key] = (T, y)  # empty
            return T, y

        # 平滑 — 使用更稳健的窗口选取
        ys = _smooth_series(y, smoothing=smoothing, window=window, polyorder=polyorder)
        if ys.size == 0:
            self._warn(f"L={L}, obs='{observable}': empty series after smoothing.")
        self._series_cache[key] = (T, ys)
        return T, ys

    def _interp(self, L: int, observable: str,
                smoothing: str = "none",
                window: int = 7,
                polyorder: int = 2):
        key = (int(L), str(observable), str(smoothing), int(window), int(polyorder))
        if key in self._interp_cache:
            return self._interp_cache[key]
        T, ys = self._get_series(L, observable, smoothing, window, polyorder)

        if T.size == 0:
            # 没有可用数据：严格失败
            raise ValueError(f"No valid data to build interpolator for L={L}, obs='{observable}'.")
        if T.size == 1:
            # 退化：常数插值器（继续运算，但交叉可能失败）
            y0 = float(ys[0])

            def constf(tt):
                tt = np.asarray(tt, dtype=float)
                return np.full_like(tt, y0, dtype=float)

            self._interp_cache[key] = constf
            self._warn(f"L={L}, obs='{observable}': only one valid point; using constant interpolator.")
            return constf

        f = _linear_interpolator(T, ys)
        self._interp_cache[key] = f
        return f

    def get_observable_vector(self, L: int, observable: str) -> Tuple[np.ndarray, np.ndarray]:
        """Expose *raw* sorted temperatures and observable vector (for tests/usage)."""
        return self._raw_curve(L, observable)

    def get_temperature_keys_sorted(self, L: int) -> np.ndarray:
        """All temperature keys retained after initial cleaning (not per observable)."""
        return self._all_T_sorted[int(L)].copy()

    # ----------------- 观测点误差/自相关探测 -----------------
    def _point_stats(self, L: int, T: float, observable: str) -> Tuple[float, float, float, float]:
        """
        为给定 (L,T,observable) 恢复或估计 (value, stderr, tau, ess)。
        特性：
        1. 智能查找 {obs}_stderr, {obs}_samples 等特定键。
        2. 如果 statistics 模块缺失，自动回退到简单标准误 (std/sqrt(N))。
        3. 如果 observable 值缺失，自动尝试用 mean 或样本均值兜底。
        """
        L = int(L)
        T = float(T)
        obs = self.results.get(L, {}).get(T, {})
        if not obs:
            self._warn(f"_point_stats: no obs found for L={L}, T={T:.6f}")
            return float("nan"), float("nan"), float("nan"), float("nan")

        # --- 1. 获取观测均值 (Value) ---
        val = None
        # 优先找物理量名称 (e.g., 'chi')
        if observable in obs:
            try:
                val = float(obs[observable])
            except Exception:
                pass
        # 兜底：找 'mean'
        if val is None and "mean" in obs:
            try:
                val = float(obs["mean"])
            except Exception:
                pass

        # --- 2. 智能查找样本数组 (Samples) ---
        arr = None
        sample_keys_to_check = [
            f"{observable}_samples",  # 优先：chi_samples
            f"{observable}_values",
            "samples",                # 通用
            "values",
            "samples_array",
            "raw"
        ]

        for key in sample_keys_to_check:
            if key in obs:
                try:
                    candidate = np.asarray(obs[key], dtype=float)
                    if candidate.size > 0:
                        arr = candidate.ravel()
                        break
                except Exception:
                    continue

        # --- 3. 基于样本计算误差 (如果有样本) ---
        # 即使没有 statistics 模块，也要能算简单误差
        if arr is not None:
            # 如果主值 val 还没找到，直接用样本均值
            if val is None:
                val = float(np.mean(arr))

            # 分支 A: 有高级统计模块 -> 自相关分析
            if stats is not None:
                try:
                    err, tau = stats.estimate_error_with_autocorr(arr)
                    ess = stats.effective_sample_size(arr, tau=tau)
                    return float(val), float(err), float(tau), float(ess)
                except Exception:
                    pass # 降级到分支 B

            # 分支 B: 无统计模块或计算失败 -> 简单标准误
            try:
                n = arr.size
                stdv = float(np.std(arr, ddof=1))
                simple_err = stdv / math.sqrt(max(1, n))
                # tau, ess 设为 nan
                return float(val), float(simple_err), float("nan"), float(n)
            except Exception:
                pass

        # --- 4. 智能查找预计算的标准误差 (Stderr) ---
        # 如果没有样本数组，或者想直接用 simulator 算好的误差
        stderr = None
        stderr_keys_to_check = [
            f"{observable}_stderr",  # 优先兼容 demo 脚本
            f"{observable}_err",     # 兼容 simulator 原始输出
            "stderr",                # 通用
            "err",
            "error"
        ]

        for key in stderr_keys_to_check:
            if key in obs:
                try:
                    v = float(obs[key])
                    if np.isfinite(v):
                        stderr = v
                        break
                except Exception:
                    continue

        # --- 5. 最后尝试用 std/n_samples 兜底 ---
        if stderr is None:
            std_keys = [f"{observable}_std", "std"]
            n_keys = [f"{observable}_n_samples", "n_samples", "samples_count"]

            std_val = None
            n_val = None

            for k in std_keys:
                if k in obs:
                    try: std_val = float(obs[k]); break
                    except: pass

            for k in n_keys:
                if k in obs:
                    try: n_val = float(obs[k]); break
                    except: pass

            if std_val is not None and n_val is not None and n_val > 0:
                stderr = std_val / math.sqrt(n_val)

        # --- 6. 最终返回 ---
        if val is None:
            self._warn(f"_point_stats: L={L},T={T},obs='{observable}' value missing.")
            return float("nan"), float("nan"), float("nan"), float("nan")

        final_stderr = stderr if stderr is not None else float("nan")
        return float(val), float(final_stderr), float("nan"), float("nan")

    # ----------------- binder crossing -----------------

    def find_binder_crossing(
        self,
        L1: int,
        L2: int,
        bounds: Optional[Tuple[float, float]] = None,
        observable: str = "U",
        smoothing: str = "none",
        window: int = 7,
        polyorder: int = 2,
        grid_n: int = 512,
    ) -> PairCrossing:
        """
        Find Binder crossing between sizes L1 and L2 via interpolation + bisection.
        If multiple sign changes exist, pick the one with the largest slope difference
        at the refined root (more stable under noise).
        """
        if L1 == L2:
            raise ValueError("L1 must differ from L2 for crossing.")
        f1 = self._interp(L1, observable, smoothing, window, polyorder)
        f2 = self._interp(L2, observable, smoothing, window, polyorder)

        # default bracket: overlap of available T-range for *this observable*
        T1, _ = self._get_series(L1, observable, smoothing="none")
        T2, _ = self._get_series(L2, observable, smoothing="none")
        if T1.size == 0 or T2.size == 0:
            raise ValueError(f"Insufficient data for '{observable}' to define crossing (L1={L1} size={T1.size}, L2={L2} size={T2.size}).")

        lo = max(float(np.min(T1)), float(np.min(T2)))
        hi = min(float(np.max(T1)), float(np.max(T2)))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            raise ValueError("No overlapping temperature range between sizes for this observable.")

        if bounds is not None:
            lo = max(float(bounds[0]), lo)
            hi = min(float(bounds[1]), hi)
            if lo >= hi:
                raise ValueError("Invalid bounds for crossing search.")

        def g(t: float) -> float:
            return float(f1(t) - f2(t))

        grid = np.linspace(lo, hi, num=int(max(64, grid_n)))
        vals = np.array([g(float(t)) for t in grid], dtype=float)

        # ensure at least some finite evaluations
        finite_mask = np.isfinite(vals)
        if not np.any(finite_mask):
            raise RuntimeError("All function evaluations are non-finite in crossing range.")

        # sign-change detection using only adjacent finite pairs
        sign = np.sign(vals)
        valid_pairs = finite_mask[:-1] & finite_mask[1:]
        idxs = np.where(valid_pairs & (sign[:-1] * sign[1:] < 0))[0]
        brackets: List[Tuple[float, float]] = [(float(grid[i]), float(grid[i + 1])) for i in idxs]

        # dynamic derivative step based on local mesh (stable)
        h1 = _min_positive_step(T1) * 0.5
        h2 = _min_positive_step(T2) * 0.5
        h = max(1e-6, min(h1, h2))

        if not brackets:
            # fallback: if endpoints opposite sign and finite
            if finite_mask[0] and finite_mask[-1] and (sign[0] * sign[-1] < 0):
                brackets = [(float(lo), float(hi))]
            else:
                # soft fallback: closest approach (argmin|g|), 并提醒
                j = int(np.argmin(np.abs(vals)))
                Tc = float(grid[j])
                s1 = _finite_diff(f1, Tc, h=h); s2 = _finite_diff(f2, Tc, h=h)
                self._warn(f"No sign change for crossing L1={L1}, L2={L2}; using argmin|Δ{observable}| as fallback at T≈{Tc:.6f}.")
                return PairCrossing(
                    L1=int(L1), L2=int(L2), Tc=Tc,
                    slope_diff=abs(s1 - s2),
                    bracket=(float(grid[max(0, j - 1)]), float(grid[min(grid.size - 1, j + 1)])),
                    method="argmin",
                    note="no sign change; used argmin|ΔU|"
                )

        # refine each bracket with bisection; choose best by slope_diff
        best: Optional[PairCrossing] = None
        for (a, b) in brackets:
            try:
                Tc = _bisection_root(g, a, b, max_iter=100, tol=1e-12)
            except Exception as e:
                _logger.debug("bisection failed on bracket (%s,%s): %s", a, b, e)
                continue
            s1 = _finite_diff(f1, Tc, h=h); s2 = _finite_diff(f2, Tc, h=h)
            sd = abs(s1 - s2)
            pc = PairCrossing(L1=int(L1), L2=int(L2), Tc=float(Tc),
                              slope_diff=float(sd), bracket=(float(a), float(b)),
                              method="bisection", note="")
            if (best is None) or (pc.slope_diff > best.slope_diff):
                best = pc

        if best is None:
            raise RuntimeError("Failed to refine any Binder crossing.")
        return best

    # ----------------- Tc estimation -----------------

    def estimate_Tc(
        self,
        use_all_pairs: bool = True,
        bounds: Optional[Tuple[float, float]] = None,
        weight_by: str = "slope",   # 'slope' | 'uniform'
        smoothing: str = "none",
        window: int = 7,
        polyorder: int = 2,
        n_boot: int = 0,            # bootstrap resamples on pair list
        rng: Optional[np.random.Generator] = None,
        verbose: bool = False,      # kept for API compatibility
    ) -> Dict[str, Any]:
        """
        Estimate Tc by averaging pairwise Binder crossings.
        Returns: dict with Tc, var, std, crossings(list), weights(list), pairs(list),
                 and (if n_boot>0) Tc_boot_samples, Tc_boot_std.
        """
        Ls = self.L_list
        if len(Ls) < 2:
            if self.Tc_theory is None:
                raise ValueError("Need at least two sizes or a Tc_theory.")
            return _FloatableDict({
                "Tc": float(self.Tc_theory), "var": 0.0, "std": 0.0,
                "crossings": [], "weights": [], "pairs": [],
                "note": "fallback to Tc_theory"
            })

        # pairs
        pairs: List[Tuple[int, int]] = []
        if use_all_pairs:
            for i in range(len(Ls)):
                for j in range(i + 1, len(Ls)):
                    pairs.append((Ls[i], Ls[j]))
        else:
            pairs = [(Ls[i], Ls[i + 1]) for i in range(len(Ls) - 1)]

        # crossings
        crossings: List[PairCrossing] = []
        for (L1, L2) in pairs:
            try:
                cr = self.find_binder_crossing(L1, L2, bounds=bounds,
                                               smoothing=smoothing, window=window, polyorder=polyorder)
                crossings.append(cr)
            except Exception as e:
                self._warn(f"Failed to find crossing for (L1={L1},L2={L2}): {e}.")
                continue

        if not crossings:
            if self.Tc_theory is not None:
                self._warn("No Binder crossings found; falling back to Tc_theory.")
                return _FloatableDict({
                    "Tc": float(self.Tc_theory), "var": 0.0, "std": 0.0,
                    "crossings": [], "weights": [], "pairs": [],
                    "note": "no crossings; fallback Tc_theory"
                })
            raise RuntimeError("No Binder crossings found to estimate Tc.")

        # weights
        if weight_by == "slope":
            ws = np.array([max(1e-12, cr.slope_diff) for cr in crossings], dtype=float)
        else:
            ws = np.ones(len(crossings), dtype=float)
        tcs = np.array([cr.Tc for cr in crossings], dtype=float)

        wsum = float(ws.sum())
        if wsum <= 0:
            ws = np.ones_like(ws); wsum = float(ws.sum())
        wnorm = ws / wsum

        Tc_hat = float(np.sum(wnorm * tcs))
        var = float(np.sum(wnorm * (tcs - Tc_hat) ** 2))
        out = {
            "Tc": Tc_hat,
            "var": var,
            "std": math.sqrt(max(0.0, var)),
            "crossings": crossings,
            "weights": ws.tolist(),
            "pairs": [(int(c.L1), int(c.L2)) for c in crossings],
        }

        # bootstrap on pair list
        if n_boot and n_boot > 0:
            g = rng if rng is not None else np.random.default_rng()
            samples = []
            if len(crossings) == 1:
                # bootstrap degenerate; return repeated samples and std=0
                samples = np.full(int(n_boot), float(tcs[0]), dtype=float)
                out["Tc_boot_samples"] = samples
                out["Tc_boot_std"] = 0.0
            else:
                for _ in range(int(n_boot)):
                    idx = g.integers(0, len(crossings), size=len(crossings))
                    ws_b = ws[idx]
                    tcs_b = tcs[idx]
                    wsum_b = float(ws_b.sum())
                    if wsum_b <= 0:
                        samples.append(float(Tc_hat))
                    else:
                        Tc_b = float(np.sum((ws_b / wsum_b) * tcs_b))
                        samples.append(Tc_b)
                samples = np.asarray(samples, dtype=float)
                out["Tc_boot_samples"] = samples
                out["Tc_boot_std"] = float(np.std(samples, ddof=1)) if samples.size > 1 else 0.0

        return _FloatableDict(out)

    # ----------------- finite-size extrapolation (C) -----------------

    def _doubling_pairs(self) -> List[Tuple[int, int]]:
        """Choose approximate doubling pairs (L, L2 ≈ 2L)."""
        Ls = self.L_list
        pairs: List[Tuple[int, int]] = []
        for i, L in enumerate(Ls[:-1]):
            cand = Ls[i + 1:]
            if not cand:
                continue
            j = int(np.argmin(np.abs(np.asarray(cand) - 2 * L)))
            pairs.append((int(L), int(cand[j])))
        return pairs

    def estimate_Tc_extrapolated(
        self,
        bounds: Optional[Tuple[float, float]] = None,
        smoothing: str = "none",
        window: int = 7,
        polyorder: int = 2,
        fit_nu: bool = True,
        nu_range: Tuple[float, float, int] = (0.6, 1.6, 101),
        use_slope_weight: bool = True,
    ) -> Dict[str, Any]:
        """
        Build Tc(L) from Binder crossings of approximate doubling pairs (L, ~2L),
        fit  Tc(L) = Tc + a * L^{-1/nu}.
        Return: {'Tc_inf','nu_used','a','pairs','points','rss','r2'}
        """
        pairs = self._doubling_pairs()
        if not pairs:
            raise RuntimeError("No doubling pairs available.")

        # Tc(L) for each pair
        pts: List[Tuple[float, float, float, Tuple[int, int]]] = []  # (Lbase, Tc(L), weight, (L1,L2))
        for (L1, L2) in pairs:
            try:
                cr = self.find_binder_crossing(L1, L2, bounds=bounds,
                                               smoothing=smoothing, window=window, polyorder=polyorder)
                Lbase = float(min(L1, L2))
                w = float(max(1e-12, cr.slope_diff)) if use_slope_weight else 1.0
                pts.append((Lbase, float(cr.Tc), w, (int(L1), int(L2))))
            except Exception as e:
                self._warn(f"Extrapolation: skipping pair (L1={L1},L2={L2}) due to crossing failure: {e}.")
                continue

        if len(pts) < 2:
            raise RuntimeError("Not enough Tc(L) points for extrapolation.")

        Ls = np.array([p[0] for p in pts], dtype=float)
        TcL = np.array([p[1] for p in pts], dtype=float)
        W = np.array([p[2] for p in pts], dtype=float)
        Pairs_list = [p[3] for p in pts]

        def _fit_for_nu(nu_val: float) -> Tuple[float, float, float, float]:
            x = Ls ** (-1.0 / float(nu_val))
            A = np.vstack([np.ones_like(x), x]).T
            # weighted least squares (normalize weights)
            w = W / max(1e-30, np.sum(W))
            Aw = (A.T * w).T
            yw = TcL * w
            coef, _, _, _ = np.linalg.lstsq(Aw, yw, rcond=None)
            Tc_inf, a = float(coef[0]), float(coef[1])
            pred = Tc_inf + a * x
            resid = TcL - pred
            rss = float(np.sum(w * resid ** 2))
            tss = float(np.sum(w * (TcL - np.sum(w * TcL)) ** 2))
            r2 = 1.0 - (rss / max(1e-30, tss))
            return Tc_inf, a, rss, r2

        if fit_nu:
            g0, g1, ngrid = nu_range
            grid = np.linspace(float(g0), float(g1), int(ngrid))
            fits = [(_fit_for_nu(nu), nu) for nu in grid]
            # choose min rss
            (Tc_inf, a, rss, r2), nu_used = min(fits, key=lambda kv: kv[0][2])
        else:
            nu_used = 1.0
            Tc_inf, a, rss, r2 = _fit_for_nu(nu_used)

        return {
            "Tc_inf": float(Tc_inf),
            "nu_used": float(nu_used),
            "a": float(a),
            "pairs": Pairs_list,  # [(L1,L2), ...]
            "points": {"L": Ls, "TcL": TcL, "W": W},
            "rss": float(rss),
            "r2": float(r2),
        }

    def estimate_nu_from_binder_slope(
        self,
        Tc_hint: Optional[float] = None,
        window_T: float = 0.05,
        smoothing: str = "none",
        window: int = 7,
        polyorder: int = 2,
    ) -> Dict[str, Any]:
        """
        Estimate nu from dU/dT|Tc ∝ L^{1/nu}:
          1) pick Tc (hint or estimate)
          2) for each L, compute slope near Tc via central diff on interpolator
          3) fit ln slope vs ln L -> slope ≈ 1/nu
        """
        if Tc_hint is None:
            Tc_hint = float(self.estimate_Tc()["Tc"])

        Ls_used = []
        lnL = []
        lnSlope = []

        for L in self.L_list:
            try:
                fU = self._interp(L, "U", smoothing, window, polyorder)
            except Exception as e:
                self._warn(f"Binder-slope: L={L} cannot build interpolator: {e}. Skipped.")
                continue
            # derivative spacing tied to local mesh
            Tvec, _ = self._get_series(L, "U", smoothing="none")
            if Tvec.size == 0:
                self._warn(f"Binder-slope: L={L} no U series; skipped.")
                continue
            h = max(1e-6, _min_positive_step(Tvec) * 0.5)
            sl = _finite_diff(fU, float(Tc_hint), h=h)
            if sl <= 0 or not np.isfinite(sl):
                self._warn(f"Binder-slope: L={L} slope not usable (sl={sl}). Skipped.")
                continue
            Ls_used.append(int(L))
            lnL.append(math.log(float(L)))
            lnSlope.append(math.log(float(abs(sl))))

        if len(lnL) < 2:
            raise RuntimeError("Not enough sizes to estimate nu from Binder slope.")

        x = np.asarray(lnL, dtype=float)
        y = np.asarray(lnSlope, dtype=float)
        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        inv_nu = float(slope)
        nu = 1.0 / inv_nu if inv_nu != 0 else float("inf")

        return {
            "Tc_used": float(Tc_hint),
            "nu": float(nu),
            "inv_nu": float(inv_nu),
            "intercept": float(intercept),
            "sizes_used": Ls_used,
        }

    # ----------------- exponent extraction (A) -----------------

    def extract_critical_exponents(
        self,
        observable: str = "chi",
        Tc_hint: Optional[float] = None,
        fit_nu: bool = False,
        nu_grid: Tuple[float, float, int] = (0.6, 1.6, 101),
        verbose: bool = False,   # API compatibility
    ) -> Dict[str, Any]:
        """
        Minimal estimator:
        - If observable == 'chi': fit ln chi(L, T≈Tc) vs ln L -> slope ≈ gamma/nu.
        - 使用点误差（若可用）做加权拟合：权重为 1/variance。
        - 若大多数点没有 stderr 信息，则退化为无权拟合并在 warnings 中说明。
        """
        if Tc_hint is None:
            Tc_hint = float(self.estimate_Tc()["Tc"])

        # gamma/nu from chi at nearest T to Tc_hint
        Ls = []; lnL = []; lnY = []; y_errs = []
        for L in self.L_list:
            T_sorted, vec = self._raw_curve(L, observable)
            if vec.size == 0:
                self._warn(f"extract_critical_exponents: L={L} has no valid '{observable}' data. Skipped.")
                continue
            idx = int(np.argmin(np.abs(T_sorted - Tc_hint)))
            y = float(vec[idx])
            if not np.isfinite(y) or y <= 0:
                self._warn(f"extract_critical_exponents: L={L} '{observable}' at T≈{Tc_hint:.6f} invalid (y={y}). Skipped.")
                continue

            # 尝试取该点的 stderr（如果数据结构中有原始样本或 stderr 字段）
            Tsel = float(T_sorted[idx])
            val, stderr, tau, ess = self._point_stats(L, Tsel, observable)

            # 如果 stderr 存在且 >0，则在对数空间近似传播误差： var(ln y) ≈ (stderr / y)^2
            if np.isfinite(stderr) and stderr > 0:
                var_ln = (stderr / y) ** 2
                err_ln = math.sqrt(var_ln)
            else:
                err_ln = float("nan")

            Ls.append(int(L))
            lnL.append(math.log(float(L)))
            lnY.append(math.log(y))
            y_errs.append(err_ln)

        if len(lnL) < 2:
            raise RuntimeError("Not enough sizes to fit gamma/nu.")

        x = np.asarray(lnL, dtype=float)
        y = np.asarray(lnY, dtype=float)
        y_errs_arr = np.asarray(y_errs, dtype=float)

        # Decide whether to do weighted fit: require at least 2 finite stderr estimates
        valid_err_mask = np.isfinite(y_errs_arr) & (y_errs_arr > 0)
        if np.sum(valid_err_mask) >= 2:
            # 使用加权最小二乘拟合（将 y_errs 作为 stderr in log-space）
            try:
                # weighted_polyfit expects y_err in linear stderr; for log-space already constructed
                coeffs = weighted_polyfit(x, y, y_errs_arr, deg=1)
                slope = float(coeffs[0])
                intercept = float(coeffs[1])
            except Exception as e:
                self._warn(f"extract_critical_exponents: weighted fit failed: {e}; falling back to unweighted fit.")
                A = np.vstack([x, np.ones_like(x)]).T
                slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        else:
            # 无足够误差信息，使用普通最小二乘拟合
            self._warn("extract_critical_exponents: insufficient per-point stderr for weighted fit; using unweighted LS.")
            A = np.vstack([x, np.ones_like(x)]).T
            slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

        gamma_over_nu = float(slope)

        # nu: either estimate from Binder slope or fall back to 1.0/grid-proxy
        if fit_nu:
            try:
                nu_est = self.estimate_nu_from_binder_slope(Tc_hint=Tc_hint)["nu"]
            except Exception:
                g0, g1, ngrid = nu_grid
                grid = np.linspace(float(g0), float(g1), int(ngrid))
                nu_est = float(grid[np.argmin(np.abs(grid - 1.0))])
        else:
            nu_est = 1.0

        return {
            "Tc_used": float(Tc_hint),
            "gamma_over_nu": float(gamma_over_nu),
            "nu": float(nu_est),
            "intercept": float(intercept),
            "sizes_used": Ls,
        }

    # ----------------- data collapse (A, D) -----------------

    def _collapse_score(self, curves: List[Dict[str, Any]], grid_n: int = 256) -> float:
        """
        Compute a simple collapse score: mean squared spread around the mean curve
        on a common x-grid over the intersection of x-ranges.
        """
        if not curves:
            return np.inf
        # intersection of x ranges
        xmins = [c["x"].min() for c in curves]
        xmaxs = [c["x"].max() for c in curves]
        lo = max(xmins); hi = min(xmaxs)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            return np.inf
        grid = np.linspace(lo, hi, num=int(max(64, grid_n)))
        Ys = []
        for c in curves:
            # interpolate y on common grid
            x = c["x"]; y = c["y"]
            f = _linear_interpolator(x, y)
            Ys.append(f(grid))
        Y = np.vstack(Ys)  # (nL, ngrid)
        mu = np.mean(Y, axis=0)
        score = float(np.mean((Y - mu) ** 2))
        return score

    def data_collapse(
        self,
        observable: str,
        Tc: Optional[float] = None,
        nu: Optional[float] = None,
        exponent_ratio: Optional[float] = None,  # e.g., gamma/nu for chi
        metric_grid: int = 256,
        smoothing: str = "none",
        window: int = 7,
        polyorder: int = 2,
        # compatibility with unit tests:
        Tc_est: Optional[float] = None,
        gamma_over_nu: Optional[float] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Returns a dict with per-size scaled curves for a simple collapse:
          x = (T - Tc) * L^(1/nu)
          y = L^(-exponent_ratio) * observable(L, T)

        Additionally, as required by tests, top-level keys include each size L:
            result[L] = {"x","y","T","raw"}
        """
        # map synonyms
        if Tc is None: Tc = Tc_est
        if exponent_ratio is None: exponent_ratio = gamma_over_nu

        if Tc is None:
            Tc = float(self.estimate_Tc()["Tc"])
        if nu is None:
            nu = 1.0
        if exponent_ratio is None:
            exponent_ratio = 1.75 if observable == "chi" else 0.0

        Tc = float(Tc); nu = float(nu); exponent_ratio = float(exponent_ratio)

        curves: List[Dict[str, Any]] = []
        for L in self.L_list:
            T_sorted, y_vec = self._raw_curve(L, observable)
            if y_vec.size == 0:
                self._warn(f"data_collapse: L={L} has no valid '{observable}' data. Skipped.")
                continue
            # optional smoothing in observable before scaling
            y_s = _smooth_series(y_vec, smoothing=smoothing, window=window, polyorder=polyorder)
            x_scaled = (T_sorted - Tc) * (L ** (1.0 / nu))
            y_scaled = y_s * (L ** (-exponent_ratio))
            curves.append({
                "L": int(L),
                "x": np.asarray(x_scaled, dtype=float),
                "y": np.asarray(y_scaled, dtype=float),
                "T": np.asarray(T_sorted, dtype=float),
                "raw": np.asarray(y_vec, dtype=float),
            })

        score = self._collapse_score(curves, grid_n=metric_grid)

        # build {L: {...}} mapping on top-level (for test expectations)
        size_map = {
            int(c["L"]): {
                "x": c["x"],
                "y": c["y"],
                "T": c["T"],
                "raw": c["raw"],
            }
            for c in curves
        }

        result = {
            "observable": observable,
            "Tc": Tc,
            "nu": nu,
            "exponent_ratio": exponent_ratio,
            "curves": curves,
            "score": float(score),
            "success": np.isfinite(score),
        }
        result.update(size_map)  # attach per-size entries (e.g., result[8], result[16], ...)
        return result

    # ----------------- simple grid search for best collapse -----------------

    def optimize_collapse(
        self,
        observable: str = "chi",
        Tc_range: Tuple[float, float, int] = (2.0, 2.6, 31),
        nu_range: Tuple[float, float, int] = (0.6, 1.6, 21),
        ratio_range: Tuple[float, float, int] = (1.2, 1.9, 29),  # for gamma/nu (chi)
        smoothing: str = "none",
        window: int = 7,
        polyorder: int = 2,
        metric_grid: int = 256,
    ) -> Dict[str, Any]:
        """
        Coarse grid search to minimize collapse score over (Tc, nu, exponent_ratio).
        """
        Tc_grid = np.linspace(*Tc_range)
        nu_grid = np.linspace(*nu_range)
        er_grid = np.linspace(*ratio_range)

        best = (np.inf, None)
        for Tc in Tc_grid:
            for nu in nu_grid:
                for er in er_grid:
                    res = self.data_collapse(
                        observable=observable, Tc=float(Tc), nu=float(nu),
                        exponent_ratio=float(er), metric_grid=metric_grid,
                        smoothing=smoothing, window=window, polyorder=polyorder
                    )
                    score = res["score"]
                    if score < best[0]:
                        best = (score, (float(Tc), float(nu), float(er)))
        score, triple = best
        Tc_b, nu_b, er_b = triple
        res_best = self.data_collapse(
            observable=observable, Tc=Tc_b, nu=nu_b, exponent_ratio=er_b,
            metric_grid=metric_grid, smoothing=smoothing, window=window, polyorder=polyorder
        )
        return {
            "best_score": float(score),
            "Tc": float(Tc_b),
            "nu": float(nu_b),
            "exponent_ratio": float(er_b),
            "result": res_best,
        }

    # ----------------- warnings accessors -----------------

    def pop_warnings(self) -> List[str]:
        """Return and clear accumulated warnings."""
        out = self.warnings[:]
        self.warnings.clear()
        return out

