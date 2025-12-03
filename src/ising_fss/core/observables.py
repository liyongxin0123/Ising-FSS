# -*- coding: utf-8 -*-
"""
    二维 Ising 模型物理观测量精确计算模块

本模块提供了针对二维 Ising 模型构型的物理量计算函数。

核心特性：
    - 精度控制：所有累积量（能量、磁化）在计算过程中强制使用 `int64` 防止溢出，最终结果统一转换为 `numpy.float64` 返回。
    - 后端适配：自动检测并支持 NumPy (CPU) 和 CuPy/JAX (GPU) 后端，支持零拷贝数据传输。
    - 批量处理：提供 `calculate_observables_batch` 接口，支持 (R, L, L) 形状的并行计算。

返回字段（calculate_observables）：
    E_total     : float64，总能量
    M_total     : int64，总磁化
    E_per_spin  : float64，每 spin 能量
    m           : float64，平均磁化 m = M/N
    abs_m       : float64，|m|
    C           : float64，比热（需传入温度）
    chi         : float64，磁化率（需传入温度）
"""

from __future__ import annotations

from typing import Any, Tuple, Dict, Optional, List, Union
import numpy as np

# optional GPU/JAX backends
try:
    import cupy as cp
    HAS_CUPY = True
except Exception:
    cp = None  # type: ignore
    HAS_CUPY = False

try:
    import jax.numpy as jnp  # type: ignore
    HAS_JAX = True
except Exception:
    jnp = None  # type: ignore
    HAS_JAX = False

__all__ = [
    "calculate_observables",
    "calculate_observables_batch",
    "calculate_binder_cumulant",
    "calculate_specific_heat_per_spin",
    "calculate_susceptibility_per_spin",
    "_energy_total_numpy",
    "_observables_for_simulator",
]


# ---------------------------------------------------------------------
# 基础：总能量（NumPy 实现，单样本）
# ---------------------------------------------------------------------
def _energy_total_numpy(latt: Any, h: float = 0.0) -> np.float64:
    """
    计算单个构型 (L,L) 的总能量（NumPy实现），包含外场项。
    返回 np.float64（总能量，不是每自旋能量）。
    """
    a = np.asarray(latt)
    if a.size == 0:
        raise ValueError("lattice must be non-empty")
    # 使用宽整数以避免 int8 溢出
    ai = a.astype(np.int64, copy=False)
    # right: shift columns, down: shift rows（只计算每条键一次）
    right = np.roll(ai, -1, axis=1)
    down = np.roll(ai, -1, axis=0)
    # bond energy: - sum_{<ij>} s_i s_j  with each bond counted once by using right+down
    e_bond = -np.sum(ai * (right + down), dtype=np.int64)
    m_tot = np.sum(ai, dtype=np.int64)
    e_tot = float(e_bond) - float(h) * float(m_tot)
    return np.float64(e_tot)


# ---------------------------------------------------------------------
# CuPy 后端（单样本或批量）
# ---------------------------------------------------------------------
def _energy_total_cupy(latt: Any, h: float = 0.0) -> Union[np.float64, np.ndarray]:
    """
    使用 CuPy 计算总能量。
    支持单样本 (L,L) -> 返回 np.float64
            批量   (R,L,L) -> 返回 np.ndarray(dtype=np.float64, shape=(R,))
    如果传入已经是 cupy.ndarray，则不会做隐式 host->device 拷贝（只在必要时 .get()/asnumpy）。
    """
    if not HAS_CUPY:
        raise RuntimeError("CuPy is not available")
    x = cp.asarray(latt, dtype=cp.int64)
    if x.ndim == 2:
        right = cp.roll(x, -1, axis=1)
        down = cp.roll(x, -1, axis=0)
        e_bond = -cp.sum(x * (right + down))
        m_tot = cp.sum(x)
        # 将 cupy scalar 转为 numpy.float64
        val = e_bond - float(h) * m_tot
        # val 可能是 cupy scalar，使用 get() 或 asnumpy
        try:
            return np.float64(val.get())
        except Exception:
            return np.float64(cp.asnumpy(val))
    elif x.ndim == 3:
        # batch: shape (R, L, L) ; roll 的轴为最后两个轴 (-1, -2) -> 等价于 axis=(1,2)
        right = cp.roll(x, -1, axis=2)
        down = cp.roll(x, -1, axis=1)
        e_bond = -cp.sum(x * (right + down), axis=(1, 2))  # cupy array shape (R,)
        m_tot = cp.sum(x, axis=(1, 2))                     # cupy array shape (R,)
        res = cp.asnumpy(e_bond - float(h) * m_tot).astype(np.float64)
        return res
    else:
        raise ValueError(f"CuPy lattice must be 2D or 3D, got {x.ndim}D")


# ---------------------------------------------------------------------
# JAX 后端（单/批量），结果转回 numpy
# ---------------------------------------------------------------------
def _energy_total_jax(latt: Any, h: float = 0.0) -> Union[np.float64, np.ndarray]:
    """
    使用 JAX 实现能量计算（若可用），结果转回 numpy。
    支持单样本 (L,L) -> np.float64 或 批量 (R,L,L) -> np.ndarray(dtype=np.float64).
    """
    if not HAS_JAX:
        raise RuntimeError("JAX is not available")
    x = jnp.asarray(latt)
    if x.ndim == 2:
        right = jnp.roll(x, -1, axis=1)
        down = jnp.roll(x, -1, axis=0)
        e_bond = -jnp.sum(x * (right + down))
        m_tot = jnp.sum(x)
        val = e_bond - float(h) * m_tot
        return np.float64(np.asarray(val))
    elif x.ndim == 3:
        right = jnp.roll(x, -1, axis=2)
        down = jnp.roll(x, -1, axis=1)
        e_bond = -jnp.sum(x * (right + down), axis=(1, 2))
        m_tot = jnp.sum(x, axis=(1, 2))
        return np.asarray(e_bond - float(h) * m_tot, dtype=np.float64)
    else:
        raise ValueError("JAX lattice must be 2D or 3D")


# ---------------------------------------------------------------------
# 单样本观测量（返回标量 np.float64）
# ---------------------------------------------------------------------
def calculate_observables(lattice: Any, h: float = 0.0, prefer_gpu: bool = False) -> Dict[str, np.float64]:
    """
    计算单个构型的能量和磁化等观测量，返回值均为 np.float64。
    返回包含字段:
      E_total, E_per_spin, M_total, m (M/N), abs_m (|M|/N), M2 (m^2), M4 (m^4)
    prefer_gpu: 若 True，优先使用 GPU 后端（CuPy 或 JAX）当可用。
    """
    # 路径选择：优先 CuPy（传入为 cupy.ndarray 且 prefer_gpu），其次 JAX（若 prefer_gpu）
    if prefer_gpu and HAS_CUPY and isinstance(lattice, cp.ndarray):
        # 在 GPU 上计算能量（返回 np.float64），对磁化量尽量只做一次 device -> host
        E_tot = _energy_total_cupy(lattice, h=h)  # np.float64
        # M_total: 先在 device 上求和再取回
        try:
            M_tot = np.float64(int(lattice.sum().get()))
        except Exception:
            M_tot = np.float64(cp.asnumpy(cp.sum(lattice)).item())
        N = int(lattice.size)
    elif prefer_gpu and HAS_JAX and (isinstance(lattice, jnp.ndarray) if HAS_JAX else False):
        E_tot = _energy_total_jax(lattice, h=h)
        # M_total: 转一次回 host
        M_tot = np.float64(np.asarray(jnp.sum(lattice)).item())
        N = int(lattice.size)
    else:
        # CPU 路径（NumPy）
        E_tot = _energy_total_numpy(lattice, h=h)
        M_tot = np.float64(np.sum(np.asarray(lattice).astype(np.int64)))
        N = int(np.asarray(lattice).size)

    if N == 0:
        raise ValueError("lattice must be non-empty")

    # 强制转换并返回标准化标量
    E_total64 = np.float64(E_tot)
    E_per_spin64 = np.float64(E_total64 / np.float64(N))
    M_total64 = np.float64(M_tot)
    m = np.float64(float(M_total64) / float(N))
    abs_m = np.float64(abs(float(M_total64)) / float(N))  # |M|/N
    m2 = np.float64(m * m)
    m4 = np.float64(m2 * m2)

    return {
        "E_total": E_total64,
        "E_per_spin": E_per_spin64,
        "M_total": M_total64,
        "m": m,
        "abs_m": abs_m,
        "M2": m2,
        "M4": m4,
    }


# ---------------------------------------------------------------------
# 批量计算：输入 (R, L, L) -> 返回 (E_array, M_array) dtype=np.float64
# ---------------------------------------------------------------------
def _normalize_h_for_batch(h: Any, R: int, xp) -> Any:
    """
    Ensure h becomes an array of length R in the provided array module xp (np or cp).
    xp should implement full/asarray semantics like numpy/cupy.
    """
    if np.isscalar(h):
        return xp.full((R,), float(h), dtype=xp.float64)
    h_arr = xp.asarray(h, dtype=xp.float64)
    if h_arr.ndim == 0:
        return xp.full((R,), float(h_arr), dtype=xp.float64)
    if h_arr.shape != (R,):
        raise ValueError(f"h must be scalar or shape (R,), got {h_arr.shape} with R={R}")
    return h_arr


def calculate_observables_batch(lattices: Any, h: Any = 0.0, prefer_gpu: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    批量计算能量与磁化（高性能实现）。
    Input:
      lattices: array-like, shape (R, L, L), entries in {-1,+1}
      h: scalar or length-R array-like
      prefer_gpu: if True and CuPy available, try to use GPU
    Returns:
      (E_array, M_array) both numpy arrays dtype=np.float64, shape (R,)
      E_array 是总能量（not per spin），M_array 是总磁化量 M_tot
    """
    # 如果传入 cupy 且 prefer_gpu -> 在 device 上就地算
    if prefer_gpu and HAS_CUPY and isinstance(lattices, cp.ndarray):
        x = cp.asarray(lattices, dtype=cp.int64)
        if x.ndim != 3 or x.shape[1] != x.shape[2]:
            raise ValueError("lattices must have shape (R, L, L)")
        R = int(x.shape[0])
        if R == 0:
            return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)
        # roll 的 axis 指向最后两个维度 (L,L) 即 axis=(1,2) 或等价的 (-2,-1)
        right = cp.roll(x, -1, axis=2)
        down = cp.roll(x, -1, axis=1)
        e_bond = -cp.sum(x * (right + down), axis=(1, 2))  # (R,)
        M = cp.sum(x, axis=(1, 2))                         # (R,)
        h_vec = _normalize_h_for_batch(h, R, xp=cp)
        E = e_bond - h_vec * M
        return cp.asnumpy(E).astype(np.float64), cp.asnumpy(M).astype(np.float64)

    # 将输入转为 numpy（CPU 路径或为一次性把整个批次拷到 GPU）
    arr = np.asarray(lattices)
    if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
        raise ValueError("lattices must have shape (R, L, L)")
    R = int(arr.shape[0])
    if R == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)

    # 如果 prefer_gpu 并且 CuPy 可用，但传入的是 numpy -> 将整个批次一次性移至 GPU（减少多次来回）
    if prefer_gpu and HAS_CUPY:
        x = cp.asarray(arr, dtype=cp.int64)
        right = cp.roll(x, -1, axis=2)
        down = cp.roll(x, -1, axis=1)
        e_bond = -cp.sum(x * (right + down), axis=(1, 2))
        M = cp.sum(x, axis=(1, 2))
        h_vec = _normalize_h_for_batch(h, R, xp=cp)
        E = e_bond - h_vec * M
        return cp.asnumpy(E).astype(np.float64), cp.asnumpy(M).astype(np.float64)

    # CPU 路径：全部在 host 上计算（使用 int64 做中间计算以避免溢出）
    x = arr.astype(np.int64, copy=False)
    # roll 最后两个轴 (L,L) -> axis=(1,2)
    right = np.roll(x, -1, axis=2)
    down = np.roll(x, -1, axis=1)
    e_bond = -np.sum(x * (right + down), axis=(1, 2), dtype=np.int64)  # shape (R,)
    M = np.sum(x, axis=(1, 2), dtype=np.int64)                          # shape (R,)
    h_vec = _normalize_h_for_batch(h, R, xp=np)
    E = e_bond.astype(np.float64) - h_vec.astype(np.float64) * M.astype(np.float64)
    return np.asarray(E, dtype=np.float64), np.asarray(M, dtype=np.float64)


# ---------------------------------------------------------------------
# 模拟器友好包装（返回 per-spin / m 等，键名与 simulator 期望一致）
# ---------------------------------------------------------------------
def _observables_for_simulator(latt: Any, h: float) -> Dict[str, np.float64]:
    """
    用于模拟器接口的包装：返回每自旋观测（E per spin, m, absM 等），全部 np.float64。
    返回字典键: "E" (E per spin), "M" (m per spin), "absM", "M2","M4"
    """
    out = calculate_observables(latt, h=h, prefer_gpu=False)
    return {
        "E": np.float64(out["E_per_spin"]),
        "M": np.float64(out["m"]),
        "absM": np.float64(out["abs_m"]),
        "M2": np.float64(out["M2"]),
        "M4": np.float64(out["M4"]),
    }


# ---------------------------------------------------------------------
# 统计学辅助（返回 np.float64）
# ---------------------------------------------------------------------
def calculate_binder_cumulant(m_series: Any) -> np.float64:
    """
    Binder cumulant: U = 1 - <m^4> / (3 <m^2>^2)
    Input m_series can be array-like of per-spin magnetizations.
    返回 np.float64。
    """
    m = np.asarray(m_series, dtype=np.float64).ravel()
    if m.size == 0:
        return np.float64(0.0)
    m2_mean = float(np.mean(m * m))
    if np.isclose(m2_mean, 0.0, atol=1e-20):
        return np.float64(0.0)
    m4_mean = float(np.mean(m * m * m * m))
    return np.float64(1.0 - (m4_mean / (3.0 * (m2_mean * m2_mean))))


def calculate_specific_heat_per_spin(E_series: Any, T: float, N: int, E_is_per_spin: bool) -> np.float64:
    """
    使用能量涨落估计比热（per spin）。
      - If E_series contains energies per spin: var(E_per_spin) * N / T^2
      - If E_series contains total energies: var(E_total) / (N * T^2)
    返回 np.float64。
    """
    if N <= 0:
        raise ValueError("N must be positive")
    if T <= 0:
        raise ValueError("T must be positive")

    E = np.asarray(E_series, dtype=np.float64).ravel()
    if E.size == 0:
        return np.float64(0.0)
    var = float(np.var(E, ddof=0))
    if E_is_per_spin:
        # var(E_per_spin) * N / T^2
        return np.float64((float(N) * var) / (float(T) * float(T)))
    else:
        # var(E_total) / (N * T^2)
        return np.float64(var / (float(N) * float(T) * float(T)))


def calculate_susceptibility_per_spin(M_series: Any, T: float, N: int, M_is_per_spin: bool) -> np.float64:
    """
    使用磁化涨落估计磁化率（per spin）。
      - If M_series contains m_per_spin: var(m) * N / T
      - If M_series contains M_total: var(M) / (N * T)
    返回 np.float64。
    """
    if N <= 0:
        raise ValueError("N must be positive")
    if T <= 0:
        raise ValueError("T must be positive")

    M = np.asarray(M_series, dtype=np.float64).ravel()
    if M.size == 0:
        return np.float64(0.0)
    var = float(np.var(M, ddof=0))
    if M_is_per_spin:
        return np.float64((float(N) * var) / float(T))
    else:
        return np.float64(var / (float(N) * float(T)))


# ---------------------------------------------------------------------
# Smoke tests（保留现有测试用例并额外增强边界检查）
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("--- observables.py smoke tests ---")

    # 1. int8 overflow safety test
    Lbig = 512
    big = np.ones((Lbig, Lbig), dtype=np.int8)
    Etot = _energy_total_numpy(big, h=0.0)
    Mtot = int(np.sum(big, dtype=np.int64))
    Nbig = Lbig * Lbig
    assert np.isclose(Etot, np.float64(-2 * Nbig))
    assert int(Mtot) == Nbig
    print(f"L={Lbig} all+1 test passed (E_tot={Etot}, M_tot={Mtot})")

    # 2. abs_m fix verification test (L=4)
    L_small = 4
    test_latt = -np.ones((L_small, L_small), dtype=np.int8)
    test_latt[0, 0] = 1
    test_latt[0, 1] = 1
    test_latt[1, 0] = 1

    obs = calculate_observables(test_latt, h=0.0)
    print(f"Test lattice M_total = {obs['M_total']}")
    print(f"Test lattice m = {obs['m']}")
    print(f"Test lattice abs_m = {obs['abs_m']}")

    assert np.isclose(obs["M_total"], -10.0)
    assert np.isclose(obs["m"], -0.625)
    assert np.isclose(obs["abs_m"], 0.625)
    assert not np.isclose(obs["abs_m"], 1.0)
    print("abs_m fix test passed.")

    # 3. Batch calculation test (R=2, L=4)
    batch_lattices = np.stack([test_latt, np.ones_like(test_latt)], axis=0)
    E_batch, M_batch = calculate_observables_batch(batch_lattices, h=0.0)

    print(f"Batch E: {E_batch}")
    print(f"Batch M: {M_batch}")

    obs_test_latt_E0 = calculate_observables(test_latt, h=0.0)["E_total"]
    obs_ones_latt_E0 = calculate_observables(np.ones_like(test_latt), h=0.0)["E_total"]

    assert np.isclose(M_batch[0], -10.0)
    assert np.isclose(M_batch[1], 16.0)
    assert np.isclose(E_batch[1], -32.0)
    print("Batch calculation test passed.")

