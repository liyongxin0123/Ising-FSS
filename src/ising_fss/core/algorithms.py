# -*- coding: utf-8 -*-
"""
    二维 Ising 模型经典 MCMC 更新算法（CPU 向量化实现）

    本模块实现了二维 Ising 模型的经典马尔可夫链蒙特卡洛 (MCMC) 更新算法。
为了保证科学计算的可复现性，所有随机性均通过显式传递 `numpy.random.Generator` 或 `replica_seed` 控制。

支持算法：
    - ``metropolis_sweep``: 棋盘格 Metropolis-Hastings（任意 h）
    - ``wolff``: 单簇翻转算法（仅 h = 0，临界区极强去相关）
    - ``swendsen_wang``: 多簇并行翻转（仅 h = 0）

实现功能：
    - 可复现：所有随机性通过显式 ``numpy.random.Generator`` 或 ``replica_seed`` 控制
    - 支持外磁场 h ≠ 0（Metropolis）与 h = 0（Wolff / Swendsen-Wang）
    - 棋盘格分解（Checkerboard）消除数据竞争，支持并行更新
    - Numba JIT 加速 + 预分配随机数数组
    - 精确统计 RNG 消耗量与接受率，便于误差分析与审计
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, List

import math
import warnings

import numpy as np

# ----------------------- 可选 Numba 加速 -----------------------
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def deco(f):
            return f
        return deco

# -----------------------
# 随机类型与 BitGenerator 检测
# -----------------------
from numpy.random import Generator  # type: ignore
try:
    from numpy.random import Philox  # type: ignore
    _HAS_PHILOX = True
except Exception:
    Philox = None  # type: ignore
    _HAS_PHILOX = False
    warnings.warn(
        "numpy.random.Philox not available in this NumPy. Falling back to default_rng. "
        "For best reproducibility with many parallel streams consider a NumPy that exposes Philox.",
        UserWarning,
    )
# SeedSequence 用于 spawn 子种子
from numpy.random import SeedSequence

# -----------------------
# 类型别名
# -----------------------
MoveMeta = Dict[str, Any]


# -----------------------
# 随机种子 / Generator 辅助
# -----------------------
def _seed32(seed: Optional[int]) -> int:
    """
    将任意整数截断为 32-bit 无符号整数，便于与 GPU/其它实现位宽一致。
    若 seed 为 None 则抛出 ValueError（调用方应拒绝 None）。
    """
    if seed is None:
        raise ValueError("seed must be an integer (not None)")
    try:
        s = int(seed)
    except Exception:
        raise ValueError("seed must be convertible to int")
    return int(s) & 0xFFFFFFFF


def _seed_to_generator(seed: Optional[int]) -> np.random.Generator:
    """
    根据整数种子构造一个 numpy.random.Generator。 
    如果可用，优先使用 Philox 位生成器；否则回退至 default_rng(seed32)。
    """
    if seed is None:
        raise ValueError("seed must not be None (explicit replica_seed required)")
    s32 = _seed32(seed)
    if _HAS_PHILOX:
        try:
            bg = Philox(s32)
            return Generator(bg)
        except Exception:
            warnings.warn("Philox construction failed; falling back to default_rng.", UserWarning)
    return np.random.default_rng(int(s32))

def spawn_replica_seeds(master_seed: int, n_replicas: int) -> List[int]:
    """
    使用 SeedSequence.spawn 方法，从主种子 (master_seed) 派生出 $n$ 个副本子 32 位种子。
    旨在跨 NumPy 版本实现可移植性。返回值是一个介于 $[0, 2^{32})$ 之间的整数列表。
    提示 ：
    推荐的做法是仅调用一次 spawn（例如，在调度程序 [dispatcher] 中），并将派生的副本种子 (replica_seeds) 分配给下游模块，以确保结果的可重现性。
    """
    if master_seed is None:
        raise ValueError("master_seed must be provided")
    ss = SeedSequence(int(master_seed))
    children = ss.spawn(int(n_replicas))
    out: List[int] = []
    for ch_idx, ch in enumerate(children):
        # 1) prefer child.entropy if present
        ent = getattr(ch, "entropy", None)
        if ent is not None:
            try:
                e0 = int(ent[0]) if hasattr(ent, "__len__") else int(ent)
                out.append(e0 & 0xFFFFFFFF)
                continue
            except Exception:
                pass
        # 2) try generate_state if available (recent numpy)
        gen_state = getattr(ch, "generate_state", None)
        if callable(gen_state):
            try:
                st = ch.generate_state(1)  # expected uint32 array on recent numpy
                out.append(int(st[0]) & 0xFFFFFFFF)
                continue
            except Exception:
                pass
        # 3) last-resort deterministic fallback (stable across invocations)
        out.append((int(master_seed) ^ 0x9e3779b97f4a7c15 ^ ch_idx) & 0xFFFFFFFF)
    return out


# -----------------------
# 算法名称规范化
# -----------------------
def normalize_algo_name(name: str) -> str:
    if name is None:
        raise ValueError("Algorithm name must be provided")
    s = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if s in ("metro", "metropolis_sweep", "metropolissweep"):
        return "metropolis_sweep"
    if "wolff" in s:
        return "wolff"
    if "swendsen" in s or s in ("sw", "swendsen_wang", "swendsenwang", "swendsen_wang_cluster"):
        return "swendsen_wang"
    return s


# -----------------------
# 能量 / 观测量（NumPy 实现）
# - _energy_total_numpy -> total energy (NOT divided by N)
# - _energy_per_spin_numpy -> energy per spin (density)
# - _observables_numpy uses per-spin energy
# -----------------------
def _energy_total_numpy(latt: np.ndarray, h: float = 0.0) -> float:
    """
    计算全部能量 E_total = E_bond - h * M_total (NOT divided by N).
    """
    a = np.asarray(latt)
    if a.size == 0:
        raise ValueError("lattice must be non-empty")
    ai = a.astype(np.int64)
    right = np.roll(ai, -1, axis=1)
    down = np.roll(ai, -1, axis=0)
    e_bond = -int(np.sum(ai * (right + down)))
    m_tot = int(np.sum(ai))
    e_tot = float(e_bond) - float(h) * float(m_tot)
    return e_tot


def _energy_per_spin_numpy(latt: np.ndarray, h: float = 0.0) -> float:
    """
    计算单个构型能量 (E/N): (E_bond - h * M_total) / N.
    """
    a = np.asarray(latt)
    if a.size == 0:
        raise ValueError("lattice must be non-empty")
    N = float(a.size)
    ai = a.astype(np.int64)
    right = np.roll(ai, -1, axis=1)
    down = np.roll(ai, -1, axis=0)
    e_bond = -int(np.sum(ai * (right + down)))
    m_tot = int(np.sum(ai))
    return (float(e_bond) - float(h) * float(m_tot)) / N


def _observables_numpy(latt: np.ndarray, h: float = 0.0) -> Dict[str, float]:
    """
    计算各个物理可测量（单个构型）：能量/磁化强度。
    Keys: "E", "M", "absM", "M2", "M4"
    """
    e = _energy_per_spin_numpy(latt, h)
    a = np.asarray(latt)
    N = float(a.size)
    m_tot = int(np.sum(a.astype(np.int64)))
    m = float(m_tot) / N
    absM = float(abs(m_tot)) / N
    return {"E": float(e), "M": float(m), "absM": float(absM), "M2": float(m * m), "M4": float(m * m * m * m)}


# -----------------------
# JIT 内核：Metropolis（棋盘格 odd–even），只消费接受随机数
# -----------------------
@njit(cache=True, fastmath=False)
def _metropolis_checkerboard_sweep_jit(
    u_acc: np.ndarray, latt: np.ndarray, beta: float, h: float
) -> Tuple[np.ndarray, int]:
    """
    Metropolis 棋盘格 JIT：两次子格（odd/even）依次更新。
    输入：
      - u_acc: 长度 >= N 的接受随机数（不再随机选址）
      - latt: int8 矩阵 (L,L)
    返回：
      (updated_latt, accepts)
    """
    L = latt.shape[0]
    N = L * L
    # 开发期早发现集成错误；几乎无运行时开销（numba 中可抛异常）
    if u_acc.size < N:
        raise ValueError("u_acc size must be at least N=L*L for checkerboard sweep")

    accepts = 0
    t = 0
    for parity in (0, 1):
        for i in range(L):
            # 当前行的起点列：奇偶位与 parity 异或
            j0 = (i + parity) & 1
            for j in range(j0, L, 2):
                s = int(latt[i, j])
                ip = i + 1
                if ip == L:
                    ip = 0
                im = i - 1
                if im < 0:
                    im = L - 1
                jp = j + 1
                if jp == L:
                    jp = 0
                jm = j - 1
                if jm < 0:
                    jm = L - 1
                neigh = int(latt[ip, j]) + int(latt[im, j]) + int(latt[i, jp]) + int(latt[i, jm])
                dE = 2.0 * s * (neigh + h)
                # 接受判据：dE<=0 或 u < exp(-β dE)
                if dE <= 0.0:
                    latt[i, j] = -latt[i, j]
                    accepts += 1
                else:
                    if u_acc[t] < math.exp(-beta * dE):
                        latt[i, j] = -latt[i, j]
                        accepts += 1
                t += 1
    return latt, accepts


# -----------------------
# JIT 内核：Wolff / Swendsen–Wang
# -----------------------
@njit(cache=True)
def _wolff_update_jit(u: np.ndarray, latt: np.ndarray, beta: float) -> Tuple[np.ndarray, int]:
    """
    Wolff 单簇 JIT 内核（仅在 h=0 时物理正确）。
    u: 随机数组，wrapper 必须提供充足长度（建议 len(u) >= 4*N，以避免在簇极端扩展时越界）
    返回 (updated_latt, cluster_size)
    """
    L = latt.shape[0]
    N = L * L
    p_add = 1.0 - math.exp(-2.0 * beta)

    r0 = int(u[0] * N)
    if r0 >= N:
        r0 = N - 1
    i0 = r0 // L
    j0 = r0 - i0 * L
    spin0 = int(latt[i0, j0])

    stack_i = np.empty(N, dtype=np.int32)
    stack_j = np.empty(N, dtype=np.int32)
    in_cluster = np.zeros((L, L), dtype=np.uint8)
    top = 0
    stack_i[top] = i0
    stack_j[top] = j0
    top += 1
    in_cluster[i0, j0] = 1
    size = 0
    k = 1   # u 索引（u[0] 已用于选起点）
    while top > 0:
        top -= 1
        i = int(stack_i[top])
        j = int(stack_j[top])
        size += 1
        ip, im = (i + 1) % L, (i - 1) % L
        jp, jm = (j + 1) % L, (j - 1) % L

        # 逐个邻居检查并根据 u[k] 决定是否加入簇（且只对同向自旋考虑）
        if in_cluster[ip, j] == 0 and int(latt[ip, j]) == spin0:
            if u[k] < p_add:
                in_cluster[ip, j] = 1
                stack_i[top] = ip
                stack_j[top] = j
                top += 1
            k += 1
        if in_cluster[im, j] == 0 and int(latt[im, j]) == spin0:
            if u[k] < p_add:
                in_cluster[im, j] = 1
                stack_i[top] = im
                stack_j[top] = j
                top += 1
            k += 1
        if in_cluster[i, jp] == 0 and int(latt[i, jp]) == spin0:
            if u[k] < p_add:
                in_cluster[i, jp] = 1
                stack_i[top] = i
                stack_j[top] = jp
                top += 1
            k += 1
        if in_cluster[i, jm] == 0 and int(latt[i, jm]) == spin0:
            if u[k] < p_add:
                in_cluster[i, jm] = 1
                stack_i[top] = i
                stack_j[top] = jm
                top += 1
            k += 1

    # 翻转簇内所有自旋
    for ii in range(L):
        for jj in range(L):
            if in_cluster[ii, jj] == 1:
                latt[ii, jj] = -latt[ii, jj]
    return latt, size


# -----------------------
# DSU 工具（numba / python fallback）
# -----------------------
if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _dsu_find(parent: np.ndarray, x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    @njit(cache=True)
    def _dsu_unite(parent: np.ndarray, sizes: np.ndarray, a: int, b: int) -> None:
        ra = _dsu_find(parent, a)
        rb = _dsu_find(parent, b)
        if ra == rb:
            return
        if sizes[ra] < sizes[rb]:
            tmp = ra
            ra = rb
            rb = tmp
        parent[rb] = ra
        sizes[ra] += sizes[rb]
else:
    def _dsu_find(parent: list, x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _dsu_unite(parent: list, sizes: list, a: int, b: int) -> None:
        ra = _dsu_find(parent, a)
        rb = _dsu_find(parent, b)
        if ra == rb:
            return
        if sizes[ra] < sizes[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        sizes[ra] += sizes[rb]


@njit(cache=True)
def _swendsen_wang_jit(u: np.ndarray, latt: np.ndarray, beta: float) -> Tuple[np.ndarray, int]:
    """
    Swendsen–Wang JIT 内核（仅在 h=0 时物理正确）。
    u: 随机数组，wrapper 会生成足够长度的 u（建议 len(u) >= 4*N）
    返回 (updated_latt, flipped_sites_count)
    """
    L = latt.shape[0]
    N = L * L
    parent = np.arange(N, dtype=np.int32)
    sizes = np.ones(N, dtype=np.int32)
    p_add = 1.0 - math.exp(-2.0 * beta)

    t = 0
    # 对每个格子考虑右与下边（避免双计）
    for i in range(L):
        for j in range(L):
            s = int(latt[i, j])
            jp = (j + 1) % L
            ip = (i + 1) % L
            # 右边
            if int(latt[i, jp]) == s:
                if u[t] < p_add:
                    _dsu_unite(parent, sizes, i * L + j, i * L + jp)
            t += 1
            # 下边
            if int(latt[ip, j]) == s:
                if u[t] < p_add:
                    _dsu_unite(parent, sizes, i * L + j, ip * L + j)
            t += 1

    # 根据抛硬币决定是否翻转
    root_flip = np.zeros(N, dtype=np.uint8)
    for x in range(N):
        if parent[x] == x:
            if u[t] < 0.5:
                root_flip[x] = 1
            t += 1

    n_flip = 0
    # 执行翻转
    for i in range(L):
        for j in range(L):
            idx = i * L + j
            r = idx
            while parent[r] != r:
                parent[r] = parent[parent[r]]
                r = parent[r]
            if root_flip[r] == 1:
                latt[i, j] = -latt[i, j]
                n_flip += 1
    return latt, n_flip


# -----------------------
# Python 回退实现（wrapper-level）
# -----------------------
def _metropolis_sweep_py(
    latt: np.ndarray, beta: float, rng: np.random.Generator, h: float = 0.0
) -> Tuple[np.ndarray, MoveMeta]:
    """
    Python 实现的 Metropolis 棋盘格 sweep（非 JIT 路径）。
    返回 meta 包含 "accepted","attempts" 以及 "rng_consumed"（单位：uniform_draws）。
    对于一个 full sweep，使用 **N** 个 uniform draws（仅接受随机数）。
    """
    arr = np.ascontiguousarray(np.asarray(latt, dtype=np.int8).copy())
    L = arr.shape[0]
    N = L * L
    # 生成等量随机数
    u_acc = rng.random(N)
    # 正常应恒为真；若集成错误（例如错误传参）尽早暴露
    assert u_acc.size >= N, "internal error: u_acc shorter than N"
    if NUMBA_AVAILABLE:
        new, accepts = _metropolis_checkerboard_sweep_jit(u_acc, arr, float(beta), float(h))
        return new, {"accepted": int(accepts), "attempts": int(N), "rng_consumed": int(N)}

    # 纯 Python 棋盘格
    accepts = 0
    t = 0
    for parity in (0, 1):
        for i in range(L):
            j0 = (i + parity) & 1
            for j in range(j0, L, 2):
                s = int(arr[i, j])
                ip, im = (i + 1) % L, (i - 1) % L
                jp, jm = (j + 1) % L, (j - 1) % L
                neigh = int(arr[ip, j]) + int(arr[im, j]) + int(arr[i, jp]) + int(arr[i, jm])
                dE = 2.0 * s * (neigh + h)
                if dE <= 0.0 or (u_acc[t] < math.exp(-beta * dE)):
                    arr[i, j] = -arr[i, j]
                    accepts += 1
                t += 1
    return arr, {"accepted": int(accepts), "attempts": int(N), "rng_consumed": int(N)}


def _wolff_update_py(latt: np.ndarray, beta: float, rng: np.random.Generator) -> Tuple[np.ndarray, MoveMeta]:
    """
    Python 实现的 Wolff 单簇更新（仅在 h=0 时物理正确）。
    为兼容 JIT 并避免在簇极端扩展时越界，生成长度为 4*N 的随机数组 u，并**全程消费 u**，
    使回退路径与 JIT 在随机消耗口径上一致（rng_consumed=4*N）。
    返回 meta 包含 "cluster_size" 以及 "rng_consumed"（单位：uniform_draws）。
    注意：簇算法内部随机消耗是可变的；此处 wrapper 生成 4*N 的 uniform draws 并记录该值作为诊断。
    """
    arr = np.ascontiguousarray(np.asarray(latt, dtype=np.int8).copy())
    L = arr.shape[0]
    N = L * L
    # 生成随机数组并调用 JIT（若可用）
    u = rng.random(4 * N)
    if NUMBA_AVAILABLE:
        new, size = _wolff_update_jit(u, arr, float(beta))
        return new, {"cluster_size": int(size), "rng_consumed": int(4 * N)}
    
    # Python 回退实现（全程消费 u）
    p_add = 1.0 - math.exp(-2.0 * beta)
    t = 0
    r0 = int(u[t] * N)
    t += 1
    if r0 >= N:
        r0 = N - 1
    i0 = r0 // L
    j0 = r0 - i0 * L
    spin0 = int(arr[i0, j0])
    cluster = set([(i0, j0)])
    stack = [(i0, j0)]
    while stack:
        i, j = stack.pop()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = (i + di) % L, (j + dj) % L
            if (ni, nj) not in cluster and arr[ni, nj] == spin0:
                if t >= u.size:
                    raise RuntimeError("wolff fallback: random pool exhausted; increase safety factor (currently 4*N)")
                if u[t] < p_add:
                    cluster.add((ni, nj))
                    stack.append((ni, nj))
                t += 1
    for (i, j) in cluster:
        arr[i, j] = -arr[i, j]
    return arr, {"cluster_size": int(len(cluster)), "rng_consumed": int(4 * N)}


def _swendsen_wang_py(latt: np.ndarray, beta: float, rng: np.random.Generator) -> Tuple[np.ndarray, MoveMeta]:
    """
    Python 实现的 Swendsen–Wang 更新（仅在 h=0 时物理正确）。
    生成长度为 4*N 的随机数组 u 以保证足够随机数（wrapper 记录 rng_consumed=4*N）。
    """
    arr = np.ascontiguousarray(np.asarray(latt, dtype=np.int8).copy())
    L = arr.shape[0]
    N = L * L
    u = rng.random(4 * N)
    if NUMBA_AVAILABLE:
        new, flips = _swendsen_wang_jit(u, arr, float(beta))
        return new, {"flipped_sites": int(flips), "rng_consumed": int(4 * N)}
    parent = list(range(N))
    size = [1] * N

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    p_add = 1.0 - math.exp(-2.0 * beta)
    t = 0
    for i in range(L):
        for j in range(L):
            s = int(arr[i, j])
            jp = (j + 1) % L
            ip = (i + 1) % L
            if int(arr[i, jp]) == s:
                if u[t] < p_add:
                    union(i * L + j, i * L + jp)
            t += 1
            if int(arr[ip, j]) == s:
                if u[t] < p_add:
                    union(i * L + j, ip * L + j)
            t += 1

    root_flip = [0] * N
    for x in range(N):
        if parent[x] == x:
            if u[t] < 0.5:
                root_flip[x] = 1
            t += 1

    flips = 0
    for i in range(L):
        for j in range(L):
            idx = i * L + j
            r = idx
            while parent[r] != r:
                parent[r] = parent[parent[r]]
                r = parent[r]
            if root_flip[r]:
                arr[i, j] = -arr[i, j]
                flips += 1
    return arr, {"flipped_sites": int(flips), "rng_consumed": int(4 * N)}


# -----------------------
# 算法注册（以规范化名称为键）
# NOTE:
#   - "metropolis_sweep" 使用 **棋盘格** 更新（odd–even）
#   - attempts == N (= L*L) per sweep
#   - rng_consumed == N uniform_draws per sweep
# wrapper 接口： (latt, beta, rng, h) -> (latt_out, meta)
# 注意：这里的 rng 是 np.random.Generator，由上层从显式 seed 创建
# meta 必须包含 "rng_consumed"（单位 uniform_draws）作为诊断
# -----------------------
_DEF_ALGOS: Dict[str, Callable[..., Tuple[np.ndarray, MoveMeta]]] = {
    "metropolis_sweep": lambda latt, beta, rng, h=0.0: _metropolis_sweep_py(latt, beta, rng, h),
    "wolff": lambda latt, beta, rng, h=0.0: _wolff_update_py(latt, beta, rng),
    "swendsen_wang": lambda latt, beta, rng, h=0.0: _swendsen_wang_py(latt, beta, rng),
}


def get_algorithm(name: str) -> Callable[..., Tuple[np.ndarray, MoveMeta]]:
    n = normalize_algo_name(name)
    if n not in _DEF_ALGOS:
        raise ValueError(f"Unknown algorithm: {name} (normalized -> '{n}'). Known algorithms: {list(_DEF_ALGOS.keys())}")
    return _DEF_ALGOS[n]


@dataclass
class MoveInfo:
    algo: str
    accepted: Optional[int] = None
    cluster_size: Optional[int] = None
    flipped_sites: Optional[int] = None
    rng_consumed: Optional[int] = None
    rng_model: Optional[str] = None
    replica_seed: Optional[int] = None


# -----------------------
# 统一入口 apply_move（支持 rng 或 replica_seed，两者兼容）
# -----------------------
def apply_move(
    lattice: Any,
    algo: Any,
    beta: float,
    replica_seed: Optional[int] = None,
    rng: Optional[Generator] = None,
    h: float = 0.0,
) -> Tuple[np.ndarray, MoveInfo]:
    """
    统一入口：支持两种 RNG 输入方式（优先级如下）：
      1. rng: 已初始化的 numpy.random.Generator（优先）
      2. replica_seed: 整数种子（会由 _seed_to_generator 构造 Generator）

    参数：
      - lattice: (L,L) array-like
      - algo: 字符串或可调用（若为字符串，会 normalize 并从注册表取 wrapper）
      - beta: float
      - replica_seed: 可选整数种子（若 rng 提供则忽略）
      - rng: 可选的 np.random.Generator；若提供则优先使用；如果同时提供 replica_seed 则 replica_seed 被忽略且发出警告
      - h: 外场

    返回：
      (lattice_out as np.ndarray(dtype=np.int8), MoveInfo)
    """
    # 解析算法
    if isinstance(algo, str):
        name = normalize_algo_name(algo)
        # 早期拒绝簇算法在 h != 0 时使用
        if abs(float(h)) > 1e-12 and name in ("wolff", "swendsen_wang"):
            warnings.warn(f"Requested cluster algorithm '{algo}' with h={h} != 0: disallowed.", UserWarning, stacklevel=2)
            raise ValueError(f"Algorithm '{algo}' is not allowed when external field h != 0. Use 'metropolis_sweep' instead.")
        f = get_algorithm(name)
    else:
        f = algo
        name = getattr(algo, "__name__", "<callable>")

    if not callable(f):
        raise ValueError("Algorithm is not callable")

    # 构造或验证 RNG 
    used_seed: Optional[int] = None
    used_rng: Optional[np.random.Generator] = None

    # RNG 处理优先级：rng 优先，若 rng 为 None 则使用 replica_seed 构造 Generator。
    if rng is not None:
        if not isinstance(rng, Generator):
            raise ValueError("rng must be a numpy.random.Generator if provided.")
        used_rng = rng
    else:
        if replica_seed is None:
            raise ValueError("Either 'rng' (Generator) or 'replica_seed' (int) must be provided for deterministic RNG.")
        used_seed = int(replica_seed)
        used_rng = _seed_to_generator(used_seed)

    # 在引入 JIT/backends 前，确保构型数据存储连续
    lattice_in = np.ascontiguousarray(np.asarray(lattice, dtype=np.int8))

    # 调用 wrapper（所有 wrapper 接受 (latt, beta, rng, h)） 
    result = f(lattice_in, float(beta), used_rng, float(h))
    if isinstance(result, tuple) and len(result) == 2:
        latt_out, meta = result
    else:
        latt_out, meta = result, {}

    info = MoveInfo(algo=name)
    # record actual bit-generator class name for provenance
    try:
        info.rng_model = type(used_rng.bit_generator).__name__
    except Exception:
        info.rng_model = "unknown"
    if used_seed is not None:
        info.replica_seed = int(used_seed)

    if isinstance(meta, dict):
        if "accepted" in meta:
            info.accepted = int(meta["accepted"])
        if "cluster_size" in meta:
            info.cluster_size = int(meta["cluster_size"])
        if "flipped_sites" in meta:
            info.flipped_sites = int(meta["flipped_sites"])
        if "accepts" in meta and info.accepted is None:
            info.accepted = int(meta["accepts"])
        if "rng_consumed" in meta:
            try:
                info.rng_consumed = int(meta["rng_consumed"])
            except Exception:
                info.rng_consumed = None

    return np.asarray(latt_out, dtype=np.int8), info


# -----------------------
# 批量接口 update_batch（使用 replica_seeds）
# -----------------------
def update_batch(
    spins_batch: Any,
    beta: float,
    *,
    replica_seeds: Sequence[int],
    algo: str = "metropolis_sweep",
    h: float = 0.0,
    n_sweeps: int = 1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    在 CPU 上对 batch spins (R,L,L) 做 n_sweeps 次更新（每个副本用对应的 replica_seed）。
    强制要求 replica_seeds 的长度与 R 一致，且其中元素不可为 None。
    返回：
      spins_out: np.ndarray (R,L,L) dtype int8
      info: dict 包含 per_replica MoveMeta，其中每个 meta 会包含 rng_consumed（单位 uniform_draws）
    """
    arr = np.ascontiguousarray(np.asarray(spins_batch, dtype=np.int8))
    if arr.ndim != 3:
        raise ValueError("spins_batch must be 3D array (R,L,L)")
    R, L1, L2 = arr.shape
    if L1 != L2:
        raise ValueError("spins must be square (L,L)")
    if replica_seeds is None:
        raise ValueError("replica_seeds must be provided and cannot be None")
    if len(replica_seeds) != R:
        raise ValueError("replica_seeds length must equal number of replicas R")

    for idx, s in enumerate(replica_seeds):
        if s is None:
            raise ValueError(f"replica_seeds[{idx}] is None: all replica_seeds must be explicit integers for reproducibility")

    algo_name = normalize_algo_name(algo)
    if abs(float(h)) > 1e-12 and algo_name in ("wolff", "swendsen_wang"):
        raise ValueError(f"Algorithm '{algo}' is not allowed when external field h != 0. Use 'metropolis_sweep' instead.")

    out = np.empty_like(arr, dtype=np.int8)
    per_replica_meta: List[MoveMeta] = []

    # beta 检查：若为序列则长度必须为 R 或 1
    if hasattr(beta, "__len__") and not isinstance(beta, (str, bytes)):
        try:
            blen = len(beta)
        except Exception:
            blen = None
        if blen is None:
            beta_list = [float(beta)] * R
        else:
            if blen not in (1, R):
                raise ValueError("update_batch: if beta is a sequence its length must be 1 or equal to number of replicas R")
            if blen == 1:
                beta_list = [float(beta[0])] * R
            else:
                beta_list = [float(x) for x in beta]
    else:
        beta_list = [float(beta)] * R

    alg = get_algorithm(algo_name)

    for r in range(R):
        seed = int(replica_seeds[r])
        rng = _seed_to_generator(seed)
        latt = np.ascontiguousarray(arr[r], dtype=np.int8).copy()
        meta_acc: MoveMeta = {}
        total_rng_consumed = 0
        for _ in range(max(1, int(n_sweeps))):
            latt, meta = alg(latt, beta_list[r], rng, float(h))
            if isinstance(meta, dict):
                if "attempts" in meta:
                    meta_acc["attempts"] = meta_acc.get("attempts", 0) + int(meta.get("attempts", 0))
                if "accepted" in meta:
                    meta_acc["accepted"] = meta_acc.get("accepted", 0) + int(meta.get("accepted", 0))
                if "accepts" in meta:
                    meta_acc["accepted"] = meta_acc.get("accepted", 0) + int(meta.get("accepts", 0))
                if "cluster_size" in meta:
                    meta_acc["cluster_size"] = int(meta["cluster_size"])
                if "flipped_sites" in meta:
                    meta_acc["flipped_sites"] = int(meta["flipped_sites"])
                if "rng_consumed" in meta:
                    try:
                        total_rng_consumed += int(meta["rng_consumed"])
                    except Exception:
                        pass
        out[r] = latt
        if total_rng_consumed > 0:
            meta_acc["rng_consumed"] = int(total_rng_consumed)
        # provenance
        try:
            meta_acc["rng_model"] = type(rng.bit_generator).__name__
        except Exception:
            meta_acc["rng_model"] = "unknown"
        meta_acc["replica_seed"] = int(seed)
        per_replica_meta.append(meta_acc)

    info = {"per_replica": per_replica_meta}
    return out, info


# -----------------------
# 小自测（Smoke test）
# -----------------------
if __name__ == "__main__":
    # 小规模 smoke 测试：确保仅使用 replica_seeds 路径可运行 
    R = 4
    L = 8
    beta = 0.44
    seeds = [12345 + i for i in range(R)]
    batch = np.random.choice([-1, 1], size=(R, L, L)).astype(np.int8)
    print("Running batch metropolis (n_sweeps=1) using replica_seeds (checkerboard sweep, rng≈N)...")
    out, info = update_batch(batch, beta, replica_seeds=seeds, algo="metropolis_sweep", n_sweeps=1)
    print("Done. per_replica meta:", info["per_replica"])
    print("Running wolff (h must be 0):")
    try:
        outw, infow = update_batch(batch, beta, replica_seeds=seeds, algo="wolff", n_sweeps=1, h=0.0)
        print("Wolff OK sample meta:", infow["per_replica"][0])
    except Exception as e:
        print("Wolff failed (expected if h!=0):", e)
    print("Smoke test finished.")

