# -*- coding: utf-8 -*-
"""
    二维 Ising 模型副本交换蒙特卡洛模拟器 (cpu 版)

    此模块实现了基于 CPU 的 REMC 模拟器 `HybridREMCSimulator`。它通过在不同温度下并行运行多个系统副本，并定期尝试交换相邻温度的构型，来加速系统的热平衡过程并降低临界慢化现象。

实现功能:
    - 副本交换 (Parallel Tempering): 管理多个温度下的副本，支持几何级数或线性温度分布。
    - 算法支持: 支持 Metropolis (棋盘格分解)、Wolff (单簇) 和 Swendsen-Wang (多簇) 算法。
    - Slot-bound RNG: 每个温度槽 (Slot) 绑定独立的随机数生成器 (RNG) 和种子，确保并行计算的可复现性，即使副本发生交换，RNG 状态依然与温度槽绑定。
    - 自适应采样 (Auto-thinning): 内置在线自相关时间 ($\tau_{int}$) 估计，动态调整采样间隔 (`thin`)，以确保收集到统计独立的样本。
    - 断点续传 (Checkpointing): 支持保存和恢复模拟状态（包括 RNG 状态、副本构型、能量、统计计数器等）到 HDF5 文件，防止长时运行中断。
    - HDF5 落盘: 支持将模拟过程中的格点构型流式保存到 HDF5 文件。  
    - Provenance: 详细记录随机数生成器的版本、种子及消耗量，满足数据溯源需求。

注意: 
    簇算法 (Wolff/SW) 仅在无外场 (h=0) 时可用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Sequence
import json
import os
import math
import time
import sys
import platform
import getpass
import ast
import logging
import warnings

import numpy as np
import h5py

logger = logging.getLogger(__name__)

# exp(x) 在 x < -700 近似下溢，统一钳位
_MIN_EXP_ARG = -700.0
_MAX_ADVANCE_CHUNK = (1 << 31) - 1
PREFERRED_RNG = "philox"

# -------------------------
# 小工具
# -------------------------
def _as_text(x):
    """将可能为 bytes/np.bytes_ 的属性值解码为 str。"""
    if isinstance(x, (bytes, bytearray, np.bytes_)):
        try:
            return x.decode("utf-8")
        except Exception:
            return str(bytes(x))
    return x

def _advance_bitgen(bg, n_steps: int) -> Dict[str, Any]:
    """对 bit_generator 执行分块 advance(n_steps)，返回过程元信息。"""
    if not hasattr(bg, "advance"):
        raise AttributeError("bit_generator does not support advance()")
    remaining = int(n_steps)
    chunks = 0
    t0 = time.time()
    failed_chunk = None
    err = None
    try:
        while remaining > 0:
            step = min(remaining, _MAX_ADVANCE_CHUNK)
            try:
                bg.advance(step)
            except Exception as exc:
                failed_chunk = chunks
                err = f"bitgen.advance failed at chunk #{chunks} (step={step}): {exc}"
                raise RuntimeError(err) from exc
            remaining -= step
            chunks += 1
    finally:
        dt = time.time() - t0
    return {"ok": failed_chunk is None, "time_s": dt, "chunks": chunks, "failed_chunk": failed_chunk, "error": err}

def _make_generator_from_seed(seed: int, prefer: str = PREFERRED_RNG) -> np.random.Generator:
    """默认优先 Philox（如不可用退回 default_rng）。"""
    s = int(seed) & 0xFFFFFFFF
    try:
        from numpy.random import Philox  # type: ignore
        bitgen = Philox(int(s))
        gen = np.random.Generator(bitgen)
        return gen
    except Exception:
        return np.random.default_rng(int(s))

def _make_generator_from_seed_and_class(seed: int, class_name: str) -> np.random.Generator:
    """
    按名称精确构造 bit-generator，以保证 advance 语义一致。
    支持：Philox / PCG64 / SFC64 / MT19937；不支持则抛异常。
    """
    from numpy.random import Philox, PCG64, SFC64, MT19937  # type: ignore
    mapping = {
        "Philox": Philox,
        "PCG64": PCG64,
        "SFC64": SFC64,
        "MT19937": MT19937,
    }
    s = int(seed) & 0xFFFFFFFF
    if class_name not in mapping:
        raise RuntimeError(f"Unsupported bit-generator class for advance-restore: {class_name}")
    return np.random.Generator(mapping[class_name](int(s)))

def _bitgen_supports_advance(gen: np.random.Generator) -> bool:
    try:
        bg = gen.bit_generator
        return hasattr(bg, "advance") and callable(getattr(bg, "advance"))
    except Exception:
        return False

def detect_bitgen_steps_per_random(gen: np.random.Generator, n_trials: int = 10) -> Optional[float]:
    """启发式测量一次 gen.random() 消耗多少 bitgen steps"""
    def numeric_state_parts(state: dict):
        parts = {}
        for k, v in state.items():
            if isinstance(v, (list, tuple, np.ndarray)):
                try:
                    arr = np.asarray(v)
                    if np.issubdtype(arr.dtype, np.integer):
                        parts[k] = arr.astype(np.int64)
                except Exception:
                    pass
            elif isinstance(v, int):
                parts[k] = np.array([v], dtype=np.int64)
        return parts

    deltas = []
    for _ in range(max(1, int(n_trials))):
        try:
            s0 = gen.bit_generator.state
            p0 = numeric_state_parts(s0)
            _ = gen.random()
            s1 = gen.bit_generator.state
            p1 = numeric_state_parts(s1)
            found = False
            for k in p0.keys():
                if k in p1 and p0[k].shape == p1[k].shape:
                    diff = (p1[k].astype(np.int64) - p0[k].astype(np.int64))
                    pos = diff[diff > 0]
                    if pos.size > 0:
                        deltas.append(int(np.sum(pos)))
                        found = True
                        break
            if not found:
                deltas.append(None)
        except Exception:
            deltas.append(None)
    nums = [d for d in deltas if isinstance(d, int)]
    if len(nums) == 0:
        return None
    return float(sum(nums)) / len(nums)

def _estimate_rng_stride_for_algo(algo_name: str, L: int, bitgen_steps_per_uniform: Optional[int] = None) -> Optional[int]:
    """估计单个 sweep 在 bitgen 层面的 stride（步数）。"""
    n = int(L) * int(L)
    name = (algo_name or "").lower().strip()
    if name in ("metropolis", "metropolis_sweep", "metro", "metropolissweep"):
        if bitgen_steps_per_uniform is None:
            return None
        return int(n * int(bitgen_steps_per_uniform))
    if name in ("wolff", "swendsen_wang", "swendsenwang", "cluster"):
        return None
    return None

# -------------------------
# 简易 τ_int 与 ESS
# -------------------------
def _tau_int_sokal(x: np.ndarray, max_lag: Optional[int] = None) -> float:
    s = np.asarray(x, dtype=float)
    n = int(s.size)
    if n < 4:
        return 1.0
    s = s - float(np.mean(s))
    var = float(np.var(s))
    if var <= 0.0:
        return 1.0
    if max_lag is None:
        max_lag = min(n // 2, 1000)
    tau = 0.5
    for k in range(1, max_lag + 1):
        num = float(np.dot(s[:-k], s[k:])) / (n - k)
        rho = num / var
        if rho <= 0.0:
            break
        tau += rho
    return max(1.0, float(tau))

def _ess_from_series(x: np.ndarray, tau: Optional[float] = None) -> float:
    n = float(len(x))
    if n <= 0:
        return 0.0
    if tau is None:
        tau = _tau_int_sokal(np.asarray(x, dtype=float))
    return max(1.0, n / (2.0 * float(tau)))

# -------------------------
# 数据类
# -------------------------
@dataclass
class _Replica:
    beta: float
    lattice: np.ndarray  # shape (L, L) dtype int8（C contiguous）
    energy_total: float
    seed: int
    rng: np.random.Generator

# -------------------------
# 外部依赖（接口预期）
# -------------------------
try:
    # 期望接口：get_algorithm(lattice, beta, rng, h) -> (lattice_out, meta{rng_consumed:int?})
    from ..core.algorithms import get_algorithm
    from ..core.observables import _observables_for_simulator, _energy_total_numpy as _energy_total_numpy_ref
    from ..analysis import statistics as stats
except Exception as e:
    logger.error("[remc_simulator] CRITICAL: failed to import required modules: %s", e)
    raise

# -------------------------
# 物理口径（CPU 统一版，与交换一致）
# -------------------------
def _energy_total_numpy_consistent(latt: np.ndarray, h: float) -> float:
    """四邻 + 1/2 + 外场项；口径与交换一致。"""
    a = np.asarray(latt, dtype=np.int8)
    nbr = np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1)
    e_bond = -0.5 * float(np.sum(a * nbr, dtype=np.int64))
    e_field = -float(h) * float(np.sum(a, dtype=np.int64))
    return e_bond + e_field

# -------------------------
# 主体类
# -------------------------
class HybridREMCSimulator:
    """
    REMC 主控类（slot-bound RNG，显式 replica_seeds，支持 checkpoint/restore）
    """

    def __init__(
        self,
        L: int,
        T_min: float,
        T_max: float,
        num_replicas: int,
        algorithm: str = "metropolis_sweep",
        spacing: str = "geom",
        temperatures: Optional[List[float]] = None,
        h: float = 0.0,
        replica_seeds: Optional[Sequence[int]] = None,
        buffer_flush: int = 64,
        record_swap_history: bool = False,
        bitgen_steps_per_uniform: Optional[int] = None,
    ) -> None:
        # 基本参数
        self.L = int(L)
        self.N = int(L) * int(L)
        self.h = float(h)

        # 温度序列
        if temperatures is not None:
            temps = np.asarray(temperatures, dtype=float)
            if temps.ndim != 1 or temps.size < 2:
                raise ValueError("temperatures must be 1D array-like with at least 2 entries")
            if len(temps) != int(num_replicas):
                raise ValueError("temperatures length must match num_replicas")
        else:
            if spacing not in ("geom", "linear"):
                raise ValueError("spacing must be 'geom' or 'linear'")
            temps = np.geomspace(float(T_min), float(T_max), int(num_replicas)) if spacing == "geom" \
                else np.linspace(float(T_min), float(T_max), int(num_replicas))
        self.temps = temps.tolist()
        self.betas = [1.0 / float(T) for T in self.temps]

        # 算法正规化与校验
        req_algo = (algorithm or "metropolis_sweep").lower().strip().replace(" ", "_").replace("-", "_")
        if req_algo in ("metropolis", "metro", "metropolis_sweep", "metropolissweep"):
            req_algo = "metropolis_sweep"
        _cluster_algos = ("wolff", "swendsen_wang", "swendsenwang", "cluster")
        if req_algo in _cluster_algos and abs(self.h) > 1e-12:
            raise ValueError(
                f"Invalid configuration: chosen algorithm '{req_algo}' is a cluster algorithm that requires h=0, "
                f"but provided h={self.h}. Set h=0 or choose a single-spin algorithm like 'metropolis_sweep'."
            )
        self.algorithm = req_algo

        # 缓存算法句柄（减少重复 get）
        self._alg = None
        try:
            self._alg = get_algorithm(self.algorithm)
        except Exception as exc:
            logger.warning("[HybridREMCSimulator] Warning: algorithm '%s' may not be available: %s", self.algorithm, exc)

        # 强制显式 replica_seeds
        if replica_seeds is None:
            raise ValueError("This simulator requires explicit replica_seeds (one integer per replica).")
        try:
            seeds_list = [int(x) for x in list(replica_seeds)]
        except Exception:
            raise ValueError("replica_seeds must be an iterable of integers.")
        if len(seeds_list) != int(num_replicas):
            raise ValueError(f"replica_seeds length ({len(seeds_list)}) must equal num_replicas ({num_replicas}).")
        self.replica_seeds = seeds_list
        self._seed_info = {"replica_seeds": self.replica_seeds, "seed_bits": 32}

        # 初始化 replicas（slot-bound RNG）
        self.replicas: List[_Replica] = []
        for r, beta in enumerate(self.betas):
            seed = int(self.replica_seeds[r])
            rng = _make_generator_from_seed(seed)  # 运行期随机流（保持“未消耗”）
            # —— 初始化随机流解耦：用另一把派生 RNG 生成初始格点 —— #
            init_rng = _make_generator_from_seed(seed ^ 0xC2B2AE35)
            latt = (init_rng.integers(0, 2, size=(self.L, self.L), dtype=np.uint8) * 2 - 1).astype(np.int8)
            if not latt.flags["C_CONTIGUOUS"]:
                latt = np.ascontiguousarray(latt)
            e_tot = _energy_total_numpy_consistent(latt, self.h)
            self.replicas.append(_Replica(beta=float(beta), lattice=latt, energy_total=float(e_tot), seed=seed, rng=rng))

        # 交换统计
        num_pairs = max(0, len(self.replicas) - 1)
        self._swap_attempts = np.zeros(num_pairs, dtype=np.int64)
        self._swap_accepts = np.zeros(num_pairs, dtype=np.int64)
        self.record_swap_history = bool(record_swap_history)
        self._swap_history: Optional[List[List[bool]]] = [ [] for _ in range(num_pairs) ] if self.record_swap_history else None

        # IO/缓存 与 其它追踪字段
        self._results: Optional[Dict[str, Dict[str, np.ndarray]]] = None
        self.final_lattices: Optional[List[np.ndarray]] = None
        self._lattice_files: Optional[Dict[str, str]] = None
        self.buffer_flush = int(max(1, buffer_flush))
        self.warnings: List[str] = []

        # RNG 版本信息（用于 provenance）
        self._rng_versions = {"numpy": np.__version__}
        try:
            import cupy as _cupy  # type: ignore
            self._rng_versions["cupy"] = getattr(_cupy, "__version__", "unknown")
        except Exception:
            pass

        # provenance 注记
        self._provenance_notes = {
            "rng_binding": "slot",
            "rng_source": "replica_seeds",
            "swap_rng_policy": "use_left_slot_rng",
        }

        # variable-stride 标记（簇算法等）
        self.rng_consumption_variable = (self.algorithm in ("wolff", "swendsen_wang"))
        self._provenance_notes["rng_consumption_variable"] = bool(self.rng_consumption_variable)

        # per-replica bookkeeping
        nrep = len(self.replicas)
        self.rng_offset_within_sweep: List[int] = [0 for _ in range(nrep)]
        self.sweep_counts: List[int] = [0 for _ in range(nrep)]

        # sweep index（用于 checkpoint/advance）
        self.sweep_index = 0

        # bitgen_steps_per_uniform 与 stride 估计
        self.bitgen_steps_per_uniform: Optional[int] = bitgen_steps_per_uniform
        self.rng_stride_per_sweep: Optional[int] = _estimate_rng_stride_for_algo(self.algorithm, self.L, self.bitgen_steps_per_uniform)

        # advance_possible 初始策略
        if self.rng_consumption_variable:
            self.advance_possible = False
        else:
            supports = all(_bitgen_supports_advance(rep.rng) for rep in self.replicas)
            self.advance_possible = (self.rng_stride_per_sweep is not None) and supports

        # 交换阶段随机数消费计数（单位：uniform_draws）
        self._swap_rng_consumed_total: int = 0
        self._swap_rng_consumed_per_slot: List[int] = [0 for _ in range(nrep)]

        # 自适应 thin 的全局冷却期（以步数计）
        self._thin_global_last_update_step: int = -10**9

    # -------------------------
    # 核心：单次 sweep / 全体 sweep / swap
    # -------------------------
    def _sweep_one(self, rep_idx: int) -> None:
        """对单个 replica 做一次更新，使用 slot-bound 的 rep.rng。"""
        try:
            if self._alg is None:
                self.warnings.append(f"_sweep_one: algorithm '{self.algorithm}' unavailable; skip")
                return

            rep = self.replicas[rep_idx]
            latt_in = np.asarray(rep.lattice, dtype=np.int8)
            latt_out, meta = self._alg(latt_in, float(rep.beta), rep.rng, float(self.h))
            rep.lattice = np.asarray(latt_out, dtype=np.int8)
            if not rep.lattice.flags["C_CONTIGUOUS"]:
                rep.lattice = np.ascontiguousarray(rep.lattice)
            rep.energy_total = float(_energy_total_numpy_consistent(rep.lattice, self.h))

            try:
                if meta and isinstance(meta, dict) and "rng_consumed" in meta:
                    consumed = int(meta.get("rng_consumed", 0))
                    self.rng_offset_within_sweep[rep_idx] += consumed
            except Exception:
                pass

            try:
                self.sweep_counts[rep_idx] += 1
            except Exception:
                pass

        except Exception as exc:
            self.warnings.append(f"_sweep_one apply_move failed (seed={getattr(self.replicas[rep_idx], 'seed', None)}): {exc}")

    def _sweep_all(self) -> None:
        try:
            self.rng_offset_within_sweep = [0 for _ in range(len(self.replicas))]
        except Exception:
            self.rng_offset_within_sweep = [0 for _ in range(len(self.replicas))]

        for idx in range(len(self.replicas)):
            self._sweep_one(idx)

    def _attempt_swaps(self, parity: int = 0) -> None:
        """
        并行回火交换尝试。
        接受判据：Δ = (β_left - β_right) * (E_left - E_right)
                 接受概率 = min(1, exp(-Δ))；仅当 Δ>0 时抽一次 u（左槽 RNG）。
        """
        start = 0 if parity == 0 else 1
        nrep = len(self.replicas)
        for left in range(start, nrep - 1, 2):
            right = left + 1
            a = self.replicas[left]
            b = self.replicas[right]
            dbeta = a.beta - b.beta
            dE = a.energy_total - b.energy_total
            pair_idx = left
            self._swap_attempts[pair_idx] += 1

            accept = False
            try:
                Delta = dbeta * dE
                if Delta >= 0.0:
                    accept = True
                else:
                    try:
                        u = a.rng.random()
                        self._swap_rng_consumed_total += 1
                        self._swap_rng_consumed_per_slot[left] += 1
                    except Exception as exc_rng:
                        self.warnings.append(f"_attempt_swaps: RNG failure at slot {left}: {exc_rng}")
                        u = 1.0
                    # 指数参数钳位于 [-700, 0]
                    exp_arg = max(float(Delta), _MIN_EXP_ARG)
                    accept = (u < math.exp(exp_arg))
            except Exception as exc:
                self.warnings.append(f"_attempt_swaps encountered exception for pair ({left},{right}): {exc}")
                raise

            if accept:
                a_latt, a_E = a.lattice, a.energy_total
                b_latt, b_E = b.lattice, b.energy_total
                a.lattice, a.energy_total = b_latt, b_E
                b.lattice, b.energy_total = a_latt, a_E
                if not a.lattice.flags["C_CONTIGUOUS"]:
                    a.lattice = np.ascontiguousarray(a.lattice)
                if not b.lattice.flags["C_CONTIGUOUS"]:
                    b.lattice = np.ascontiguousarray(b.lattice)
                self._swap_accepts[pair_idx] += 1

            if self.record_swap_history and self._swap_history is not None:
                try:
                    self._swap_history[pair_idx].append(bool(accept))
                except Exception:
                    pass

    # -------------------------
    # 主运行循环：equilibration + production
    # -------------------------
    def run(
        self,
        equilibration_steps: int,
        production_steps: int,
        exchange_interval: int = 1,
        thin: int = 1,
        verbose: bool = False,
        save_lattices: bool = False,
        save_dir: Optional[str] = None,
        worker_id: Optional[str] = None,
        burn_in: Optional[int] = None,
        *,
        auto_thin: bool = True,
        thin_min: int = 1,
        thin_max: int = 10_000,
        tau_update_interval: Optional[int] = None,
        tau_window: int = 2048,
        unit_sanity_check: bool = True,
        compute_stats_in_run: bool = True,
    ) -> None:
        """
        运行 REMC 主循环。

        参数说明（新增）：
        - compute_stats_in_run: 是否在 run() 结束时自动调用一次 analyze()，
          并把每个温度的 C / χ / Binder U 以及交换率汇总写入 metadata JSON。
          对于非常长的模拟，如果担心 bootstrap 比较耗时，可以设为 False，
          然后在外部单独调用 sim.analyze()。
        """
        thin_local = int(max(1, thin if thin is not None else 1))

        _thin_k = int(tau_update_interval if tau_update_interval is not None else 256)
        _thin_W = int(max(256, tau_window))
        _tau_last: Dict[str, Optional[float]] = {f"T_{T:.6f}": None for T in self.temps}
        _thin_last_update_step: Dict[str, int] = {f"T_{T:.6f}": -10**9 for T in self.temps}

        # HDF5 文件准备（若需要保存格点快照）
        h5files: Dict[str, Tuple[Any, Any]] = {}
        sample_counters: Dict[str, int] = {}
        lattice_buffers: Dict[str, List[np.ndarray]] = {}
        if save_lattices:
            if save_dir is None:
                save_dir = "lattices"
            os.makedirs(save_dir, exist_ok=True)
            if worker_id is None:
                worker_id = f"worker_{os.getpid()}_{int(time.time())}"
            for T in self.temps:
                T_str = f"T_{T:.6f}"
                filename = os.path.join(save_dir, f"{worker_id}__latt_{T_str}_h{self.h:.6f}.h5")
                if os.path.exists(filename):
                    f = h5py.File(filename, "a")
                    if "lattices" in f:
                        dset = f["lattices"]
                        sample_counters[T_str] = dset.shape[0]
                    else:
                        chunk_shape = (1, self.L, self.L)
                        try:
                            dset = f.create_dataset(
                                "lattices",
                                shape=(0, self.L, self.L),
                                maxshape=(None, self.L, self.L),
                                dtype='i1',
                                chunks=chunk_shape,
                                compression='gzip', compression_opts=5, shuffle=True,
                            )
                        except TypeError:
                            dset = f.create_dataset("lattices", shape=(0, self.L, self.L),
                                                    maxshape=(None, self.L, self.L), dtype='i1', chunks=chunk_shape)
                        sample_counters[T_str] = 0
                else:
                    f = h5py.File(filename, "w")
                    chunk_shape = (1, self.L, self.L)
                    try:
                        dset = f.create_dataset(
                            "lattices",
                            shape=(0, self.L, self.L),
                            maxshape=(None, self.L, self.L),
                            dtype='i1',
                            chunks=chunk_shape,
                            compression='gzip', compression_opts=5, shuffle=True,
                        )
                    except TypeError:
                        dset = f.create_dataset("lattices", shape=(0, self.L, self.L),
                                                maxshape=(None, self.L, self.L), dtype='i1', chunks=chunk_shape)
                    sample_counters[T_str] = 0
                h5files[T_str] = (f, dset)
                lattice_buffers[T_str] = []

            # 轻量 metadata JSON（旁路）
            meta = {
                "L": int(self.L),
                "h": float(self.h),
                "temps": [float(T) for T in self.temps],
                "worker_id": worker_id,
                "replica_seeds": self._seed_info.get("replica_seeds"),
                "seed_bits": 32,
                "rng_versions": self._rng_versions,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "rng_binding": self._provenance_notes.get("rng_binding"),
                "rng_source": self._provenance_notes.get("rng_source"),
                "swap_rng_policy": self._provenance_notes.get("swap_rng_policy"),
                "rng_consumption_variable": bool(getattr(self, "rng_consumption_variable", False)),
                "python_version": sys.version,
                "platform": platform.platform(),
                "created_by": None,
                "swap_rng_consumed_total": int(self._swap_rng_consumed_total),
                "swap_rng_consumed_per_slot": list(self._swap_rng_consumed_per_slot),
                # 统一字段，便于 GPU 版对齐
                "algorithm": str(self.algorithm),
                "init_rng_decoupled": True,
            }
            try:
                meta["created_by"] = getpass.getuser()
            except Exception:
                meta["created_by"] = None

            try:
                meta_path = os.path.join(save_dir, f"{worker_id}__metadata.json")
                with open(meta_path, "w", encoding="utf-8") as mf:
                    json.dump(meta, mf, indent=2)
            except Exception:
                logger.debug("Failed to write lightweight meta JSON (non-fatal).")

        # Equilibration
        for t in range(int(equilibration_steps)):
            self._sweep_all()
            if (t + 1) % int(exchange_interval) == 0:
                self._attempt_swaps(parity=0)
                self._attempt_swaps(parity=1)
            self.sweep_index += 1
            if verbose and ((t + 1) % max(1, equilibration_steps // 5) == 0):
                print(f"[remc] equilibration progress: {t+1}/{equilibration_steps}")

        # Production
        series: Dict[str, Dict[str, List[float]]] = {}
        for T in self.temps:
            T_str = f"T_{T:.6f}"
            series[T_str] = {"E": [], "M": [], "absM": [], "M2": [], "M4": []}

        self._last_tau: Dict[str, Dict[str, float]] = {}

        step_counter = 0
        for t in range(int(production_steps)):
            self._sweep_all()
            if (t + 1) % int(exchange_interval) == 0:
                self._attempt_swaps(parity=0)
                self._attempt_swaps(parity=1)

            self.sweep_index += 1
            step_counter += 1

            if step_counter % thin_local == 0:
                for r, T in enumerate(self.temps):
                    T_str = f"T_{T:.6f}"
                    rep = self.replicas[r]

                    # —— 统一口径：能量用总能量 / Nsite；磁化由 ∑s/Nsite —— #
                    Nsite = float(rep.lattice.size)
                    m_tot = float(np.sum(rep.lattice, dtype=np.int64))
                    e_per_spin = float(rep.energy_total) / Nsite
                    m_per_spin = m_tot / Nsite
                    ob = {"E": e_per_spin, "M": m_per_spin,
                          "absM": abs(m_per_spin), "M2": m_per_spin**2, "M4": m_per_spin**4}

                    if unit_sanity_check:
                        try:
                            if abs(float(ob.get("M", 0.0))) > 1.0 + 1e-6:
                                self.warnings.append(f"[{T_str}] |m|>1 detected; check per-spin pipeline.")
                            if abs(float(ob.get("E", 0.0))) > (2.0 + abs(self.h) + 1e-3):
                                self.warnings.append(f"[{T_str}] |E| per spin > 2+|h|; check units.")
                        except Exception:
                            pass

                    for key in ("E", "M", "absM", "M2", "M4"):
                        series[T_str][key].append(float(ob.get(key, 0.0)))

                    if save_lattices:
                        latt_arr = rep.lattice
                        try:
                            latt_np = np.asarray(latt_arr, dtype=np.int8)
                            if not latt_np.flags["C_CONTIGUOUS"]:
                                latt_np = np.ascontiguousarray(latt_np)
                        except Exception as exc:
                            self.warnings.append(f"Failed to convert lattice to numpy at T={T_str}, skipping snapshot: {exc}")
                            continue
                        lattice_buffers[T_str].append(latt_np)

                        if len(lattice_buffers[T_str]) >= self.buffer_flush:
                            try:
                                f, dset = h5files[T_str]
                                cur = dset.shape[0]
                                nnew = len(lattice_buffers[T_str])
                                dset.resize(cur + nnew, axis=0)
                                dset[cur:cur + nnew, :, :] = np.stack(lattice_buffers[T_str], axis=0)
                                sample_counters[T_str] += nnew
                                lattice_buffers[T_str].clear()
                                try:
                                    self._safe_flush_and_sync(f)
                                except Exception:
                                    pass
                            except Exception as exc:
                                self.warnings.append(f"Failed to flush lattice buffer for {T_str}: {exc}")

            # —— 在线 τ_int → 自适应 thin —— #
            if auto_thin and ((t + 1) % _thin_k == 0):
                proposed: List[int] = []
                for r, T in enumerate(self.temps):
                    T_str = f"T_{T:.6f}"
                    m_ser = np.asarray(series[T_str]["absM"] or series[T_str]["M"], dtype=float)
                    if m_ser.size >= max(64, _thin_W // 4):
                        x = m_ser[-_thin_W:] if m_ser.size > _thin_W else m_ser
                        # 1) 在 thinned 序列上估计 τ_int（单位：样本 index）
                        try:
                            tau_samples = float(_tau_int_sokal(x))
                        except Exception:
                            tau_samples = 1.0

                        # 2) 把它还原成以 sweep 为单位的 τ_int
                        #    当前 thin_local 表示“每 thin_local 个 sweep 取一个样本”
                        #    所以 τ_sweeps ≈ tau_samples * thin_local
                        tau_sweeps = max(1.0, tau_samples * float(thin_local))

                        # 3) ESS 用的是样本单位的 τ（这是对“已有样本序列”的有效独立数）
                        ess = float(_ess_from_series(x, tau=tau_samples))

                        prev = _tau_last[T_str]
                        can_update = ((t + 1) - _thin_last_update_step[T_str]) >= (5 * _thin_k)
                        # 这里我们用“以 sweep 为单位的 τ”来比较是否变化较大
                        should_update = (prev is None) or (
                            abs(tau_sweeps - prev) / max(prev or 1.0, 1e-9) > 0.25
            )

                        # 4) 新的 thin 用 τ_sweeps 来算 —— 单位已经是 sweep
                        new_thin = int(
                            min(
                                thin_max,
                                max(thin_min, math.ceil(2.0 * float(tau_sweeps))),
                            )
                        )
                        proposed.append(new_thin)

                        if can_update and should_update:
                            _tau_last[T_str] = float(tau_sweeps)  # 这里记的是“每多少个 sweep 相关”
                            _thin_last_update_step[T_str] = (t + 1)

                        # 记录详细信息，方便在 metadata 里检查
                        self._last_tau[T_str] = {
                            "tau_int_samples": float(tau_samples),   # 单位：样本 index
                            "tau_int_sweeps": float(tau_sweeps),     # 单位：sweeps
                            "ESS": float(ess),
                            "thin": int(new_thin),
                        }


                if proposed:
                    new_global = max(proposed)
                    if ((t + 1) - self._thin_global_last_update_step) >= (5 * _thin_k):
                        if abs(new_global - thin_local) / max(1, thin_local) > 0.25:
                            thin_local = new_global
                            self._thin_global_last_update_step = (t + 1)

        # finalize results（E/M 时间序列）
        self._results = {T_str: {k: np.asarray(v, dtype=float) for k, v in series[T_str].items()} for T_str in series}
        self.final_lattices = [np.asarray(r.lattice, dtype=np.int8).copy() for r in self.replicas]

        # 预先在 run 结束时做一次物理量分析，供 JSON 元数据使用
        analysis_summary_for_json: Optional[Dict[str, Any]] = None
        if save_lattices and compute_stats_in_run:
            try:
                stats_all = self.analyze(verbose=False)
                thermo_stats: Dict[str, Dict[str, Any]] = {}
                for T_str, vals in stats_all.items():
                    if not (isinstance(T_str, str) and T_str.startswith("T_")):
                        continue
                    if not isinstance(vals, dict):
                        continue
                    try:
                        T_val = float(T_str.replace("T_", ""))
                    except Exception:
                        T_val = None
                    entry = {
                        "T": T_val,
                        "C": float(vals.get("C", 0.0)),
                        "C_err": float(vals.get("C_err", 0.0)),
                        "chi": float(vals.get("chi", 0.0)),
                        "chi_err": float(vals.get("chi_err", 0.0)),
                        "U": float(vals.get("U", 0.0)),
                        "n_samples": int(vals.get("n_samples", 0)),
                    }
                    thermo_stats[T_str] = entry
                swap_summary = None
                swap_block = stats_all.get("swap", None)
                if isinstance(swap_block, dict):
                    try:
                        swap_summary = {
                            "attempt": int(swap_block.get("attempt", 0)),
                            "accept": int(swap_block.get("accept", 0)),
                            "rate": float(swap_block.get("rate", 0.0)),
                            "pair_rates": list(swap_block.get("pair_rates", [])),
                        }
                    except Exception:
                        swap_summary = None
                analysis_summary_for_json = {"thermo_stats": thermo_stats}
                if swap_summary is not None:
                    analysis_summary_for_json["swap_summary"] = swap_summary
            except Exception as exc:
                self.warnings.append(f"run(): internal analyze() for JSON stats failed: {exc}")
                analysis_summary_for_json = None

        # flush 剩余 buffers 并写 provenance
        if save_lattices:
            lattice_paths: Dict[str, str] = {}
            for T_str, (f, dset) in h5files.items():
                buf = lattice_buffers.get(T_str, [])
                if buf:
                    try:
                        cur = dset.shape[0]
                        nnew = len(buf)
                        dset.resize(cur + nnew, axis=0)
                        dset[cur:cur + nnew, :, :] = np.stack(buf, axis=0)
                        sample_counters[T_str] += nnew
                        lattice_buffers[T_str].clear()
                    except Exception as exc:
                        self.warnings.append(f"Final flush failed for {T_str}: {exc}")

                try:
                    if "provenance" in f and isinstance(f["provenance"], h5py.Group):
                        prov = f["provenance"]
                    else:
                        prov = f.create_group("provenance")
                    prov.attrs["L"] = int(self.L)
                    prov.attrs["h"] = float(self.h)
                    try:
                        prov.attrs["T"] = float(T_str.replace("T_", ""))
                    except Exception:
                        prov.attrs["T"] = float("nan")
                    prov.attrs["samples_written"] = int(sample_counters.get(T_str, 0))
                    prov.attrs["worker_id"] = worker_id
                    try:
                        prov.attrs["replica_seeds"] = json.dumps(self._seed_info.get("replica_seeds"))
                    except Exception:
                        prov.attrs["replica_seeds"] = str(self._seed_info.get("replica_seeds"))
                    prov.attrs["seed_bits"] = 32
                    prov.attrs["rng_versions"] = json.dumps(self._rng_versions)
                    prov.attrs["sampler_type"] = str(self.algorithm)
                    prov.attrs["burn_in"] = int(burn_in) if burn_in is not None else -1
                    prov.attrs["thin"] = int(thin_local)
                    try:
                        prov.attrs["tau_reports_json"] = json.dumps(getattr(self, "_last_tau", {}))
                    except Exception:
                        pass
                    try:
                        prov.attrs["swap_attempts"] = json.dumps(self._swap_attempts.tolist())
                        prov.attrs["swap_accepts"] = json.dumps(self._swap_accepts.tolist())
                    except Exception:
                        try:
                            prov.attrs["swap_attempts"] = str(self._swap_attempts.tolist())
                            prov.attrs["swap_accepts"] = str(self._swap_accepts.tolist())
                        except Exception:
                            pass
                    prov.attrs["rng_binding"] = str(self._provenance_notes.get("rng_binding"))
                    prov.attrs["rng_source"] = str(self._provenance_notes.get("rng_source"))
                    prov.attrs["swap_rng_policy"] = str(self._provenance_notes.get("swap_rng_policy"))
                    prov.attrs["rng_consumption_variable"] = bool(self.rng_consumption_variable)
                    prov.attrs["swap_rng_consumed_total"] = int(self._swap_rng_consumed_total)
                    try:
                        prov.attrs["swap_rng_consumed_per_slot"] = json.dumps(self._swap_rng_consumed_per_slot)
                    except Exception:
                        prov.attrs["swap_rng_consumed_per_slot"] = str(self._swap_rng_consumed_per_slot)
                    try:
                        prov.attrs["rng_offset_within_sweep"] = json.dumps(self.rng_offset_within_sweep)
                        prov.attrs["rng_offset_within_sweep_unit"] = "meta_reported_units"
                    except Exception:
                        prov.attrs["rng_offset_within_sweep"] = str(self.rng_offset_within_sweep)
                        prov.attrs["rng_offset_within_sweep_unit"] = "meta_reported_units"
                    try:
                        bitgen_classes = [rep.rng.bit_generator.__class__.__name__ for rep in self.replicas]
                        prov.attrs["bitgen_classes"] = json.dumps(bitgen_classes)
                    except Exception:
                        try:
                            prov.attrs["bitgen_classes"] = str([rep.rng.bit_generator.__class__.__name__ for rep in self.replicas])
                        except Exception:
                            pass
                    try:
                        prov.attrs["python_version"] = sys.version
                        prov.attrs["platform"] = platform.platform()
                    except Exception:
                        pass

                    # 在 HDF5 中写入 per-spin 的 E / M 序列
                    try:
                        if self._results is not None and T_str in self._results:
                            E_arr = np.asarray(self._results[T_str].get("E", []), dtype=np.float64)
                            M_arr = np.asarray(self._results[T_str].get("M", []), dtype=np.float64)

                            # 简单一致性检查（非致命）
                            try:
                                n_latt = int(sample_counters.get(T_str, 0))
                                if n_latt not in (0, E_arr.size) and E_arr.size > 0:
                                    self.warnings.append(
                                        f"[{T_str}] samples_written={n_latt} vs len(E)={E_arr.size}; "
                                        "E/M length mismatch with lattices."
                                    )
                            except Exception:
                                pass

                            for name, arr in (("E", E_arr), ("M", M_arr)):
                                if arr.size == 0:
                                    continue
                                if name in f:
                                    try:
                                        del f[name]
                                    except Exception:
                                        pass
                                try:
                                    # 与 lattices 一样启用压缩（如不可用则退化）
                                    try:
                                        f.create_dataset(
                                            name,
                                            data=arr,
                                            dtype="f8",
                                            compression="gzip",
                                            compression_opts=5,
                                            shuffle=True,
                                        )
                                    except TypeError:
                                        f.create_dataset(name, data=arr, dtype="f8")
                                except Exception as exc_em:
                                    self.warnings.append(f"Failed to write dataset '{name}' for {T_str}: {exc_em}")
                    except Exception as exc:
                        self.warnings.append(f"Failed to write E/M observables into HDF5 for {T_str}: {exc}")

                    try:
                        self._safe_flush_and_sync(f)
                    except Exception:
                        pass
                except Exception as exc:
                    self.warnings.append(f"Failed to write provenance into HDF5 for {T_str}: {exc}")

                try:
                    filename = f.filename
                except Exception:
                    filename = os.path.join(save_dir, f"{worker_id}__{T_str}.h5")
                try:
                    f.close()
                except Exception:
                    pass
                lattice_paths[T_str] = filename

            # 更新 metadata 文件
            meta_update = {
                "samples_per_temp": sample_counters,
                "final_seeds": self._seed_info,
                "swap_attempts": self._swap_attempts.tolist(),
                "swap_accepts": self._swap_accepts.tolist(),
                "rng_versions": self._rng_versions,
                "rng_binding": self._provenance_notes.get("rng_binding"),
                "rng_source": self._provenance_notes.get("rng_source"),
                "swap_rng_policy": self._provenance_notes.get("swap_rng_policy"),
                "thin": int(thin_local),
                "rng_consumption_variable": bool(self.rng_consumption_variable),
                "swap_rng_consumed_total": int(self._swap_rng_consumed_total),
                "swap_rng_consumed_per_slot": list(self._swap_rng_consumed_per_slot),
                "algorithm": str(self.algorithm),
                "init_rng_decoupled": True,
            }
            # 将 thermodynamic 统计结果与交换汇总一并写入 JSON（如果启用了 compute_stats_in_run）
            if analysis_summary_for_json is not None:
                try:
                    meta_update.update(analysis_summary_for_json)
                except Exception:
                    pass

            try:
                meta_path = os.path.join(save_dir, f"{worker_id}__metadata.json")
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as mf:
                            try:
                                old = json.load(mf)
                            except Exception:
                                old = {}
                    except Exception:
                        old = {}
                    old.update(meta_update)
                    tmp_meta = meta_path + ".tmp"
                    with open(tmp_meta, "w", encoding="utf-8") as mf:
                        json.dump(old, mf, indent=2)
                        mf.flush()
                        try:
                            os.fsync(mf.fileno())
                        except Exception:
                            pass
                    os.replace(tmp_meta, meta_path)
                else:
                    tmp_meta = meta_path + ".tmp"
                    with open(tmp_meta, "w", encoding="utf-8") as mf:
                        json.dump(meta_update, mf, indent=2)
                        mf.flush()
                        try:
                            os.fsync(mf.fileno())
                        except Exception:
                            pass
                    os.replace(tmp_meta, meta_path)
            except Exception:
                pass
            self._lattice_files = lattice_paths

    # -------------------------
    # 安全 flush / fsync
    # -------------------------
    def _safe_flush_and_sync(self, fobj: h5py.File):
        try:
            fobj.flush()
            try:
                os.fsync(fobj.fileno())
            except Exception:
                try:
                    os.fsync(fobj.id.get_vfd_handle())  # type: ignore
                except Exception:
                    pass
        except Exception:
            pass

    # -------------------------
    # Checkpoint 保存 / 恢复（含物理态）
    # -------------------------
    def save_checkpoint(
        self,
        path: str,
        group_name: str = "remc_checkpoint",
        rng_stride_per_sweep: Optional[int] = None,
        advance_possible: Optional[bool] = None,
        max_state_bytes: int = 1_000_000
    ) -> Dict[str, Any]:
        result = {"ok": False, "saved_states": False, "group": group_name, "reason": ""}

        try:
            f = h5py.File(path, "a")
        except Exception as exc:
            raise RuntimeError(f"save_checkpoint: cannot open h5 file {path}: {exc}") from exc

        if group_name in f:
            try:
                del f[group_name]
            except Exception:
                pass
        grp = f.create_group(group_name)

        # 基本 attrs
        grp.attrs["L"] = int(self.L)
        grp.attrs["h"] = float(self.h)
        grp.attrs["temps"] = json.dumps([float(T) for T in self.temps])
        grp.attrs["algorithm"] = str(self.algorithm)

        grp.attrs["sweep_index"] = int(getattr(self, "sweep_index", 0))
        grp.attrs["sweep_counts"] = json.dumps([int(x) for x in self.sweep_counts])

        try:
            grp.attrs["replica_seeds"] = json.dumps(self._seed_info.get("replica_seeds"))
        except Exception:
            grp.attrs["replica_seeds"] = str(self._seed_info.get("replica_seeds"))
        grp.attrs["seed_bits"] = 32
        grp.attrs["rng_versions"] = json.dumps(self._rng_versions)
        grp.attrs["rng_binding"] = str(self._provenance_notes.get("rng_binding"))
        grp.attrs["rng_source"] = str(self._provenance_notes.get("rng_source"))
        grp.attrs["swap_rng_policy"] = str(self._provenance_notes.get("swap_rng_policy"))
        # 初始化随机流解耦标记（用于恢复时判定 advance 可行性）
        grp.attrs["init_rng_decoupled"] = True

        try:
            grp.attrs["swap_rng_consumed_total"] = int(self._swap_rng_consumed_total)
            grp.attrs["swap_rng_consumed_per_slot"] = json.dumps(self._swap_rng_consumed_per_slot)
            grp.attrs["swap_rng_unit"] = "uniform_draws"
        except Exception:
            grp.attrs["swap_rng_consumed_per_slot"] = str(self._swap_rng_consumed_per_slot)

        if getattr(self, "bitgen_steps_per_uniform", None) is not None:
            try:
                grp.attrs["bitgen_steps_per_uniform"] = int(self.bitgen_steps_per_uniform)
            except Exception:
                grp.attrs["bitgen_steps_per_uniform"] = str(self.bitgen_steps_per_uniform)
        elif self._swap_rng_consumed_total > 0:
            grp.attrs["advance_possible"] = False
            grp.attrs["advance_disabled_reason"] = "swap_rng_present_without_conversion_factor"

        if advance_possible is None:
            adv_flag = bool(getattr(self, "advance_possible", False))
        else:
            adv_flag = bool(advance_possible)

        if getattr(self, "rng_consumption_variable", False):
            grp.attrs["advance_possible"] = False
            grp.attrs["rng_consumption_variable"] = True
            try:
                grp.attrs["rng_offset_within_sweep"] = json.dumps(self.rng_offset_within_sweep)
                grp.attrs["rng_offset_within_sweep_unit"] = "meta_reported_units"
            except Exception:
                grp.attrs["rng_offset_within_sweep"] = str(self.rng_offset_within_sweep)
                grp.attrs["rng_offset_within_sweep_unit"] = "meta_reported_units"
        else:
            stride_to_write = int(rng_stride_per_sweep) if rng_stride_per_sweep is not None else self.rng_stride_per_sweep
            if adv_flag and stride_to_write is None:
                warnings_msg = "save_checkpoint: advance_possible=True but rng_stride_per_sweep is None; disabling advance in saved checkpoint"
                logger.warning(warnings_msg)
                self.warnings.append(warnings_msg)
                grp.attrs["advance_possible"] = False
            else:
                grp.attrs["advance_possible"] = bool(adv_flag)
                if stride_to_write is not None:
                    grp.attrs["rng_stride_per_sweep"] = int(stride_to_write)
            grp.attrs["rng_stride_per_sweep_unit"] = "bitgen_steps"
            try:
                grp.attrs["rng_offset_within_sweep"] = json.dumps(self.rng_offset_within_sweep)
                grp.attrs["rng_offset_within_sweep_unit"] = "meta_reported_units"
            except Exception:
                grp.attrs["rng_offset_within_sweep"] = str(self.rng_offset_within_sweep)
                grp.attrs["rng_offset_within_sweep_unit"] = "meta_reported_units"

        try:
            bitgen_classes = [rep.rng.bit_generator.__class__.__name__ for rep in self.replicas]
            grp.attrs["bitgen_classes"] = json.dumps(bitgen_classes)
        except Exception:
            try:
                grp.attrs["bitgen_classes"] = str([getattr(rep.rng.bit_generator, "__class__", None).__name__ for rep in self.replicas])
            except Exception:
                pass

        # 保存每个副本的 lattice 与 energy_total（关键：物理态）
        try:
            latt_stack = np.stack([rep.lattice for rep in self.replicas], axis=0).astype(np.int8, copy=False)
            if not latt_stack.flags["C_CONTIGUOUS"]:
                latt_stack = np.ascontiguousarray(latt_stack)
            grp.create_dataset(
                "lattices", data=latt_stack, dtype='i1',
                compression='gzip', compression_opts=5, shuffle=True
            )
            grp.create_dataset(
                "energies", data=np.asarray([rep.energy_total for rep in self.replicas], dtype=np.float64)
            )
        except Exception as exc:
            self.warnings.append(f"save_checkpoint: failed to write lattices/energies: {exc}")

        # 序列化每个 replica 的 bit_generator.state
        rng_states_serial = []
        saved_states = False
        rng_state_trimmed_any = False
        trim_keys = []
        try:
            for rep in self.replicas:
                try:
                    st = rep.rng.bit_generator.state
                    obj = self._serialize_rng_state(st)
                    txt = json.dumps(obj)
                    if len(txt.encode("utf-8")) > max_state_bytes:
                        if isinstance(obj, dict):
                            small = {}
                            for k in ("state", "key", "counter", "status", "pos"):
                                if k in obj:
                                    small[k] = obj[k]
                                    if k not in trim_keys:
                                        trim_keys.append(k)
                            if not small:
                                small = {"note": "state_too_large", "len_bytes": len(txt.encode("utf-8"))}
                                rng_state_trimmed_any = True
                            else:
                                rng_state_trimmed_any = True
                            obj = small
                        else:
                            obj = {"note": "state_too_large", "len_bytes": len(txt.encode("utf-8"))}
                            rng_state_trimmed_any = True
                        txt = json.dumps(obj)
                        self.warnings.append(f"save_checkpoint: RNG state for seed {rep.seed} trimmed due to size")
                    rng_states_serial.append(txt)
                except Exception as exc:
                    rng_states_serial.append(json.dumps({"error": str(exc)}))
            dt = h5py.string_dtype(encoding='utf-8')
            ds = grp.create_dataset("rng_states", shape=(len(rng_states_serial),), dtype=dt)
            ds[:] = rng_states_serial
            grp.attrs["rng_states_format"] = "json_str_per_item_v1"
            grp.attrs["saved_rng_states_version"] = 1
            saved_states = True
        except Exception:
            try:
                grp.attrs["rng_states_json"] = json.dumps(rng_states_serial)
                grp.attrs["rng_states_format"] = "json_str_per_item_v1"
                grp.attrs["saved_rng_states_version"] = 1
                saved_states = True
            except Exception:
                try:
                    grp.attrs["rng_states_json"] = str(rng_states_serial)[:4096]
                    saved_states = False
                except Exception:
                    saved_states = False
        grp.attrs["saved_rng_states"] = bool(saved_states)

        if rng_state_trimmed_any:
            grp.attrs["rng_state_trimmed"] = True
            try:
                grp.attrs["rng_state_trim_keys"] = json.dumps(trim_keys)
            except Exception:
                grp.attrs["rng_state_trim_keys"] = str(trim_keys)
            grp.attrs["rng_state_trim_reason"] = "exceeded_max_state_bytes"
            grp.attrs["advance_possible"] = False
            grp.attrs["advance_disabled_reason"] = "rng_state_trimmed"

        try:
            grp.create_dataset("swap_attempts", data=self._swap_attempts.astype(np.int64))
            grp.create_dataset("swap_accepts", data=self._swap_accepts.astype(np.int64))
        except Exception:
            try:
                grp.attrs["swap_attempts"] = json.dumps(self._swap_attempts.tolist())
                grp.attrs["swap_accepts"] = json.dumps(self._swap_accepts.tolist())
            except Exception:
                pass

        try:
            self._safe_flush_and_sync(f)
        except Exception:
            pass
        try:
            f.close()
        except Exception:
            pass

        result["saved_states"] = bool(saved_states)
        result["ok"] = True
        return result

    def restore_from_checkpoint(
        self,
        path: str,
        group_name: str = "remc_checkpoint",
        rng_stride_per_sweep: Optional[int] = None,
        bitgen_steps_per_uniform: Optional[int] = None,
        max_advance_pretest_time_s: float = 60.0
    ) -> Dict[str, Any]:
        result = {"method": "none", "ok": False, "sweep_index": None, "reason": ""}

        if not os.path.exists(path):
            msg = f"restore_from_checkpoint: file not found: {path}"
            self.warnings.append(msg)
            result["reason"] = msg
            return result

        try:
            f = h5py.File(path, "r")
        except Exception as exc:
            msg = f"restore_from_checkpoint: cannot open h5 file {path}: {exc}"
            self.warnings.append(msg)
            result["reason"] = msg
            return result

        if group_name not in f:
            msg = f"restore_from_checkpoint: group '{group_name}' not found in {path}"
            self.warnings.append(msg)
            try:
                f.close()
            except Exception:
                pass
            result["reason"] = msg
            return result

        grp = f[group_name]

        # --- 健壮解码 + 强校验：L/h/temps/algorithm/replica_seeds ---
        L_file = int(grp.attrs.get("L", self.L))
        if L_file != self.L:
            raise ValueError(f"Checkpoint L={L_file} mismatches instance L={self.L}")

        h_file = float(grp.attrs.get("h", self.h))
        if abs(h_file - self.h) > 1e-12:
            raise ValueError(f"Checkpoint h={h_file} mismatches instance h={self.h}")

        temps_raw = grp.attrs.get("temps", None)
        if temps_raw is not None:
            ttxt = _as_text(temps_raw)
            try:
                temps_file = json.loads(ttxt)
            except Exception:
                temps_file = ast.literal_eval(ttxt)
            tfile = list(map(float, temps_file))
            if len(tfile) != len(self.temps) or any(abs(a-b) > 1e-12 for a,b in zip(tfile, self.temps)):
                raise ValueError("Checkpoint temps mismatch current simulator temps")

        alg_raw = grp.attrs.get("algorithm", self.algorithm)
        alg_file = _as_text(alg_raw)
        if str(alg_file) != str(self.algorithm):
            raise ValueError(f"Checkpoint algorithm='{alg_file}' mismatches current '{self.algorithm}'")

        # RNG 版本一致性提示（非阻断）
        try:
            rv_raw = grp.attrs.get("rng_versions", "{}")
            rv_txt = _as_text(rv_raw)
            saved_versions = json.loads(rv_txt)
            if saved_versions.get("numpy") and saved_versions["numpy"] != np.__version__:
                self.warnings.append(f"RNG version mismatch: file numpy={saved_versions['numpy']} vs runtime {np.__version__}")
        except Exception:
            pass

        # 解析并读取 replica_seeds（以 checkpoint 为真，必要时覆盖实例）
        file_replica_seeds = None
        try:
            seeds_raw = grp.attrs.get("replica_seeds", None)
            if isinstance(seeds_raw, (bytes, bytearray, np.bytes_)):
                seeds_raw = seeds_raw.decode("utf-8")
            file_replica_seeds = json.loads(seeds_raw) if isinstance(seeds_raw, str) else None
        except Exception:
            file_replica_seeds = None

        if file_replica_seeds is not None:
            file_seeds_list = [int(x) for x in file_replica_seeds]
            # 长度必须匹配；否则认为 checkpoint / 实例不兼容
            if len(file_seeds_list) != len(self.replica_seeds):
                raise ValueError(
                    f"Checkpoint replica_seeds length ({len(file_seeds_list)}) "
                    f"mismatches current simulator seeds length ({len(self.replica_seeds)})"
                )
            if file_seeds_list != self.replica_seeds:
                # 关键改动：不再报错，而是用 checkpoint 里的种子覆盖实例
                msg = (
                    "restore_from_checkpoint: replica_seeds in checkpoint differ from "
                    "current simulator seeds; overriding instance seeds with checkpoint seeds."
                )
                self.warnings.append(msg)
                logger.warning(msg)
                self.replica_seeds = file_seeds_list
                # 更新 seed_info 与每个 replica.seed（用于之后 advance / 重建 RNG）
                self._seed_info["replica_seeds"] = self.replica_seeds
                for rep, s in zip(self.replicas, self.replica_seeds):
                    rep.seed = int(s)

        # sweep_index / per-replica counts
        sweep_index = int(grp.attrs.get("sweep_index", 0))
        result["sweep_index"] = sweep_index
        try:
            sc_raw = grp.attrs.get("sweep_counts", None)
            if sc_raw is not None:
                sc_txt = _as_text(sc_raw)
                sc_list = json.loads(sc_txt)
                if isinstance(sc_list, list) and len(sc_list) == len(self.replicas):
                    self.sweep_counts = [int(x) for x in sc_list]
        except Exception:
            pass

        # stride/advance flags
        file_advance_possible = bool(grp.attrs.get("advance_possible", False))
        file_stride = None
        try:
            if "rng_stride_per_sweep" in grp.attrs:
                file_stride = int(grp.attrs.get("rng_stride_per_sweep"))
        except Exception:
            file_stride = None
        if rng_stride_per_sweep is not None:
            file_stride = int(rng_stride_per_sweep)

        # bitgen_steps_per_uniform
        file_bitgen_steps_per_uniform = None
        try:
            if bitgen_steps_per_uniform is not None:
                file_bitgen_steps_per_uniform = int(bitgen_steps_per_uniform)
            else:
                attr = grp.attrs.get("bitgen_steps_per_uniform", None)
                if attr is not None:
                    at = _as_text(attr)
                    try:
                        file_bitgen_steps_per_uniform = int(at)
                    except Exception:
                        try:
                            file_bitgen_steps_per_uniform = int(json.loads(str(at)))
                        except Exception:
                            file_bitgen_steps_per_uniform = None
        except Exception:
            file_bitgen_steps_per_uniform = None

        # ★ 修复点：把文件中的 stride / conversion 因子回写到实例，便于恢复后继续运行/再次存档
        if file_stride is not None:
            self.rng_stride_per_sweep = int(file_stride)
        if file_bitgen_steps_per_uniform is not None:
            self.bitgen_steps_per_uniform = int(file_bitgen_steps_per_uniform)

        # 读取 bit-generator 类名序列（用于 advance 精确构造）
        bitgen_classes = None
        try:
            raw = grp.attrs.get("bitgen_classes", None)
            if raw is not None:
                bitgen_classes = json.loads(_as_text(raw))
        except Exception:
            bitgen_classes = None

        # swap RNG 消耗（单位：uniform_draws）
        per_slot_swap_counts: List[int] = [0 for _ in range(len(self.replica_seeds))]
        try:
            raw = grp.attrs.get("swap_rng_consumed_per_slot", None)
            raw = _as_text(raw)
            if raw is not None:
                per_slot_swap_counts = list(json.loads(raw))
        except Exception:
            per_slot_swap_counts = [0 for _ in range(len(self.replica_seeds))]
        swap_unit = _as_text(grp.attrs.get("swap_rng_unit", "uniform_draws"))
        if swap_unit != "uniform_draws":
            # 未知单位时，保守禁用 advance
            file_advance_possible = False
            result["reason"] = "unknown_swap_rng_unit"

        # ★ 修复点：将 swap 随机数消耗计数同步回实例（总计也同步）
        try:
            self._swap_rng_consumed_per_slot = [int(x) for x in per_slot_swap_counts]
            self._swap_rng_consumed_total = int(sum(self._swap_rng_consumed_per_slot))
        except Exception:
            pass

        # variable RNG 消费
        file_rng_var = bool(grp.attrs.get("rng_consumption_variable", False))
        if file_rng_var:
            self.warnings.append("restore_from_checkpoint: file indicates variable RNG consumption; skipping advance-based restore")
            file_advance_possible = False

        # 初始化随机流是否解耦
        init_decoupled = bool(grp.attrs.get("init_rng_decoupled", False))
        if not init_decoupled:
            # 旧存档可能未解耦，advance 难以精确对齐；优先保守禁用。
            file_advance_possible = False
            result["reason"] = "init_rng_not_decoupled"

        # --- 先恢复物理态：lattices/energies（并做形状&连续性检查） ---
        if "lattices" not in grp:
            msg = "restore_from_checkpoint: lattices dataset missing"
            self.warnings.append(msg)
            result["reason"] = msg
            try:
                f.close()
            except Exception:
                pass
            return result
        latt = grp["lattices"][()]
        expect_shape = (len(self.replicas), self.L, self.L)
        if tuple(latt.shape) != expect_shape:
            raise ValueError(f"Checkpoint lattices shape {latt.shape} mismatches {expect_shape}")
        energies = None
        if "energies" in grp:
            energies = grp["energies"][()].astype(np.float64).ravel()
            if energies.shape[0] != len(self.replicas):
                raise ValueError(f"Checkpoint energies length {energies.shape[0]} mismatches replicas {len(self.replicas)}")

        for i, rep in enumerate(self.replicas):
            rep.lattice = np.asarray(latt[i], dtype=np.int8)
            if not rep.lattice.flags["C_CONTIGUOUS"]:
                rep.lattice = np.ascontiguousarray(rep.lattice)
            rep.energy_total = float(energies[i] if energies is not None else _energy_total_numpy_consistent(rep.lattice, self.h))

        # --- advance 预检（在物理态恢复后进行） ---
        restored_via_advance = False
        advance_pretest_meta: Dict[str, Any] = {"per_replica": [], "total_time_s": 0.0, "total_chunks": 0, "ok": False, "notes": []}

        if file_advance_possible and (file_stride is not None):
            # 必须能够按文件类名构造 RNG；缺失类名时保守放弃 advance
            if not (isinstance(bitgen_classes, list) and len(bitgen_classes) >= len(self.replicas)):
                file_advance_possible = False
                advance_pretest_meta["notes"].append("missing_or_invalid_bitgen_classes")

        if file_advance_possible and (file_stride is not None):
            tmp_gens = []
            tmp_ok = True
            total_time = 0.0
            total_chunks = 0
            total_advances = []
            try:
                for i, seed in enumerate(self.replica_seeds):
                    # 优先逐副本 sweep_counts（若缺失则用全局 sweep_index）
                    cnt_i = int(self.sweep_counts[i]) if i < len(self.sweep_counts) else int(sweep_index)

                    # move offset 与 swap 消耗换算（需要 conversion 因子）
                    offset_i = 0
                    per_replica_offsets = None
                    try:
                        off_raw = grp.attrs.get("rng_offset_within_sweep", None)
                        if off_raw is not None:
                            off_txt = _as_text(off_raw)
                            try:
                                per_replica_offsets = json.loads(off_txt)
                            except Exception:
                                per_replica_offsets = ast.literal_eval(off_txt)
                        if per_replica_offsets is not None:
                            offset_i = int(per_replica_offsets[i])
                    except Exception:
                        offset_i = 0

                    offset_bits = 0
                    swap_bits = 0
                    if file_bitgen_steps_per_uniform is not None:
                        try:
                            offset_bits = int(offset_i) * int(file_bitgen_steps_per_uniform)
                        except Exception:
                            offset_bits = 0
                            advance_pretest_meta["notes"].append(f"replica_{i}_offset_conversion_failed")
                        try:
                            if i < len(per_slot_swap_counts):
                                swap_bits = int(per_slot_swap_counts[i]) * int(file_bitgen_steps_per_uniform)
                        except Exception:
                            swap_bits = 0
                            advance_pretest_meta["notes"].append(f"replica_{i}_swap_conversion_failed")
                    else:
                        if sum(per_slot_swap_counts) > 0:
                            tmp_ok = False
                            advance_pretest_meta["notes"].append("swap_present_but_no_conversion_factor;abort_advance")
                            break

                    total_advance_i = int(cnt_i) * int(file_stride) + int(offset_bits) + int(swap_bits)
                    total_advances.append(total_advance_i)

                    # —— 按文件记录的 bitgen 类构造 —— #
                    try:
                        cls = bitgen_classes[i]
                        tmp = _make_generator_from_seed_and_class(seed, cls)
                    except Exception as exc:
                        tmp_ok = False
                        advance_pretest_meta["per_replica"].append({"seed": seed, "ok": False, "error": f"bitgen_class_unsupported: {exc}"})
                        self.warnings.append(f"advance test failed for seed {seed}: unsupported bitgen class ({exc})")
                        break

                    bg = tmp.bit_generator
                    if not hasattr(bg, "advance"):
                        tmp_ok = False
                        advance_pretest_meta["per_replica"].append({"seed": seed, "ok": False, "error": "no_advance_support"})
                        self.warnings.append(f"restore_from_checkpoint: bitgen for seed {seed} lacks advance(), aborting advance pretest")
                        break
                    try:
                        adv_meta = _advance_bitgen(bg, total_advance_i)
                        total_time += float(adv_meta.get("time_s", 0.0))
                        total_chunks += int(adv_meta.get("chunks", 0))
                        advance_pretest_meta["per_replica"].append({"seed": seed, "ok": bool(adv_meta.get("ok", False)), "meta": adv_meta, "advance_steps": total_advance_i})
                        if not adv_meta.get("ok", False):
                            tmp_ok = False
                            break
                        tmp_gens.append(tmp)
                    except Exception as exc:
                        tmp_ok = False
                        advance_pretest_meta["per_replica"].append({"seed": seed, "ok": False, "error": str(exc), "advance_steps": total_advance_i})
                        self.warnings.append(f"advance test failed for seed {seed}: {exc}")
                        logger.warning("advance test failed for seed %s: %s", seed, exc)
                        break
            except Exception as exc:
                tmp_ok = False
                advance_pretest_meta.setdefault("error", str(exc))
                self.warnings.append(f"advance pre-test exception: {exc}")
                logger.warning("advance pre-test exception: %s", exc)

            advance_pretest_meta["total_time_s"] = float(total_time)
            advance_pretest_meta["total_chunks"] = int(total_chunks)
            advance_pretest_meta["per_replica_advances"] = total_advances
            advance_pretest_meta["ok"] = bool(tmp_ok)

            if advance_pretest_meta["total_time_s"] > float(max_advance_pretest_time_s):
                tmp_ok = False
                advance_pretest_meta["ok"] = False
                advance_pretest_meta["notes"].append(f"pretest_time_exceeded_{max_advance_pretest_time_s}s")
                self.warnings.append(f"advance pretest exceeded max time {max_advance_pretest_time_s}s; falling back to state-based restore")

            if tmp_ok and len(tmp_gens) == len(self.replicas):
                for rep, tmp in zip(self.replicas, tmp_gens):
                    rep.rng = tmp
                self.sweep_index = sweep_index
                try:
                    self.sweep_counts = [int(cnt) for cnt in self.sweep_counts]  # 已从文件读取
                except Exception:
                    pass
                restored_via_advance = True
                result["method"] = "advance"
                result["ok"] = True
                result["reason"] = "advanced_generators_replaced"
                result["advance_pretest"] = advance_pretest_meta
            else:
                restored_via_advance = False
                result["advance_pretest"] = advance_pretest_meta

        # --- 若 advance 失败/不可行：state-based 恢复 RNG ---
        if not restored_via_advance:
            rng_states_list = self._read_rng_states_from_h5(grp)
            if rng_states_list is None:
                msg = "restore_from_checkpoint: no usable rng_states found and advance restore not possible"
                self.warnings.append(msg)
                result["reason"] = msg
                try:
                    f.close()
                except Exception:
                    pass
                return result

            applied_all = True
            failed_indices = []
            for idx, rep in enumerate(self.replicas):
                try:
                    item = rng_states_list[idx]
                except Exception:
                    item = None
                if item is None:
                    applied_all = False
                    failed_indices.append(idx)
                    self.warnings.append(f"restore_from_checkpoint: missing rng state for replica {idx}")
                    continue
                try:
                    if isinstance(item, str):
                        parsed = json.loads(item)
                    else:
                        parsed = item
                except Exception:
                    try:
                        parsed = ast.literal_eval(str(item))
                    except Exception:
                        parsed = item
                try:
                    state_obj = self._deserialize_rng_state(parsed)

                    # ★ 修复点：若档案记录的 bitgen 类与当前不同，先重建同类 RNG 再赋 state
                    try:
                        if isinstance(bitgen_classes, list) and len(bitgen_classes) > idx:
                            want = str(bitgen_classes[idx])
                            have = rep.rng.bit_generator.__class__.__name__
                            if want != have:
                                rep.rng = _make_generator_from_seed_and_class(rep.seed, want)
                    except Exception:
                        pass

                    try:
                        rep.rng.bit_generator.state = state_obj
                    except Exception as exc_assign:
                        applied_all = False
                        failed_indices.append(idx)
                        self.warnings.append(f"restore_from_checkpoint: failed to assign rng.state for replica {idx}: {exc_assign}")
                        logger.warning("Failed to assign rng.state for replica %d: %s", idx, exc_assign)
                except Exception as exc:
                    applied_all = False
                    failed_indices.append(idx)
                    self.warnings.append(f"restore_from_checkpoint: deserialization error for replica {idx}: {exc}")
                    logger.warning("Deserialization error for replica %d: %s", idx, exc)

            if applied_all:
                self.sweep_index = sweep_index
                try:
                    self.sweep_counts = [int(cnt) for cnt in self.sweep_counts]
                except Exception:
                    pass
                result["method"] = "state"
                result["ok"] = True
                result["reason"] = "restored_from_saved_states"
            else:
                result["method"] = "state"
                result["ok"] = False
                result["reason"] = "partial_or_failed_state_restore"
                result["failed_state_indices"] = failed_indices

        # 恢复 swap attempts/accepts（若有），并做长度检查
        try:
            target_len = max(0, len(self.replicas) - 1)
            if "swap_attempts" in grp:
                self._swap_attempts = np.asarray(grp["swap_attempts"][()], dtype=np.int64)
            else:
                sa = grp.attrs.get("swap_attempts", None)
                if isinstance(sa, (bytes, bytearray, np.bytes_)):
                    sa = sa.decode("utf-8")
                if sa is not None:
                    self._swap_attempts = np.asarray(json.loads(sa), dtype=np.int64)

            if "swap_accepts" in grp:
                self._swap_accepts = np.asarray(grp["swap_accepts"][()], dtype=np.int64)
            else:
                sc = grp.attrs.get("swap_accepts", None)
                if isinstance(sc, (bytes, bytearray, np.bytes_)):
                    sc = sc.decode("utf-8")
                if sc is not None:
                    self._swap_accepts = np.asarray(json.loads(sc), dtype=np.int64)

            if self._swap_attempts.shape[0] != target_len or self._swap_accepts.shape[0] != target_len:
                self.warnings.append("swap stats length mismatch; resetting swap arrays")
                self._swap_attempts = np.zeros(target_len, dtype=np.int64)
                self._swap_accepts = np.zeros(target_len, dtype=np.int64)
        except Exception as exc:
            self.warnings.append(f"restore swap stats failed: {exc}")

        try:
            f.close()
        except Exception:
            pass
        return result

    # -------------------------
    # 序列化 / 反序列化 RNG state（JSON 友好）
    # -------------------------
    def _serialize_rng_state(self, state_obj: Any) -> Dict[str, Any]:
        def _conv(o):
            if o is None:
                return None
            if isinstance(o, (str, int, float, bool)):
                return o
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (list, tuple)):
                return [_conv(x) for x in o]
            if isinstance(o, dict):
                return {str(k): _conv(v) for k, v in o.items()}
            try:
                json.dumps(o)
                return o
            except Exception:
                return str(o)
        return _conv(state_obj)

    def _deserialize_rng_state(self, serial_obj: Any) -> Any:
        def _try_array(x):
            if isinstance(x, list) and len(x) > 0:
                if all(isinstance(el, int) for el in x):
                    minv, maxv = min(x), max(x)
                    if minv >= 0:
                        if maxv < 2**32:
                            return np.asarray(x, dtype=np.uint32)
                        else:
                            return np.asarray(x, dtype=np.uint64)
                    else:
                        if minv >= -2**31 and maxv < 2**31:
                            return np.asarray(x, dtype=np.int32)
                        else:
                            return np.asarray(x, dtype=np.int64)
                if all(isinstance(el, (int, float)) for el in x):
                    return np.asarray(x, dtype=np.float64)
                return np.asarray(x, dtype=object)
            return x

        if serial_obj is None:
            return None
        if isinstance(serial_obj, dict):
            out = {}
            for k, v in serial_obj.items():
                if isinstance(v, list):
                    out[k] = self._deserialize_rng_state(v)
                elif isinstance(v, dict):
                    out[k] = self._deserialize_rng_state(v)
                else:
                    out[k] = v
            return out
        if isinstance(serial_obj, list):
            return _try_array(serial_obj)
        return serial_obj

    # -------------------------
    # 读取 HDF5 中的 rng_states
    # -------------------------
    def _read_rng_states_from_h5(self, grp: h5py.Group) -> Optional[List[Any]]:
        if "rng_states" in grp:
            try:
                ds = grp["rng_states"]
                raw = ds[()]
                if isinstance(raw, (bytes, bytearray)):
                    s = raw.decode('utf-8')
                    return json.loads(s)
                elif isinstance(raw, np.ndarray):
                    arr = []
                    for item in raw:
                        if item is None:
                            arr.append(None); continue
                        if isinstance(item, (bytes, bytearray, np.bytes_)):
                            try:
                                text = item.decode('utf-8')
                            except Exception:
                                text = str(item)
                        else:
                            text = str(item)
                        try:
                            arr.append(json.loads(text))
                        except Exception:
                            try:
                                arr.append(ast.literal_eval(text))
                            except Exception:
                                arr.append(text)
                    return arr
                else:
                    s = str(raw)
                    try:
                        return json.loads(s)
                    except Exception:
                        try:
                            return ast.literal_eval(s)
                        except Exception:
                            return None
            except Exception:
                pass

        try:
            attr = grp.attrs.get("rng_states_json", None)
            if attr is not None:
                if isinstance(attr, (bytes, bytearray)):
                    attr = attr.decode('utf-8')
                try:
                    return json.loads(attr)
                except Exception:
                    try:
                        return ast.literal_eval(attr)
                    except Exception:
                        return None
        except Exception:
            pass

        try:
            attr2 = grp.attrs.get("rng_states", None)
            if attr2 is not None:
                if isinstance(attr2, (bytes, bytearray)):
                    attr2 = attr2.decode('utf-8')
                try:
                    return json.loads(attr2)
                except Exception:
                    try:
                        return ast.literal_eval(attr2)
                    except Exception:
                        return None
        except Exception:
            pass

        return None


    def analyze(
        self,
        verbose: bool = False,
        method: str = "bootstrap",
        n_bootstrap: int = 400,
    ) -> dict:
        """
        对 REMC 采样结果做统计，返回结构大致为：

            {
              "T_2.250000": {
                  "T": 2.25,
                  "E": <E per spin mean>,
                  "E_err": <stderr(E)>,
                  "M": <M per spin mean>,
                  "M_err": <stderr(M)>,
                  "C": <heat capacity>,
                  "C_err": <stderr(C)>,
                  "chi": <susceptibility>,
                  "chi_err": <stderr(chi)>,
                  "U": <Binder cumulant>,
                  "U_err": 0.0,      # 目前先给 0，有需要可做 bootstrap
                  "n_samples": N,
                  "E_samples": np.ndarray(shape=(N,)),
                  "M_samples": np.ndarray(shape=(N,)),
                  "C_samples": np.ndarray(shape=(N,)),
                  "chi_samples": np.ndarray(shape=(N,)),
              },
              "swap": {...},
              ...
            }

        其中 *_err / *_samples 会被 FSSAnalyzer 用来做 weighted LS。
        """
        if getattr(self, "_results", None) is None:
            return {"error": "run() has not been executed or produced no samples"}

        out: Dict[str, Any] = {}

        for T_str, data in self._results.items():
            # -------------------------------
            # 取出时间序列（每自旋）
            # -------------------------------
            e = np.asarray(data.get("E", np.asarray([])), dtype=float)
            m = np.asarray(data.get("M", np.asarray([])), dtype=float)
            m2 = np.asarray(data.get("M2", m * m), dtype=float)
            m4 = np.asarray(data.get("M4", m2 * m2), dtype=float)

            N_samples = int(e.size)
            if N_samples == 0:
                # 什么都没有就跳过
                continue

            # -------------------------------
            # 基本参数
            # -------------------------------
            try:
                T = float(T_str.replace("T_", ""))
            except Exception:
                T = 1.0
            beta = 1.0 / T
            N_site = float(self.N)

            # -------------------------------
            # 一阶统计
            # -------------------------------
            mean_e = float(np.mean(e))
            mean_m = float(np.mean(m))

            mean_m2 = float(np.mean(m2))
            mean_m4 = float(np.mean(m4))

            # 方差（注意保证非负）
            var_e = float(np.mean(e ** 2) - mean_e ** 2)
            var_e = max(var_e, 0.0)

            var_m = float(mean_m2 - mean_m ** 2)
            var_m = max(var_m, 0.0)

            # 比热 / 磁化率（per spin）
            C_point = float((beta ** 2) * N_site * var_e)
            chi_point = float(beta * N_site * var_m)

            # Binder U
            if mean_m2 <= 1e-15:
                U = 0.0
            else:
                U = 1.0 - mean_m4 / (3.0 * (mean_m2 ** 2 + 1e-16))
            U = float(U)

            # -------------------------------
            # 统计误差（默认先给简单 stderr，再尝试 bootstrap）
            # -------------------------------
            # E/M 的简单标准误差（未做自相关修正）
            if N_samples > 1:
                E_err = float(np.std(e, ddof=1) / np.sqrt(N_samples))
                M_err = float(np.std(m, ddof=1) / np.sqrt(N_samples))
            else:
                E_err = 0.0
                M_err = 0.0

            # 先用“样本化”的 C_i, chi_i 做一个粗略 stderr，
            # 以便在没有 statistics 模块时也能给出误差条。
            # 注意：这是近似的，但对于 FSS 的加权拟合已经足够。
            C_samples = (
                (beta ** 2)
                * N_site
                * ((e - mean_e) ** 2)  # 其实这里减不减 mean_e^2 差别不大
            )
            chi_samples = (
                beta * N_site * ((m ** 2) - mean_m2 + var_m)
            )  # 保证平均值接近 chi_point

            if N_samples > 1:
                C_err = float(np.std(C_samples, ddof=1) / np.sqrt(N_samples))
                chi_err = float(np.std(chi_samples, ddof=1) / np.sqrt(N_samples))
            else:
                C_err = 0.0
                chi_err = 0.0

            # 如果提供了 statistics 模块，就再用 moving block bootstrap
            # 覆盖掉上面的粗略估计。
            if N_samples > 4 and stats is not None and method == "bootstrap":
                try:
                    def _C_func(s):
                        s = np.asarray(s, dtype=float)
                        mu = np.mean(s)
                        var = np.mean(s ** 2) - mu ** 2
                        return float((beta ** 2) * N_site * var)

                    bl_e = stats.estimate_block_len(e)
                    C_val, C_std, _ = stats.moving_block_bootstrap_error(
                        e,
                        _C_func,
                        block_len=int(bl_e),
                        n_bootstrap=n_bootstrap,
                    )
                    C_point = float(C_val)
                    C_err = float(C_std)
                except Exception:
                    pass

                try:
                    def _chi_func(s):
                        s = np.asarray(s, dtype=float)
                        mu = np.mean(s)
                        mu2 = np.mean(s ** 2)
                        var = mu2 - mu ** 2
                        return float(beta * N_site * var)

                    bl_m = stats.estimate_block_len(m)
                    chi_val, chi_std, _ = stats.moving_block_bootstrap_error(
                        m,
                        _chi_func,
                        block_len=int(bl_m),
                        n_bootstrap=n_bootstrap,
                    )
                    chi_point = float(chi_val)
                    chi_err = float(chi_std)
                except Exception:
                    pass

            # Binder U 的误差目前先不估计
            U_err = 0.0

            # -------------------------------
            # 写回 out[T_str]
            # -------------------------------
            out[T_str] = {
                "T": float(T),
                "E": mean_e,
                "E_stderr": float(E_err),
                "M": mean_m,
                "M_stderr": float(M_err),
                "C": float(C_point),
                "C_stderr": float(C_err),
                "chi": float(chi_point),
                "chi_stderr": float(chi_err),
                "U": float(U),
                "U_stderr": float(U_err),
                "n_samples": int(N_samples),
                # 原始时间序列
                "E_samples": e.copy(),
                "M_samples": m.copy(),
                "C_samples": np.asarray(C_samples, dtype=float).copy(),
                "chi_samples": np.asarray(chi_samples, dtype=float).copy(),
            }

        # -------------------------------
        # swap 统计 / 元信息部分保持之前的逻辑
        # -------------------------------
        total_attempt = int(np.sum(self._swap_attempts))
        total_accept = int(np.sum(self._swap_accepts))
        out["swap"] = {
            "rate": total_accept / max(1, total_attempt),
            "attempts": self._swap_attempts.tolist(),
            "accepts": self._swap_accepts.tolist(),
        }

        out["field"] = float(self.h)
        #  out["replica_counters"] = [int(x) for x in self.replica_counters]
        out["rng_model"] = getattr(self, "rng_model", "unknown")
        out["rng_unit"] = getattr(self, "rng_unit", "unknown")
        out["advance_possible"] = bool(getattr(self, "advance_possible", False))

        if getattr(self, "warnings", None):
            out["warnings"] = list(self.warnings)

        if verbose:
            print(
                f"[remc.analyze] sweep_index={getattr(self, 'sweep_index', -1)} "
                f"swap_rate={out['swap']['rate']:.4f} "
                f"total_attempts={total_attempt} total_accepts={total_accept}"
            )

        return out

