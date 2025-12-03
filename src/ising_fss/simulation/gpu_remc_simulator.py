# -*- coding: utf-8 -*-
"""
二维 Ising 模型副本交换蒙特卡洛模拟器 (gpu 版)

此模块实现了基于 CuPy 的 GPU 加速 REMC 模拟器 `GPU_REMC_Simulator`。它专为大规模并行模拟设计，
能够利用 GPU 的高吞吐量同时处理数千个 Ising 模型副本，并执行高效的并行回火 (Parallel Tempering)。

核心功能:
    - GPU 驻留: 所有副本的自旋构型 (Spins) 和能量 (Energies) 全程在 GPU 显存中维护和更新，仅在必须进行交换判定或
      数据 I/O 时将极少量数据回传至 Host 内存，最大限度减少 Host-Device 通信开销。
    - 向量化 Metropolis: 利用 `gpu_algorithms.metropolis_update_batch` 接口，通过棋盘格分解
      (Checkerboard Decomposition) 和向量化操作，在 GPU 上并行更新所有副本。
    - Slot-bound RNG: 采用 Slot-bound 随机数生成策略。每个温度槽 (Slot) 拥有独立的 RNG 状态。虽然底层使用
      向量化生成，但通过精心的种子管理 (显式 replica_seeds + slot 绑定)，保证了模拟的可复现性。
    - Checkpoint: 使用 JSON + NPZ 双文件保存 GPU 端状态（自旋、能量、部分 RNG 状态）。
    - 物理一致性: 能量计算与交换判据采用统一的物理口径 (包含外场项)，与 CPU 版本保持一致。

整体设计与 CPU HybridREMCSimulator 在随机数与 checkpoint 语义上保持一致：
    - 由外部提供的 replica_seeds 完全决定随机数流。
    - 每个温度槽绑定一个独立 RNG（slot-bound）。
    - 初始格点由 seed^const 派生的独立 RNG 生成，解耦初始化与运行期随机流。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Sequence
import json
import os
import math
import time
import inspect
import sys
import platform
import getpass
import numpy as np
import logging

logger = logging.getLogger(__name__)

# --- cupy 可选导入 ---
_cp_available = True
try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None  # type: ignore
    _cp_available = False
    logger.warning(
        "cupy not available at import time; GPU operations will raise if attempted. "
        "For testing, inject a fake 'cupy' into sys.modules before importing this module."
    )

# 下游 GPU 算法模块
try:
    from ..core import gpu_algorithms as ga  # type: ignore
except Exception as e:
    raise ImportError("gpu_remc_simulator requires gpu_algorithms module (GPU implementations).") from e

# observables / energy helper（回退用）
try:
    from ..core.observables import _observables_for_simulator  # type: ignore
except Exception:
    def _observables_for_simulator(latt, h):
        """Host 回退观测量计算"""
        ai = np.asarray(latt, dtype=np.int64)
        nbr = (
            np.roll(ai, 1, 0)
            + np.roll(ai, -1, 0)
            + np.roll(ai, 1, 1)
            + np.roll(ai, -1, 1)
        )
        e_bond = -0.5 * np.sum(ai * nbr)
        M = float(np.sum(ai))
        E = float(e_bond) - float(h) * M
        return {
            "E": E,
            "M": M,
            "absM": float(abs(M)),
            "M2": float(M * M),
            "M4": float(M ** 4),
        }

try:
    from ..analysis import statistics as stats
except Exception:
    stats = None

_MIN_EXP_ARG = -700.0
PREFERRED_RNG = "philox"


# -----------------------
# RNG 小工具（与 CPU 版保持一致）
# -----------------------
def _make_generator_from_seed(seed: int, prefer: str = PREFERRED_RNG) -> np.random.Generator:
    """
    优先使用 Philox（若可用），否则回退到 default_rng。
    与 CPU HybridREMCSimulator 保持一致，以便在有需要时也可以利用 bitgen 级别的信息。
    """
    s = int(seed) & 0xFFFFFFFF
    try:
        from numpy.random import Philox  # type: ignore
        bitgen = Philox(int(s))
        gen = np.random.Generator(bitgen)
        return gen
    except Exception:
        return np.random.default_rng(int(s))


@dataclass
class _GReplica:
    beta: float
    seed: int
    host_rng: np.random.Generator


class GPU_REMC_Simulator:
    """
    GPU REMC 主控器（基于 gpu_algorithms），严格 slot-bound RNG 语义。
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

        # 温度列
        if temperatures is not None:
            temps = np.asarray(temperatures, dtype=float)
            if temps.ndim != 1 or temps.size < 2:
                raise ValueError(
                    "temperatures must be 1D array-like with at least 2 entries"
                )
            if temps.size != int(num_replicas):
                raise ValueError("temperatures length must match num_replicas")
        else:
            if spacing not in ("geom", "linear"):
                raise ValueError("spacing must be 'geom' or 'linear'")
            if spacing == "geom":
                temps = np.geomspace(float(T_min), float(T_max), int(num_replicas))
            else:
                temps = np.linspace(float(T_min), float(T_max), int(num_replicas))
        self.temps = temps.tolist()
        for T in self.temps:
            if float(T) <= 0.0:
                raise ValueError("Temperatures must be positive.")
        if any(self.temps[i] >= self.temps[i + 1] for i in range(len(self.temps) - 1)):
            raise ValueError("temperatures must be strictly increasing for REMC/PT")
        self.betas = [1.0 / float(T) for T in self.temps]

        # 算法校验
        req_algo = (
            algorithm or "metropolis_sweep"
        ).lower().strip().replace(" ", "_").replace("-", "_")
        _cluster_algos = ("wolff", "swendsen_wang", "swendsenwang", "cluster")
        if req_algo in _cluster_algos and abs(self.h) > 1e-12:
            raise ValueError(f"Cluster algorithms require h=0, but h={self.h} provided.")
        if req_algo in ("metropolis_sweep", "metropolis", "metro", "metropolissweep"):
            req_algo = "metropolis_sweep"
        self.algorithm = req_algo

        # Seeds（显式传入，控制全局随机性）
        if replica_seeds is None:
            raise ValueError("GPU_REMC_Simulator requires explicit replica_seeds.")
        try:
            seeds_list = [int(x) for x in list(replica_seeds)]
        except Exception:
            raise ValueError("replica_seeds must be an iterable of integers.")
        if len(seeds_list) != int(num_replicas):
            raise ValueError(
                f"replica_seeds length ({len(seeds_list)}) must equal num_replicas ({num_replicas})."
            )
        self.replica_seeds = seeds_list
        self._seed_info = {"replica_seeds": self.replica_seeds, "seed_bits": 32}

        # Host RNGs (Slot-bound，使用 Philox 优先，与 CPU 版保持一致)
        self._slot_rngs: List[np.random.Generator] = [
            _make_generator_from_seed(int(s)) for s in self.replica_seeds
        ]

        # Spins Init (Host) —— 使用与 slot_rng 解耦的种子（seed ^ const）
        R = int(num_replicas)
        host_spins = np.empty((R, self.L, self.L), dtype=np.int8)
        _init_choices = np.array([-1, 1], dtype=np.int8)
        for r in range(R):
            seed = int(self.replica_seeds[r])
            init_rng = _make_generator_from_seed(seed ^ 0xC2B2AE35)
            host_spins[r] = init_rng.choice(_init_choices, size=(self.L, self.L))

        self._spins_host = host_spins
        self._spins = None
        self._device_spins_initialized = False

        # Replicas Meta
        self.replicas: List[_GReplica] = []
        for r, beta in enumerate(self.betas):
            self.replicas.append(
                _GReplica(
                    beta=float(beta),
                    seed=int(self.replica_seeds[r]),
                    host_rng=self._slot_rngs[r],
                )
            )

        self.replica_counters: List[int] = [0 for _ in range(R)]
        self.rng_model: str = "gpu_default_rng_slot_bound"
        self.rng_unit: str = "uniform_draws"
        # GPU 这边不尝试 bitgen.advance，因此 advance_possible 固定 False
        self.advance_possible: bool = False

        # Swap stats
        num_pairs = max(0, R - 1)
        self._swap_attempts = np.zeros(num_pairs, dtype=np.int64)
        self._swap_accepts = np.zeros(num_pairs, dtype=np.int64)
        self.record_swap_history = bool(record_swap_history)
        self._swap_history: Optional[List[List[bool]]] = (
            [[] for _ in range(num_pairs)] if self.record_swap_history else None
        )
        self._swap_rng_consumed_total: int = 0
        self._swap_rng_consumed_per_slot: List[int] = [0 for _ in range(R)]

        self._device_counters = None

        self._results: Optional[Dict[str, Dict[str, np.ndarray]]] = None
        self.final_lattices: Optional[List[np.ndarray]] = None
        self._lattice_files: Optional[Dict[str, str]] = None
        self.buffer_flush = int(max(1, buffer_flush))
        self.warnings: List[str] = []

        self._provenance_notes = {
            "rng_binding": "slot",
            "rng_source": "replica_seeds",
            "swap_rng_policy": "use_left_slot_rng",
            "gpu_sampling": "gpu_algorithms.metropolis_update_batch",
            "init_rng_decoupled": True,
        }
        self._rng_versions = {
            "numpy": np.__version__,
            "cupy": getattr(cp, "__version__", "unknown") if cp is not None else "not_installed",
        }

        # Interface check
        missing = []
        if not hasattr(ga, "metropolis_update_batch"):
            missing.append("metropolis_update_batch")
        if not hasattr(ga, "device_energy"):
            self.warnings.append(
                "gpu_algorithms does not provide device_energy(); host fallback will be used."
            )
        if missing:
            self.warnings.append(f"gpu_algorithms missing: {missing}.")

        self.sweep_index = 0
        self.rng_offset_within_sweep = 0

    # ------------------------------------------------------------------
    # 基础工具
    # ------------------------------------------------------------------
    def _ensure_device_spins(self):
        if self._device_spins_initialized:
            return
        if cp is None:
            raise ImportError("cupy is required.")
        try:
            self._spins = cp.ascontiguousarray(cp.asarray(self._spins_host))
            self._device_spins_initialized = True
        except Exception as exc:
            raise RuntimeError(f"Failed to transfer spins to device: {exc}") from exc

    def _log_warn(self, msg: str):
        self.warnings.append(msg)
        logger.warning(msg)

    # 更稳健的参数探测
    def _ga_accepts_param(self, name: str) -> bool:
        try:
            sig = inspect.signature(ga.metropolis_update_batch)
            return name in sig.parameters
        except Exception as exc:
            self._log_warn(
                f"_ga_accepts_param inspect.signature failed: {exc}; assuming param not accepted."
            )
            return False

    # 解析 meta 并更新计数器
    def _parse_ga_meta_and_update_counters(self, device_counters):
        """
        解析下游返回的 device_counters / meta 信息并更新
        self._device_counters / self.replica_counters / rng_offset_within_sweep。
        """
        meta_candidate = None
        try:
            if isinstance(device_counters, tuple) and len(device_counters) == 2:
                self._device_counters, meta_candidate = device_counters
                is_valid_dc = False
                try:
                    keys_attr = getattr(self._device_counters, "keys", None)
                    if keys_attr is not None:
                        keys_iter = keys_attr() if callable(keys_attr) else keys_attr
                        if "accepts" in (
                            list(keys_iter)
                            if not isinstance(keys_iter, (list, tuple))
                            else keys_iter
                        ):
                            is_valid_dc = True
                    else:
                        if "accepts" in dict(self._device_counters):
                            is_valid_dc = True
                except Exception:
                    pass

                if not is_valid_dc:
                    self._log_warn(
                        "gpu_algorithms returned unexpected device_counters structure; ignoring device_counters."
                    )
                    self._device_counters = None
            else:
                meta_candidate = device_counters

            if isinstance(meta_candidate, dict):
                if "rng_model" in meta_candidate:
                    try:
                        self.rng_model = str(meta_candidate["rng_model"])
                    except Exception:
                        pass
                if "replica_counters" in meta_candidate:
                    try:
                        new_rc = np.asarray(
                            meta_candidate["replica_counters"], dtype=np.int64
                        )
                        if new_rc.size == len(self.replica_counters):
                            self.replica_counters = [int(x) for x in new_rc.tolist()]
                        else:
                            self._log_warn(
                                "gpu_algorithms returned replica_counters with unexpected length; ignored."
                            )
                    except Exception:
                        self._log_warn(
                            "failed to parse replica_counters from gpu_algorithms meta; ignored."
                        )
                if "rng_consumed" in meta_candidate:
                    try:
                        consumed = np.asarray(meta_candidate["rng_consumed"])
                        if consumed.ndim == 0:
                            add_val = int(consumed)
                            for i in range(len(self.replica_counters)):
                                self.replica_counters[i] += add_val
                            self.rng_offset_within_sweep += int(
                                add_val * len(self.replica_counters)
                            )
                        else:
                            for i, v in enumerate(consumed.tolist()):
                                if i < len(self.replica_counters):
                                    self.replica_counters[i] += int(v)
                            self.rng_offset_within_sweep += int(np.sum(consumed))
                    except Exception:
                        self._log_warn(
                            "failed to parse rng_consumed from gpu_algorithms meta; ignored."
                        )
            else:
                try:
                    arr = np.asarray(meta_candidate)
                    if arr.size == len(self.replica_counters):
                        for i, v in enumerate(arr.tolist()):
                            self.replica_counters[i] += int(v)
                        self.rng_offset_within_sweep += int(np.sum(arr))
                    elif arr.ndim == 0:
                        add_val = int(arr)
                        for i in range(len(self.replica_counters)):
                            self.replica_counters[i] += add_val
                        self.rng_offset_within_sweep += int(
                            add_val * len(self.replica_counters)
                        )
                except Exception:
                    pass
        except Exception as exc:
            self._log_warn(f"Exception while parsing gpu_algorithms meta: {exc}")

    # ------------------------------------------------------------------
    # GPU sweep wrapper
    # ------------------------------------------------------------------
    def _gpu_sweep_batch(self, n_sweeps: int = 1, checkerboard: bool = True):
        self._ensure_device_spins()

        R = len(self.replicas)
        beta_list = [rep.beta for rep in self.replicas]

        try:
            beta_arr = cp.asarray(beta_list, dtype=float)
        except Exception:
            beta_arr = beta_list

        if self._device_counters is None and hasattr(ga, "init_device_counters"):
            try:
                self._device_counters = ga.init_device_counters(R)
            except Exception as exc:
                self._log_warn(
                    f"init_device_counters failed: {exc}; proceeding with device_counters=None"
                )
                self._device_counters = None

        base_kwargs = dict(
            n_sweeps=int(n_sweeps),
            replica_seeds=self.replica_seeds,
            device_counters=self._device_counters,
            checkerboard=checkerboard,
            h=self.h,
        )
        if self._ga_accepts_param("sweep_start"):
            base_kwargs["sweep_start"] = int(self.sweep_index)

        accept_counters_param = self._ga_accepts_param("replica_counters")
        call_kwargs_with_counters = dict(base_kwargs)
        try:
            rc_np = np.asarray(self.replica_counters, dtype=np.int64)
            try:
                rc_dev = cp.asarray(rc_np)
                call_kwargs_with_counters["replica_counters"] = rc_dev
            except Exception:
                call_kwargs_with_counters["replica_counters"] = rc_np
        except Exception:
            call_kwargs_with_counters = dict(base_kwargs)
            accept_counters_param = False

        spins_out = None
        device_counters_ret = None

        if accept_counters_param:
            try:
                try:
                    spins_out, device_counters_ret = ga.metropolis_update_batch(
                        self._spins, beta_arr, **call_kwargs_with_counters
                    )
                except TypeError:
                    spins_out, device_counters_ret = ga.metropolis_update_batch(
                        self._spins, beta_list, **call_kwargs_with_counters
                    )
            except Exception as exc:
                self._log_warn(
                    f"metropolis_update_batch with replica_counters exception: {exc}; fallback."
                )
                spins_out = None
                device_counters_ret = None

        if spins_out is None:
            try:
                try:
                    spins_out, device_counters_ret = ga.metropolis_update_batch(
                        self._spins, beta_arr, **base_kwargs
                    )
                except TypeError:
                    spins_out, device_counters_ret = ga.metropolis_update_batch(
                        self._spins, beta_list, **base_kwargs
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"gpu_algorithms.metropolis_update_batch (fallback) failed: {exc}"
                ) from exc

        if spins_out is None:
            raise RuntimeError(
                "gpu_algorithms.metropolis_update_batch returned None for spins_out."
            )

        try:
            if cp is not None and not isinstance(spins_out, cp.ndarray):
                spins_out = cp.asarray(spins_out)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to convert spins_out to cupy.ndarray: {exc}"
            ) from exc
        self._spins = cp.ascontiguousarray(spins_out)

        self._parse_ga_meta_and_update_counters(device_counters_ret)

    # ------------------------------------------------------------------
    # swap (host-side)
    # ------------------------------------------------------------------
    def _attempt_swaps_host(self, parity: int = 0):
        R = len(self.replicas)
        E_host = None
        try:
            if hasattr(ga, "device_energy"):
                if not self._device_spins_initialized:
                    if cp is None:
                        raise RuntimeError("cupy not available")
                    else:
                        self._ensure_device_spins()
                E_dev = ga.device_energy(self._spins, h=self.h)
                try:
                    E_host = cp.asnumpy(E_dev).astype(float).ravel()
                except Exception:
                    E_host = np.asarray(E_dev).astype(float).ravel()
                if E_host.size != R:
                    self._log_warn(f"device_energy shape mismatch {E_host.size} vs {R}")
                    E_host = None
            else:
                E_host = None
        except Exception as exc:
            self._log_warn(f"device_energy failed: {exc}; fallback")
            E_host = None

        if E_host is None:
            try:
                if not self._device_spins_initialized:
                    spins_host = self._spins_host
                else:
                    spins_host = cp.asnumpy(self._spins)
                E_host = np.array(
                    [
                        float(_observables_for_simulator(sp, self.h).get("E", 0.0))
                        for sp in spins_host
                    ],
                    dtype=float,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Host-side swap energy calc failed: {exc}"
                ) from exc

        start = 0 if parity == 0 else 1
        for left in range(start, R - 1, 2):
            right = left + 1
            a_beta = self.replicas[left].beta
            b_beta = self.replicas[right].beta
            dE = float(E_host[left]) - float(E_host[right])
            dbeta = a_beta - b_beta
            pair_idx = left
            self._swap_attempts[pair_idx] += 1

            Delta = dbeta * dE
            accept = False
            if Delta >= 0.0:
                accept = True
            else:
                try:
                    u = float(self._slot_rngs[left].random())
                    self._swap_rng_consumed_total += 1
                    self._swap_rng_consumed_per_slot[left] += 1
                except Exception as exc:
                    raise RuntimeError(
                        f"Slot host RNG failure at slot {left}: {exc}"
                    )
                exp_arg = max(float(Delta), _MIN_EXP_ARG)
                accept = (u < math.exp(exp_arg))

            if accept:
                swapped = False
                if self._device_spins_initialized and cp is not None:
                    try:
                        try:
                            self._spins[[left, right]] = self._spins[[right, left]]
                        except Exception:
                            tmp = self._spins[left].copy()
                            self._spins[left] = self._spins[right]
                            self._spins[right] = tmp
                        swapped = True
                    except Exception as exc:
                        self._log_warn(f"device swap failed: {exc}; trying host-copy")
                if not swapped:
                    try:
                        if not self._device_spins_initialized:
                            tmp = self._spins_host[left].copy()
                            self._spins_host[left] = self._spins_host[right]
                            self._spins_host[right] = tmp
                        else:
                            spins_host = cp.asnumpy(self._spins)
                            tmp = spins_host[left].copy()
                            spins_host[left] = spins_host[right]
                            spins_host[right] = tmp
                            self._spins = cp.ascontiguousarray(cp.asarray(spins_host))
                    except Exception as exc2:
                        raise RuntimeError(
                            f"Host-copy swap failed: {exc2}"
                        ) from exc2
                self._swap_accepts[pair_idx] += 1
                E_host[left], E_host[right] = E_host[right], E_host[left]

            if self.record_swap_history and self._swap_history is not None:
                try:
                    if pair_idx < len(self._swap_history):
                        self._swap_history[pair_idx].append(bool(accept))
                except Exception:
                    pass
        return

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
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
    ) -> None:
        """
        GPU 版 run 与 CPU 版接口保持一致，但当前实现：
          - 忽略 auto_thin / tau_* 参数，不做自适应 thin（thin 固定为传入值）。
          - unit_sanity_check 生效：对每个采样点做简单的单位检查。
        """
        R = len(self.replicas)
        thin_local = int(max(1, thin if thin is not None else 1))

        if auto_thin:
            # 当前 GPU 版本不做自适应 thinning，只提示一次
            self._log_warn(
                "[GPU_REMC_Simulator.run] auto_thin=True is not implemented on GPU; using fixed thin="
                f"{thin_local}."
            )

        # ---------------------- HDF5 文件准备 ----------------------
        h5files: Dict[str, Tuple[Any, Any]] = {}
        sample_counters: Dict[str, int] = {}
        lattice_buffers: Dict[str, List[np.ndarray]] = {}
        if save_lattices:
            try:
                import h5py
            except Exception as exc:
                raise ImportError("save_lattices requires h5py installed") from exc
            if save_dir is None:
                save_dir = "lattices"
            os.makedirs(save_dir, exist_ok=True)
            if worker_id is None:
                worker_id = f"gpu_worker_{os.getpid()}_{int(time.time())}"
            for T in self.temps:
                T_str = f"T_{T:.6f}"
                filename = os.path.join(
                    save_dir, f"{worker_id}__latt_{T_str}_h{self.h:.6f}.h5"
                )
                chunk_shape = (1, self.L, self.L)
                if os.path.exists(filename):
                    f = h5py.File(filename, "a")
                    if "lattices" in f:
                        dset = f["lattices"]
                        sample_counters[T_str] = dset.shape[0]
                    else:
                        try:
                            dset = f.create_dataset(
                                "lattices",
                                shape=(0, self.L, self.L),
                                maxshape=(None, self.L, self.L),
                                dtype="i1",
                                chunks=chunk_shape,
                                compression="gzip",
                                compression_opts=5,
                                shuffle=True,
                            )
                        except TypeError:
                            dset = f.create_dataset(
                                "lattices",
                                shape=(0, self.L, self.L),
                                maxshape=(None, self.L, self.L),
                                dtype="i1",
                                chunks=chunk_shape,
                            )
                        sample_counters[T_str] = 0
                else:
                    f = h5py.File(filename, "w")
                    try:
                        dset = f.create_dataset(
                            "lattices",
                            shape=(0, self.L, self.L),
                            maxshape=(None, self.L, self.L),
                            dtype="i1",
                            chunks=chunk_shape,
                            compression="gzip",
                            compression_opts=5,
                            shuffle=True,
                        )
                    except TypeError:
                        dset = f.create_dataset(
                            "lattices",
                            shape=(0, self.L, self.L),
                            maxshape=(None, self.L, self.L),
                            dtype="i1",
                            chunks=chunk_shape,
                        )
                    sample_counters[T_str] = 0
                h5files[T_str] = (f, dset)
                lattice_buffers[T_str] = []

            # 初始 metadata JSON（轻量）
            meta = {
                "L": int(self.L),
                "h": float(self.h),
                "temps": [float(T) for T in self.temps],
                "algorithm": str(self.algorithm),
                "worker_id": worker_id,
                "replica_seeds": self._seed_info.get("replica_seeds"),
                "seed_bits": 32,
                "rng_versions": self._rng_versions,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "rng_binding": self._provenance_notes.get("rng_binding"),
                "rng_source": self._provenance_notes.get("rng_source"),
                "swap_rng_policy": self._provenance_notes.get("swap_rng_policy"),
                "rng_consumption_variable": False,
                "python_version": sys.version,
                "platform": platform.platform(),
                "created_by": None,
                "swap_rng_consumed_total": int(self._swap_rng_consumed_total),
                "swap_rng_consumed_per_slot": list(self._swap_rng_consumed_per_slot),
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
                pass

        # ---------------- Equilibration ----------------
        for t in range(int(equilibration_steps)):
            self.rng_offset_within_sweep = 0
            self._gpu_sweep_batch(n_sweeps=1, checkerboard=True)
            self.sweep_index += 1
            if (t + 1) % int(exchange_interval) == 0:
                self._attempt_swaps_host(parity=0)
                self._attempt_swaps_host(parity=1)
            if verbose and ((t + 1) % max(1, equilibration_steps // 5) == 0):
                print(
                    f"[gpu_remc] equilibration progress: {t+1}/{equilibration_steps}"
                )

        # ---------------- Production ----------------
        series: Dict[str, Dict[str, List[float]]] = {}
        for T in self.temps:
            T_str = f"T_{T:.6f}"
            series[T_str] = {"E": [], "M": [], "absM": [], "M2": [], "M4": []}

        for t in range(int(production_steps)):
            self.rng_offset_within_sweep = 0
            self._gpu_sweep_batch(n_sweeps=1, checkerboard=True)
            self.sweep_index += 1

            if (t + 1) % int(exchange_interval) == 0:
                self._attempt_swaps_host(parity=0)
                self._attempt_swaps_host(parity=1)

            if (self.sweep_index % thin_local) == 0:
                Nsite = float(self.N)
                E_vec, M_vec = None, None
                try:
                    if self._device_spins_initialized and hasattr(ga, "device_energy"):
                        E_dev = ga.device_energy(self._spins, h=self.h)
                        if hasattr(ga, "device_magnetization"):
                            M_dev = ga.device_magnetization(self._spins)
                        else:
                            M_dev = None
                        E_vec = cp.asnumpy(E_dev).astype(float).ravel()
                        if M_dev is not None:
                            M_vec = cp.asnumpy(M_dev).astype(float).ravel()
                except Exception as exc:
                    self._log_warn(f"device observables failed: {exc}")
                    E_vec, M_vec = None, None

                if (E_vec is None) or (M_vec is None):
                    # Host fallback
                    try:
                        spins_host = (
                            cp.asnumpy(self._spins)
                            if self._device_spins_initialized
                            else self._spins_host
                        )
                    except Exception:
                        spins_host = np.zeros((R, self.L, self.L), dtype=np.int8)
                    E_host = np.empty((R,), dtype=float)
                    M_host = np.empty((R,), dtype=float)
                    for r in range(R):
                        ob = _observables_for_simulator(spins_host[r], self.h)
                        E_host[r] = float(ob.get("E", 0.0))
                        M_host[r] = float(
                            ob.get("M", float(np.sum(spins_host[r])))
                        )
                    if E_vec is None:
                        E_vec = E_host
                    if M_vec is None:
                        M_vec = M_host

                for r, T in enumerate(self.temps):
                    T_str = f"T_{T:.6f}"
                    E_ps = float(E_vec[r]) / Nsite
                    M_ps = float(M_vec[r]) / Nsite

                    # 单位 sanity check（与 CPU 版类似）
                    if unit_sanity_check:
                        try:
                            if abs(M_ps) > 1.0 + 1e-6:
                                self.warnings.append(
                                    f"[GPU {T_str}] |m|>1 detected; check per-spin pipeline."
                                )
                            if abs(E_ps) > (2.0 + abs(self.h) + 1e-3):
                                self.warnings.append(
                                    f"[GPU {T_str}] |E| per spin > 2+|h|; check units."
                                )
                        except Exception:
                            pass

                    series[T_str]["E"].append(E_ps)
                    series[T_str]["M"].append(M_ps)
                    series[T_str]["absM"].append(abs(M_ps))
                    series[T_str]["M2"].append(M_ps ** 2)
                    series[T_str]["M4"].append(M_ps ** 4)

                    if save_lattices:
                        try:
                            latt = (
                                cp.asnumpy(self._spins[r])
                                if self._device_spins_initialized
                                else self._spins_host[r]
                            )
                            lattice_buffers[T_str].append(
                                np.asarray(latt, dtype=np.int8).copy()
                            )
                        except Exception:
                            continue
                        if len(lattice_buffers[T_str]) >= self.buffer_flush:
                            try:
                                f, dset = h5files[T_str]
                                cur = dset.shape[0]
                                nnew = len(lattice_buffers[T_str])
                                dset.resize(cur + nnew, axis=0)
                                dset[cur : cur + nnew, :, :] = np.stack(
                                    lattice_buffers[T_str], axis=0
                                )
                                sample_counters[T_str] += nnew
                                lattice_buffers[T_str].clear()
                                try:
                                    self._safe_flush_and_sync(f)
                                except Exception:
                                    pass
                            except Exception:
                                pass

        # 保存 E/M 序列到 _results
        self._results = {
            T_str: {
                k: np.asarray(v, dtype=float) for k, v in series[T_str].items()
            }
            for T_str in series
        }
        try:
            final_spins_host = (
                cp.asnumpy(self._spins)
                if self._device_spins_initialized
                else self._spins_host
            )
            self.final_lattices = [
                np.asarray(final_spins_host[r], dtype=np.int8).copy()
                for r in range(len(self.replicas))
            ]
        except Exception:
            self.final_lattices = [None for _ in range(len(self.replicas))]

        # --- 预先分析一次，用于写入 JSON meta（C/chi/U & 交换率） ---
        analysis_summary_for_json: Optional[Dict[str, Any]] = None
        if save_lattices:
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
                self.warnings.append(f"GPU run(): internal analyze() for JSON stats failed: {exc}")
                analysis_summary_for_json = None

        # ---------------- Finalize HDF5 & metadata ----------------
        if save_lattices:
            import h5py  # type: ignore
            lattice_paths = {}
            for T in self.temps:
                T_str = f"T_{T:.6f}"
                f, dset = h5files[T_str]
                buf = lattice_buffers.get(T_str, [])
                if buf:
                    try:
                        cur = dset.shape[0]
                        nnew = len(buf)
                        dset.resize(cur + nnew, axis=0)
                        dset[cur : cur + nnew, :, :] = np.stack(
                            buf, axis=0
                        )
                        sample_counters[T_str] += nnew
                        lattice_buffers[T_str].clear()
                    except Exception as exc:
                        self.warnings.append(f"GPU final flush failed for {T_str}: {exc}")

                # provenance + E/M 写入
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
                    # 这里统一命名：HDF5 和 JSON 中都叫 samples_per_temp
                    prov.attrs["samples_per_temp"] = int(sample_counters.get(T_str, 0))
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
                    prov.attrs["rng_binding"] = str(self._provenance_notes.get("rng_binding"))
                    prov.attrs["rng_source"] = str(self._provenance_notes.get("rng_source"))
                    prov.attrs["swap_rng_policy"] = str(self._provenance_notes.get("swap_rng_policy"))
                    prov.attrs["rng_consumption_variable"] = False
                    prov.attrs["swap_rng_consumed_total"] = int(self._swap_rng_consumed_total)
                    try:
                        prov.attrs["swap_rng_consumed_per_slot"] = json.dumps(self._swap_rng_consumed_per_slot)
                    except Exception:
                        prov.attrs["swap_rng_consumed_per_slot"] = str(self._swap_rng_consumed_per_slot)
                    try:
                        prov.attrs["python_version"] = sys.version
                        prov.attrs["platform"] = platform.platform()
                    except Exception:
                        pass

                    # E/M 序列写入 HDF5
                    try:
                        if self._results is not None and T_str in self._results:
                            E_arr = np.asarray(self._results[T_str].get("E", []), dtype=np.float64)
                            M_arr = np.asarray(self._results[T_str].get("M", []), dtype=np.float64)

                            try:
                                n_latt = int(sample_counters.get(T_str, 0))
                                if n_latt not in (0, E_arr.size) and E_arr.size > 0:
                                    self.warnings.append(
                                        f"[GPU {T_str}] samples_written={n_latt} vs len(E)={E_arr.size}; "
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
                                    self.warnings.append(
                                        f"GPU Failed to write dataset '{name}' for {T_str}: {exc_em}"
                                    )
                    except Exception as exc:
                        self.warnings.append(
                            f"GPU Failed to write E/M observables into HDF5 for {T_str}: {exc}"
                        )

                    try:
                        self._safe_flush_and_sync(f)
                    except Exception:
                        pass
                except Exception as exc:
                    self.warnings.append(f"GPU Failed to write provenance into HDF5 for {T_str}: {exc}")

                try:
                    filename = f.filename
                except Exception:
                    filename = os.path.join(save_dir, f"{worker_id}__{T_str}.h5")
                try:
                    f.close()
                except Exception:
                    pass
                lattice_paths[T_str] = filename

            # 最终 metadata 更新
            meta_update = {
                # JSON 里也叫 samples_per_temp，与 HDF5 一致
                "samples_per_temp": sample_counters,
                # 不再写入 final_seeds，避免与 replica_seeds 命名冗余
                "swap_attempts": self._swap_attempts.tolist(),
                "swap_accepts": self._swap_accepts.tolist(),
                "rng_versions": self._rng_versions,
                "rng_binding": self._provenance_notes.get("rng_binding"),
                "rng_source": self._provenance_notes.get("rng_source"),
                "swap_rng_policy": self._provenance_notes.get("swap_rng_policy"),
                "thin": int(thin_local),
                "rng_consumption_variable": False,
                "swap_rng_consumed_total": int(self._swap_rng_consumed_total),
                "swap_rng_consumed_per_slot": list(self._swap_rng_consumed_per_slot),
                "algorithm": str(self.algorithm),
                "init_rng_decoupled": True,
            }
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
    def _safe_flush_and_sync(self, fobj):
        try:
            fobj.flush()
            try:
                os.fsync(fobj.fileno())
            except Exception:
                try:
                    fobj_id = getattr(fobj, "id", None)
                    if fobj_id is not None and hasattr(fobj_id, "get_vfd_handle"):
                        os.fsync(fobj_id.get_vfd_handle())  # type: ignore
                except Exception:
                    pass
        except Exception:
            pass

    # =========================
    # 快照与存档
    # =========================
    def _snapshot_spins_and_energies(self) -> Tuple[np.ndarray, np.ndarray]:
        R = len(self.replicas)
        try:
            spins_host = (
                cp.asnumpy(self._spins)
                if self._device_spins_initialized and cp is not None
                else self._spins_host
            )
            spins_host = np.ascontiguousarray(spins_host, dtype=np.int8)
        except Exception as exc:
            raise RuntimeError(f"Failed to snapshot spins: {exc}") from exc

        E = None
        try:
            if self._device_spins_initialized and hasattr(ga, "device_energy"):
                E_dev = ga.device_energy(self._spins, h=self.h)
                E = np.asarray(
                    cp.asnumpy(E_dev) if cp is not None else E_dev,
                    dtype=np.float64,
                ).ravel()
                if E.size != R:
                    E = None
        except Exception:
            E = None
        if E is None:
            E = np.empty((R,), dtype=np.float64)
            for r in range(R):
                E[r] = float(
                    _observables_for_simulator(spins_host[r], self.h).get("E", 0.0)
                )
        return spins_host, np.asarray(E, dtype=np.float64).ravel()

    def save_checkpoint(self, path: str) -> None:
        """
        保存 GPU REMC 的 checkpoint：
          - {path}.npz : spins & energies（物理态）
          - {path}     : JSON meta（包含 replica_seeds / RNG 版本 / 设备 RNG 状态等）
        """
        # NPZ
        state_npz = path + ".npz"
        spins_host, energies = self._snapshot_spins_and_energies()
        try:
            tmp_npz = state_npz + ".tmp.npz"
            np.savez_compressed(
                tmp_npz,
                spins=spins_host.astype(np.int8, copy=False),
                energies=energies.astype(np.float64, copy=False),
            )
            os.replace(tmp_npz, state_npz)
        except Exception as exc:
            raise RuntimeError(f"Failed to write NPZ: {exc}") from exc

        # JSON
        data = {
            "L": int(self.L),
            "h": float(self.h),
            "temps": [float(T) for T in self.temps],
            "algorithm": str(self.algorithm),
            "replica_seeds": [int(x) for x in self.replica_seeds],
            "replica_counters": [int(x) for x in self.replica_counters],
            "sweep_index": int(getattr(self, "sweep_index", -1)),
            "rng_model": str(self.rng_model),
            "rng_unit": str(self.rng_unit),
            "advance_possible": bool(self.advance_possible),
            "rng_versions": self._rng_versions,
            "provenance_notes": self._provenance_notes,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "swap_rng_consumed_total": int(self._swap_rng_consumed_total),
            "swap_rng_consumed_per_slot": [
                int(x) for x in self._swap_rng_consumed_per_slot
            ],
            "rng_per_sweep": int(self.N),
            "state_file": os.path.basename(state_npz),
        }

        # 设备 RNG 状态：只有在可序列化时才标记为已写入
        rng_states_written = False
        try:
            if hasattr(ga, "get_device_rng_state"):
                state_obj = ga.get_device_rng_state()
                try:
                    json.dumps(state_obj)
                    data["device_rng_state"] = state_obj
                    rng_states_written = True
                except Exception:
                    data["device_rng_state_repr"] = repr(state_obj)
        except Exception as exc:
            self._log_warn(f"ga.get_device_rng_state failed: {exc}")
        data["rng_states_written"] = bool(rng_states_written)

        tmp = path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, path)
        except Exception as exc:
            raise RuntimeError(f"Failed to write JSON: {exc}") from exc

    def _advance_host_rng(self, rng: np.random.Generator, n: int) -> None:
        """
        用“消耗 n 个 random() 调用”的方式前进 host RNG。
        这是一个通用、bitgen 无关的前进方法（虽然可能较慢，但只在 checkpoint 恢复时使用）。
        """
        CHUNK = 1_000_000
        remain = int(max(0, n))
        while remain > 0:
            k = CHUNK if remain > CHUNK else remain
            _ = rng.random(k)
            remain -= k

    def restore_from_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        从 JSON + NPZ checkpoint 恢复：
          - 首先严格校验 L / h / temps / algorithm / replica_seeds 是否与当前实例匹配；
          - 然后恢复 replica_counters / sweep_index / 设备 RNG 状态 / host swap RNG 状态 / spins。
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as exc:
            raise RuntimeError(f"Failed to read JSON: {exc}") from exc

        notes: Dict[str, Any] = {"path": path, "restored": [], "warnings": []}

        # 0. 物理与配置的一致性检查
        L_file = int(data.get("L", self.L))
        if L_file != self.L:
            raise ValueError(f"Checkpoint L={L_file} mismatches instance L={self.L}")
        h_file = float(data.get("h", self.h))
        if abs(h_file - self.h) > 1e-12:
            raise ValueError(
                f"Checkpoint h={h_file} mismatches instance h={self.h}"
            )

        temps_file = data.get("temps", None)
        if temps_file is not None:
            tfile = [float(x) for x in temps_file]
            if len(tfile) != len(self.temps) or any(
                abs(a - b) > 1e-12 for a, b in zip(tfile, self.temps)
            ):
                raise ValueError("Checkpoint temps mismatch current simulator temps")

        alg_file = str(data.get("algorithm", self.algorithm))
        if alg_file != self.algorithm:
            raise ValueError(
                f"Checkpoint algorithm='{alg_file}' mismatches current '{self.algorithm}'"
            )

        # 1. Seeds
        ck_seeds = data.get("replica_seeds")
        if ck_seeds is not None:
            ck_list = [int(x) for x in ck_seeds]
            if ck_list != self.replica_seeds:
                raise ValueError(
                    "Checkpoint replica_seeds mismatch current simulator seeds"
                )
            notes["restored"].append("replica_seeds_checked")

        # 2. Replica Counters
        ck_counters = data.get("replica_counters")
        if ck_counters and len(ck_counters) == len(self.replica_counters):
            self.replica_counters = [int(x) for x in ck_counters]
            notes["restored"].append("replica_counters")

        self.sweep_index = int(data.get("sweep_index", self.sweep_index))
        notes["restored"].append("sweep_index")

        # 3. Device RNG
        if (
            data.get("rng_states_written")
            and hasattr(ga, "set_device_rng_state")
            and "device_rng_state" in data
        ):
            try:
                ga.set_device_rng_state(data["device_rng_state"])
                notes["restored"].append("device_rng_state")
            except Exception as exc:
                notes["warnings"].append(f"set_device_rng_state failed: {exc}")

        # 4. Host RNG (Swap)
        try:
            self._swap_rng_consumed_total = int(
                data.get("swap_rng_consumed_total", self._swap_rng_consumed_total)
            )
            pr = data.get("swap_rng_consumed_per_slot", None)
            if isinstance(pr, list) and len(pr) == len(self._slot_rngs):
                pr_int = [int(x) for x in pr]
                for i, cnt in enumerate(pr_int):
                    self._advance_host_rng(self._slot_rngs[i], cnt)
                self._swap_rng_consumed_per_slot = pr_int
                notes["restored"].append("host_rng_advanced")
        except Exception as exc:
            notes["warnings"].append(f"host RNG advance failed: {exc}")

        # 5. Physical State (NPZ)
        state_fn = data.get("state_file")
        npz_path = (
            os.path.join(os.path.dirname(path), state_fn)
            if state_fn
            else path + ".npz"
        )
        if os.path.exists(npz_path):
            try:
                arr = np.load(npz_path)
                if "spins" in arr.files:
                    spins = np.asarray(arr["spins"], dtype=np.int8, order="C")
                    if spins.shape == (len(self.replicas), self.L, self.L):
                        self._spins_host = spins
                        if cp is not None:
                            self._spins = cp.ascontiguousarray(
                                cp.asarray(self._spins_host)
                            )
                            self._device_spins_initialized = True
                        notes["restored"].append("spins")
                    else:
                        notes["warnings"].append("spins shape mismatch")
            except Exception as exc:
                notes["warnings"].append(f"NPZ load failed: {exc}")
        else:
            notes["warnings"].append("NPZ file missing")

        if self.warnings:
            notes["warnings"].extend(self.warnings)
        return notes

    def analyze(
        self, verbose: bool = False, method: str = "bootstrap", n_bootstrap: int = 400
    ) -> dict:
        if getattr(self, "_results", None) is None:
            return {"error": "run() has not been executed or produced no samples"}

        out: Dict[str, Any] = {}

        # --- 1. 物理量统计 (E, M, C, chi, U) + 误差 + samples ---
        for T_str, data in self._results.items():
            if not T_str.startswith("T_"):
                continue

            e = np.asarray(data.get("E", np.asarray([])), dtype=float)
            m = np.asarray(data.get("M", np.asarray([])), dtype=float)
            m2 = np.asarray(data.get("M2", np.asarray([])), dtype=float)
            m4 = np.asarray(data.get("M4", np.asarray([])), dtype=float)

            N_samples = int(e.size)
            if N_samples == 0:
                continue

            try:
                T = float(T_str.replace("T_", ""))
            except Exception:
                T = 1.0
            beta = 1.0 / T
            N_site = int(self.N)

            # --- 平均值 ---
            mean_e = float(np.mean(e))
            mean_m = float(np.mean(m)) if m.size else 0.0

            if m2.size:
                mean_m2 = float(np.mean(m2))
            else:
                mean_m2 = float(np.mean(m ** 2)) if m.size else 0.0

            # --- 方差 & 点估计 C, χ ---
            var_e = max(
                0.0,
                float(np.mean(e ** 2) - mean_e ** 2),
            )
            if m.size:
                var_m = max(0.0, mean_m2 - mean_m ** 2)
            else:
                var_m = 0.0

            C_point = float((beta ** 2) * N_site * var_e)
            chi_point = float(beta * N_site * var_m)

            # --- Binder U ---
            m4_mean = float(np.mean(m4)) if m4.size else float(np.mean(m ** 4)) if m.size else 0.0
            if mean_m2 <= 1e-15:
                U = 0.0
            else:
                U = 1.0 - m4_mean / (3.0 * (mean_m2 ** 2 + 1e-16))
            U = float(U)

            # --- 误差：先给朴素 sqrt(var/N) 兜底 ---
            E_err = float(np.sqrt(var_e / max(1, N_samples)))
            M_err = float(np.sqrt(var_m / max(1, N_samples))) if m.size else 0.0
            C_err = 0.0
            chi_err = 0.0

            # --- 若有 statistics 模块，用 bootstrap 修正 ---
            if stats is not None and method == "bootstrap":
                try:
                    if N_samples > 4:
                        def _E_func(s):
                            return float(np.mean(s))

                        bl_e = stats.estimate_block_len(e)
                        E_val, E_std, _ = stats.moving_block_bootstrap_error(
                            e,
                            _E_func,
                            block_len=int(bl_e),
                            n_bootstrap=n_bootstrap,
                        )
                        mean_e = float(E_val)
                        E_err = float(E_std)
                except Exception:
                    pass

                try:
                    if m.size > 4:
                        def _M_func(s):
                            return float(np.mean(s))

                        bl_m = stats.estimate_block_len(m)
                        M_val, M_std, _ = stats.moving_block_bootstrap_error(
                            m,
                            _M_func,
                            block_len=int(bl_m),
                            n_bootstrap=n_bootstrap,
                        )
                        mean_m = float(M_val)
                        M_err = float(M_std)
                except Exception:
                    pass

                try:
                    if N_samples > 4:
                        def _C_func(s):
                            s = np.asarray(s, dtype=float)
                            return float(
                                (beta ** 2)
                                * N_site
                                * (np.mean(s ** 2) - np.mean(s) ** 2)
                            )

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
                    if m.size > 4:
                        def _chi_func(s):
                            s = np.asarray(s, dtype=float)
                            mu = np.mean(s)
                            return float(
                                beta * N_site * (np.mean(s ** 2) - mu ** 2)
                            )

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

            # --- 写入 out[T_str] ---
            out[T_str] = {
                "T": float(T),
                "E": mean_e,
                "E_err": float(E_err),
                "M": mean_m,
                "M_err": float(M_err),
                "C": float(C_point),
                "C_err": float(C_err),
                "chi": float(chi_point),
                "chi_err": float(chi_err),
                "U": float(U),
                "n_samples": int(N_samples),
                "E_samples": e.copy(),
                "M_samples": m.copy(),
            }

        # --- 2. Swap 统计 ---
        total_attempt = int(np.sum(self._swap_attempts))
        total_accept = int(np.sum(self._swap_accepts))
        out["swap"] = {
            "rate": total_accept / max(1, total_attempt),
            "attempts": self._swap_attempts.tolist(),
            "accepts": self._swap_accepts.tolist(),
        }

        # --- 3. 一些元数据 ---
        out["rng_versions"] = self._rng_versions
        out["field"] = float(self.h)
        if getattr(self, "_lattice_files", None) is not None:
            out["final_lattices_paths"] = self._lattice_files
        out["provenance_notes"] = self._provenance_notes
        out["replica_counters"] = [int(x) for x in self.replica_counters]
        out["rng_model"] = self.rng_model
        out["rng_unit"] = self.rng_unit
        out["advance_possible"] = bool(self.advance_possible)
        out["swap_rng_consumed_total"] = int(self._swap_rng_consumed_total)
        out["swap_rng_consumed_per_slot"] = list(self._swap_rng_consumed_per_slot)

        if self.warnings:
            out["warnings"] = list(self.warnings)

        if verbose:
            print(f"[gpu_remc.analyze] swap_rate={out['swap']['rate']:.4f}")

        return out



    def close(self):
        try:
            self._spins = None
            self._device_counters = None
            if cp is not None:
                cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

