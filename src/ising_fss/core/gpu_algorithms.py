# -*- coding: utf-8 -*-
"""
    GPU 加速二维 Ising 模型蒙特卡洛更新（CuPy 实现）

    本模块提供了基于 NVIDIA GPU 的二维 Ising 模型蒙特卡洛（Monte Carlo）更新算法。基于 CuPy 实现的高性能 Ising 模型演化内核。

实现算法：
    - 向量化 Metropolis: 采用棋盘格分解(Checkerboard Decomposition)，将晶格分为红/黑两组子格交替更新，避免并行更新时的邻居数据依赖冲突，确保满足细致平衡（Detailed Balance）。

实现功能：
    - Look-Up Table (LUT): 预计算玻尔兹曼因子 $e^{-\\beta \\Delta E}$，减少指数运算。
    - Philox RNG: 强制种子：严禁使用全局随机状态，必须显式传入种子列表。
    - 集成 `cupy.random.Philox`，提供统计性质优异的并行随机数生成流。默认情况下，每个副本绑定一个独立的 `Philox` 随机数生成器。这保证了即使改变副本数量或并行度，单个副本的演化轨迹依然是确定且可复现的。
    - 零同步设计: 核心循环完全在设备端执行，仅在必要时回传统计量，最大化总线带宽利用率。
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional, Sequence, List
import warnings
import numpy as np  # 必需导入

try:
    import cupy as cp  # type: ignore

    # 尝试导入 Philox BitGenerator (兼容不同 CuPy 版本)
    Philox = None
    for name in ("Philox4x3210", "Philox"):
        if hasattr(cp.random, name):
            Philox = getattr(cp.random, name)
            break
    _HAS_CUPY_PHILOX = Philox is not None
except Exception as e:
    raise ImportError(
        "gpu_algorithms requires CuPy but it is not available in the environment."
    ) from e

# ===== 与dispatcher 的gpu 检查模块相适应 =====
def gpu_available() -> bool:
    """
    轻量 GPU 可用性检查：

    - 本模块能被导入说明 CuPy 至少可以 import；
    - 再检测是否有 >=1 块 CUDA 设备；
    - 额外尝试做一次极小的 GPU 运算，防止“有驱动但不能用”的极端情况。
    """
    try:
        # 1) 是否有设备
        ndev = cp.cuda.runtime.getDeviceCount()
        if ndev <= 0:
            return False

        # 2) 简单试算：在 device 上分配一点内存并做一次运算
        x = cp.arange(4, dtype=cp.float32)
        y = (x * 2).sum()
        _ = float(y.get())  # 强制一次 host 同步，确保 kernel 真跑完

        return True
    except Exception:
        # 任何异常一律视为 GPU 不可用（保守策略）
        return False
# ===== gpu_available() 结束 =====

# exp 指数的安全区间
_MIN_EXP_ARG = -700.0
_MAX_EXP_ARG = 700.0
# 近零/标量容差
_EPS_H = 1e-12
_EPS_BETA = 1e-15


# =========================
# 工具函数（RNG 工厂）
# =========================
def _make_cupy_generator(seed: int) -> cp.random.Generator:
    """
    构造 CuPy Generator。优先使用 Philox（必须显式传 seed）。
    保证可复现： 不  尝试无参 Philox() 构造。
    """
    seed32 = int(seed) & 0xFFFFFFFF

    if _HAS_CUPY_PHILOX and Philox is not None:
        # 只尝试显式传 seed 的构造签名
        for ctor_call in (lambda s: Philox(seed=s), lambda s: Philox(s)):
            try:
                bitgen = ctor_call(seed32)
                return cp.random.Generator(bitgen)
            except Exception:
                continue

        # 如果都失败：发出警告，准备回退
        warnings.warn(
            f"Philox construction with seed={seed32} failed; "
            "falling back to cp.random.default_rng(seed).",
            RuntimeWarning,
        )

    # Fallback (仍然是有 seed 的)
    try:
        return cp.random.default_rng(seed32)
    except Exception as e:
        raise RuntimeError(
            "CuPy does not provide random.default_rng or a usable Philox constructor. "
            "Refusing to fall back to global RNG."
        ) from e


def init_device_counters(R: int) -> Dict[str, Any]:
    return {
        "accepts": cp.zeros((R,), dtype=cp.int64),
        "attempts": cp.zeros((R,), dtype=cp.int64),
        "E_sum": cp.zeros((R,), dtype=cp.float64),
        "M_sum": cp.zeros((R,), dtype=cp.float64),
    }


def get_and_reset_counters(device_counters: Dict[str, Any]) -> Dict[str, Any]:
    host: Dict[str, np.ndarray] = {}
    for k in ("accepts", "attempts", "E_sum", "M_sum"):
        v = device_counters[k]
        # 防御性：如果是 CuPy 数组则 get()，否则直接转 numpy
        if hasattr(v, "get"):
            host[k] = v.get()
        else:
            host[k] = np.asarray(v)

    # 设备端原地清零 (仅对支持 .fill 的数组)
    if hasattr(device_counters["accepts"], "fill"):
        device_counters["accepts"].fill(0)
        device_counters["attempts"].fill(0)
        device_counters["E_sum"].fill(0.0)
        device_counters["M_sum"].fill(0.0)
    return host


# =========================
# 观测量（设备端）
# =========================
def device_energy(spins: cp.ndarray, h: float = 0.0) -> cp.ndarray:
    s = spins.astype(cp.int32, copy=False)
    nbr = (
        cp.roll(s, 1, axis=1)
        + cp.roll(s, -1, axis=1)
        + cp.roll(s, 1, axis=2)
        + cp.roll(s, -1, axis=2)
    )
    E_bond = -0.5 * cp.sum(s * nbr, axis=(1, 2))
    M = cp.sum(s, axis=(1, 2))
    E_total = E_bond - (float(h) * M)
    return E_total.astype(cp.float64)


def device_magnetization(spins: cp.ndarray) -> cp.ndarray:
    s = spins.astype(cp.int32, copy=False)
    M = cp.sum(s, axis=(1, 2))
    return M.astype(cp.float64)


# =========================
# 主更新（Metropolis）
# =========================
def metropolis_update_batch(
    spins: cp.ndarray,
    beta,
    *,
    n_sweeps: int = 1,
    replica_seeds: Optional[Sequence[int]] = None,
    replica_counters: Optional[Sequence[int]] = None,
    device_counters: Optional[Dict[str, Any]] = None,
    checkerboard: bool = True,
    legacy_metropolis: bool = False,
    h: float = 0.0,
    vectorized_rng: bool = False,
    rng_chunk_replicas: Optional[int] = None,
    sweep_start: int = 0,
    precision: str = "float32",
) -> Tuple[cp.ndarray, Dict[str, Any]]:
    """
    在 GPU (CuPy) 上对批量 spins (R,L,L) 执行 n_sweeps 次 Metropolis 更新。
    """
    if legacy_metropolis:
        raise RuntimeError("GPU module does not support CPU fallback.")

    # ---------- 1. 参数与形状校验 ----------
    spins = cp.ascontiguousarray(cp.asarray(spins).astype(cp.int8, copy=False))
    if spins.ndim != 3:
        raise ValueError("spins must have shape (R, L, L)")
    R, L1, L2 = spins.shape
    if L1 != L2:
        raise ValueError("spins must be square")
    L, N_sites = L1, L1 * L1

    if int(n_sweeps) < 1:
        raise ValueError("n_sweeps must be >= 1")

    # 精度设置
    dtype_float = cp.float32 if precision == "float32" else cp.float64

    # Seeds 处理
    if replica_seeds is None:
        raise ValueError("replica_seeds required")
    seeds: List[int] = []
    try:
        for s in replica_seeds:
            seeds.append(int(s) & 0xFFFFFFFF)
    except Exception:
        raise ValueError("Invalid replica_seeds")
    if len(seeds) != R:
        raise ValueError("replica_seeds length mismatch")

    # Beta 处理: 优先在 Host 端处理，减少 Device 同步
    beta_is_scalar = False
    beta_scalar = 0.0

    if hasattr(beta, "__len__") and not isinstance(beta, (str, bytes)):
        try:
            # 优先转 numpy (Host)
            beta_host = np.array(beta, dtype=np.float64)
            if beta_host.size != R:
                raise ValueError("beta length mismatch")
            b0 = beta_host[0]
            # 在 Host 端判断标量
            if np.all(np.abs(beta_host - b0) < _EPS_BETA):
                beta_is_scalar = True
                beta_scalar = float(b0)
            # 最后再转 Device
            beta_arr = cp.asarray(beta_host).reshape((R, 1, 1))
        except Exception:
            # Fallback: 若 beta 已经是 cupy array 或其他
            beta_arr = cp.asarray(beta, dtype=cp.float64).reshape((R, 1, 1))
            b0_dev = float(beta_arr[0, 0, 0].get())  # 这里会有一次同步
            beta_is_scalar = bool(
                cp.all(cp.abs(beta_arr - b0_dev) < _EPS_BETA).item()
            )
            beta_scalar = b0_dev if beta_is_scalar else None
    else:
        beta_scalar = float(beta)
        beta_is_scalar = True
        beta_arr = cp.full((R, 1, 1), beta_scalar, dtype=cp.float64)

    # 若不使用 LUT，强制转换 beta_arr 以匹配计算精度
    use_lut = (abs(float(h)) < _EPS_H) and beta_is_scalar
    if not use_lut:
        beta_arr = beta_arr.astype(dtype_float, copy=False)

    # Counters 初始化
    if device_counters is None:
        device_counters = init_device_counters(R)
    else:
        for k in ("accepts", "attempts", "E_sum", "M_sum"):
            if k not in device_counters:
                dtype = cp.int64 if k in ("accepts", "attempts") else cp.float64
                device_counters[k] = cp.zeros((R,), dtype=dtype)
            else:
                device_counters[k] = cp.asarray(device_counters[k])

    # 掩膜预计算
    idx = cp.arange(L, dtype=cp.int32)
    ii, jj = cp.meshgrid(idx, idx, indexing="ij")
    parity_mask = [(((ii + jj) & 1) == 0), (((ii + jj) & 1) == 1)]
    mask_count = [
        N_sites // 2 if L % 2 == 0 else (N_sites + 1) // 2,
        N_sites // 2,
    ]

    # ---------- 2. RNG 初始化与缓存 ----------
    gens_slot: Optional[List[cp.random.Generator]] = None
    gen_vec: Optional[cp.random.Generator] = None

    if vectorized_rng:
        comb = 0
        for ss in seeds:
            comb ^= int(ss)
        derived = (comb ^ (int(sweep_start) * 1315423911)) & 0xFFFFFFFF
        gen_vec = _make_cupy_generator(int(derived))

        backend_name = getattr(
            gen_vec.bit_generator, "__class__", type(gen_vec.bit_generator)
        ).__name__
        device_counters["rng_model"] = f"{backend_name}_vectorized_combined_seed"
    else:
        gens_slot = []
        for r in range(R):
            seed_here = (int(seeds[r]) ^ (int(sweep_start) * 1315423911)) & 0xFFFFFFFF
            gens_slot.append(_make_cupy_generator(seed_here))

        backend_name = getattr(
            gens_slot[0].bit_generator, "__class__", type(gens_slot[0].bit_generator)
        ).__name__
        device_counters["rng_model"] = f"{backend_name}_slot_bound"

    # LUT 准备
    if use_lut:
        dE_values = cp.asarray([-8.0, -4.0, 0.0, 4.0, 8.0], dtype=cp.float64)
        arg = cp.clip(-float(beta_scalar) * dE_values, _MIN_EXP_ARG, _MAX_EXP_ARG)
        lut_prob = cp.exp(arg).astype(dtype_float)

    # Chunking 设置
    if rng_chunk_replicas is not None:
        chunk = int(rng_chunk_replicas)
    else:
        chunk = R

    # Buffer 预分配
    est_bytes = chunk * L * L * np.dtype(dtype_float).itemsize
    if est_bytes > (1 << 28):  # > 256MB
        warnings.warn(
            f"High GPU memory usage for RNG buffer: {est_bytes / 1e6:.1f} MB. "
            "Consider reducing rng_chunk_replicas.",
            ResourceWarning,
        )

    u_blk_buffer = cp.zeros((chunk, L, L), dtype=dtype_float)

    # ---------- 3. 主循环 ----------
    for _sweep in range(int(n_sweeps)):
        if checkerboard:
            for parity in (0, 1):
                mask = parity_mask[parity]
                mc = int(mask_count[parity])

                s = spins.astype(cp.int32, copy=False)
                nbr = (
                    cp.roll(s, 1, 1)
                    + cp.roll(s, -1, 1)
                    + cp.roll(s, 1, 2)
                    + cp.roll(s, -1, 2)
                )

                if use_lut:
                    dE_data = (2 * s * nbr).astype(cp.int32, copy=False)
                else:
                    dE_data = (
                        2.0
                        * s.astype(dtype_float, copy=False)
                        * (nbr.astype(dtype_float, copy=False) + float(h))
                    )

                if vectorized_rng:
                    for start in range(0, R, chunk):
                        end = min(start + chunk, R)
                        real_chunk = end - start

                        # 生成随机数
                        u_vals = gen_vec.random(
                            (real_chunk, mc), dtype=dtype_float
                        )

                        # 复用 Buffer
                        u_blk = u_blk_buffer[:real_chunk]
                        u_blk[:, mask] = u_vals

                        _apply_update(
                            spins,
                            start,
                            end,
                            u_blk,
                            use_lut,
                            dE_data,
                            lut_prob if use_lut else None,
                            beta_arr,
                            device_counters,
                            mask,
                            dtype_float,
                        )
                        device_counters["attempts"][start:end] += mc

                else:
                    # Slot-bound (Chunked Batch)
                    for start in range(0, R, chunk):
                        end = min(start + chunk, R)
                        real_chunk = end - start

                        # 批量生成随机数 (List + Stack)
                        u_vals_list = [
                            gens_slot[r].random((mc,), dtype=dtype_float)
                            for r in range(start, end)
                        ]
                        u_vals_stack = cp.stack(u_vals_list, axis=0)

                        u_blk = u_blk_buffer[:real_chunk]
                        u_blk[:, mask] = u_vals_stack

                        _apply_update(
                            spins,
                            start,
                            end,
                            u_blk,
                            use_lut,
                            dE_data,
                            lut_prob if use_lut else None,
                            beta_arr,
                            device_counters,
                            mask,
                            dtype_float,
                        )
                        device_counters["attempts"][start:end] += mc

        else:
            # Non-checkerboard (Full Sweep)
            s = spins.astype(cp.int32, copy=False)
            nbr = (
                cp.roll(s, 1, 1)
                + cp.roll(s, -1, 1)
                + cp.roll(s, 1, 2)
                + cp.roll(s, -1, 2)
            )

            if use_lut:
                dE_data = (2 * s * nbr).astype(cp.int32, copy=False)
            else:
                dE_data = (
                    2.0
                    * s.astype(dtype_float, copy=False)
                    * (nbr.astype(dtype_float, copy=False) + float(h))
                )

            if vectorized_rng:
                for start in range(0, R, chunk):
                    end = min(start + chunk, R)
                    real_chunk = end - start

                    u_blk = u_blk_buffer[:real_chunk]
                    u_blk[:] = gen_vec.random(
                        (real_chunk, L, L), dtype=dtype_float
                    )

                    _apply_update(
                        spins,
                        start,
                        end,
                        u_blk,
                        use_lut,
                        dE_data,
                        lut_prob if use_lut else None,
                        beta_arr,
                        device_counters,
                        None,
                        dtype_float,
                    )
                    device_counters["attempts"][start:end] += N_sites

            else:
                for start in range(0, R, chunk):
                    end = min(start + chunk, R)
                    real_chunk = end - start

                    u_vals_list = [
                        gens_slot[r].random((L, L), dtype=dtype_float)
                        for r in range(start, end)
                    ]
                    u_blk = u_blk_buffer[:real_chunk]
                    u_blk[:] = cp.stack(u_vals_list, axis=0)

                    _apply_update(
                        spins,
                        start,
                        end,
                        u_blk,
                        use_lut,
                        dE_data,
                        lut_prob if use_lut else None,
                        beta_arr,
                        device_counters,
                        None,
                        dtype_float,
                    )
                    device_counters["attempts"][start:end] += N_sites

        # 累加观测
        E_inst = device_energy(spins, h=h)
        M_inst = device_magnetization(spins)
        device_counters["E_sum"] += E_inst.astype(
            device_counters["E_sum"].dtype, copy=False
        )
        device_counters["M_sum"] += M_inst.astype(
            device_counters["M_sum"].dtype, copy=False
        )

    # 结果封装
    rng_per_sweep = N_sites
    rng_consumed_per_rep = int(n_sweeps) * int(rng_per_sweep)

    # 让 rng_consumed 与其他计数器类型一致：设备端 CuPy 数组 (int64)
    device_counters["rng_consumed"] = cp.full(
        (int(R),),
        int(rng_consumed_per_rep),
        dtype=cp.int64,
    )

    if replica_counters is not None:
        try:
            rc_arr = cp.asarray(replica_counters, dtype=cp.int64)
            updated = (rc_arr + device_counters["rng_consumed"]).get()
            meta = {"replica_counters": [int(x) for x in updated.tolist()]}
            return spins, (device_counters, meta)
        except Exception:
            pass

    return spins, device_counters


# --- 辅助函数：提取重复的 update 逻辑 ---
def _apply_update(
    spins: cp.ndarray,
    start: int,
    end: int,
    u_blk: cp.ndarray,
    use_lut: bool,
    dE_data: cp.ndarray,
    lut_prob: Optional[cp.ndarray],
    beta_arr: cp.ndarray,
    device_counters: Dict[str, Any],
    mask: Optional[cp.ndarray],
    dtype_float,
) -> None:
    if use_lut:
        idx_map_blk = ((dE_data[start:end] + 8) // 4).astype(cp.int32, copy=False)
        prob_blk = lut_prob[idx_map_blk]
        accept_blk = (dE_data[start:end] <= 0) | (u_blk < prob_blk)
    else:
        arg_blk = (
            -beta_arr[start:end].astype(dtype_float, copy=False) * dE_data[start:end]
        )
        prob_blk = cp.exp(cp.clip(arg_blk, _MIN_EXP_ARG, _MAX_EXP_ARG))
        accept_blk = (dE_data[start:end] <= 0.0) | (u_blk < prob_blk)

    if mask is not None:
        flip_mask_blk = accept_blk & mask
    else:
        flip_mask_blk = accept_blk

    mult_blk = cp.where(flip_mask_blk, -1, 1).astype(spins.dtype, copy=False)
    spins[start:end] *= mult_blk

    # Device-side accumulation
    device_counters["accepts"][start:end] += cp.sum(
        flip_mask_blk, axis=(1, 2)
    ).astype(device_counters["accepts"].dtype)


def _apply_update_single(
    spins: cp.ndarray,
    r: int,
    u_r: cp.ndarray,
    use_lut: bool,
    dE_int: Optional[cp.ndarray],
    dE_float: Optional[cp.ndarray],
    lut_prob: Optional[cp.ndarray],
    beta_arr: cp.ndarray,
    device_counters: Dict[str, Any],
    mask: Optional[cp.ndarray],
    dtype_float,
) -> None:
    # 保留用于可能的单副本回退
    pass


def legacy_metropolis_cpu(*args,  **kwargs) -> None:
    raise RuntimeError("GPU module does not support CPU fallback.")

