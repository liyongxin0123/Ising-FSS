# -*- coding: utf-8 -*-
"""
    跨晶格尺寸并行任务调度器（有限尺寸标度分析专用）

本模块负责跨 $L$ (晶格尺寸) 的任务调度，用于有限尺寸标度分析 (FSS) 的数据生产。

实现功能：
    - 强制使用 ``multiprocessing.get_context("spawn")``，兼容 CUDA、无 fork 死锁
    - 从主种子自动派生子任务种子（SeedSequence），保证全局可复现
    - 每个 L 独立运行完整 REMC 模拟，支持断点续传
    - 自动生成 checkpoint 文件名（含物理参数哈希），避免冲突
    - 任务粒度灵活：支持单 L 多副本或多 L 并行
    - 确定性随机性: 使用 `numpy.random.SeedSequence` 从主种子 (Master Seed) 派生出子种子，确保所有子进程的 RNG 状态可预测且无重叠。

主要入口:
- `across_L`: 并行运行多个 L 的模拟任务并汇总结果。
"""

from __future__ import annotations

import os
import sys
import traceback
import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import h5py

try:
    # numpy >= 1.17
    from numpy.random import SeedSequence  # type: ignore
except Exception:  # 极端防御
    SeedSequence = None  # type: ignore


# -----------------------------------------------------------------------------
# 动态导入模拟器类
# -----------------------------------------------------------------------------
def _load_simulator_class():
    """
    宽容地尝试导入常见的模拟器类，按优先级寻找并返回类对象。
    找不到时抛出 ImportError。
    """
    # 1) 包内相对导入（首选，避免串到 site-packages/工作目录同名模块）
    try:
        from . import remc_simulator as _rem  # type: ignore
        for name in ("HybridREMCSimulator", "REMC", "GPU_REMC_Simulator"):
            if hasattr(_rem, name):
                return getattr(_rem, name)
    except Exception:
        pass

    try:
        from . import gpu_remc_simulator as _grem  # type: ignore
        for name in ("GPU_REMC_Simulator", "HybridREMCSimulator"):
            if hasattr(_grem, name):
                return getattr(_grem, name)
    except Exception:
        pass

    # 2) 绝对导入（兜底）
    attempts = [
        ("remc_simulator", "HybridREMCSimulator"),
        ("remc_simulator", "REMC"),
        ("gpu_remc_simulator", "GPU_REMC_Simulator"),
        ("gpu_remc_simulator", "HybridREMCSimulator"),
        ("remc_simulator", "GPU_REMC_Simulator"),
    ]
    for modname, attr in attempts:
        try:
            module = __import__(modname, fromlist=[attr])
            cls = getattr(module, attr)
            return cls
        except Exception:
            continue

    raise ImportError(
        "无法导入模拟器类：请确保 remc_simulator.HybridREMCSimulator 或 "
        "gpu_remc_simulator.GPU_REMC_Simulator 可用。"
    )


# -----------------------------------------------------------------------------
# Checkpoint 命名与路径工具
# -----------------------------------------------------------------------------
def _short_hash_for_temps(temps: Optional[Sequence[float]]) -> str:
    """
    对显式温度列生成短哈希（8 hex），用于文件名防冲突；temps 为 None 则返回空串。
    """
    if temps is None:
        return ""
    try:
        arr = np.asarray(list(temps), dtype=np.float64).tobytes()
    except Exception:
        arr = repr(list(temps)).encode("utf-8", errors="ignore")
    h = hashlib.sha1(arr).hexdigest()[:8]
    return h


def _checkpoint_basename(
    L: int,
    T_min: float,
    T_max: float,
    num_replicas: int,
    algorithm: str,
    h: float,
    spacing: str,
    temps: Optional[Sequence[float]],
) -> str:
    """
    生成稳定的 checkpoint 基名（不含目录）。显式温度列会附加短哈希。
    """
    algo = str(algorithm).strip().replace(" ", "_")
    spc = str(spacing).strip()
    if temps is None:
        return (
            f"remc_L{L}_T{T_min:.6f}-{T_max:.6f}_R{num_replicas}_h{h:.6f}_{algo}_{spc}.ckpt.json"
        )
    else:
        th = _short_hash_for_temps(temps)
        return f"remc_L{L}_Texpl_{len(temps)}R_h{h:.6f}_{algo}_{spc}_{th}.ckpt.json"


def _checkpoint_path(
    checkpoint_dir: str,
    basename: Optional[str],
    L: int,
    T_min: float,
    T_max: float,
    num_replicas: int,
    algorithm: str,
    h: float,
    spacing: str,
    temps: Optional[Sequence[float]],
) -> str:
    """
    组合得到完整 checkpoint 路径。若显式提供 basename 则直接使用。
    """
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    if basename and basename.strip():
        name = basename.strip()
        # 确保 .json 结尾（与模拟器实现一致）
        if not name.endswith(".json"):
            if name.endswith(".ckpt"):
                name = name + ".json"
            elif not name.endswith(".ckpt.json"):
                name = name + ".ckpt.json"
    else:
        name = _checkpoint_basename(
            L, T_min, T_max, num_replicas, algorithm, h, spacing, temps
        )
    return os.path.join(checkpoint_dir, name)


# -----------------------------------------------------------------------------
# 从 checkpoint 读取 replica_seeds（用于保证恢复时种子精确一致）
# -----------------------------------------------------------------------------
def _load_replica_seeds_from_checkpoint(path: str) -> Optional[List[int]]:
    """
    尝试从 checkpoint HDF5 文件中读取 replica_seeds。

    返回:
      - list[int]：成功解析的种子序列
      - None：文件不存在 / 结构不符 / 无法解析时
    """
    if not os.path.exists(path):
        return None
    try:
        f = h5py.File(path, "r")
    except Exception:
        return None

    try:
        if "remc_checkpoint" not in f:
            return None
        grp = f["remc_checkpoint"]
        seeds_raw = grp.attrs.get("replica_seeds", None)
        if seeds_raw is None:
            return None
        if isinstance(seeds_raw, (bytes, bytearray, np.bytes_)):
            try:
                seeds_raw = seeds_raw.decode("utf-8")
            except Exception:
                seeds_raw = str(seeds_raw)
        if not isinstance(seeds_raw, str):
            return None
        try:
            vals = json.loads(seeds_raw)
        except Exception:
            return None
        return [int(x) for x in vals]
    except Exception:
        return None
    finally:
        try:
            f.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Worker: run a single-L simulation (Pool worker)  —— 顶层函数，spawn 友好
# -----------------------------------------------------------------------------
def run_single_L_simulation(args: Sequence[Any]) -> Tuple[int, Any]:
    """
    Worker function suitable for Pool.map/imap.
    Accepts a sequence of positional arguments for backward compatibility.

    Returns:
      (L, results) where results is sim.analyze() output, or {"error": "..."} on failure.

    Supported positional patterns (robust parsing):
      legacy minimal (>=8): (L, T_min, T_max, num_replicas, equil, prod, algorithm, nproc [, seed])
      extended: L, T_min, T_max, num_replicas, equil, prod, algorithm, nproc,
                h, exchange_interval, thin, spacing, temperatures_or_None, [child_seed],
                [replica_seeds], [checkpoint_dir], [resume_if_exists], [checkpoint_basename], [checkpoint_final]
    """
    L_val = None
    sim = None  # 用于异常时尝试保存紧急 checkpoint
    try:
        if len(args) < 8:
            raise ValueError(f"期待至少 8 个位置参数，收到 {len(args)}: {args}")

        # 基础参数
        (
            L_val,
            T_min,
            T_max,
            num_replicas,
            equilibration,
            production,
            algorithm,
            nproc,
        ) = args[:8]

        # 默认
        h = 0.0
        exchange_interval = 1
        thin = 1
        spacing = "geom"
        temperatures = None
        child_seed = None
        replica_seeds = None  # 每个 replica 的 seed 列表

        # 新增：checkpoint 相关
        checkpoint_dir: Optional[str] = None
        resume_if_exists: bool = True
        checkpoint_basename: Optional[str] = None
        checkpoint_final: bool = True  # 运行完成后是否保存

        # 解析扩展参数（宽容）
        idx = 8
        # 9 参：第 9 个要么是 child_seed，要么是 replica_seeds
        if len(args) == 9:
            candidate = args[8]
            if isinstance(candidate, (list, tuple)) and len(candidate) == int(
                num_replicas
            ):
                replica_seeds = list(candidate)
            else:
                try:
                    child_seed = int(candidate)
                except Exception:
                    child_seed = None
        elif len(args) > 9:
            try:
                if idx < len(args):
                    h = float(args[idx])
                    idx += 1
                if idx < len(args):
                    exchange_interval = int(args[idx])
                    idx += 1
                if idx < len(args):
                    thin = int(args[idx])
                    idx += 1
                if idx < len(args):
                    spacing = str(args[idx])
                    idx += 1
                if idx < len(args):
                    temperatures = args[idx]
                    idx += 1
                if idx < len(args):
                    candidate = args[idx]
                    idx += 1
                    if isinstance(candidate, (list, tuple)) and len(candidate) == int(
                        num_replicas
                    ):
                        replica_seeds = list(candidate)
                    else:
                        try:
                            child_seed = int(candidate)
                        except Exception:
                            child_seed = None
                # === 只有在前面还没有成功解析 replica_seeds 时，才尝试 maybe_replica ===
                if idx < len(args) and replica_seeds is None:
                    maybe_replica = args[idx]
                    idx += 1
                    if isinstance(maybe_replica, (list, tuple)) and len(
                        maybe_replica
                    ) == int(num_replicas):
                        replica_seeds = list(maybe_replica)
                # checkpoint 可选参数（依次追加）
                if idx < len(args):
                    checkpoint_dir = args[idx]
                    idx += 1
                if idx < len(args):
                    resume_if_exists = bool(args[idx])
                    idx += 1
                if idx < len(args):
                    checkpoint_basename = args[idx]
                    idx += 1
                if idx < len(args):
                    checkpoint_final = bool(args[idx])
                    idx += 1
            except Exception:
                pass

        # 规范化类型
        L = int(L_val)
        T_min = float(T_min)
        T_max = float(T_max)
        num_replicas = int(num_replicas)
        equilibration = int(equilibration)
        production = int(production)
        algorithm = str(algorithm)
        exchange_interval = int(exchange_interval)
        thin = int(thin)
        spacing = str(spacing)
        if temperatures is not None:
            try:
                temperatures = [float(x) for x in list(temperatures)]
            except Exception:
                pass

        # 加载模拟器类
        Simulator = _load_simulator_class()

        print(
            f"[worker pid={os.getpid()}] Starting L={L}  seed={child_seed} "
            f"replica_seeds_provided={replica_seeds is not None}  h={h}  "
            f"checkpoint={'ON' if checkpoint_dir else 'OFF'}",
            flush=True,
        )

        # 构造器尝试
        candidates: List[Dict[str, Any]] = []

        kw1 = dict(
            L=L,
            T_min=T_min,
            T_max=T_max,
            num_replicas=num_replicas,
            algorithm=algorithm,
            spacing=spacing,
            temperatures=temperatures,
            h=h,
            seed=(int(child_seed) if child_seed is not None else None),
        )
        if replica_seeds is not None:
            kw1["replica_seeds"] = replica_seeds
        candidates.append(kw1)

        kw2 = dict(
            L=L,
            T_min=T_min,
            T_max=T_max,
            num_replicas=num_replicas,
            algorithm=algorithm,
            spacing=spacing,
            h=h,
            seed=(int(child_seed) if child_seed is not None else None),
        )
        if replica_seeds is not None:
            kw2["replica_seeds"] = replica_seeds
        candidates.append(kw2)

        kw3 = dict(
            L=L,
            T_min=T_min,
            T_max=T_max,
            num_replicas=num_replicas,
            algorithm=algorithm,
        )
        if replica_seeds is not None:
            kw3["replica_seeds"] = replica_seeds
        candidates.append(kw3)

        kw4 = dict(L=L, T_min=T_min, T_max=T_max, num_replicas=num_replicas)
        if replica_seeds is not None:
            kw4["replica_seeds"] = replica_seeds
        candidates.append(kw4)

        last_exc: Optional[BaseException] = None
        for kw in candidates:
            try:
                sim = Simulator(**kw)
                break
            except TypeError as te:
                last_exc = te
                continue
            except Exception as e:
                last_exc = e
                break

        if sim is None:
            try:
                sim = Simulator()
                for attr, val in (
                    ("L", L),
                    ("T_min", T_min),
                    ("T_max", T_max),
                    ("num_replicas", num_replicas),
                    ("h", h),
                    ("spacing", spacing),
                ):
                    try:
                        if hasattr(sim, attr):
                            setattr(sim, attr, val)
                    except Exception:
                        pass
                if replica_seeds is not None:
                    try:
                        if hasattr(sim, "replica_seeds"):
                            setattr(sim, "replica_seeds", replica_seeds)
                        else:
                            sim.replica_seeds = replica_seeds  # type: ignore
                    except Exception:
                        pass
                if child_seed is not None:
                    try:
                        if hasattr(sim, "seed"):
                            setattr(sim, "seed", int(child_seed))
                        else:
                            sim.seed = int(child_seed)  # type: ignore
                    except Exception:
                        pass
            except Exception as e:
                raise RuntimeError(
                    f"无法构造模拟器实例 (last error: {last_exc})"
                ) from e

        # ---------------------------
        # Checkpoint：恢复（若启用）
        # ---------------------------
        ckpt_path: Optional[str] = None

        # === 确保 checkpoint_dir 类型正常，否则直接禁用 checkpoint ===
        if checkpoint_dir is not None and not isinstance(checkpoint_dir, str):
            print(
                f"[worker pid={os.getpid()}] L={L} 收到非法 checkpoint_dir={checkpoint_dir!r}，已禁用 checkpoint",
                flush=True,
            )
            checkpoint_dir = None

        if checkpoint_dir:
            try:
                ckpt_path = _checkpoint_path(
                    checkpoint_dir=checkpoint_dir,
                    basename=checkpoint_basename,
                    L=L,
                    T_min=T_min,
                    T_max=T_max,
                    num_replicas=num_replicas,
                    algorithm=algorithm,
                    h=h,
                    spacing=spacing,
                    temps=temperatures,
                )
                if (
                    resume_if_exists
                    and os.path.exists(ckpt_path)
                    and hasattr(sim, "restore_from_checkpoint")
                ):
                    try:
                        notes = sim.restore_from_checkpoint(
                            ckpt_path
                        )  # type: ignore[attr-defined]
                        print(
                            f"[worker pid={os.getpid()}] L={L} 恢复 checkpoint 成功：{os.path.basename(ckpt_path)} "
                            f"(method={notes.get('method', 'unknown')}, ok={notes.get('ok', False)})",
                            flush=True,
                        )
                    except Exception as exc:
                        print(
                            f"[worker pid={os.getpid()}] L={L} 恢复 checkpoint 失败：{exc}",
                            flush=True,
                        )
            except Exception as exc:
                print(
                    f"[worker pid={os.getpid()}] L={L} 准备 checkpoint 失败：{exc}",
                    flush=True,
                )
                ckpt_path = None  # 继续运行但不保存

        # ---------------------------
        # 运行
        # ---------------------------
        try:
            run_kw = dict(
                equilibration_steps=int(equilibration),
                production_steps=int(production),
                exchange_interval=int(exchange_interval),
                thin=int(thin),
                verbose=False,
            )
            try:
                sim.run(**run_kw)
            except TypeError:
                try:
                    sim.run(int(equilibration), int(production))
                except TypeError:
                    sim.run()
        except Exception as e:
            raise RuntimeError(f"模拟运行失败: {e}")

        # ---------------------------
        # Checkpoint：保存（若启用）
        # ---------------------------
        if ckpt_path and checkpoint_final and hasattr(sim, "save_checkpoint"):
            try:
                sim.save_checkpoint(ckpt_path)  # type: ignore[attr-defined]
                print(
                    f"[worker pid={os.getpid()}] L={L} 已保存 checkpoint -> {os.path.basename(ckpt_path)}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[worker pid={os.getpid()}] L={L} 保存 checkpoint 失败：{exc}",
                    flush=True,
                )

        # 分析（稳健：先带 verbose 调用，TypeError 再降级）
        try:
            if hasattr(sim, "analyze"):
                try:
                    results = sim.analyze(verbose=False)
                except TypeError:
                    results = sim.analyze()
            else:
                results = getattr(
                    sim, "results", {"note": "no analyze method; returned simulator object"}
                )
        except Exception as e:
            raise RuntimeError(f"分析步骤失败: {e}")

        print(f"[worker pid={os.getpid()}] L={L} completed", flush=True)
        return int(L), results

    except Exception as e:
        # 捕获并返回结构化错误，避免 Pool 崩溃；必要时可尝试紧急落盘
        try:
            L_ret = int(L_val) if L_val is not None else -1
        except Exception:
            L_ret = -1
        tb = traceback.format_exc()
        sys.stderr.write(f"[worker pid={os.getpid()}] ERROR L={L_ret}: {e}\n{tb}\n")
        return int(L_ret), {"error": str(e), "traceback": tb}


# -----------------------------------------------------------------------------
# across_L: top-level orchestration  —— 统一强制 spawn 的进程池
# -----------------------------------------------------------------------------
def across_L(
    L_list: Sequence[int],
    T_min: float,
    T_max: float,
    num_replicas: int,
    equilibration: int,
    production: int,
    algorithm: str,
    n_processes_per_L: int = 1,
    seed: Optional[int] = None,
    h: float = 0.0,
    exchange_interval: int = 1,
    thin: int = 1,
    spacing: str = "geom",
    temperatures: Optional[Sequence[float]] = None,
    pool_size: Optional[int] = None,
    # ==== 新增：checkpoint 相关 ====
    checkpoint_dir: Optional[str] = None,
    resume_if_exists: bool = True,
    checkpoint_basename: Optional[str] = None,
    checkpoint_final: bool = True,
) -> Dict[int, Any]:
    """
    并行跑多个 L 的模拟并汇总结果（强制 spawn 上下文）。

    新增参数：
      - checkpoint_dir: str|None  指定目录则启用 checkpoint（JSON + .npz 边车由模拟器负责）
      - resume_if_exists: bool    若为 True 且对应文件存在，则在 run() 前先恢复
      - checkpoint_basename: str|None 自定义基名（.ckpt.json / .json 均可）；不指定则按物理配置自动生成
      - checkpoint_final: bool    运行完成后是否保存（默认保存）
    """
    Ls = [int(x) for x in L_list]
    if pool_size is None:
        try:
            cpu_cnt = os.cpu_count() or 1
        except Exception:
            cpu_cnt = 1
        pool_size = min(len(Ls), cpu_cnt)
    pool_size = max(1, int(pool_size))

    temps = None if temperatures is None else [float(x) for x in list(temperatures)]
    tasks: List[Tuple[Any, ...]] = []

    # 预创建 checkpoint 目录（若启用）
    if checkpoint_dir:
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
        except Exception as exc:
            print(
                f"[parallel] 警告：无法创建 checkpoint_dir={checkpoint_dir}，将忽略 checkpoint。原因：{exc}",
                file=sys.stderr,
            )
            checkpoint_dir = None  # 继续但禁用

    # 派生 seeds（无 master 时非确定性，有 master 时确定性）
    # 同时：若存在 checkpoint 且 resume_if_exists，则优先从 checkpoint 读取 replica_seeds，
    #       保证恢复时与文件内种子完全一致，避免 replica_seeds mismatch。
    if seed is None or SeedSequence is None:
        # 非确定性模式（未给 master seed）
        for L in Ls:
            # 1) 若可以恢复，从 checkpoint 读 seeds
            rep_seeds: Optional[List[int]] = None
            if checkpoint_dir and resume_if_exists:
                try:
                    ckpt_path = _checkpoint_path(
                        checkpoint_dir=checkpoint_dir,
                        basename=checkpoint_basename,
                        L=int(L),
                        T_min=float(T_min),
                        T_max=float(T_max),
                        num_replicas=int(num_replicas if temps is None else len(temps)),
                        algorithm=str(algorithm),
                        h=float(h),
                        spacing=str(spacing),
                        temps=temps,
                    )
                    rep_seeds = _load_replica_seeds_from_checkpoint(ckpt_path)
                except Exception:
                    rep_seeds = None

            # 2) 若 checkpoint 中没有 seeds，则退回到随机生成
            if rep_seeds is None:
                replica_rng = np.random.default_rng()
                rep_seeds = replica_rng.integers(
                    0, 2**31 - 1, size=num_replicas
                ).tolist()

            if temps is None:
                tasks.append(
                    (
                        int(L),
                        float(T_min),
                        float(T_max),
                        int(num_replicas),
                        int(equilibration),
                        int(production),
                        str(algorithm),
                        int(n_processes_per_L),
                        float(h),
                        int(exchange_interval),
                        int(thin),
                        str(spacing),
                        None,
                        rep_seeds,
                        checkpoint_dir,
                        bool(resume_if_exists),
                        checkpoint_basename,
                        bool(checkpoint_final),
                    )
                )
            else:
                tasks.append(
                    (
                        int(L),
                        float(T_min),
                        float(T_max),
                        len(temps),
                        int(equilibration),
                        int(production),
                        str(algorithm),
                        int(n_processes_per_L),
                        float(h),
                        int(exchange_interval),
                        int(thin),
                        str(spacing),
                        temps,
                        rep_seeds,
                        checkpoint_dir,
                        bool(resume_if_exists),
                        checkpoint_basename,
                        bool(checkpoint_final),
                    )
                )
    else:
        # 确定性模式：从 master SeedSequence 派生（推荐）
        ss_master = SeedSequence(int(seed))
        children = ss_master.spawn(len(Ls))
        for idx, L in enumerate(Ls):
            child_ss = children[idx]
            # child_seed（给模拟器可能用到的“主种子”，GPU 版本常用）
            try:
                st = child_ss.generate_state(1, dtype="uint32")
                child_seed = int(st[0])
            except Exception:
                ent = getattr(child_ss, "entropy", None)
                if isinstance(ent, int):
                    child_seed = int(ent & 0xFFFFFFFF)
                else:
                    child_seed = abs(hash(repr(child_ss))) & 0xFFFFFFFF

            # 1) 尝试从 checkpoint 读取 replica_seeds（若存在）
            rep_seeds: Optional[List[int]] = None
            if checkpoint_dir and resume_if_exists:
                try:
                    ckpt_path = _checkpoint_path(
                        checkpoint_dir=checkpoint_dir,
                        basename=checkpoint_basename,
                        L=int(L),
                        T_min=float(T_min),
                        T_max=float(T_max),
                        num_replicas=int(num_replicas if temps is None else len(temps)),
                        algorithm=str(algorithm),
                        h=float(h),
                        spacing=str(spacing),
                        temps=temps,
                    )
                    rep_seeds = _load_replica_seeds_from_checkpoint(ckpt_path)
                except Exception:
                    rep_seeds = None

            # 2) 若 checkpoint 未提供，则用 SeedSequence 派生每个 replica 的 seed
            if rep_seeds is None:
                try:
                    child_replica_ss = child_ss.spawn(int(num_replicas))
                    rep_seeds = []
                    for rch in child_replica_ss:
                        try:
                            st = rch.generate_state(1, dtype="uint32")
                            rep_seeds.append(int(st[0]))
                        except Exception:
                            ent = getattr(rch, "entropy", None)
                            if isinstance(ent, int):
                                rep_seeds.append(int(ent & 0xFFFFFFFF))
                            else:
                                rep_seeds.append(
                                    abs(hash(repr(rch))) & 0xFFFFFFFF
                                )
                except Exception:
                    try:
                        tmp_rng = np.random.default_rng(int(child_seed))
                        rep_seeds = tmp_rng.integers(
                            0, 2**31 - 1, size=int(num_replicas)
                        ).tolist()
                    except Exception:
                        rep_seeds = [
                            (abs(hash(f"{child_seed}_{i}")) & 0xFFFFFFFF)
                            for i in range(int(num_replicas))
                        ]

            if temps is None:
                tasks.append(
                    (
                        int(L),
                        float(T_min),
                        float(T_max),
                        int(num_replicas),
                        int(equilibration),
                        int(production),
                        str(algorithm),
                        int(n_processes_per_L),
                        float(h),
                        int(exchange_interval),
                        int(thin),
                        str(spacing),
                        None,
                        int(child_seed),
                        rep_seeds,
                        checkpoint_dir,
                        bool(resume_if_exists),
                        checkpoint_basename,
                        bool(checkpoint_final),
                    )
                )
            else:
                tasks.append(
                    (
                        int(L),
                        float(T_min),
                        float(T_max),
                        len(temps),
                        int(equilibration),
                        int(production),
                        str(algorithm),
                        int(n_processes_per_L),
                        float(h),
                        int(exchange_interval),
                        int(thin),
                        str(spacing),
                        temps,
                        int(child_seed),
                        rep_seeds,
                        checkpoint_dir,
                        bool(resume_if_exists),
                        checkpoint_basename,
                        bool(checkpoint_final),
                    )
                )

    if not tasks:
        return {}

    # 强制使用 spawn 上下文
    from multiprocessing import get_context

    ctx = get_context("spawn")
    results_map: Dict[int, Any] = {}

    with ctx.Pool(processes=pool_size) as pool:
        try:
            for L_res, data in pool.imap_unordered(run_single_L_simulation, tasks):
                results_map[int(L_res)] = data
        except Exception:
            traceback.print_exc()
            raise

    return results_map


# -----------------------------------------------------------------------------
# CLI quick demo (small smoke test)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import multiprocessing as mp

    # 在创建任何子进程/进程池之前，强制全局使用 spawn（双保险）
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # 已设置过（如在交互环境/上层应用），忽略
        pass
    mp.freeze_support()  # Windows/macOS 友好

    demo_Ls = [16, 32]
    try:
        out = across_L(
            L_list=demo_Ls,
            T_min=1.6,
            T_max=3.0,
            num_replicas=8,
            equilibration=200,
            production=500,
            algorithm="wolff",  # 注意：簇算法要求 h=0
            seed=20251020,
            h=0.0,
            exchange_interval=5,
            thin=2,
            spacing="geom",
            temperatures=None,
            pool_size=None,
            # ---- 演示 checkpoint（可选）----
            checkpoint_dir="checkpoints_demo",
            resume_if_exists=True,
            checkpoint_basename=None,  # 让系统按物理配置自动命名
            checkpoint_final=True,
        )
        print("\n=== across_L demo 返回键 ===")
        print(sorted(out.keys()))
        for L in demo_Ls:
            res = out.get(L)
            if isinstance(res, dict) and "error" in res:
                print(f"L={L} ERROR: {res.get('error')}")
            else:
                print(f"L={L} OK (结果结构由模拟器.analyze 决定)")
    except Exception as e:
        print("Demo failed (可能是模拟器不可用):", e, file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

