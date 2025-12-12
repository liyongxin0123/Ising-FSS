
# -*- coding: utf-8 -*-
"""
run_dispatcher_remc.py
----------------------------------------------------------
功能与原脚本相同：
  - 统一规范/校验：算法名、PBC+棋盘 L 偶数、h≠0 禁簇；
  - 生成温表（显式或区间+spacing）；
  - 用 SeedSequence 由 master 派生 per-replica 种子（单进程单 worker）；
  - 根据 BACKEND=auto/cpu/gpu 选择 REMC 执行器（GPU 失败时可回退）;
  - 运行 Hybrid/GPU REMC，并将中间结果写入 run_dir/tmp 方便后续合并。

用法：
    python examples/run_dispatcher_remc.py
"""

from __future__ import annotations
from pathlib import Path
import json, time, warnings
from typing import Dict, Any, List

import numpy as np
from numpy.random import SeedSequence

# 1) dispatcher：用于算法名规范化、GPU 可用性检测、种子派发等辅助
from ising_fss.simulation import dispatcher
# 2) REMC 执行器
from ising_fss.simulation.remc_simulator import HybridREMCSimulator
# 3) （可选）GPU 执行器
try:
    from ising_fss.simulation.gpu_remc_simulator import GPU_REMC_Simulator
except Exception:
    GPU_REMC_Simulator = None  # 按需回退

# ====================== 可配置区域 ======================
OUTDIR        = Path("runs/dispatcher_remc_remc")
L             = 64              # PBC+Metropolis 需偶数
H_FIELD       = 0.0
ALGO          = "metropolis"    # h=0 也可 wolff；GPU 仅支持 Metropolis
BOUNDARY      = "pbc"
BACKEND       = "auto"          # 'cpu'|'gpu'|'auto'
STRICT        = True            # 违反口径立即报错；想宽松回退可改 False

# 温表：显式 or 由区间+spacing 生成（二选一）
TEMPS_EXPL: List[float] | None = None
T_MIN, T_MAX, N_REPLICAS, SPACING = 2.0, 2.6, 24, "geom"

# MC 参数
EQUIL, PROD, THIN, EXC_INT = 8192, 20000, 8, 10

# 可复现
MASTER_SEED   = 20251112
RUN_TAG       = "cpu_or_gpu_auto"  # 出现在输出目录名里

# ====================== 辅助函数 ======================
def _normalize_and_check() -> str:
    algo_norm = dispatcher.normalize_algo_name(ALGO)
    # h≠0 禁簇
    if algo_norm in dispatcher.CLUSTER_ALGOS and abs(H_FIELD) > dispatcher.H_TOL:
        msg = f"Cluster algorithm '{algo_norm}' invalid for h={H_FIELD} != 0."
        if STRICT:
            raise ValueError(msg)
        warnings.warn(msg + " → fallback to 'metropolis_sweep'.")
        algo_norm = "metropolis_sweep"
    # PBC+棋盘 Metropolis 要求 L 为偶数
    if BOUNDARY == "pbc" and algo_norm == "metropolis_sweep" and (L % 2 == 1):
        msg = "PBC + checkerboard Metropolis requires even L."
        if STRICT:
            raise ValueError(msg)
        warnings.warn(msg)
    return algo_norm

def _gen_temperatures() -> List[float]:
    if TEMPS_EXPL is not None:
        arr = np.asarray(TEMPS_EXPL, dtype=float)
        if arr.ndim != 1 or arr.size < 2:
            raise ValueError("TEMPS_EXPL must be 1D with length >= 2")
        return arr.tolist()
    if N_REPLICAS < 2 or T_MIN <= 0 or T_MAX <= 0 or T_MIN >= T_MAX:
        raise ValueError("Invalid temperature ladder parameters.")
    if str(SPACING).lower().startswith("geom"):
        g = (T_MAX / T_MIN) ** (1.0 / (N_REPLICAS - 1))
        return [float(T_MIN * (g ** k)) for k in range(N_REPLICAS)]
    return np.linspace(T_MIN, T_MAX, N_REPLICAS, dtype=float).tolist()

def _select_backend(algo_norm: str) -> str:
    d = (BACKEND or "auto").lower()
    if d == "cpu":
        return "cpu"
    if d == "gpu":
        if algo_norm != "metropolis_sweep":
            if STRICT:
                raise ValueError("GPU backend currently only supports 'metropolis_sweep'.")
            warnings.warn("GPU backend only supports 'metropolis_sweep'; falling back to CPU.")
            return "cpu"
        if not dispatcher.gpu_available():
            if STRICT:
                raise RuntimeError("Requested GPU backend but GPU/CuPy not available.")
            warnings.warn("GPU/CuPy not available; falling back to CPU.")
            return "cpu"
        return "gpu"
    # auto
    if algo_norm == "metropolis_sweep" and dispatcher.gpu_available():
        return "gpu"
    return "cpu"

# ====================== 主流程 ======================
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # 1) 统一“规划”
    algo_norm = _normalize_and_check()
    temperatures = _gen_temperatures()
    backend = _select_backend(algo_norm)

    # 单 worker：直接派生 per-replica 种子
    replica_seeds = dispatcher.make_replica_seeds(MASTER_SEED, len(temperatures))

    # 简要 metadata（与原 plan.metadata() 等价用途）
    plan_meta: Dict[str, Any] = {
        "algo": ALGO,
        "algo_norm": algo_norm,
        "backend": backend,
        "boundary": BOUNDARY,
        "L": L,
        "h": float(H_FIELD),
        "temperatures": temperatures,
        "replicas": len(temperatures),
        "master_seed": int(MASTER_SEED),
    }

    # 2) 输出目录（唯一）
    run_dir = OUTDIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{RUN_TAG}_{backend}_{algo_norm}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tmp").mkdir(exist_ok=True)

    with open(run_dir / "plan.metadata.json", "w", encoding="utf-8") as f:
        json.dump(plan_meta, f, indent=2, ensure_ascii=False)

    # 3) 根据 backend 选择执行器
    SimulatorClass = HybridREMCSimulator
    if backend == "gpu" and GPU_REMC_Simulator is not None:
        SimulatorClass = GPU_REMC_Simulator
    elif backend == "gpu" and GPU_REMC_Simulator is None:
        warnings.warn("GPU backend selected but GPU_REMC_Simulator not importable; falling back to CPU.")
        SimulatorClass = HybridREMCSimulator

    sim = SimulatorClass(
        L=L,
        T_min=min(temperatures), T_max=max(temperatures),
        num_replicas=len(temperatures),
        algorithm=algo_norm,
        temperatures=temperatures,        # 显式温标
        h=H_FIELD,
        replica_seeds=replica_seeds,      # **链绑定 RNG**
    )

    sim.run(
        equilibration_steps=EQUIL,
        production_steps=PROD,
        exchange_interval=EXC_INT,
        thin=THIN,
        verbose=False,
        save_lattices=True,
        save_dir=str(run_dir / "tmp"),
        worker_id=f"dispatcher_single",
    )

    # 4)（可选）合并/导出
    try:
        from ising_fss.data.data_manager import _orchestrate_worker_merge
        final_h5 = _orchestrate_worker_merge(
            output_base_dir=run_dir,
            tmp_subdir="tmp",
            merged_subdir="merged",
        )
        print("✓ merged HDF5:", final_h5)
    except Exception as e:
        print("[INFO] merge skipped / failed:", e)

if __name__ == "__main__":
    main()

