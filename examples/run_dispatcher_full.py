
# -*- coding: utf-8 -*-
"""
test_dispatcher_full.py
-----------------------
用当前 dispatcher（无 resolve()）实现与 run_dispatcher_full.py **等价功能**的测试脚本：
- 校验：算法名规范化、PBC+棋盘格要求 L 偶数、h≠0 禁簇；
- 规划：生成全局温标（linear/geom）、按 worker 切片；
- 派发：使用 dispatcher.make_replica_seeds(master→worker→replica) 确定性派生种子；
- 执行：每个 worker 启动 HybridREMCSimulator.run(...)；
- 合并：尝试用 data_manager._orchestrate_worker_merge 流式合并到一个 HDF5；
- 跨平台：spawn + __main__ 守卫，macOS/Windows 可运行。

用法：
    python examples/test_dispatcher_full.py           # 单进程
    python examples/test_dispatcher_full.py mp        # 多进程（按 N_WORKERS 切片）
"""

from __future__ import annotations
import os, sys, time, json, logging, multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from numpy.random import SeedSequence

# 依赖库
from ising_fss.simulation import dispatcher as disp
from ising_fss.simulation.remc_simulator import HybridREMCSimulator

try:
    from ising_fss.data.data_manager import _orchestrate_worker_merge
except Exception:
    _orchestrate_worker_merge = None

# ====================== 可配置区域 ======================
L              = 64
H_FIELD        = 0.0
ALGO           = "metropolis"     # 'metropolis' | 'wolff' | 'swendsen_wang'（h≠0 禁簇）
BOUNDARY       = "pbc"
STRICT         = True

# 温度配置（二选一）：EXPLICIT_TEMPS 或 (T_MIN, T_MAX, REPLICAS, SPACING)
EXPLICIT_TEMPS: List[float] | None = None
T_MIN, T_MAX, REPLICAS, SPACING    = 2.0, 2.6, 24, "geom"  # 'geom' 或 'linear'

# Monte Carlo 步数 / 交换
EQUIL, PROD   = 4_096, 10_000
THIN          = 8
EXC_INT       = 10

# 并行与可复现
MASTER_SEED   = 20251113
N_WORKERS     = 2                 # mp 模式下建议 <= 物理核数
RUN_TAG       = "remc_full"
OUT_BASE      = Path("runs/test_dispatcher_full")

# ====================== 辅助函数 ======================
def _norm_algo(a: str) -> str:
    return disp.normalize_algo_name(a)

def _check_physics(algo_norm: str):
    if algo_norm in disp.CLUSTER_ALGOS and abs(H_FIELD) > disp.H_TOL:
        msg = f"Cluster algorithm '{algo_norm}' invalid for h={H_FIELD} != 0."
        if STRICT:
            raise ValueError(msg)
        else:
            import warnings; warnings.warn(msg + " → fallback to 'metropolis_sweep'.")
            return "metropolis_sweep"
    if BOUNDARY == "pbc" and algo_norm == "metropolis_sweep" and (L % 2 == 1):
        msg = "PBC + checkerboard Metropolis requires even L."
        if STRICT:
            raise ValueError(msg)
        else:
            import warnings; warnings.warn(msg)
    return algo_norm

def _gen_temperatures(temps: List[float] | None, T_min: float, T_max: float, n_total: int, spacing: str) -> List[float]:
    if temps is not None:
        arr = np.asarray(temps, dtype=float)
        if arr.ndim != 1 or arr.size < 2:
            raise ValueError("temperatures must be 1D, len >= 2")
        return arr.tolist()
    if n_total < 2 or T_min <= 0 or T_max <= 0 or T_min >= T_max:
        raise ValueError("Invalid temperature ladder params")
    if str(spacing).lower().startswith("geom"):
        g = (T_max / T_min) ** (1.0 / (n_total - 1))
        return [float(T_min * (g ** k)) for k in range(n_total)]
    return np.linspace(T_min, T_max, n_total, dtype=float).tolist()

def _spawn_safe():
    # macOS/Windows 默认 spawn，需要 __main__ 守卫；设置 spawn 更稳健
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    if os.name == "nt":
        mp.freeze_support()

def _setup_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("test_dispatcher_full")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt); sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    try:
        fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
        fh.setFormatter(fmt); fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    except Exception:
        pass
    return logger

# ====================== Worker 执行 ======================
def _worker_exec(worker_id: int, L: int, h_field: float, algo_norm: str,
                 temps: List[float], replica_seeds: List[int],
                 run_tmp_dir: Path, logger_name: str) -> Dict[str, Any]:
    logger = logging.getLogger(logger_name)
    worker_dir = run_tmp_dir / f"worker_{worker_id:03d}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[W{worker_id}] temps={len(temps)} [{min(temps):.4f},{max(temps):.4f}], rngs={len(replica_seeds)}")

    sim = HybridREMCSimulator(
        L=L,
        T_min=min(temps), T_max=max(temps),
        num_replicas=len(temps),
        temperatures=temps,
        algorithm=algo_norm,
        h=h_field,
        replica_seeds=replica_seeds,
    )
    sim.run(
        equilibration_steps=EQUIL,
        production_steps=PROD,
        exchange_interval=EXC_INT,
        thin=THIN,
        verbose=False,
        save_lattices=True,
        save_dir=str(worker_dir),
        worker_id=f"tdW{worker_id}",
    )
    return {"status": "success", "worker_id": worker_id, "dir": str(worker_dir)}

# ====================== 主流程 ======================
def main(multi_process: bool = False):
    _spawn_safe()

    algo_norm = _norm_algo(ALGO)
    algo_norm = _check_physics(algo_norm)

    # 全局温标与 worker 切片
    n_workers = int(N_WORKERS if multi_process else 1)
    temps_all = _gen_temperatures(EXPLICIT_TEMPS, T_MIN, T_MAX, REPLICAS, SPACING)
    chunks = np.array_split(np.asarray(temps_all, dtype=float), n_workers)

    # 输出目录
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    run_dir = OUT_BASE / f"{time.strftime('%Y%m%d-%H%M%S')}_{RUN_TAG}_{algo_norm}_L{L}"
    (run_dir / "tmp").mkdir(parents=True, exist_ok=True)
    logger = _setup_logging(run_dir)

    # 简要 metadata
    meta = {
        "algo": ALGO, "algo_norm": algo_norm, "backend": "auto(gpu if available)",
        "L": L, "h": H_FIELD, "boundary": BOUNDARY,
        "temperatures": temps_all, "spacing": SPACING,
        "master_seed": MASTER_SEED, "replicas": len(temps_all),
        "n_workers": n_workers,
        "gpu_available": bool(disp.gpu_available()),
    }
    with open(run_dir / "plan.metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # 为每个 worker 派生独立 worker_seed，再派生 replica_seeds
    w_seqs = SeedSequence(int(MASTER_SEED)).spawn(n_workers)
    tasks: List[Tuple[int, int, float, str, List[float], List[int], Path, str]] = []
    for wid, (wseq, ch) in enumerate(zip(w_seqs, chunks)):
        worker_seed = int(wseq.generate_state(1, dtype=np.uint32)[0])
        temps = ch.tolist()
        # 用 dispatcher.make_replica_seeds 由 worker_seed 派生本 worker 的 per-replica 种子
        replica_seeds = disp.make_replica_seeds(worker_seed, len(temps))
        tasks.append((wid, L, H_FIELD, algo_norm, temps, replica_seeds, run_dir / "tmp", logger.name))

    # 执行
    t0 = time.time()
    if multi_process and len(tasks) > 1:
        logger.info(f"启动多进程：workers = {len(tasks)}")
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(tasks)) as pool:
            results = pool.starmap(_worker_exec, tasks)
    else:
        logger.info("单进程执行")
        results = [_worker_exec(*tasks[0])]

    ok = sum(1 for r in results if r.get("status") == "success")
    logger.info(f"完成：{ok}/{len(results)} workers，用时 {time.time()-t0:.2f}s")

    # 合并（可选）
    if _orchestrate_worker_merge is not None:
        try:
            final_h5 = _orchestrate_worker_merge(
                output_base_dir=run_dir,
                tmp_subdir="tmp",
                merged_subdir="merged",
            )
            logger.info(f"✓ 合并完成：{final_h5}")
        except Exception as e:
            logger.warning(f"合并失败/跳过：{e}")
    else:
        logger.info("未找到 _orchestrate_worker_merge，跳过合并。")

    logger.info(f"输出目录：{run_dir}")

if __name__ == "__main__":
    multi = len(sys.argv) > 1 and sys.argv[1].lower().startswith("m")
    main(multi_process=multi)

