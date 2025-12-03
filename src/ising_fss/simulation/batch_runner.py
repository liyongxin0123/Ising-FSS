# -*- coding: utf-8 -*-
"""
批量生产任务启动器（兼容单机多卡与集群）

实现功能：
    - 自动创建唯一 worker 目录（含 PID + UUID）
    - 支持 demo 模式（不启动真实模拟器）
    - 三种运行模式：
        • run_workers       : 启动 N 个子进程
        • run_workers_demo  : 生成假数据用于测试合并
        • merge             : 合并所有 worker 输出
    - 与 data_manager 深度集成

Examples:
    $ python -m ising_fss.simulation.batch_runner --outdir run1 --mode run_workers --nworkers 8
    $ python -m ising_fss.simulation.batch_runner --outdir run1 --mode merge
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import uuid
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Union

import numpy as np

# 可选导入项目内 orchestrator
try:
    from ..data import data_manager  # type: ignore
except Exception:
    data_manager = None  # type: ignore

# 仅尝试明确的 HybridREMCSimulator
try:
    from .remc_simulator import HybridREMCSimulator  # type: ignore
except Exception:
    HybridREMCSimulator = None  # type: ignore

# 通过环境变量可强制 demo（不走模拟器）
FORCE_DEMO = os.getenv("ISING_FSS_FORCE_DEMO", "0") == "1"


# ----------------------------- 基础工具 -----------------------------
def make_unique_save_dir(base_out: Union[str, Path], tag: Optional[str] = None) -> str:
    tag_s = f"_{tag}" if tag else ""
    base = Path(base_out)
    base.mkdir(parents=True, exist_ok=True)
    worker_id = f"worker_pid{os.getpid()}_{uuid.uuid4().hex[:8]}{tag_s}"
    save_dir = base / "tmp" / worker_id
    save_dir.mkdir(parents=True, exist_ok=True)
    return str(save_dir)


def _atomic_write_text(path: Union[str, Path], text: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dirp = str(path.parent)
    with tempfile.NamedTemporaryFile("w", encoding=encoding, dir=dirp, delete=False) as tf:
        tmpname = tf.name
        tf.write(text)
        try:
            tf.flush()
            os.fsync(tf.fileno())
        except Exception:
            pass
    os.replace(tmpname, str(path))


def _atomic_write_bytes(path: Union[str, Path], data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dirp = str(path.parent)
    with tempfile.NamedTemporaryFile("wb", dir=dirp, delete=False) as tf:
        tmpname = tf.name
        tf.write(data)
        try:
            tf.flush()
            os.fsync(tf.fileno())
        except Exception:
            pass
    os.replace(tmpname, str(path))


def _generate_replica_seeds(num_replicas: int, base_seed: Optional[int] = None) -> List[int]:
    """
    生成长度为 num_replicas 的显式整数种子列表。
    - 若给出 base_seed，尽可能用 numpy.random.SeedSequence 派生确定性子种子；
    - 否则使用 default_rng 非确定生成。
    """
    try:
        from numpy.random import SeedSequence  # type: ignore
        if base_seed is not None:
            ss = SeedSequence(int(base_seed))
            children = ss.spawn(int(num_replicas))
            out: List[int] = []
            for ch in children:
                try:
                    st = ch.generate_state(1, dtype="uint32")
                    out.append(int(st[0]))
                except Exception:
                    out.append(abs(hash(repr(ch))) & 0xFFFFFFFF)
            return out
    except Exception:
        pass
    rng = np.random.default_rng(int(base_seed) if base_seed is not None else None)
    return rng.integers(0, 2**31 - 1, size=int(num_replicas)).astype(np.int64).tolist()


def _infer_T_window_around(T_center: float, width: float = 0.6) -> Tuple[float, float]:
    """
    从中心温度推一个对称窗口（默认 ±0.3）。
    """
    c = float(T_center)
    half = float(width) / 2.0
    return max(0.0, c - half), max(c + half, c + 1e-6)


# ----------------------------- demo 写入 -----------------------------
def _write_demo_outputs(base_path: Path, L: int, T: float, worker_index: int, tidx: int) -> None:
    """
    仅用于 demo：尽量写 HDF5（若 h5py 可用），否则写 NPZ，此外总是写 scalars_*.json。
    """
    try:
        # HDF5
        try:
            import h5py  # type: ignore
            arr = (np.arange(L * L).reshape((L, L)) + worker_index + tidx).astype(np.int32)
            tmp_h5 = base_path / f"result_{tidx}.h5.tmp"
            with h5py.File(str(tmp_h5), "w") as hf:
                grp = hf.require_group("entries").require_group(f"L={L}").require_group(f"T={T}")
                ds = grp.create_dataset("lattice", data=arr)
                grp.attrs["T"] = float(T)
                grp.attrs["h"] = float(0.0)
                ds.attrs["created_by"] = f"demo_worker_{worker_index}"
            os.replace(str(tmp_h5), str(base_path / f"result_{tidx}.h5"))
            wrote = True
        except Exception:
            wrote = False

        if not wrote:
            # NPZ
            tmp_npz = base_path / f"result_{tidx}.npz.tmp"
            np.savez_compressed(str(tmp_npz), lattice=(worker_index + tidx))
            os.replace(str(tmp_npz), str(base_path / f"result_{tidx}.npz"))

        # scalars
        scal = {
            "worker_index": int(worker_index),
            "task_index": int(tidx),
            "L": int(L),
            "T": float(T),
            "samples": int(100 + worker_index + tidx),
            "timestamp": time.time(),
        }
        _atomic_write_text(base_path / f"scalars_{tidx}.json", json.dumps(scal, ensure_ascii=False, indent=2))
    except Exception as e:
        _atomic_write_text(base_path / f"error_{tidx}.txt", f"demo write failed: {e}")


# ----------------------------- 子进程入口 -----------------------------
def run_worker_process(
    out_base: Union[str, Path],
    worker_index: int,
    L: int = 8,
    T: float = 2.27,              # demo 兼容：中心温度；若使用模拟器则派生 T_min/T_max
    equilibration: int = 100,
    production: int = 1000,
    exchange_interval: int = 10,
    thin: int = 1,
    algorithm: str = "metropolis_sweep",
    spacing: str = "geom",
    num_replicas: int = 4,
    h: float = 0.0,
    base_seed: Optional[int] = None,     # 可选：用于确定性派生 replica_seeds
    tasks: int = 1,                      # demo 兼容：每个 worker 写多个 demo 任务
    save_lattices: bool = False,         # 若用模拟器，是否保存格点
    bitgen_steps_per_uniform: Optional[int] = None,  # 若你已测得转换因子，可传入
) -> None:
    """
    子进程：若可用则运行 HybridREMCSimulator（显式 replica_seeds），否则 demo 写文件。
    """
    save_dir = Path(make_unique_save_dir(out_base, tag=f"w{worker_index}"))
    print(f"[worker {worker_index}] save_dir -> {save_dir}", flush=True)

    # 强制 demo 或未找到模拟器 -> demo 路径
    if FORCE_DEMO or (HybridREMCSimulator is None):
        if HybridREMCSimulator is None and not FORCE_DEMO:
            _atomic_write_text(save_dir / "sim_unrecognized.txt",
                               "remc_simulator not available; falling back to demo")
        for t in range(tasks):
            _write_demo_outputs(save_dir, L=L, T=T, worker_index=worker_index, tidx=t)
        print(f"[worker {worker_index}] demo outputs written -> {save_dir}", flush=True)
        return

    # 使用模拟器路径：准备参数
    try:
        T_min, T_max = _infer_T_window_around(T_center=T, width=0.6)  # ±0.3 的小窗口
        num_replicas = int(max(2, num_replicas))
        replica_seeds = _generate_replica_seeds(num_replicas=num_replicas, base_seed=base_seed)

        # 构造模拟器（使用你提供的显式签名）
        sim = HybridREMCSimulator(
            L=int(L),
            T_min=float(T_min),
            T_max=float(T_max),
            num_replicas=int(num_replicas),
            algorithm=str(algorithm),
            spacing=str(spacing),
            temperatures=None,                 # 使用 T_min/T_max+spacing
            h=float(h),
            replica_seeds=replica_seeds,       # 关键：显式种子
            buffer_flush=64,
            record_swap_history=False,
            bitgen_steps_per_uniform=(int(bitgen_steps_per_uniform) if bitgen_steps_per_uniform is not None else None),
        )

        # 运行（可按需保存格点）
        worker_id = f"worker_{os.getpid()}_{worker_index}"
        sim.run(
            equilibration_steps=int(equilibration),
            production_steps=int(production),
            exchange_interval=int(exchange_interval),
            thin=int(thin),
            verbose=False,
            save_lattices=bool(save_lattices),
            save_dir=str(save_dir) if save_lattices else None,
            worker_id=worker_id,
        )

        # 分析（非必须，但便于 smoke 的“有返回/有副产物”）
        try:
            _ = sim.analyze(verbose=False)
        except Exception as e_an:
            _atomic_write_text(save_dir / "analyze_warning.txt", f"analyze() failed or skipped: {e_an}")

        print(f"[worker {worker_index}] sim.run completed -> {save_dir}", flush=True)
    except Exception as e:
        # 构造/运行失败，写错误并回退 demo（保证 smoke 不中断）
        _atomic_write_text(save_dir / "error.txt", f"{e}")
        for t in range(tasks):
            _write_demo_outputs(save_dir, L=L, T=T, worker_index=worker_index, tidx=t)
        print(f"[worker {worker_index}] simulator failed, fallback to demo -> {save_dir}", flush=True)


# ----------------------------- 多进程 demo 入口 -----------------------------
def run_workers_demo(outdir: str, nworkers: int = 4, tasks_per_worker: int = 1, timeout: Optional[float] = None) -> None:
    """
    启动 nworkers 个子进程：若可用则运行模拟器；否则 demo 写文件。
    （统一 spawn 上下文）
    """
    print(f"Running demo: {nworkers} workers x {tasks_per_worker} tasks -> base outdir {outdir}")
    ctx = mp.get_context("spawn")
    procs: List[mp.Process] = []
    try:
        for i in range(int(nworkers)):
            p = ctx.Process(
                target=run_worker_process,
                args=(outdir, i),
                kwargs={"L": 8, "T": 2.27, "equilibration": 10, "production": 10,
                        "tasks": int(tasks_per_worker), "num_replicas": 4, "algorithm": "metropolis_sweep",
                        "save_lattices": False},
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join(timeout)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received; terminating workers...", flush=True)
        for p in procs:
            try: p.terminate()
            except Exception: pass
        for p in procs:
            try: p.join(1.0)
            except Exception: pass
    finally:
        for p in procs:
            if p.is_alive():
                try: p.join(0.1)
                except Exception: pass
    print("Demo workers finished.")


# ----------------------------- 合并（回退版） -----------------------------
def _fallback_merge_tmp_dirs(base_out: str) -> str:
    base = Path(base_out)
    tmp = base / "tmp"
    merged_root = base / "merged"
    merged_root.mkdir(parents=True, exist_ok=True)
    manifest = {"merged_at": time.time(), "entries": []}
    if not tmp.exists():
        raise FileNotFoundError(f"No tmp directory found under {base_out}")
    for child in tmp.iterdir():
        if not child.is_dir():
            continue
        dest = merged_root / child.name
        if dest.exists():
            dest = merged_root / f"{child.name}_{uuid.uuid4().hex[:6]}"
        try:
            shutil.copytree(child, dest)
            manifest["entries"].append({"src": str(child), "dest": str(dest)})
        except Exception:
            dest.mkdir(parents=True, exist_ok=True)
            for f in child.iterdir():
                try:
                    if f.is_file():
                        shutil.copy2(f, dest / f.name)
                except Exception:
                    pass
            manifest["entries"].append({"src": str(child), "dest": str(dest), "partial": True})
    manifest_path = merged_root / "manifest.json"
    _atomic_write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))
    return str(manifest_path)


# ----------------------------- CLI -----------------------------
def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Batch runner (spawn-only) with HybridREMCSimulator support and demo fallback.")
    parser.add_argument("--outdir", "-o", type=str, default="out", help="output base dir (will contain tmp/ and merged/)")
    parser.add_argument("--mode", "-m", type=str, choices=["run_workers_demo", "merge", "run_workers"], default="run_workers_demo")
    parser.add_argument("--nworkers", type=int, default=4, help="number of parallel workers for demo/run")
    parser.add_argument("--tasks", type=int, default=1, help="tasks per worker for demo path")
    parser.add_argument("--L", type=int, default=8, help="lattice size (passed to simulator)")
    parser.add_argument("--T", type=float, default=2.27, help="center temperature (sim path will derive a window)")
    parser.add_argument("--equil", type=int, default=100, help="equilibration steps for sim.run")
    parser.add_argument("--prod", type=int, default=1000, help="production steps for sim.run")
    parser.add_argument("--exchange_interval", type=int, default=10, help="exchange interval for REMC")
    parser.add_argument("--thin", type=int, default=1, help="thin sampling interval")
    parser.add_argument("--replicas", type=int, default=8, help="num replicas for simulator")
    parser.add_argument("--algo", type=str, default="metropolis_sweep", help="algorithm name")
    parser.add_argument("--spacing", type=str, default="geom", choices=["geom", "linear"], help="temperature spacing")
    parser.add_argument("--h", type=float, default=0.0, help="magnetic field")
    parser.add_argument("--seed", type=int, default=None, help="master seed for deriving replica_seeds (optional)")
    parser.add_argument("--save_lattices", action="store_true", help="if set, simulator will write lattice snapshots")
    args = parser.parse_args(argv)

    outdir = args.outdir

    if args.mode == "run_workers_demo":
        run_workers_demo(outdir, nworkers=args.nworkers, tasks_per_worker=args.tasks)
        print(f"Now you can call: python -m ising_fss.simulation.batch_runner --outdir {outdir} --mode merge")
        return

    if args.mode == "merge":
        if data_manager is not None:
            try:
                manifest_path = data_manager._orchestrate_worker_merge(outdir)  # type: ignore
                print("Merged file/manifest written to:", manifest_path)
                return
            except Exception as e:
                print("data_manager merge failed, falling back to local merge:", e, file=sys.stderr)
        try:
            manifest = _fallback_merge_tmp_dirs(outdir)
            print("Merged manifest written to:", manifest)
        except Exception as e:
            print("Merge failed:", e, file=sys.stderr)
        return

    if args.mode == "run_workers":
        ctx = mp.get_context("spawn")
        procs: List[mp.Process] = []
        try:
            for i in range(int(args.nworkers)):
                p = ctx.Process(
                    target=run_worker_process,
                    args=(outdir, i),
                    kwargs=dict(
                        L=int(args.L), T=float(args.T),
                        equilibration=int(args.equil), production=int(args.prod),
                        exchange_interval=int(args.exchange_interval), thin=int(args.thin),
                        algorithm=str(args.algo), spacing=str(args.spacing),
                        num_replicas=int(args.replicas), h=float(args.h),
                        base_seed=(int(args.seed) if args.seed is not None else None),
                        tasks=1, save_lattices=bool(args.save_lattices),
                    ),
                )
                p.start()
                procs.append(p)
            for p in procs:
                p.join()
            print("All worker processes finished. You can now run --mode merge to combine results.")
        except KeyboardInterrupt:
            print("KeyboardInterrupt received; terminating workers...", flush=True)
            for p in procs:
                try: p.terminate()
                except Exception: pass
            for p in procs:
                try: p.join(1.0)
                except Exception: pass
        return


if __name__ == "__main__":
    # Windows/macOS 下被直接执行为脚本时的安全导入主模块保护
    mp.freeze_support()
    main()

