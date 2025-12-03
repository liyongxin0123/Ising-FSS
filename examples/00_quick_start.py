# examples/quick_start.py
"""
Quick start: 最简单的一步 REMC 示例

- 在 CPU 上用 HybridREMCSimulator 跑一个小系统 (L=16, R=8)
- 不依赖 Config 系统，直接用裸参数
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from ising_fss.simulation.remc_simulator import HybridREMCSimulator
from ising_fss.simulation.dispatcher import make_replica_seeds


def main():
    L = 16
    T_min, T_max = 1.0, 3.5
    num_replicas = 3

    # 生成确定性的副本种子
    replica_seeds = make_replica_seeds(master_seed=42, n_replicas=num_replicas)

    sim = HybridREMCSimulator(
        L=L,
        T_min=T_min,
        T_max=T_max,
        num_replicas=num_replicas,
        algorithm="metropolis_sweep",
        h=0.0,
        replica_seeds=replica_seeds,
    )

    sim.run(
        equilibration_steps=500,
        production_steps=2000,
        exchange_interval=5,
        thin=5,
        save_lattices=False,
        save_dir="runs/quick_start",
        worker_id="quick_start",
    )

    results = sim.analyze(verbose=False)

    # 只数一数有多少个温度条目（排除 swap / seeds 等全局项）
    T_keys = sorted(k for k in results.keys() if isinstance(k, str) and k.startswith("T_"))
    print(f"Got {len(T_keys)} temperature entries\n")

    # 打印每个温度点的主要观测量
    print("Per-temperature observables:")
    for k in T_keys:
        v = results[k]
        T = float(k.replace("T_", ""))
        C = v["C"]; C_err = v["C_err"]
        chi = v["chi"]; chi_err = v["chi_err"]
        U = v["U"]
        n = v["n_samples"]
        print(
            f"{k} (T={T:.6f}): "
            f"C = {C:.4f} ± {C_err:.4f}, "
            f"chi = {chi:.4f} ± {chi_err:.4f}, "
            f"U = {U:.4f}, "
            f"n_samples = {n}"
        )

    # 交换统计信息
    swap = results.get("swap", {})
    print("\nSwap statistics:")
    print(f"  total attempts = {swap.get('attempt', 0)}")
    print(f"  total accepts  = {swap.get('accept', 0)}")
    print(f"  overall rate   = {swap.get('rate', 0.0):.4f}")
    pair_rates = swap.get("pair_rates", [])
    temps = swap.get("temps", [])
    for i, r in enumerate(pair_rates):
        if i + 1 < len(temps):
            print(
                f"  pair {i}: T={temps[i]:.4f} <-> T={temps[i+1]:.4f}, "
                f"accept rate = {r:.4f}"
            )

    # 如果有 warning，也打印出来看看
    if "warnings" in results:
        print("\nWarnings:")
        for w in results["warnings"]:
            print("  -", w)


if __name__ == "__main__":
    main()

