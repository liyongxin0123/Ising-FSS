# examples/pipelines/run_path_A_independent.py
"""
路径 A：多温度独立 Metropolis 采样（非 REMC），用于生成 ML 数据。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from ising_fss.core.algorithms import update_batch, spawn_replica_seeds  # 你已有的接口
from ising_fss.data.data_manager import save_ml_dataset  # 假设有类似函数


def simulate_independent(
    L: int,
    temps: List[float],
    n_configs_per_T: int,
    n_sweeps_per_sample: int,
    out_h5: Path,
):
    R = len(temps)
    spins_batch = np.random.choice([-1, 1], size=(R, L, L)).astype(np.int8)
    seeds = spawn_replica_seeds(master_seed=1234, n_replicas=R)

    records = []
    for i in range(n_configs_per_T):
        update_batch(
            spins_batch=spins_batch,
            beta=[1.0 / T for T in temps],
            replica_seeds=seeds,
            algo="metropolis_sweep",
            h=0.0,
            n_sweeps=n_sweeps_per_sample,
        )
        records.append(spins_batch.copy())

    configs = np.stack(records, axis=0)  # (n_configs, R, L, L)
    save_ml_dataset(configs=configs, temps=temps, out_path=str(out_h5))


def main():
    L = 32
    temps = np.linspace(1.6, 3.2, 40).tolist()
    simulate_independent(
        L=L,
        temps=temps,
        n_configs_per_T=1000,
        n_sweeps_per_sample=10,
        out_h5=Path("runs/pathA_independent_L32.h5"),
    )


if __name__ == "__main__":
    main()

