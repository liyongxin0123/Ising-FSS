# examples/cpu_remc_basic.py
"""
CPU REMC 基本示例：HybridREMCSimulator + make_replica_seeds
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
    T_min, T_max = 2.0, 2.6
    num_replicas = 8

    replica_seeds = make_replica_seeds(master_seed=2024, n_replicas=num_replicas)
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
        equilibration_steps=1000,
        production_steps=5000,
        exchange_interval=5,
        thin=5,
        save_lattices=False,
        save_dir="runs/cpu_basic",
        worker_id="cpu_basic",
    )
    stats = sim.analyze(verbose=False)
    print("Finished CPU REMC. #temps =", len(stats))


if __name__ == "__main__":
    main()

