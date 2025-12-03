# examples/config/run_sim_with_from_args.py
"""
使用 Config.from_args() + 命令行 --preset / --set / ENV 来驱动 REMC。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from ising_fss.utils.config import from_args, validate_config
from ising_fss.simulation.remc_simulator import HybridREMCSimulator
from ising_fss.simulation.dispatcher import make_replica_seeds


def main():
    cfg = from_args()  # 会解析 --preset / --config / --set / ENV 等
    has_warning, warning_list = validate_config(cfg)
    for w in warning_list:
        print("[config warning]", w)

    s = cfg.simulation
    replica_seeds = make_replica_seeds(master_seed=s.seed or 0, n_replicas=s.num_replicas)

    sim = HybridREMCSimulator(
        L=s.L,
        T_min=s.T_min,
        T_max=s.T_max,
        num_replicas=s.num_replicas,
        algorithm=s.algorithm,
        h=s.h_field,
        replica_seeds=replica_seeds,
    )

    sim.run(
        equilibration_steps=s.equilibration,
        production_steps=s.production,
        exchange_interval=s.exchange_interval,
        thin=s.sampling_interval,
        save_lattices=True,
        save_dir=str(Path(cfg.data.output_dir) / "raw_from_args"),
        worker_id="from_args",
    )

    print("Run finished. Output dir:", cfg.data.output_dir)


if __name__ == "__main__":
    main()

