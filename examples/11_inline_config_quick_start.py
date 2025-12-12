# examples/inline_config_quick_start.py
"""
在脚本中直接构造 Config，然后用其中的 simulation 配置跑一次 REMC。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from ising_fss.utils.config import SimulationConfig, DataConfig, Config, validate_config
from ising_fss.simulation.remc_simulator import HybridREMCSimulator
from ising_fss.simulation.dispatcher import make_replica_seeds


def main():
    # ---- 1. 构造 Config ----
    sim_cfg = SimulationConfig(
        L=32,
        T_min=2.0,
        T_max=2.6,
        num_replicas=12,
        h_field=0.0,
        algorithm="metropolis",  # → 'metropolis_sweep'
        boundary="pbc",
        backend="cpu",
        equilibration=2000,
        production=8000,
        exchange_interval=5,
        sampling_interval=5,
    )
    data_cfg = DataConfig(
        L=32,
        T_range=(2.0, 2.6),
        n_T=12,
        n_configs=2000,
        output_dir="data/config_inline_demo",
        export_pytorch=False,
    )
    cfg = Config(simulation=sim_cfg, data=data_cfg)

    has_warning, warning_list = validate_config(cfg)
    for w in warning_list:
        print("[config warning]", w)

    # ---- 2. 构造模拟器 ----
    s = cfg.simulation
    replica_seeds = make_replica_seeds(master_seed=1234, n_replicas=s.num_replicas)

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
        save_dir=str(Path(data_cfg.output_dir) / "raw"),
        worker_id="inline_cfg",
    )

    print("Done. Raw REMC data written under", data_cfg.output_dir)


if __name__ == "__main__":
    main()

