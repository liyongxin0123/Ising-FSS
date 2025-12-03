# examples/pipelines/gpu_large_scale_fss.py
"""
使用 GPU REMC 对大 L 系统做 FSS 的骨架示例。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from ising_fss.simulation.dispatcher import make_replica_seeds, gpu_available
from ising_fss.simulation.gpu_remc_simulator import GPU_REMC_Simulator  # noqa: E402
from ising_fss.analysis.fss_analyzer import FSSAnalyzer


def run_one_L(L: int, outdir: Path) -> Dict[str, Any]:
    T_min, T_max = 2.0, 2.6
    num_replicas = 64
    replica_seeds = make_replica_seeds(master_seed=L * 10, n_replicas=num_replicas)
    sim = GPU_REMC_Simulator(
        L=L,
        T_min=T_min,
        T_max=T_max,
        num_replicas=num_replicas,
        algorithm="metropolis",
        h=0.0,
        replica_seeds=replica_seeds,
    )
    sim.run(
        equilibration_steps=20000,
        production_steps=100000,
        exchange_interval=10,
        thin=50,
        save_lattices=False,
        save_dir=str(outdir / f"L{L}"),
        worker_id=f"gpu_L{L}",
    )
    return sim.analyze(verbose=False)


def main():
    import argparse, json

    parser = argparse.ArgumentParser()
    parser.add_argument("--L_list", type=int, nargs="+", default=[64, 96, 128])
    parser.add_argument("--outdir", default="runs/gpu_large_scale_fss")
    args = parser.parse_args()

    if not gpu_available():
        print("❌ GPU 不可用，本示例无法运行。")
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results_all: Dict[int, Dict[str, Any]] = {}
    for L in args.L_list:
        print(f"=== GPU REMC for L={L} ===")
        results_all[L] = run_one_L(L, outdir)

    with open(outdir / "raw_results.json", "w") as f:
        json.dump(results_all, f, indent=2, default=float)

    analyzer = FSSAnalyzer(results_all)
    Tc_est = analyzer.estimate_Tc("U")
    with open(outdir / "Tc_est.json", "w") as f:
        json.dump(Tc_est, f, indent=2, default=float)

    print("Done. See", outdir)


if __name__ == "__main__":
    main()

