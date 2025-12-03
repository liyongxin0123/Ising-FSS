# examples/parallel_across_L.py
"""
parallel.across_L：多 L 并行 + checkpoint 恢复示例
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from ising_fss.simulation.parallel import across_L


def main():
    L_list = [16, 32, 64]
    out_ckpt = Path("runs/parallel_ckpt")
    out_ckpt.mkdir(parents=True, exist_ok=True)

    results = across_L(
        L_list=L_list,
        T_min=2.0,
        T_max=2.6,
        num_replicas=32,
        equilibration=2000,
        production=5000,
        algorithm="wolff",
        exchange_interval=5,
        thin=5,
        n_processes_per_L=1,
        checkpoint_dir=str(out_ckpt),
        checkpoint_final=True,
        resume_if_exists=True,
    )

    print("\nSummary:")
    for L, res in results.items():
        if isinstance(res, dict) and "error" in res:
            print(f" L={L}: ERROR -> {res['error']}")
        else:
            swap = res.get("swap", {})
            print(f" L={L}: swap rate ≈ {swap.get('rate', 'N/A')}")


if __name__ == "__main__":
    main()

