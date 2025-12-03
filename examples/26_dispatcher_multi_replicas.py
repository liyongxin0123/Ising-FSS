# examples/dispatcher_multi_replicas.py
"""
dispatcher.apply_move_batch: 多副本批量更新示例
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from ising_fss.simulation import dispatcher


def main():
    R, L = 8, 16
    betas = [1.0 / 2.269] * R
    spins_batch = np.random.choice([-1, 1], size=(R, L, L)).astype(np.int8)

    replica_seeds = dispatcher.make_replica_seeds(master_seed=999, n_replicas=R)

    new_batch, meta = dispatcher.apply_move_batch(
        spins_batch,
        betas,
        replica_seeds=replica_seeds,
        algo="metropolis_sweep",
        backend="cpu",
        n_sweeps=10,
    )

    print("Batch update done.")
    print("meta keys:", meta.keys())


if __name__ == "__main__":
    main()

