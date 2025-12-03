# examples/dispatcher_single_replica.py
"""
dispatcher.apply_move: 单副本一步更新示例
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
    L = 16
    beta = 1.0 / 2.269
    spins = np.random.choice([-1, 1], size=(L, L)).astype(np.int8)

    new_spins, info = dispatcher.apply_move(
        spins,
        beta,
        replica_seed=123,
        algo="metropolis_sweep",
        backend="auto",
    )

    print("Single replica update done.")
    print("Accepted moves:", info.get("accepted", "N/A"))


if __name__ == "__main__":
    main()

