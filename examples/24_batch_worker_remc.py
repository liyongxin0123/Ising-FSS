# examples/batch_worker_remc.py
"""
直接在 Python 脚本中调用 batch_runner.main(argv) 启动多 worker REMC。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from ising_fss.simulation import batch_runner


def main():
    outdir = Path("runs/batch_worker_demo")
    outdir.mkdir(parents=True, exist_ok=True)

    argv = [
        "--mode", "run_workers",
        "--outdir", str(outdir),
        "--nworkers", "2",
        "--L", "32",
        "--T", "2.269",
        "--equil", "2000",
        "--prod", "5000",
        "--exchange_interval", "5",
        "--thin", "10",
        "--replicas", "16",
        "--algo", "metropolis_sweep",
        "--spacing", "geom",
        "--h", "0.0",
        "--save_lattices",
    ]
    batch_runner.main(argv)
    print("Workers finished. You can now run merge via 05_batch_demo_cli.py or CLI.")


if __name__ == "__main__":
    main()

