# examples/analysis/check_tnn_file.py
"""
检查单个 tnn_L*.npz 文件的内容，并画出简单曲线。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("npz", help="tnn_L*.npz file")
    args = parser.parse_args()

    path = Path(args.npz)
    data = np.load(path)
    print("keys:", list(data.keys()))
    print("L =", data["L"])
    print("temperatures shape:", data["temperatures"].shape)

    T = data["temperatures"]
    E, M = data["E"], data["M"]

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(T, E, "o-")
    plt.xlabel("T")
    plt.ylabel("E")
    plt.subplot(1, 2, 2)
    plt.plot(T, M, "o-")
    plt.xlabel("T")
    plt.ylabel("M")
    plt.tight_layout()
    out_png = path.with_suffix(".check.png")
    plt.savefig(out_png, dpi=200)
    print("Saved preview to", out_png)


if __name__ == "__main__":
    main()

