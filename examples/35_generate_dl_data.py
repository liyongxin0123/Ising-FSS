# examples/generate_dl_data.py
"""
从 REMC HDF5 输出生成适合 PyTorch 的训练集格式。

假定输入目录里已经有一个或多个 worker 写出的 .h5 文件，或者 batch_runner merge 后的 final_ml_data.h5。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Union

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from ising_fss.data.config_io import load_configs_hdf5, export_for_pytorch


def _find_h5(root: Path) -> Path:
    # 优先找 final_ml_data.h5，其次任意 .h5
    cand = list(root.rglob("final_ml_data.h5"))
    if cand:
        return cand[0]
    cand = list(root.rglob("*.h5"))
    if not cand:
        raise FileNotFoundError(f"No .h5 found under {root}")
    return cand[0]


def generate_from_hdf5(
    raw_dir: Union[str, Path],
    out_dir: Union[str, Path],
    normalize: bool = True,
    dtype: str = "uint8",
):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_path = _find_h5(raw_dir)
    print("Using HDF5:", h5_path)
    ds = load_configs_hdf5(str(h5_path), load_configs=False)

    export_for_pytorch(
        ds,
        save_dir=str(out_dir),
        split_ratio=0.8,
        dtype=dtype,
        normalize=normalize,
        verbose=True,
    )
    print("PyTorch dataset written to", out_dir)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True, help="目录，里面有 REMC 的 HDF5")
    parser.add_argument("--out_dir", required=True, help="输出 PyTorch 数据集目录")
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--dtype", default="uint8")
    args = parser.parse_args()

    generate_from_hdf5(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        normalize=not args.no_normalize,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()

