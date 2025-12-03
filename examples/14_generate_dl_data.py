#!/usr/bin/env python3
"""
从 Config 出发，一键生成用于 DL 的 HDF5 + PyTorch 数据集。

- 第一步：根据 Config 跑 REMC（如果 raw_dir 里还没有 .h5）
- 第二步：直接在本文件中，从 HDF5 读出 configs，并导出为 PyTorch 友好的布局
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Union

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from ising_fss.utils.config import from_args, validate_config
from ising_fss.simulation.remc_simulator import HybridREMCSimulator
from ising_fss.simulation.dispatcher import make_replica_seeds
from ising_fss.data.config_io import load_configs_hdf5, export_for_pytorch

logger = logging.getLogger("generate_dl_data")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

PathLike = Union[str, Path]


def _flatten_configs(configs: np.ndarray) -> np.ndarray:
    """
    将 HDF5 里读出的 configs 统一成 (N, L, L)。

    支持两种典型布局：
        - (N, L, L)
        - (n_h, n_T, n_c, L, L)  -> 展平成 (N, L, L)
    """
    arr = np.asarray(configs)
    if arr.ndim == 3:
        return arr
    if arr.ndim == 5:
        n_h, n_T, n_c, Lx, Ly = arr.shape
        return arr.reshape(n_h * n_T * n_c, Lx, Ly)
    raise ValueError(f"Unexpected configs ndim={arr.ndim}, expected 3 or 5.")


def _export_pytorch_from_hdf5(
    raw_dir: PathLike,
    out_dir: PathLike,
    *,
    normalize: bool = True,
    dtype: str = "uint8",
    split_ratio: float = 0.8,
    seed: int = 0,
) -> None:
    """
    从 REMC 生成的 HDF5 原始晶格文件中，构造一个 PyTorch 友好的数据集。

    raw_dir 下应当有若干 .h5 文件（由 HybridREMCSimulator 保存）。
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(raw_dir.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found under {raw_dir}")

    logger.info("Found %d HDF5 files under %s", len(h5_files), raw_dir)

    exported = False
    for h5 in h5_files:
        logger.info("Try loading configs from %s", h5)
        try:
            ds_raw = load_configs_hdf5(h5, load_configs=True, load_obs=True)
        except Exception as exc:
            logger.warning("load_configs_hdf5 failed for %s: %s", h5, exc)
            continue

        if "configs" not in ds_raw:
            logger.warning("No 'configs' field in %s; skip.", h5)
            continue

        configs = _flatten_configs(np.asarray(ds_raw["configs"]))
        if configs.ndim != 3:
            logger.warning("Unexpected configs ndim=%d in %s; skip.", configs.ndim, h5)
            continue

        N, Lx, Ly = configs.shape
        if Lx != Ly:
            logger.warning("Non-square lattice (%d x %d) in %s; skip.", Lx, Ly, h5)
            continue

        logger.info("Configs shape: N=%d, L=%d", N, Lx)

        # 为了简单/稳健，标签和观测量先全 0 占位，完全由下游任务自由使用
        temps = np.zeros(N, dtype=np.float32)
        fields = np.zeros(N, dtype=np.float32)
        energy = np.zeros(N, dtype=np.float32)
        magnetization = np.zeros(N, dtype=np.float32)

        ds_pt = {
            "configs": configs,
            "temperatures": temps,
            "fields": fields,
            "energy": energy,
            "magnetization": magnetization,
            "parameters": {
                "L": int(Lx),
                "n_configs": int(N),
                "generator": "config.generate_dl_data",
                "source_file": str(h5),
            },
        }

        logger.info(
            "Exporting PyTorch dataset to %s (normalize=%s, dtype=%s, split_ratio=%.3f, seed=%d)...",
            out_dir,
            normalize,
            dtype,
            split_ratio,
            seed,
        )

        export_for_pytorch(
            ds_pt,
            out_dir,
            split_ratio=split_ratio,
            normalize=normalize,
            dtype=dtype,
            seed=seed,
        )

        exported = True
        logger.info("PyTorch export succeeded from %s", h5)
        break

    if not exported:
        raise RuntimeError(
            f"Failed to export PyTorch dataset: no suitable HDF5 file "
            f"with 'configs' found under {raw_dir}"
        )


def main():
    # 1. 从命令行 / YAML 读取 Config
    cfg = from_args()

    has_problem, warning_list = validate_config(cfg)
    for w in warning_list:
        print("[config warning]", w)

    s = cfg.simulation
    d = cfg.data

    out_root = Path(d.output_dir)
    raw_dir = out_root / "raw"
    pt_dir = out_root / "pytorch"

    # 2. 如果 raw_dir 下没有 .h5，就跑一次 REMC
    if not any(raw_dir.glob("*.h5")):
        raw_dir.mkdir(parents=True, exist_ok=True)
        replica_seeds = make_replica_seeds(
            master_seed=s.seed or 0,
            n_replicas=s.num_replicas,
        )

        logger.info(
            "Running REMC: L=%d, T∈[%.3f, %.3f], replicas=%d, eq=%d, prod=%d, thin=%d",
            s.L, s.T_min, s.T_max, s.num_replicas, s.equilibration, s.production, s.sampling_interval
        )

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
            save_dir=str(raw_dir),
            worker_id="dl_from_config",
        )
        logger.info("REMC finished. Raw HDF5 saved under %s", raw_dir)
    else:
        logger.info("Found existing .h5 files under %s, skip REMC simulation.", raw_dir)

    # 3. 从 HDF5 导出 PyTorch 数据
    #   尝试从 DataConfig 里读出一些参数，不存在就用默认值
    normalize = getattr(d, "normalize", True)
    dtype = getattr(d, "dtype", "uint8")
    split_ratio = getattr(d, "split_ratio", 0.8)
    seed = getattr(s, "seed", 0) or 0

    _export_pytorch_from_hdf5(
        raw_dir=raw_dir,
        out_dir=pt_dir,
        normalize=normalize,
        dtype=dtype,
        split_ratio=split_ratio,
        seed=seed,
    )

    print("Done. Raw REMC data in", raw_dir)
    print("      PyTorch-ready dataset in", pt_dir)


if __name__ == "__main__":
    main()

