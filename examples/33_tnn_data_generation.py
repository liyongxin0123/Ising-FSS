# examples/tnn_data_generation.py
"""
为张量网络 (TNN) / TNR 生成多 L 的热力学统计数据 (NPZ)。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Any, Mapping, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from ising_fss.simulation.parallel import across_L
from ising_fss.utils.logger import ExperimentLogger
from ising_fss.visualization.styles import publication_style


def _iter_temp_items(
    data_L: Mapping[Any, Mapping[str, Any]]
) -> List[Tuple[float, Any]]:
    items: List[Tuple[float, Any]] = []
    for k in data_L.keys():
        if isinstance(k, (int, float)):
            T_val = float(k)
        elif isinstance(k, str):
            if k.startswith("T_"):
                try:
                    T_val = float(k[2:])
                except ValueError:
                    continue
            else:
                try:
                    T_val = float(k)
                except ValueError:
                    continue
        else:
            continue
        items.append((T_val, k))
    items.sort(key=lambda x: x[0])
    return items


def export_tnn_data(results: Dict[int, Dict[float, Dict]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for L, data_L in results.items():
        if not data_L:
            continue
        if isinstance(data_L, dict) and "error" in data_L:
            print(f"⚠️ 跳过 L={L} (模拟失败: {data_L['error']})")
            continue
        temp_items = _iter_temp_items(data_L)
        if not temp_items:
            print(f"⚠️ L={L} 未找到温度键，跳过")
            continue

        T_vals = [T for T, _ in temp_items]
        n_T = len(T_vals)
        arrays: Dict[str, np.ndarray] = {
            "temperatures": np.asarray(T_vals, dtype=np.float64),
            "L": np.int64(L),
        }
        keys = ["E", "M", "C", "chi", "U", "E_err", "M_err", "C_err", "chi_err"]
        for name in keys:
            arr = np.full(n_T, np.nan, dtype=np.float64)
            for i, (_T, orig) in enumerate(temp_items):
                try:
                    val = data_L[orig].get(name, np.nan)
                except Exception:
                    val = np.nan
                arr[i] = float(val) if val is not None else np.nan
            arrays[name] = arr

        fname = out_dir / f"tnn_L{L}.npz"
        np.savez_compressed(fname, **arrays)
        print(f"✓ 导出 L={L}: {fname}")


def plot_overview(results: Dict[int, Dict], out_path: str):
    with publication_style():
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        L_list = sorted(results.keys())
        colors = plt.cm.viridis(np.linspace(0, 1, len(L_list)))
        for idx, L in enumerate(L_list):
            data_L = results[L]
            if isinstance(data_L, dict) and "error" in data_L:
                continue
            temp_items = _iter_temp_items(data_L)
            if not temp_items:
                continue
            Ts = [T for T, _orig in temp_items]
            Es = [data_L[orig].get("E", np.nan) for _, orig in temp_items]
            Ms = [data_L[orig].get("M", np.nan) for _, orig in temp_items]
            Cs = [data_L[orig].get("C", np.nan) for _, orig in temp_items]
            Xs = [data_L[orig].get("chi", np.nan) for _, orig in temp_items]
            kw = dict(marker=".", ls="-", color=colors[idx], label=f"L={L}", alpha=0.8)
            axes[0].plot(Ts, Es, **kw)
            axes[1].plot(Ts, Ms, **kw)
            axes[2].plot(Ts, Cs, **kw)
            axes[3].plot(Ts, Xs, **kw)

        axes[0].set_ylabel("E")
        axes[1].set_ylabel("M")
        axes[2].set_ylabel("C")
        axes[3].set_ylabel("chi")
        for ax in axes:
            ax.set_xlabel("T")
            ax.legend(fontsize="small")
            ax.axvline(2.269185, color="gray", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        print("📊 概览图已保存:", out_path)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--L_list", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--T_min", type=float, default=2.0)
    parser.add_argument("--T_max", type=float, default=2.6)
    parser.add_argument("--n_T", type=int, default=32)
    parser.add_argument("--outdir", default="data_tnn")
    parser.add_argument("--algo", default="wolff")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--high_precision", action="store_true")
    args = parser.parse_args()

    equil, prod, thin = 5000, 20000, 10
    if args.quick:
        equil, prod = 500, 1000
    if args.high_precision:
        equil, prod = 20000, 100000

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = ExperimentLogger("tnn_gen", output_dir=str(out_dir)).logger

    logger.info(
        f"L={args.L_list}, T=[{args.T_min},{args.T_max}], n_T={args.n_T}, algo={args.algo}"
    )
    t0 = time.time()
    results = across_L(
        L_list=args.L_list,
        T_min=args.T_min,
        T_max=args.T_max,
        num_replicas=args.n_T,
        equilibration=equil,
        production=prod,
        algorithm=args.algo,
        exchange_interval=5,
        thin=thin,
        n_processes_per_L=1,
        checkpoint_dir=str(out_dir / "ckpt"),
        checkpoint_final=True,
    )
    logger.info(f"模拟完成，用时 {time.time()-t0:.1f}s")

    export_tnn_data(results, out_dir / "npz")
    try:
        plot_overview(results, str(out_dir / "overview.png"))
    except Exception as e:  # noqa: BLE001
        logger.error(f"绘图失败: {e}")

    import pickle
    with open(out_dir / "raw_results.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()

