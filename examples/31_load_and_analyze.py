# examples/load_and_analyze.py
"""
从 REMC 输出目录 / HDF5 / NPZ 加载数据，做 E/M 时间序列 和 FSS 统计量作图。

使用方式：
  1) 目录模式（多温度统计 + 从 JSON 读误差/交换率）：
       python load_and_analyze.py /path/to/remc_output_dir

  2) 单个 HDF5（单温度，画 E(t)/M(t) 随样本变化）：
       python load_and_analyze.py /path/to/worker__latt_T_2.350000_h0.000000.h5

  3) tnn_L*.npz 旧格式（整体统计 E/M/C/χ vs T）：
       python load_and_analyze.py /path/to/tnn_L64.npz
"""

from __future__ import annotations

import sys
import re
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

# 让 examples/* 能找到项目里的 src/
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from ising_fss.data.config_io import load_configs_hdf5
from ising_fss.core.observables import _energy_total_numpy as energy_fn


# ------------------------------------------------------------------
# 一些小工具
# ------------------------------------------------------------------
def _parse_worker_T_h_from_name(name: str) -> Optional[Tuple[str, float, float]]:
    """
    从文件名中解析 worker 前缀 / T / h

    期望格式类似：
        cpu_yaml_demo__latt_T_2.350000_h0.000000.h5

    返回:
        (worker_prefix, T, h) 或 None
    """
    m = re.match(r"(.+?)__latt_T_([-0-9.]+)_h([-0-9.]+)\.h5$", name)
    if not m:
        return None
    worker = m.group(1)
    T = float(m.group(2))
    h = float(m.group(3))
    return worker, T, h


def _compute_observables_from_configs(configs: np.ndarray,
                                      T: float,
                                      h: float) -> dict:
    """
    给定某个温度下的全部构型 (N, L, L)、温度 T、外场 h，
    计算 E(t)、M(t)、以及 C(T)、chi(T)、Binder U(T)。

    返回 dict:
        {
            "T": T,
            "h": h,
            "E_series": E_per_spin_array,  # shape (N,)
            "M_series": M_per_spin_array,  # shape (N,)
            "E_mean": ...,
            "M_mean": ...,
            "C": ...,
            "chi": ...,
            "U": ...,
            "n_samples": N,
        }
    """
    configs = np.asarray(configs)
    assert configs.ndim == 3, f"configs must be (N,L,L), got {configs.shape}"
    N_samples, L, _ = configs.shape
    N_site = L * L
    beta = 1.0 / float(T)

    E = np.empty(N_samples, dtype=np.float64)
    M = np.empty(N_samples, dtype=np.float64)

    for i, cfg in enumerate(configs):
        spins = np.asarray(cfg, dtype=np.int8)
        # 总能量
        e_tot = energy_fn(spins, h=h)
        # 每自旋能量 / 磁化
        E[i] = e_tot / N_site
        M[i] = spins.mean()

    # 一阶统计
    E_mean = float(np.mean(E))
    M_mean = float(np.mean(M))

    # 比热 C(T) 和磁化率 χ(T)（简单方差，不考虑自相关修正）
    var_E = float(np.var(E))
    var_M = float(np.var(M))

    C = beta * beta * N_site * var_E
    chi = beta * N_site * var_M

    # Binder 累积量 U
    m2 = np.mean(M ** 2)
    m4 = np.mean(M ** 4)
    if m2 <= 1e-15:
        U = 0.0  # 非常接近高温极限 / m≈0，防止数值爆炸
    else:
        U = 1.0 - m4 / (3.0 * (m2 ** 2 + 1e-16))

    out = {
        "T": float(T),
        "h": float(h),
        "E_series": E,
        "M_series": M,
        "E_mean": E_mean,
        "M_mean": M_mean,
        "C": float(C),
        "chi": float(chi),
        "U": float(U),
        "n_samples": int(N_samples),
    }
    return out


def _load_thermo_from_metadata(meta_path: Path) -> Optional[Dict[str, Any]]:
    """
    从 worker__metadata.json 中读取 thermo_stats / swap 信息。

    期望 JSON 中包含字段：
      - "thermo_stats": {
            "T_2.350000": {
                "T": 2.35,
                "C":...,"C_err":...,
                "chi":...,"chi_err":...,
                "U":...,
                "n_samples":...   # 或 "samples_per_temp"
            }, ...
        }
      - "swap_summary" 或 "swap": {
            "rate": float,
            "attempts": [...],
            "accepts": [...],
            "pair_rates": [...]   # 若存在
        }

    返回 dict 或 None：
        {
            "temps": np.array([...]),
            "C": np.array([...]),
            "C_err": np.array([...]),
            "chi": np.array([...]),
            "chi_err": np.array([...]),
            "U": np.array([...]),
            "n_samples": np.array([...], dtype=int),
            "swap": { ... }  # 可能不存在
        }
    """
    if not meta_path.is_file():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as exc:
        print(f"[warning] 读取 metadata {meta_path} 失败: {exc}")
        return None

    thermo = meta.get("thermo_stats", None)
    if not isinstance(thermo, dict) or not thermo:
        return None

    temps: List[float] = []
    C_list: List[float] = []
    C_err_list: List[float] = []
    chi_list: List[float] = []
    chi_err_list: List[float] = []
    U_list: List[float] = []
    n_samples_list: List[int] = []

    for key, entry in thermo.items():
        if not isinstance(entry, dict):
            continue
        # T 优先用 entry["T"]，否则从 key "T_2.350000" 里解析
        T_val = entry.get("T", None)
        if T_val is None:
            try:
                T_val = float(str(key).replace("T_", ""))
            except Exception:
                continue
        try:
            temps.append(float(T_val))
            C_list.append(float(entry.get("C", 0.0)))
            C_err_list.append(float(entry.get("C_err", 0.0)))
            chi_list.append(float(entry.get("chi", 0.0)))
            chi_err_list.append(float(entry.get("chi_err", 0.0)))
            U_list.append(float(entry.get("U", 0.0)))
            # 兼容 n_samples / samples_per_temp 两种命名
            n_s = entry.get("n_samples", entry.get("samples_per_temp", 0))
            n_samples_list.append(int(n_s))
        except Exception:
            continue

    if not temps:
        return None

    # 按温度排序
    order = np.argsort(np.asarray(temps, dtype=float))
    temps_arr = np.asarray(temps, dtype=float)[order]
    C_arr = np.asarray(C_list, dtype=float)[order]
    C_err_arr = np.asarray(C_err_list, dtype=float)[order]
    chi_arr = np.asarray(chi_list, dtype=float)[order]
    chi_err_arr = np.asarray(chi_err_list, dtype=float)[order]
    U_arr = np.asarray(U_list, dtype=float)[order]
    n_samples_arr = np.asarray(n_samples_list, dtype=int)[order]

    swap_block = meta.get("swap_summary", None)
    if swap_block is None:
        swap_block = meta.get("swap", None)

    return {
        "temps": temps_arr,
        "C": C_arr,
        "C_err": C_err_arr,
        "chi": chi_arr,
        "chi_err": chi_err_arr,
        "U": U_arr,
        "n_samples": n_samples_arr,
        "swap": swap_block,
    }


# ------------------------------------------------------------------
# 单个 HDF5：画 E(t)/M(t) 时间序列（你提到的用法）
# ------------------------------------------------------------------
def plot_em_series_from_single_hdf5(h5_path: Path, out_png: Optional[Path] = None):
    """
    用法：
        python load_and_analyze.py /path/to/worker__latt_T_2.350000_h0.000000.h5

    效果：
        对这个单一温度下的所有采样，画出
          - E_per_spin(t) 随样本编号 t 的变化；
          - M_per_spin(t) 随样本编号 t 的变化。
    """
    h5_path = h5_path.resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)

    ds = load_configs_hdf5(str(h5_path), load_configs=True, load_obs=False)
    configs = np.asarray(ds["configs"])
    parsed = _parse_worker_T_h_from_name(h5_path.name)
    if parsed is not None:
        worker, T_from_name, h_from_name = parsed
    else:
        worker, T_from_name, h_from_name = "unknown", None, None

    T_ds = ds.get("T", None)
    h_ds = ds.get("h", None)

    T = T_ds if T_ds is not None else T_from_name
    h = h_ds if h_ds is not None else h_from_name
    if T is None:
        raise RuntimeError(f"无法从 {h5_path.name} 中解析温度 T")
    if h is None:
        h = 0.0

    obs = _compute_observables_from_configs(configs, T=float(T), h=float(h))

    E_series = obs["E_series"]
    M_series = obs["M_series"]
    n_samples = obs["n_samples"]

    x = np.arange(n_samples)

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax[0].plot(x, E_series, "-", lw=0.8)
    ax[0].axhline(obs["E_mean"], color="red", ls="--", alpha=0.7,
                  label=f"<E>={obs['E_mean']:.4f}")
    ax[0].set_ylabel("E per spin")
    ax[0].legend()

    ax[1].plot(x, M_series, "-", lw=0.8)
    ax[1].axhline(obs["M_mean"], color="red", ls="--", alpha=0.7,
                  label=f"<m>={obs['M_mean']:.4f}")
    ax[1].set_ylabel("m per spin")
    ax[1].set_xlabel("sample index")
    ax[1].legend()

    fig.suptitle(
        f"E/M series at T={obs['T']:.6f}, h={obs['h']:.6f} (worker={worker})",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if out_png is None:
        out_png = h5_path.with_suffix(".png")
    plt.savefig(out_png, dpi=200)
    print("Saved time-series plot to", out_png)


# ------------------------------------------------------------------
# 多温度：对单个 worker 的 HDF5 + JSON 进行统计和作图
# ------------------------------------------------------------------
def plot_worker_from_hdf5_group(worker_prefix: str,
                                files_to_process: List[Path],
                                meta_path: Optional[Path] = None,
                                out_prefix: Optional[Path] = None):
    """
    对某个 worker 的全部 HDF5 文件（不同 T）进行汇总：
      - 计算每个 T 的 <E>, <m>, C, chi, U（raw，无误差修正）；
      - 若 meta_path JSON 里有 thermo_stats / swap，则再画
        1) 带误差条的 C/chi/U vs T
        2) swap 统计图
      - 输出若干 png：
        <out_prefix>_obs.png, <out_prefix>_binder.png, <out_prefix>_thermo_meta.png, <out_prefix>_swap.png
    """
    files_to_process = sorted(
        files_to_process,
        key=lambda fp: _parse_worker_T_h_from_name(fp.name)[1]
        if _parse_worker_T_h_from_name(fp.name) is not None
        else 0.0,
    )
    if not files_to_process:
        print(f"[warning] worker={worker_prefix} 没有任何 HDF5 文件可用。")
        return

    if out_prefix is None:
        out_prefix = files_to_process[0].with_suffix("")

    # 对每个 T 文件计算统计量
    results: List[dict] = []
    for fpath in files_to_process:
        ds = load_configs_hdf5(str(fpath), load_configs=True, load_obs=False)
        configs = np.asarray(ds["configs"])

        # 优先从 ds 里拿 T / h；没有则从文件名里解析
        T_ds = ds.get("T", None)
        h_ds = ds.get("h", None)

        parsed = _parse_worker_T_h_from_name(fpath.name)
        if parsed is not None:
            _, T_from_name, h_from_name = parsed
        else:
            T_from_name, h_from_name = None, None

        T = T_ds if T_ds is not None else T_from_name
        h = h_ds if h_ds is not None else h_from_name
        if T is None:
            raise RuntimeError(f"无法从 {fpath.name} 中解析温度 T")
        if h is None:
            h = 0.0  # 默认 h=0

        obs = _compute_observables_from_configs(configs, T=float(T), h=float(h))
        results.append(obs)

        print(
            f"[worker={worker_prefix}] {fpath.name}: "
            f"T={obs['T']:.6f}, h={obs['h']:.6f}, "
            f"n={obs['n_samples']}, "
            f"<E>={obs['E_mean']:.6f}, <m>={obs['M_mean']:.6f}, "
            f"C={obs['C']:.6f}, chi={obs['chi']:.6f}, U={obs['U']:.6f}"
        )

    # 按 T 排序并画 E(T)/M(T)/C(T)/chi(T)
    results_sorted = sorted(results, key=lambda d: d["T"])
    temps = np.array([r["T"] for r in results_sorted], dtype=float)
    E_mean = np.array([r["E_mean"] for r in results_sorted], dtype=float)
    M_mean = np.array([r["M_mean"] for r in results_sorted], dtype=float)
    C_vals = np.array([r["C"] for r in results_sorted], dtype=float)
    chi_vals = np.array([r["chi"] for r in results_sorted], dtype=float)
    U_vals = np.array([r["U"] for r in results_sorted], dtype=float)

    # ----------------- 图 1：E, m, C, chi (raw) -----------------
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    ax = ax.flatten()

    ax[0].plot(temps, E_mean, "o-", ms=3)
    ax[0].set_ylabel("E per spin")

    ax[1].plot(temps, M_mean, "o-", ms=3)
    ax[1].set_ylabel("m per spin")

    ax[2].plot(temps, C_vals, "o-", ms=3)
    ax[2].set_ylabel("C (raw)")

    ax[3].plot(temps, chi_vals, "o-", ms=3)
    ax[3].set_ylabel("chi (raw)")

    for a in ax:
        a.set_xlabel("T")
        a.axvline(2.269185, color="gray", ls="--", alpha=0.5)

    fig.suptitle(f"REMC observables (worker={worker_prefix})", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_obs = out_prefix.with_name(out_prefix.name + "_obs.png")
    plt.savefig(out_obs, dpi=200)
    print("Saved plot:", out_obs)

    # ----------------- 图 2：Binder U(T) (raw) -----------------
    if len(temps) > 0:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(temps, U_vals, "o-", ms=3)
        ax2.set_xlabel("T")
        ax2.set_ylabel("Binder U (raw)")
        ax2.axvline(2.269185, color="gray", ls="--", alpha=0.5)
        ax2.set_title(f"Binder cumulant U(T) (worker={worker_prefix})")
        plt.tight_layout()
        out_binder = out_prefix.with_name(out_prefix.name + "_binder.png")
        plt.savefig(out_binder, dpi=200)
        print("Saved plot:", out_binder)

    # ----------------- 图 3/4：从 metadata.json 读取 thermo_stats + swap -----------------
    meta_info = None
    if meta_path is not None:
        meta_info = _load_thermo_from_metadata(meta_path)

    # 3.1 thermo_stats: C/χ/U 带误差
    if meta_info is not None:
        temps_m = meta_info["temps"]
        C_m = meta_info["C"]
        C_err_m = meta_info["C_err"]
        chi_m = meta_info["chi"]
        chi_err_m = meta_info["chi_err"]
        U_m = meta_info["U"]

        fig3, ax3 = plt.subplots(3, 1, figsize=(6, 9), sharex=True)
        ax3[0].errorbar(temps_m, C_m, yerr=C_err_m, fmt="o-", ms=3)
        ax3[0].set_ylabel("C")
        ax3[0].axvline(2.269185, color="gray", ls="--", alpha=0.5)

        ax3[1].errorbar(temps_m, chi_m, yerr=chi_err_m, fmt="o-", ms=3)
        ax3[1].set_ylabel("chi")
        ax3[1].axvline(2.269185, color="gray", ls="--", alpha=0.5)

        ax3[2].plot(temps_m, U_m, "o-", ms=3)
        ax3[2].set_ylabel("Binder U")
        ax3[2].set_xlabel("T")
        ax3[2].axvline(2.269185, color="gray", ls="--", alpha=0.5)

        fig3.suptitle(
            f"Thermo observables from metadata (worker={worker_prefix})",
            fontsize=12,
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        out_thermo = out_prefix.with_name(out_prefix.name + "_thermo_meta.png")
        plt.savefig(out_thermo, dpi=200)
        print("Saved plot:", out_thermo)

        # 3.2 swap 统计
        swap_block = meta_info.get("swap", None)
        if isinstance(swap_block, dict):
            rate_global = float(swap_block.get("rate", 0.0))
            pair_rates = swap_block.get("pair_rates", None)
            if pair_rates is not None:
                pair_rates = np.asarray(pair_rates, dtype=float)
            else:
                attempts = np.asarray(swap_block.get("attempts", []), dtype=float)
                accepts = np.asarray(swap_block.get("accepts", []), dtype=float)
                if attempts.size and accepts.size and attempts.size == accepts.size:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        pr = np.where(attempts > 0, accepts / attempts, 0.0)
                    pair_rates = pr
                else:
                    pair_rates = np.array([])

            fig4, ax4 = plt.subplots(figsize=(6, 4))
            if pair_rates.size > 0:
                temps_mid = (temps_m[:-1] + temps_m[1:]) / 2.0
                if temps_mid.size == pair_rates.size:
                    ax4.plot(temps_mid, pair_rates, "o-", ms=3, label="pair swap rate")
                    ax4.set_xlabel("mid T of pair")
                else:
                    ax4.plot(
                        np.arange(pair_rates.size),
                        pair_rates,
                        "o-",
                        ms=3,
                        label="pair swap rate",
                    )
                    ax4.set_xlabel("pair index")
            else:
                ax4.set_xlabel("pair index")

            ax4.axhline(
                rate_global,
                color="red",
                ls="--",
                label=f"global rate={rate_global:.3f}",
            )
            ax4.set_ylabel("swap rate")
            ax4.set_title(f"Swap statistics (worker={worker_prefix})")
            ax4.legend()
            plt.tight_layout()

            out_swap = out_prefix.with_name(out_prefix.name + "_swap.png")
            plt.savefig(out_swap, dpi=200)
            print("Saved plot:", out_swap)


# ------------------------------------------------------------------
# tnn_L*.npz 的旧路径保留
# ------------------------------------------------------------------
def plot_from_tnn_npz(npz_path: Path, out_png: Optional[Path] = None):
    data = np.load(npz_path)
    T = data["temperatures"]
    E = data["E"]
    M = data["M"]
    C = data["C"]
    chi = data["chi"]

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    ax = ax.flatten()
    ax[0].plot(T, E, "o-")
    ax[0].set_ylabel("E")
    ax[1].plot(T, M, "o-")
    ax[1].set_ylabel("M")
    ax[2].plot(T, C, "o-")
    ax[2].set_ylabel("C")
    ax[3].plot(T, chi, "o-")
    ax[3].set_ylabel("chi")

    for a in ax:
        a.set_xlabel("T")
        a.axvline(2.269185, color="gray", ls="--", alpha=0.5)

    plt.tight_layout()
    if out_png is not None:
        plt.savefig(out_png, dpi=200)
        print("Saved plot to", out_png)
    else:
        plt.show()


# ------------------------------------------------------------------
# 目录模式：给一个 remc_simulator 输出目录，自动识别并作图
# ------------------------------------------------------------------
def analyze_remc_output_dir(dir_path: Path):
    """
    给 remc_simulator / GPU_REMC_Simulator 的输出目录，比如：

        examples/runs/L64_from_yaml/

    目录中包含：
      - <worker>__latt_T_..._h....h5
      - <worker>__metadata.json

    本函数会：
      1. 找到所有匹配 HDF5，按 worker 分组；
      2. 对每个 worker，调用 plot_worker_from_hdf5_group(...) 进行作图。
    """
    dir_path = dir_path.resolve()
    if not dir_path.is_dir():
        raise NotADirectoryError(dir_path)

    # 收集该目录下所有符合命名约定的 HDF5
    groups: Dict[str, List[Path]] = defaultdict(list)
    for f in dir_path.iterdir():
        if not f.is_file():
            continue
        if not f.name.endswith(".h5"):
            continue
        parsed = _parse_worker_T_h_from_name(f.name)
        if parsed is None:
            continue
        worker, T, h = parsed
        groups[worker].append(f)

    if not groups:
        print(
            f"[warning] 目录 {dir_path} 下没有匹配模式 'xxx__latt_T_..._h....h5' 的 HDF5 文件。"
        )
        return

    # 对每个 worker 分别作图
    for worker, files in groups.items():
        meta_path = dir_path / f"{worker}__metadata.json"
        print(
            f"[dir] worker='{worker}' 发现 {len(files)} 个温度文件，"
            f"metadata={'存在' if meta_path.is_file() else '不存在'}"
        )

        out_prefix = (dir_path / f"{worker}__remc_summary").with_suffix("")

        plot_worker_from_hdf5_group(
            worker_prefix=worker,
            files_to_process=files,
            meta_path=meta_path if meta_path.is_file() else None,
            out_prefix=out_prefix,
        )


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        help="remc 输出目录 / 单个 HDF5 文件 / tnn_L*.npz",
    )
    args = parser.parse_args()

    path = Path(args.path)

    if path.is_dir():
        # 目录模式：多温度统计
        analyze_remc_output_dir(path)
        return

    if path.suffix.lower() == ".npz":
        # tnn 旧格式
        plot_from_tnn_npz(path, out_png=path.with_suffix(".png"))
    elif path.suffix.lower() == ".h5":
        # 单个 HDF5：时间序列模式 E(t)/M(t)
        plot_em_series_from_single_hdf5(path, out_png=path.with_suffix(".png"))
    else:
        raise RuntimeError(
            f"不支持的文件类型: {path.suffix} (期望目录 / .h5 / .npz)"
        )


if __name__ == "__main__":
    main()

