# examples/cpu_remc_large_scale_fss.py
"""
基于 CPU / HybridREMCSimulator 的 REMC → FSS 管线脚本。

目标：
- 行为尽量模仿 gpu_large_scale_fss.py（42_gpu_large_scale_fss.py）：
  * 支持多次运行同一个 outdir，自动在 raw_results.json 里“追加样本”；
  * 每次 run 之后都用 FSSAnalyzer 做一次 Tc / γ/ν / 数据塌缩分析；
  * 把 Binder U 的 crossing 信息写入 Tc_est.json。
- 区别：
  * 这里用的是 HybridREMCSimulator（CPU / 混合实现），而不是 GPU 版模拟器；
  * 暂不做 checkpoint 恢复（可以以后再按 remc_simulator 的接口加上）。
"""

from __future__ import annotations

import sys
import json
import math
from pathlib import Path
from typing import Dict, Any

import numpy as np

# CuPy 是可选的：没有 GPU 也不会影响 CPU 版脚本
try:
    import cupy as cp  # type: ignore
    from cupy import ndarray as cupy_ndarray  # type: ignore
except Exception:
    cp = None
    cupy_ndarray = None

# ---------- sys.path 设置 ----------
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from ising_fss.simulation.remc_simulator import HybridREMCSimulator
from ising_fss.simulation.dispatcher import make_replica_seeds
from ising_fss.analysis.fss_analyzer import FSSAnalyzer


# ---------- json.dump helper ----------
def json_default(o):
    """
    让 json.dump 能处理 numpy / cupy / set 等类型：
      - numpy 标量 → Python 标量
      - numpy / cupy 数组 → list
      - 其它不认识的 → repr(o)
    """
    # numpy 标量
    if isinstance(o, (np.floating, np.integer)):
        return o.item()

    # numpy 数组
    if isinstance(o, np.ndarray):
        return o.tolist()

    # cupy 数组
    if cp is not None and cupy_ndarray is not None:
        if isinstance(o, cupy_ndarray):  # type: ignore[attr-defined]
            try:
                return cp.asnumpy(o).tolist()  # type: ignore[attr-defined]
            except Exception:
                return repr(o)

    # 0-d array / 其它“有 item() 的标量”
    if hasattr(o, "shape") and getattr(o, "shape", None) == () and hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            pass

    # set → list
    if isinstance(o, set):
        return list(o)

    # 兜底：字符串表示
    return repr(o)


# ---------- 原始 analyze() → FSSAnalyzer 输入格式 ----------

def to_fss_format(res_raw: Dict[str, Any]) -> Dict[float, Dict[str, Any]]:
    """
    将 REMC 模拟器的原始 analyze() 输出转换为 FSSAnalyzer 需要的格式：

        输入：res_raw = {
            "T_2.100000": {...},
            "T_2.225664": {...},
            "swap": {...},
            "field": 0.0,
            ...
        }

        输出：{
            2.100000: {...},
            2.225664: {...},
            ...
        }

    只保留 key 形如 "T_..." 且 value 为 dict 的条目。
    并且在这里尽量把标量 / 数组都转成 float64，避免精度退化。
    """
    out: Dict[float, Dict[str, Any]] = {}

    for key, val in res_raw.items():
        if not (isinstance(key, str) and key.startswith("T_") and isinstance(val, dict)):
            continue
        try:
            T = float(key.split("_", 1)[1])
        except Exception:
            continue

        obs: Dict[str, Any] = {}
        for k, x in val.items():
            # 标量类：转成 numpy.float64（或 Python float 也等价于双精度）
            if isinstance(x, (int, float, np.floating)):
                obs[k] = np.float64(x)
            # numpy 数组：转成 float64 数组
            elif isinstance(x, np.ndarray):
                obs[k] = np.asarray(x, dtype=np.float64)
            # cupy 数组：先搬到 host，再转 float64
            elif cp is not None and cupy_ndarray is not None and isinstance(x, cupy_ndarray):  # type: ignore[attr-defined]
                obs[k] = cp.asnumpy(x).astype(np.float64)  # type: ignore[attr-defined]
            else:
                # 其它类型（比如字符串、整数列表、元组）原样保留
                obs[k] = x

        out[np.float64(T)] = obs

    return out


# ---------- 合并多次 run：old + new ----------

def merge_analyze_for_one_L(
    old_L: Dict[str, Any],
    new_L: Dict[str, Any],
    L: int,
) -> Dict[str, Any]:
    """
    把同一个 L（例如 L=128）在多次 run 中得到的 analyze() 结果合并：

    - 对每个温度块 "T_xxx"：
        * old 和 new 中的 E_samples / M_samples 拼接（以 float64 存储）；
        * 用拼接后的序列重新计算：E, M, C, chi, U, n_samples 等；
        * E_err, M_err 用简单 sqrt(var/N) 兜底（不做 bootstrap），
          这样与 GPU 版 analyze() 的逻辑保持一致的量纲；
    - 对 swap：
        * 若 attempts / accepts 维度一致，则直接逐对相加；
        * 否则保留 new_L["swap"]。
    - 对其它键（field、rng_versions 等）：
        * 优先使用 new_L 中的条目；
        * old_L 中有而 new_L 中没有的键会被保留。
    """
    N_site = int(L) * int(L)
    merged: Dict[str, Any] = {}

    # 先遍历“新结果”，逐个 key 合并
    for key, new_block in new_L.items():
        # --- 温度块 T_xxx ---
        if isinstance(key, str) and key.startswith("T_") and isinstance(new_block, dict):
            old_block = old_L.get(key, {})

            # 明确用 float64
            e_old = np.asarray(old_block.get("E_samples", []), dtype=np.float64)
            e_new = np.asarray(new_block.get("E_samples", []), dtype=np.float64)
            m_old = np.asarray(old_block.get("M_samples", []), dtype=np.float64)
            m_new = np.asarray(new_block.get("M_samples", []), dtype=np.float64)

            if e_old.size or e_new.size:
                if e_old.size and e_new.size:
                    e_all = np.concatenate([e_old, e_new])
                else:
                    e_all = e_old if e_old.size else e_new
            else:
                e_all = np.asarray([], dtype=np.float64)

            if m_old.size or m_new.size:
                if m_old.size and m_new.size:
                    m_all = np.concatenate([m_old, m_new])
                else:
                    m_all = m_old if m_old.size else m_new
            else:
                m_all = np.asarray([], dtype=np.float64)

            if e_all.size == 0:
                # 没有样本，就直接使用 new_block
                merged[key] = new_block
                continue

            # 温度 T 的确定优先级：new_block["T"] > old_block["T"] > 从 key 解析
            T_val_raw = None
            if isinstance(new_block.get("T", None), (int, float, np.floating)):
                T_val_raw = float(new_block["T"])
            elif isinstance(old_block.get("T", None), (int, float, np.floating)):
                T_val_raw = float(old_block["T"])
            if T_val_raw is None:
                T_val_raw = float(key.split("_", 1)[1])

            T_val = np.float64(T_val_raw)
            beta = np.float64(1.0) / T_val

            mean_e = np.float64(np.mean(e_all))
            if m_all.size:
                mean_m = np.float64(np.mean(m_all))
            else:
                mean_m = np.float64(0.0)

            m2 = m_all ** 2 if m_all.size else np.asarray([], dtype=np.float64)
            m4 = m_all ** 4 if m_all.size else np.asarray([], dtype=np.float64)
            mean_m2 = np.float64(np.mean(m2)) if m2.size else np.float64(0.0)

            var_e = max(np.float64(0.0), np.float64(np.mean(e_all ** 2) - mean_e ** 2))
            if m_all.size:
                var_m = max(np.float64(0.0), mean_m2 - mean_m ** 2)
            else:
                var_m = np.float64(0.0)

            C_point = (beta ** 2) * np.float64(N_site) * var_e
            chi_point = beta * np.float64(N_site) * var_m

            if mean_m2 <= np.float64(1e-15):
                U = np.float64(0.0)
            else:
                m4_mean = np.float64(np.mean(m4)) if m4.size else np.float64(0.0)
                U = np.float64(1.0) - m4_mean / (np.float64(3.0) * (mean_m2 ** 2 + np.float64(1e-16)))

            N_samples = int(e_all.size)
            E_err = np.float64(math.sqrt(float(var_e) / max(1, N_samples)))
            if m_all.size:
                M_err = np.float64(math.sqrt(float(var_m) / max(1, N_samples)))
            else:
                M_err = np.float64(0.0)

            merged[key] = {
                "T": float(T_val),
                "E": float(mean_e),
                "E_err": float(E_err),
                "M": float(mean_m),
                "M_err": float(M_err),
                "C": float(C_point),
                "C_err": 0.0,   # 如需 bootstrap，可在后处理阶段做
                "chi": float(chi_point),
                "chi_err": 0.0,
                "U": float(U),
                "n_samples": int(N_samples),
                "E_samples": e_all,  # 这里保留为 float64 数组
                "M_samples": m_all,
            }

        # --- swap 统计 ---
        elif key == "swap" and isinstance(new_block, dict):
            old_block = old_L.get("swap", {})
            a_old = np.asarray(old_block.get("attempts", []), dtype=np.int64)
            a_new = np.asarray(new_block.get("attempts", []), dtype=np.int64)
            c_old = np.asarray(old_block.get("accepts", []), dtype=np.int64)
            c_new = np.asarray(new_block.get("accepts", []), dtype=np.int64)

            if a_old.size and a_new.size and a_old.size == a_new.size:
                a_all = (a_old + a_new)
                if c_old.size and c_old.size == c_new.size:
                    c_all = (c_old + c_new)
                else:
                    c_all = c_new
                merged[key] = {
                    "attempts": a_all,
                    "accepts": c_all,
                    "total_attempts": int(np.sum(a_all)),
                    "total_accepts": int(np.sum(c_all)),
                }
            else:
                merged[key] = new_block

        # --- 其它键：优先 new，其次 old ---
        else:
            if key in old_L and key not in merged:
                # old 里有、new 里没有的键，先放 old
                merged[key] = old_L[key]
            # new 中的值覆盖 old
            merged[key] = new_block

    # 再把 old_L 里遗漏的键补上
    for key, old_block in old_L.items():
        if key not in merged:
            merged[key] = old_block

    return merged


# ---------- CPU 版：跑单个 L 的 REMC ----------

def run_one_L(L: int, outdir: Path, args) -> Dict[str, Any]:
    """
    跑单个 L 的 HybridREMCSimulator REMC，返回 sim.analyze() 的原始结果：
        {
          "T_2.100000": {...},
          "T_2.225664": {...},
          "swap": {...},
          "field": 0.0,
          ...
        }
    """
    T_min = float(args.T_min)
    T_max = float(args.T_max)
    num_replicas = int(args.num_replicas)

    replica_seeds = make_replica_seeds(master_seed=10_000 + int(L), n_replicas=num_replicas)

    print(
        f"\n=== 运行 REMC (CPU 版): L={L}, "
        f"T∈[{T_min}, {T_max}], replicas={num_replicas}, algo=metropolis_sweep ==="
    )

    sim = HybridREMCSimulator(
        L=L,
        T_min=T_min,
        T_max=T_max,
        num_replicas=num_replicas,
        algorithm="metropolis_sweep",
        h=0.0,
        replica_seeds=replica_seeds,
    )

    # 每个 L 单独一个子目录，用于保存 lattices（若启用）
    save_dir_L = outdir / f"L{L}"
    save_dir_L.mkdir(parents=True, exist_ok=True)

    sim.run(
        equilibration_steps=int(args.equil_steps),
        production_steps=int(args.prod_steps),
        exchange_interval=int(args.exchange_interval),
        thin=int(args.thin),
        verbose=bool(args.verbose),
        save_lattices=bool(args.save_lattices),
        save_dir=str(save_dir_L),
        worker_id=f"cpu_L{L}",
        auto_thin=bool(getattr(args, "auto_thin", False)),
        thin_min=int(getattr(args, "thin_min", 1)),
        thin_max=int(getattr(args, "thin_max", 10_000)),
        tau_update_interval=int(getattr(args, "tau_update_interval", 256)),
        tau_window=int(getattr(args, "tau_window", 2048)),
    )

    res = sim.analyze(verbose=False)
    return res


# ---------- 小工具：按条目换行打印 Tc_est 结果 ----------

def _pretty_print_Tc_est(label: str, est: Dict[str, Any]) -> None:
    print(f"[INFO] {label} 结果:")

    if not isinstance(est, dict):
        print(f"  {est}")
        return

    for key in ("Tc", "var", "std"):
        if key in est:
            print(f"  {key}: {est[key]}")

    if "weights" in est:
        print("  weights:")
        try:
            for w in est["weights"]:
                print(f"    - {w}")
        except TypeError:
            print(f"    {est['weights']}")

    if "pairs" in est:
        print("  pairs:")
        try:
            for pair in est["pairs"]:
                try:
                    L1, L2 = pair
                    print(f"    - ({L1}, {L2})")
                except Exception:
                    print(f"    - {pair}")
        except TypeError:
            print(f"    {est['pairs']}")

    if "crossings" in est:
        print("  crossings:")
        try:
            for c in est["crossings"]:
                try:
                    L1 = getattr(c, "L1", None)
                    L2 = getattr(c, "L2", None)
                    Tc_c = getattr(c, "Tc", None)
                    slope_diff = getattr(c, "slope_diff", None)
                    bracket = getattr(c, "bracket", None)
                    method = getattr(c, "method", "")
                    note = getattr(c, "note", "")

                    line = "    - "
                    if L1 is not None and L2 is not None:
                        line += f"L1={L1}, L2={L2}, "
                    if Tc_c is not None:
                        try:
                            line += f"Tc={Tc_c:.6f}, "
                        except Exception:
                            line += f"Tc={Tc_c}, "
                    if slope_diff is not None:
                        try:
                            line += f"slope_diff={slope_diff:.3f}, "
                        except Exception:
                            line += f"slope_diff={slope_diff}, "
                    if bracket is not None:
                        line += f"bracket={bracket}, "
                    if method:
                        line += f"method={method}"
                    if note:
                        line += f", note={note}"
                    print(line)
                except Exception:
                    print(f"    - {c}")
        except TypeError:
            print(f"    {est['crossings']}")

    for key, value in est.items():
        if key in ("Tc", "var", "std", "weights", "pairs", "crossings"):
            continue
        print(f"  {key}: {value}")


# ---------- 基于 raw_results 的 FSS 分析 ----------

def run_fss_analysis_from_raw(
    results_all_raw: Dict[str, Dict[str, Any]],
    outdir: Path,
    Tc_theory: float = 2.269185,
) -> Dict[str, Any]:
    """
    使用合并后的 raw_results 做 FSS 分析：
      - 先用 to_fss_format 转成 FSSAnalyzer 输入形式；
      - 再补充 *_stderr 字段；
      - 然后跑 Tc / γ/ν / 数据塌缩。
    返回 estimate_Tc('U') 的完整字典。
    """
    print("\n=== 基于合并后的 raw_results 构建 FSSAnalyzer ===")

    results_all_fss: Dict[int, Dict[float, Dict[str, Any]]] = {}
    for L_key, block in results_all_raw.items():
        try:
            L_int = int(L_key)
        except Exception:
            continue

        fss_block = to_fss_format(block)

        # 给 FSSAnalyzer 补上 *_stderr 字段（沿用 *_err）
        for obs in fss_block.values():
            if not isinstance(obs, dict):
                continue
            for base in ("E", "M", "C", "chi", "U"):
                err_key = f"{base}_err"
                stderr_key = f"{base}_stderr"
                if err_key in obs and stderr_key not in obs:
                    val = obs[err_key]
                    if isinstance(val, (int, float, np.floating)):
                        obs[stderr_key] = float(val)

        results_all_fss[L_int] = fss_block

    if not results_all_fss:
        print("⚠️ 没有可用的 FSS 数据（可能所有 L 都为空？）")
        return {}

    analyzer = FSSAnalyzer(results_all_fss, Tc_theory=Tc_theory)

    # 1) Binder U 交叉 → Tc 估计
    Tc_val = None
    Tc_est: Dict[str, Any] = {}
    try:
        est = analyzer.estimate_Tc("U")
        if isinstance(est, dict):
            Tc_est = est
            Tc_val = float(est.get("Tc", None))
            _pretty_print_Tc_est("estimate_Tc('U')", est)
        else:
            Tc_val = float(est)
            Tc_est = {"Tc": Tc_val}
            print(f"[INFO] estimate_Tc('U') 得到 Tc ≈ {Tc_val:.6f}")
    except Exception as e:
        print("[WARN] estimate_Tc('U') 失败:", e)

    if Tc_val is None:
        Tc_val = Tc_theory
        print(f"[INFO] 使用理论 Tc = {Tc_val:.6f} 作为后续拟合基准")
    else:
        print(f"[INFO] 估计 Tc ≈ {Tc_val:.6f} (理论值 Tc≈{Tc_theory})")

    # 2) 用 χ 的 FSS 拟合 γ/ν
    gamma_over_nu = None
    try:
        expo = analyzer.extract_critical_exponents(
            observable="chi",
            Tc_hint=Tc_val,
            fit_nu=False,  # ν 已知为 1 的情形下，只拟合 γ/ν 更稳
        )
        print("exponents (from chi):", expo)

        for k in ["gamma_over_nu", "exponent_ratio", "exponent"]:
            if k in expo:
                gamma_over_nu = float(expo[k])
                print(f"[INFO] 识别到 {k} ≈ {gamma_over_nu:.4f}")
                break
    except TypeError:
        expo = analyzer.extract_critical_exponents("chi")
        print("exponents (from chi):", expo)
        for k in ["gamma_over_nu", "exponent_ratio", "exponent"]:
            if k in expo:
                gamma_over_nu = float(expo[k])
                print(f"[INFO] 识别到 {k} ≈ {gamma_over_nu:.4f}")
                break
    except Exception as e:
        print("[WARN] 提取临界指数失败:", e)

    if gamma_over_nu is not None:
        print(
            "[INFO] 理论值 γ/ν ≈ 1.75; "
            f"当前拟合得到 γ/ν ≈ {gamma_over_nu:.4f}"
        )
        if gamma_over_nu < 0:
            print("[WARN] γ/ν < 0 明显违背物理常识，说明采样或拟合还有问题。")
    else:
        print("[WARN] 未能从 expo 中识别出 γ/ν，后续 data collapse 将使用理论值。")
        gamma_over_nu = 1.75

    # 3) 做一次 χ 的数据塌缩
    print("\n=== chi 数据塌缩 (CPU 版) ===")
    if not hasattr(analyzer, "data_collapse"):
        print("[INFO] 当前 FSSAnalyzer 未实现 data_collapse，跳过该步骤。")
    else:
        try:
            collapse = analyzer.data_collapse(
                observable="chi",
                Tc=Tc_val,
                nu=1.0,                # 2D Ising 的理论 ν = 1
                exponent_ratio=gamma_over_nu,
            )
            print("data_collapse keys:", list(collapse.keys()))
            if "score" in collapse:
                print(f"collapse score ≈ {collapse['score']:.6g}")
                print("（score 越小通常代表塌缩质量越好，仅供相对比较）")
        except Exception as e:
            print("[WARN] data_collapse 调用失败:", e)

    # 写 Tc_est.json
    Tc_path = outdir / "Tc_est.json"
    try:
        with open(Tc_path, "w", encoding="utf-8") as f:
            json.dump(Tc_est, f, indent=2, default=json_default, ensure_ascii=False)
        print(f"✅ Tc 估计与配对 crossing 信息已写入 {Tc_path}")
    except Exception as exc:
        print(f"❌ 写 Tc_est.json 失败: {exc}")

    return Tc_est


# ---------- main：整体管线 ----------

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--L_list", type=int, nargs="+", default=[16, 32, 64],
                        help="要跑的 L 列表，例如: --L_list 16 32 64")
    parser.add_argument("--outdir", default="runs/cpu_large_scale_fss",
                        help="输出目录（raw_results.json / Tc_est.json / lattices 等）")

    # 物理 & 模拟参数（默认取你原来 demo 的那一组）
    parser.add_argument("--T_min", type=float, default=2.1)
    parser.add_argument("--T_max", type=float, default=2.5)
    parser.add_argument("--num_replicas", type=int, default=16)

    parser.add_argument("--equil_steps", type=int, default=20_000,
                        help="预热步数（sweeps）")
    parser.add_argument("--prod_steps", type=int, default=100_000,
                        help="生产阶段总 sweeps 数（不包含预热）")
    parser.add_argument("--exchange_interval", type=int, default=5,
                        help="每隔多少 sweeps 尝试一次 replica 交换")

    #  parser.add_argument("--thin", type=int, default=20,
    #                      help="初始 thinning 间隔（sweeps）。若 --auto_thin，则作为起始 thin。")
    parser.add_argument("--thin", type=int, default=200,
                        help="初始 thinning 间隔（sweeps）。若 --auto_thin，则作为起始 thin。")

    # 自适应 thin 相关参数（HybridREMCSimulator 也支持）
    parser.add_argument("--auto_thin", action="store_true",
                        help="启用在线估计 τ_int 的自适应 thinning。")
    parser.add_argument("--thin_min", type=int, default=1,
                        help="自适应 thinning 的最小值（单位：sweeps）。")
    parser.add_argument("--thin_max", type=int, default=10_000,
                        help="自适应 thinning 的最大值（单位：sweeps）。")
    parser.add_argument("--tau_update_interval", type=int, default=256,
                        help="每隔多少个 production sweeps 做一次 τ_int 更新。")
    parser.add_argument("--tau_window", type=int, default=2048,
                        help="估计 τ_int 时使用的窗口长度（最大历史样本数）。")

    # I/O & 其它
    parser.add_argument("--save_lattices", action="store_true",
                        help="是否把 lattice 轨迹写入 HDF5（每个温度一个文件）。")
    parser.add_argument("--verbose", action="store_true",
                        help="打印一些进度信息。")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CPU REMC → FSSAnalyzer → Tc / γ/ν / 数据塌缩")
    print("=" * 70)
    print(
        f"参数概览：L_list={args.L_list}, T∈[{args.T_min},{args.T_max}], "
        f"replicas={args.num_replicas}, equil={args.equil_steps}, prod={args.prod_steps}, thin={args.thin}"
    )

    # ---------- 读取旧的 raw_results.json（用于合并样本） ----------
    raw_path = outdir / "raw_results.json"
    prev_all_raw: Dict[str, Any] = {}
    if raw_path.exists():
        try:
            with open(raw_path, "r", encoding="utf-8") as f:
                prev_all_raw = json.load(f)
            if not isinstance(prev_all_raw, dict):
                prev_all_raw = {}
        except Exception as exc:
            print(f"⚠️ 读取已有 raw_results.json 失败，将从空白开始: {exc}")
            prev_all_raw = {}
    else:
        prev_all_raw = {}

    # ---------- 本次 run 的（或合并后的）结果 ----------
    results_all_raw: Dict[str, Dict[str, Any]] = {}

    for L in args.L_list:
        print(f"\n=== REMC for L={L} ===")
        res_new = run_one_L(L, outdir, args)

        L_key = str(L)
        if L_key in prev_all_raw:
            print(f"[L={L}] 🔄 与 raw_results.json 中旧样本进行合并（追加模式）")
            merged = merge_analyze_for_one_L(prev_all_raw[L_key], res_new, L)
        else:
            merged = res_new

        results_all_raw[L_key] = merged

    # 把这次没有跑到的 L（但旧结果里存在的）搬过来
    for L_key, block in prev_all_raw.items():
        if L_key not in results_all_raw:
            results_all_raw[L_key] = block

    # ---------- 写回合并后的 raw_results.json ----------
    try:
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(results_all_raw, f, indent=2, default=json_default, ensure_ascii=False)
        print(f"✅ 合并后的统计结果已写入 {raw_path}")
    except Exception as exc:
        print(f"❌ 写 raw_results.json 失败: {exc}")
        return

    # ---------- FSS 分析 ----------
    Tc_est = run_fss_analysis_from_raw(results_all_raw, outdir=outdir)
    print("Done. See", outdir)


if __name__ == "__main__":
    main()

