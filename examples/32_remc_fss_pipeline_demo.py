# examples/analysis/remc_fss_demo.py
"""
物理版示例：REMC → FSSAnalyzer → Tc / γ/ν / 数据塌缩

注意：
- 这是“相对物理靠谱”的 demo，而不是快速单元测试。
- 默认参数会比 demo_remc_fss_pipeline.py 跑得久很多（视机器性能，可能是分钟级甚至更长）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np

# ---- 保证可以直接从源码导入 ising_fss ----
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from ising_fss.simulation.remc_simulator import HybridREMCSimulator
from ising_fss.simulation.dispatcher import make_replica_seeds
from ising_fss.analysis.fss_analyzer import FSSAnalyzer


# -----------------------------
# 1. 单个 L 的 REMC + 分析
# -----------------------------

def run_single_L(
    L: int,
    T_min: float,
    T_max: float,
    num_replicas: int = 16,
    equil_steps: int = 20_000,
    prod_steps: int = 80_000,
    thin: int = 20,
    exchange_interval: int = 5,
    algorithm: str = "metropolis_sweep",
) -> Dict[float, Dict[str, Any]]:
    """
    跑单一晶格尺寸 L 的 REMC，并返回：
        { T: {obs_dict}, ... }

    其中 obs_dict 中会尽可能包含：
        - E, M, C, chi, U 及其误差：
            E_err, M_err, C_err, chi_err, U_err
        - 以及可选的样本数组：
            E_samples, M_samples, C_samples, chi_samples, ...
    """
    print(
        f"\n=== 运行 REMC (物理版): L={L}, "
        f"T∈[{T_min}, {T_max}], replicas={num_replicas}, algo={algorithm} ==="
    )

    replica_seeds = make_replica_seeds(master_seed=10_000 + L, n_replicas=num_replicas)

    sim = HybridREMCSimulator(
        L=L,
        T_min=T_min,
        T_max=T_max,
        num_replicas=num_replicas,
        replica_seeds=replica_seeds,
        algorithm=algorithm,
        h=0.0,
    )

    sim.run(
        equilibration_steps=equil_steps,
        production_steps=prod_steps,
        exchange_interval=exchange_interval,
        thin=thin,
        save_lattices=False,  # 这里只关心统计量，不落盘晶格
        verbose=False,
    )

    res = sim.analyze(verbose=False)

    temp_map: Dict[float, Dict[str, Any]] = {}

    # 标量均值
    mean_keys = ["E", "M", "C", "chi", "U"]
    # 标准误差
    err_keys = ["E_err", "M_err", "C_err", "chi_err", "U_err"]
    # 样本数组
    sample_keys = [
        "E_samples",
        "M_samples",
        "C_samples",
        "chi_samples",
    ]

    for key, val in res.items():
        if not isinstance(key, str) or not key.startswith("T_"):
            continue
        try:
            T = float(key.split("_", 1)[1])
        except Exception:
            continue

        obs: Dict[str, Any] = {}

        # ---- 1) 均值 ----
        for name in mean_keys:
            if name in val:
                try:
                    v = float(val[name])
                    if np.isfinite(v):
                        obs[name] = v
                except Exception:
                    continue

        # ---- 2) 误差条 ----
        for name in err_keys:
            if name in val:
                try:
                    v = float(val[name])
                    if not np.isfinite(v):
                        continue
                    # 原始 *_err 保留
                    obs[name] = v

                    # 关键一步：再复制一份成 *_stderr，给 FSSAnalyzer 用
                    # 例如 chi_err -> chi_stderr, C_err -> C_stderr
                    if name.endswith("_err"):
                        base = name[:-4]  # 去掉 "_err"
                        stderr_key = f"{base}_stderr"
                        obs[stderr_key] = v
                except Exception:
                    continue

        # ---- 3) 样本数组 ----
        for name in sample_keys:
            if name in val:
                try:
                    arr = np.asarray(val[name], dtype=float)
                    if arr.size > 0:
                        obs[name] = arr
                except Exception:
                    continue

        # ---- 4) 辅助信息（如 n_samples）----
        for aux_key in ["n_samples", "samples"]:
            if aux_key in val:
                try:
                    obs[aux_key] = int(val[aux_key])
                except Exception:
                    pass

        temp_map[T] = obs

    print("  收到温度点数量:", len(temp_map))
    return temp_map


# -----------------------------
# 2. 多个 L 的结果拼成 FSS 输入
# -----------------------------
def build_fss_results_for_sizes(
    L_list,
    T_min: float,
    T_max: float,
    num_replicas: int = 16,
    equil_steps: int = 20_000,
    prod_steps: int = 80_000,
    thin: int = 20,
    exchange_interval: int = 5,
    algorithm: str = "metropolis_sweep",
):
    """
    返回结构：
        results[L][T] = {obs_dict}

    obs_dict 里包含：
        - E, M, C, chi, U
        - 及其误差：E_err, M_err, C_err, chi_err, U_err
        - 以及兼容 FSSAnalyzer 的：E_stderr, M_stderr, C_stderr, chi_stderr, U_stderr
        - 以及可选的 *_samples 数组（若 analyze() 提供）。
    """
    all_results: Dict[int, Dict[float, Dict[str, Any]]] = {}
    for L in L_list:
        all_results[int(L)] = run_single_L(
            L=L,
            T_min=T_min,
            T_max=T_max,
            num_replicas=num_replicas,
            equil_steps=equil_steps,
            prod_steps=prod_steps,
            thin=thin,
            exchange_interval=exchange_interval,
            algorithm=algorithm,
        )
    return all_results


# -----------------------------
# 工具函数：按条目换行打印 Tc_est 结果
# -----------------------------
def _pretty_print_Tc_est(label: str, est: Dict[str, Any]) -> None:
    """
    按条目（key）逐行打印 estimate_Tc 返回的字典，
    对 crossings / weights / pairs 做简单展开，便于阅读。
    """
    print(f"[INFO] {label} 结果:")

    if not isinstance(est, dict):
        print(f"  {est}")
        return

    # 先打几个常用标量
    for key in ("Tc", "var", "std"):
        if key in est:
            print(f"  {key}: {est[key]}")

    # 打印权重
    if "weights" in est:
        print("  weights:")
        try:
            for w in est["weights"]:
                print(f"    - {w}")
        except TypeError:
            print(f"    {est['weights']}")

    # 打印 (L1, L2) 配对
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

    # 打印 crossings 详情
    if "crossings" in est:
        print("  crossings:")
        try:
            for c in est["crossings"]:
                # 尝试按 PairCrossing 的属性来打印
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
                    # 打印失败就直接 print 对象
                    print(f"    - {c}")
        except TypeError:
            print(f"    {est['crossings']}")

    # 其余键（如果有）也逐行打印，避免遗漏
    for key, value in est.items():
        if key in ("Tc", "var", "std", "weights", "pairs", "crossings"):
            continue
        print(f"  {key}: {value}")


# -----------------------------
# 3. FSS 分析（更偏“物理版”）
# -----------------------------
def run_fss_analysis(results: Dict[int, Dict[float, Dict[str, Any]]]):
    print("\n=== 构建 FSSAnalyzer (物理版) ===")

    analyzer = FSSAnalyzer(results, Tc_theory=2.269185)

    # -------- 1) Binder U 的交叉点 → Tc 估计 --------
    Tc_val = None
    try:
        Tc_est = analyzer.estimate_Tc("U")
        if isinstance(Tc_est, dict):
            Tc_val = float(Tc_est.get("Tc", None))
            # 这里改成按条目换行打印
            _pretty_print_Tc_est("estimate_Tc('U')", Tc_est)
        else:
            Tc_val = float(Tc_est)
            print(f"[INFO] estimate_Tc('U') 得到 Tc ≈ {Tc_val:.6f}")
    except Exception as e:
        print("[WARN] estimate_Tc('U') 失败:", e)

    if Tc_val is None:
        Tc_val = 2.269185
        print(f"[INFO] 使用理论 Tc = {Tc_val:.6f} 作为后续拟合基准")
    else:
        print(f"[INFO] 估计 Tc ≈ {Tc_val:.6f} (理论值 Tc≈2.269185)")

    # -------- 2) 提取 γ/ν （用 χ 的 FSS 标度） --------
    gamma_over_nu = None
    try:
        expo = analyzer.extract_critical_exponents(
            observable="chi",
            Tc_hint=Tc_val,
            fit_nu=False,  # ν 已知为 1 的情形下，只拟合 γ/ν 更稳一些
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

    # -------- 3) 数据塌缩（chi） --------
    print("\n=== chi 数据塌缩 (物理版) ===")
    if not hasattr(analyzer, "data_collapse"):
        print("[INFO] 当前 FSSAnalyzer 未实现 data_collapse，跳过该步骤。")
        return

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


# -----------------------------
# 4. main：一键跑“物理版”管线
# -----------------------------
def main():
    # ---- 这里是可以按需要调节的“物理参数” ----
    L_list = [16, 32, 64]    # 如果机器给力可以加到 128
    T_min, T_max = 2.1, 2.5  # 把温度区间收窄到临界附近
    num_replicas = 16        # 温度点数量（每个 L 上的 T 数目）

    equil_steps = 20_000     # 平衡 steps
    prod_steps = 80_000      # 采样 steps
    thin = 20                # 每隔 thin sweeps 取一个样本
    exchange_interval = 5    # 每 5 sweeps 尝试一次交换

    print("=" * 70)
    print("物理版示例：REMC → FSSAnalyzer → Tc / γ/ν / 数据塌缩")
    print("=" * 70)
    print(
        f"参数概览：L_list={L_list}, T∈[{T_min},{T_max}], "
        f"replicas={num_replicas}, equil={equil_steps}, prod={prod_steps}, thin={thin}"
    )

    results = build_fss_results_for_sizes(
        L_list=L_list,
        T_min=T_min,
        T_max=T_max,
        num_replicas=num_replicas,
        equil_steps=equil_steps,
        prod_steps=prod_steps,
        thin=thin,
        exchange_interval=exchange_interval,
        algorithm="metropolis_sweep",
    )

    print("\n=== results 预览 ===")
    for L, Tmap in results.items():
        print("L=", L, "| #T =", len(Tmap))

    run_fss_analysis(results)


if __name__ == "__main__":
    main()

