# examples/fss_from_multiple_gpu_raw.py
from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

# 保证能导入 ising_fss
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from ising_fss.analysis.fss_analyzer import FSSAnalyzer


def to_fss_format(res_raw: Dict[str, Any]) -> Dict[float, Dict[str, Any]]:
    """
    将 GPU/CPU REMC 的 analyze() 输出转换为 FSSAnalyzer 需要的格式：
        res_raw = {"T_2.100000": {...}, "T_2.225664": {...}, "swap": {...}, ...}
    ->
        {2.100000: {...}, 2.225664: {...}, ...}
    并把数值统一成 float64。
    """
    out: Dict[float, Dict[str, Any]] = {}
    for key, val in res_raw.items():
        if not isinstance(key, str):
            continue
        if not key.startswith("T_"):
            continue
        if not isinstance(val, dict):
            continue
        try:
            T = float(key.split("_", 1)[1])
        except Exception:
            continue

        obs: Dict[str, Any] = {}
        for k, x in val.items():
            if isinstance(x, (int, float, np.floating)):
                obs[k] = np.float64(x)
            elif isinstance(x, np.ndarray):
                obs[k] = np.asarray(x, dtype=np.float64)
            else:
                obs[k] = x
        out[np.float64(T)] = obs
    return out


def load_all_raw(outdirs) -> Dict[str, Dict[str, Any]]:
    """
    从多个 outdir 中读取 raw_results.json，并按 L_key 合并到一个大 dict：
        results_all_raw[L_key] = block
    这里假设每个 outdir 里只有一个 L（正好对应我们每次只跑一个 L 的设置）。
    """
    results_all_raw: Dict[str, Dict[str, Any]] = {}

    for odir in outdirs:
        raw_path = Path(odir) / "raw_results.json"
        if not raw_path.exists():
            print(f"[WARN] {raw_path} 不存在，跳过该目录。")
            continue
        try:
            with open(raw_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] 读取 {raw_path} 失败: {e}")
            continue

        if not isinstance(data, dict):
            print(f"[WARN] {raw_path} 内容不是 dict，跳过。")
            continue

        for L_key, block in data.items():
            # 如果后续你真的有“同一个 L 分散在多个 outdir”的需求，
            # 可以在这里用 merge_analyze_for_one_L 做合并。
            if L_key in results_all_raw:
                print(f"[INFO] L={L_key} 已存在，这里简单覆盖旧值（如需合并，可在此调用 merge_analyze_for_one_L）。")
            results_all_raw[L_key] = block

    return results_all_raw


def main():
    # 参与 FSS 的 4 个 L 所在目录（与 bash 脚本里的 outdir 一一对应）
    outdirs = [
        "runs/gpu_L32_thin20",
        "runs/gpu_L64_thin90",
        "runs/gpu_L96_thin400",
        "runs/gpu_L128_thin1800",
    ]

    results_all_raw = load_all_raw(outdirs)
    if not results_all_raw:
        print("❌ 没有读到任何 raw_results.json，检查 outdir 路径是否正确。")
        return

    # 转成 FSSAnalyzer 需要的格式：results_all_fss[L_int][T] = {obs_dict}
    results_all_fss: Dict[int, Dict[float, Dict[str, Any]]] = {}
    for L_key, block in results_all_raw.items():
        try:
            L_int = int(L_key)
        except Exception:
            print(f"[WARN] 无法将 L_key='{L_key}' 转成 int，跳过。")
            continue

        fss_block = to_fss_format(block)

        # 给 FSSAnalyzer 补上 *_stderr 字段（沿用 *_err），以兼容你之前的 FSS 逻辑
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

    print("参与 FSS 的 L 列表:", sorted(results_all_fss.keys()))

    if len(results_all_fss) < 2:
        print("⚠️ 参与 FSS 的 L 少于 2 个，Binder 交叉无法工作。")
        return

    analyzer = FSSAnalyzer(results_all_fss, Tc_theory=2.269185)

    # 1) Binder U 交叉 → Tc
    try:
        Tc_est = analyzer.estimate_Tc("U")
        print("\n=== estimate_Tc('U') 结果 ===")
        print(json.dumps(Tc_est, indent=2, ensure_ascii=False, default=str))
    except Exception as e:
        print("❌ estimate_Tc('U') 失败:", e)
        return

    # 2) 你也可以在这里继续做 γ/ν 和数据塌缩（可直接复用你之前 CPU FSS 脚本里的逻辑）

    # 若需要，把 Tc_est 写到一个统一的位置
    outdir = Path("runs/gpu_fss_L32_64_96_128")
    outdir.mkdir(parents=True, exist_ok=True)
    Tc_path = outdir / "Tc_est_combined.json"
    with open(Tc_path, "w", encoding="utf-8") as f:
        json.dump(Tc_est, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n✅ 统一 FSS 的 Tc 估计已写入 {Tc_path}")


if __name__ == "__main__":
    main()

