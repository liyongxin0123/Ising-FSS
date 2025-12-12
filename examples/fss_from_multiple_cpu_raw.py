# examples/fss_from_multiple_cpu_raw.py
from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

# ---- 基准路径：当前脚本所在目录（examples） ----
BASE = Path(__file__).resolve().parent          # .../ising-fss/examples
ROOT = BASE.parent                              # .../ising-fss

# 加入 src 到 sys.path，便于 import ising_fss
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from ising_fss.analysis.fss_analyzer import FSSAnalyzer


def to_fss_format(res_raw: Dict[str, Any]) -> Dict[float, Dict[str, Any]]:
    """
    把单个 L 的 analyze() 结果转换为 FSSAnalyzer 需要的形式：
      {"T_2.10": {...}, "T_2.12": {...}, "swap": {...}, ...}
    -> {2.10: {...}, 2.12: {...}, ...}
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
    从多个 outdir 中读取 raw_results.json，并按 L_key 组织到：
      results_all_raw[L_key] = block_for_that_L
    """
    results_all_raw: Dict[str, Dict[str, Any]] = {}

    for odir in outdirs:
        raw_path = odir / "raw_results.json"
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
            if L_key in results_all_raw:
                print(f"[INFO] L={L_key} 已存在，这里简单覆盖旧值（如需 merge，可在此调用 merge_analyze_for_one_L）。")
            results_all_raw[L_key] = block

    return results_all_raw


def main():
    # 注意：这里就是 examples/runs/... 目录
    outdirs = [
        BASE / "runs/cpu_L32_thin20",
        BASE / "runs/cpu_L64_thin90",
        BASE / "runs/cpu_L96_thin400",
        BASE / "runs/cpu_L128_thin1800",
    ]

    results_all_raw_by_L = load_all_raw(outdirs)
    if not results_all_raw_by_L:
        print("❌ 没有读到任何 raw_results.json，检查 outdir 路径是否正确。")
        return

    # 转成 FSSAnalyzer 需要的格式：results_all_fss[L_int][T] = {obs_dict}
    results_all_fss: Dict[int, Dict[float, Dict[str, Any]]] = {}
    for L_key, block in results_all_raw_by_L.items():
        try:
            L_int = int(L_key)
        except Exception:
            print(f"[WARN] 无法将 L_key='{L_key}' 转成 int，跳过。")
            continue

        fss_block = to_fss_format(block)

        # 给 FSSAnalyzer 补充 *_stderr 字段（直接沿用 *_err）
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

    # 1) Binder U → Tc
    try:
        Tc_est = analyzer.estimate_Tc("U")
        #  print("\n=== estimate_Tc('U') 结果 ===")
        print(json.dumps(Tc_est, indent=2, ensure_ascii=False, default=str))
    except Exception as e:
        print("❌ estimate_Tc('U') 失败:", e)
        return

    # 2) 把 Tc_est 存到工程根目录的 runs 下，便于统一管理结果
    fss_outdir = BASE / "runs/cpu_fss_L32_64_96_128"
    fss_outdir.mkdir(parents=True, exist_ok=True)
    Tc_path = fss_outdir / "Tc_est_combined.json"
    with open(Tc_path, "w", encoding="utf-8") as f:
        json.dump(Tc_est, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n✅ 统一 FSS 的 Tc 估计已写入 {Tc_path}")


if __name__ == "__main__":
    main()

