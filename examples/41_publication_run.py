# examples/41_publication_run0.py
"""
“论文级” FSS 生产脚本（修正版）：
- 多个 L
- 较长 REMC
- 保存 raw 结果 + FSS-friendly 结果
- 用 FSS-friendly 结果喂给 FSSAnalyzer
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from ising_fss.simulation.parallel import across_L
from ising_fss.analysis.fss_analyzer import FSSAnalyzer


# ---------- JSON 序列化 helper ----------
def json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return list(o)
    return repr(o)


# ---------- 把 across_L 的 raw 结果，转换成 FSSAnalyzer 期待的结构 ----------
def to_fss_results(
    raw: Dict[Any, Any]
) -> Dict[int, Dict[float, Dict[str, float]]]:
    """
    输入：across_L 返回的 raw 结果
          raw[L] 基本上是 sim.analyze() 的字典，包括 'T_2.000000'、'swap' 等键
    输出：FSSAnalyzer 期望的结构：
          { L : { T(float) : { 'E': ..., 'M': ..., 'C': ..., 'chi': ..., 'U': ... } } }
    """
    out: Dict[int, Dict[float, Dict[str, float]]] = {}

    for L_key, res in raw.items():
        # 1) 解析 L
        try:
            L = int(L_key)
        except Exception:
            if isinstance(L_key, int):
                L = L_key
            else:
                print(f"[WARN] skip non-int L key: {L_key!r}")
                continue

        if not isinstance(res, dict):
            print(f"[WARN] raw[{L}] is not dict, got {type(res)}; skip")
            continue

        temp_map: Dict[float, Dict[str, float]] = {}

        for key, val in res.items():
            # 只保留形如 'T_2.345000' 的键
            if not (isinstance(key, str) and key.startswith("T_")):
                continue
            try:
                T = float(key.split("_", 1)[1])
            except Exception:
                print(f"[WARN] cannot parse temperature key {key!r} at L={L}")
                continue

            if not isinstance(val, dict):
                # 理论上这里应该是 analyze() 返回的 per-T dict
                print(f"[WARN] value at L={L}, {key} is not dict ({type(val)}); skip")
                continue

            obs: Dict[str, float] = {}
            for name in ["E", "M", "C", "chi", "U"]:
                if name not in val:
                    continue
                v = val[name]
                # 如果是数组/列表，取均值
                if isinstance(v, (list, tuple, np.ndarray)):
                    try:
                        v = float(np.mean(v))
                    except Exception:
                        continue
                else:
                    try:
                        v = float(v)
                    except Exception:
                        continue
                obs[name] = v

            if obs:
                temp_map[T] = obs

        if not temp_map:
            print(f"[WARN] no valid temperature entries for L={L}; this size will be empty in FSS.")
        out[L] = temp_map

    return out


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--L_list", type=int, nargs="+", default=[16,24,32])
    parser.add_argument("--T_min", type=float, default=2.0)
    parser.add_argument("--T_max", type=float, default=2.6)
    parser.add_argument("--replicas", type=int, default=64)
    parser.add_argument("--equil", type=int, default=20000)
    parser.add_argument("--prod", type=int, default=100000)
    parser.add_argument("--algo", default="metropolis_sweep")
    parser.add_argument("--outdir", default="runs/publication_fss")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------- 1. 大规模 REMC ----------
    raw_results = across_L(
        L_list=args.L_list,
        T_min=args.T_min,
        T_max=args.T_max,
        num_replicas=args.replicas,
        equilibration=args.equil,
        production=args.prod,
        algorithm=args.algo,
        exchange_interval=10,
        thin=50,
        n_processes_per_L=1,
        checkpoint_dir=str(outdir / "ckpt"),
        checkpoint_final=True,
    )

    # ---------- 2. 保存 raw 结果（原始 analyze 输出） ----------
    raw_json = outdir / "raw_results.json"
    with raw_json.open("w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2, default=json_default)
    print(f"[INFO] raw results saved to {raw_json}")

    # ---------- 3. 转换成 FSS-friendly 结构 ----------
    fss_results = to_fss_results(raw_results)

    fss_json = outdir / "fss_results.json"
    with fss_json.open("w", encoding="utf-8") as f:
        json.dump(fss_results, f, indent=2, default=json_default)
    print(f"[INFO] FSS-friendly results saved to {fss_json}")

    # ---------- 4. FSS 分析 ----------
    analyzer = FSSAnalyzer(fss_results)

    # (1) Tc 估计
    try:
        Tc_est = analyzer.estimate_Tc("U")
        Tc_json = outdir / "Tc_est.json"
        with Tc_json.open("w", encoding="utf-8") as f:
            json.dump(Tc_est, f, indent=2, default=json_default)
        print(f"[INFO] Tc estimate saved to {Tc_json}")
    except Exception as e:
        print("[WARN] estimate_Tc('U') failed:", e)

    # (2) 临界指数示例（chi）
    try:
        expo = analyzer.extract_critical_exponents("chi")
        expo_json = outdir / "exponents_chi.json"
        with expo_json.open("w", encoding="utf-8") as f:
            json.dump(expo, f, indent=2, default=json_default)
        print(f"[INFO] critical exponents (chi) saved to {expo_json}")
    except Exception as e:
        print("[WARN] extract_critical_exponents('chi') failed:", e)

    print("Publication run finished. Results under", outdir)


if __name__ == "__main__":
    main()

