#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 GPU 管线生成的 raw_results.json 中读取结果，
复用 42_gpu_large_scale_fss.py 里的 FSS 部分，单独做一次 FSS 分析：

  raw_results.json  ---->  FSSAnalyzer  ---->  Tc_est.json

注意：
- 不跑任何模拟，只做后处理分析；
- 逻辑上等价于 42_gpu_large_scale_fss.py 末尾那段 FSS 代码。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

# 尝试导入 CuPy（保持和 42 脚本一致的环境）
try:
    import cupy as cp  # type: ignore
    from cupy import ndarray as cupy_ndarray  # type: ignore
except Exception:
    cp = None
    cupy_ndarray = None

# 项目内 FSS 分析器（与 42 脚本一致的导入方式）
try:
    from ising_fss.analysis.fss_analyzer import FSSAnalyzer  # type: ignore
except Exception:
    # 如果工程布局不同，你可以按自己的路径修改这一行
    from fss_analyzer import FSSAnalyzer  # type: ignore


# ---------- JSON 序列化 helper（拷贝自 42_gpu_large_scale_fss.py） ----------
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

    # 集合类
    if isinstance(o, (set, frozenset)):
        return list(o)

    # 兜底：转成字符串
    return repr(o)


# ---------- 42 里用的 raw→FSS 格式转换函数（原样抽出） ----------
def to_fss_format(res_raw: Dict[str, Any]) -> Dict[float, Dict[str, Any]]:
    """
    将 GPU 模拟器的原始输出转换为 FSSAnalyzer 需要的格式：

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

    转换规则：
        - 只保留键名以 "T_" 开头的温度块；
        - 键名 "T_xxx" 中的 xxx 解析为 float T；
        - 温度块内：
            * numpy 标量 → float64
            * numpy 数组 → float64 数组
            * cupy 数组 → 先搬回 host 再转 float64 数组
            * 其它类型原样保留
    """
    out: Dict[float, Dict[str, Any]] = {}

    for key, val in res_raw.items():
        if not isinstance(key, str):
            continue
        if not key.startswith("T_"):
            # 跳过 'swap', 'field', 'rng_model' 等非温度键
            continue
        if not isinstance(val, dict):
            continue
        try:
            T = float(key.split("_", 1)[1])
        except Exception:
            continue

        # 对该温度块内的字段做一次“float64 化”
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


# ---------- 主程序：只做 FSS 分析 ----------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="从 raw_results.json 读取结果并使用 FSSAnalyzer 做 Binder FSS（42_gpu_large_scale_fss.py 的 FSS 部分抽取版）"
    )
    parser.add_argument(
        "--outdir",
        default="runs/gpu_large_scale_fss",
        help="包含 raw_results.json 的输出目录（Tc_est.json 也会写在这里）",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir).resolve()
    raw_path = outdir / "raw_results.json"

    if not raw_path.exists():
        print(f"❌ 找不到 raw_results.json: {raw_path}")
        return

    # ---------- 读取 raw_results.json ----------
    try:
        with raw_path.open("r", encoding="utf-8") as f:
            results_all_raw = json.load(f)
        if not isinstance(results_all_raw, dict):
            raise ValueError("raw_results.json 顶层不是 dict")
    except Exception as exc:
        print(f"❌ 读取或解析 {raw_path} 失败: {exc}")
        return

    if not results_all_raw:
        print("⚠️ raw_results.json 为空，无法做 FSS 分析。")
        return

    # ---------- 把所有 L 的结果喂给 FSSAnalyzer ----------
    results_all_fss: Dict[int, Dict[float, Dict[str, Any]]] = {}
    for L_key, block in results_all_raw.items():
        try:
            L_int = int(L_key)
        except Exception:
            # 跳过非整数键（例如元数据）
            continue
        if not isinstance(block, dict):
            continue
        results_all_fss[L_int] = to_fss_format(block)

    if not results_all_fss:
        print("⚠️ 没有可用的 FSS 数据（可能所有 L 都为空？）")
        return

    print("可用于 FSS 的 L 尺寸：", sorted(results_all_fss.keys()))

    # 与 42_gpu_large_scale_fss.py 保持一致：直接用默认参数做 Binder FSS
    analyzer = FSSAnalyzer(results_all_fss)
    Tc_est = analyzer.estimate_Tc("U")  # 这里传入 "U" 只是保持与原脚本一致，等价于 use_all_pairs=True

    # ---------- 写出 Tc_est.json ----------
    Tc_path = outdir / "Tc_est.json"
    try:
        with Tc_path.open("w", encoding="utf-8") as f:
            json.dump(Tc_est, f, indent=2, default=json_default, ensure_ascii=False)
        print(f"✅ Tc 估计与配对 crossing 信息已写入 {Tc_path}")
    except Exception as exc:
        print(f"❌ 写 Tc_est.json 失败: {exc}")
        return

    print("Done. See", outdir)


if __name__ == "__main__":
    main()

