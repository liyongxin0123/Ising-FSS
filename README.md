# Ising FSS: 二维 Ising 模型有限尺寸标度分析工具包

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Ising FSS** 是一个现代化的、高性能的二维 Ising 模型模拟与有限尺寸标度（Finite-Size Scaling, FSS）分析工具包。  
它面向统计物理、凝聚态物理以及机器学习方向的研究者与学生，提供从 **蒙特卡洛模拟 / 副本交换 (REMC)** 到 **临界指数提取 / 数据坍缩** 的完整工作流。

---

## ✨ 核心特性

### 🚀 高性能模拟引擎

- **多算法支持（2D Ising，偶数尺寸，周期边界条件）**
  - Metropolis（棋盘格分解单自旋翻转）
  - Wolff 单簇算法
  - Swendsen–Wang 多簇算法  
  > 簇算法仅在零外场 `h = 0` 时可用；非零外场由 Metropolis 负责。

- **CPU / GPU 双后端**
  - **CPU**：使用 Numba/JIT 加速的核心更新算子，配合显式种子管理，结果完全可复现
  - **GPU**：基于 CuPy 的大规模并行实现  
    - 一次可更新数百至上千个副本  
    - 使用 checkerboard 分解避免写冲突  
    - 所有自旋构型与能量驻留在 GPU 显存中，只在必要时回传

- **副本交换蒙特卡洛（REMC / 平行回火）**
  - Slot-bound RNG：**温度槽** 与随机数流一一绑定，副本交换时只交换构型，不交换 RNG 状态
  - 显式 `replica_seeds`：用户完全控制随机性（checkpoint 恢复时严格校验）
  - CPU / GPU 版本实现语义对齐：  
    - 初始化随机流与运行期随机流解耦（`seed ^ const` 派生初始化 RNG）
    - 交换判据与能量定义在 CPU/GPU 间保持统一
  - 支持断点续传：
    - CPU：基于 HDF5 的 checkpoint（包括 lattice、能量、RNG 状态）
    - GPU：JSON + NPZ 双文件结构（元数据 + 物理态）

---

### 📊 有限尺寸标度分析

- **观测量基础管线**
  - 每一温度 `T` 对应一条时间序列：能量密度 `E`、磁化密度 `M`、`|M|`、`M²`、`M⁴`
  - `analyze()` 自动给出：
    - 比热：`C`
    - 磁化率：`χ`
    - Binder 累积量：`U`
    - 样本数：`n_samples`

- **有限尺寸标度（FSS）分析（analysis 子包）**
  - 支持以 `{L, T, observable}` 的结构组织数据，便于之后：
    - Binder 交叉点分析 (Tc 估计)
    - 临界指数拟合（例如 ν, γ/ν, β/ν）
    - 数据坍缩（Data Collapse）  
  - 时间序列统计工具：
    - 自相关时间估计（Sokal 窗口法）
    - Moving-block Bootstrap 误差估计
    - 阻塞分析（blocking）作为兜底方案

> FSS 的高层接口（如 `FSSAnalyzer`）推荐在 `analysis/` 或 `examples/` 中用脚本或 notebook 实现，
> 直接对 `remc_simulator` / `gpu_remc_simulator` 的 `analyze()` 输出进行二次处理。

---

### 🤖 深度学习集成（可选 / 拓展方向）

工具包的模拟结果（HDF5/NPZ 格式）适合直接作为深度学习数据集使用。推荐做法：

- 自行编写 / 使用示例中的 PyTorch `Dataset` / `DataLoader`
  - 惰性读取 HDF5/NPZ
  - 按需做 D4 群数据增强（旋转 + 翻转）
  - 按 `T` 或其他物理量打标签
- 支持 ±1 自旋到 [0,1] 或 [-1,1] 的可配置映射，以适配 VAE/CNN/Transformer 等模型

---

### 🔬 科学计算最佳实践

- **完全可复现**
  - 所有随机性通过显式 `replica_seeds` 控制
  - CPU/GPU 保持一致的 RNG 策略（Philox 优先，回退到 `default_rng`）
  - Checkpoint 恢复时会严格检查：
    - 系统尺寸 `L`
    - 外场 `h`
    - 温度列表 `temps`
    - 算法名称 `algorithm`
    - `replica_seeds`（不匹配时拒绝恢复）

- **稳定可靠的 I/O**
  - GPU 侧：HDF5 流式写入；NPZ + JSON 双文件（临时文件 + `os.replace`，避免中途崩溃产生半截文件）
  - CPU 侧：HDF5 流式写入 + provenance 记录组（`provenance`）


---

## 📦 安装

### 1. 基础依赖（仅 CPU）

```bash
pip install numpy scipy h5py numba pyyaml
pip install -e .  # 开发模式安装 ising-fss
```

### 2. GPU 加速（可选）

确保系统已正确安装 CUDA（或 ROCm 对应版本）。

```bash
# 根据 CUDA 版本选择
pip install cupy-cuda11x
# 或
pip install cupy-cuda12x
```

### 3. 深度学习扩展（可选）

```bash
pip install torch torchvision       # PyTorch
pip install matplotlib seaborn      # 基础可视化
# 如需要交互式图表：
# pip install plotly
```

---

## 🎯 应用示例

###  0. quick_start（remc: metropolis_sweep 算法示例，CPU）

```python
# examples/quick_start.py
"""
Quick start: 最简单的一步 REMC 示例

- 在 CPU 上用 HybridREMCSimulator 跑一个小系统 (L=16, R=8)
- 不依赖 Config 系统，直接用裸参数
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from ising_fss.simulation.remc_simulator import HybridREMCSimulator
from ising_fss.simulation.dispatcher import make_replica_seeds


def main():
    L = 16
    T_min, T_max = 1.0, 3.5
    num_replicas = 3

    # 生成确定性的副本种子
    replica_seeds = make_replica_seeds(master_seed=42, n_replicas=num_replicas)

    sim = HybridREMCSimulator(
        L=L,
        T_min=T_min,
        T_max=T_max,
        num_replicas=num_replicas,
        algorithm="metropolis_sweep",
        h=0.0,
        replica_seeds=replica_seeds,
    )

    sim.run(
        equilibration_steps=500,
        production_steps=2000,
        exchange_interval=5,
        thin=5,
        save_lattices=False,
        save_dir="runs/quick_start",
        worker_id="quick_start",
    )

    results = sim.analyze(verbose=False)

    # 只数一数有多少个温度条目（排除 swap / seeds 等全局项）
    T_keys = sorted(k for k in results.keys() if isinstance(k, str) and k.startswith("T_"))
    print(f"Got {len(T_keys)} temperature entries\n")

    # 打印每个温度点的主要观测量
    print("Per-temperature observables:")
    for k in T_keys:
        v = results[k]
        T = float(k.replace("T_", ""))
        C = v["C"]; C_err = v["C_err"]
        chi = v["chi"]; chi_err = v["chi_err"]
        U = v["U"]
        n = v["n_samples"]
        print(
            f"{k} (T={T:.6f}): "
            f"C = {C:.4f} ± {C_err:.4f}, "
            f"chi = {chi:.4f} ± {chi_err:.4f}, "
            f"U = {U:.4f}, "
            f"n_samples = {n}"
        )

    # 交换统计信息
    swap = results.get("swap", {})
    print("\nSwap statistics:")
    print(f"  total attempts = {swap.get('attempt', 0)}")
    print(f"  total accepts  = {swap.get('accept', 0)}")
    print(f"  overall rate   = {swap.get('rate', 0.0):.4f}")
    pair_rates = swap.get("pair_rates", [])
    temps = swap.get("temps", [])
    for i, r in enumerate(pair_rates):
        if i + 1 < len(temps):
            print(
                f"  pair {i}: T={temps[i]:.4f} <-> T={temps[i+1]:.4f}, "
                f"accept rate = {r:.4f}"
            )

    # 如果有 warning，也打印出来看看
    if "warnings" in results:
        print("\nWarnings:")
        for w in results["warnings"]:
            print("  -", w)


if __name__ == "__main__":
    main()

```
输出：
````
Got 3 temperature entries

Per-temperature observables:
T_1.000000 (T=1.000000): C = 4.8949 ± 0.3306, chi = 61.5623 ± 3.1173, U = 0.3386, n_samples = 400
T_1.870829 (T=1.870829): C = 0.8897 ± 0.0622, chi = 3.4607 ± 0.2231, U = 0.1404, n_samples = 400
T_3.500000 (T=3.500000): C = 0.3755 ± 0.0126, chi = 49.2940 ± 1.4780, U = 0.5535, n_samples = 400

Swap statistics:
  total attempts = 1000
  total accepts  = 985
  overall rate   = 0.9850
  pair 0: T=1.0000 <-> T=1.8708, accept rate = 0.9700
  pair 1: T=1.8708 <-> T=3.5000, accept rate = 1.0000
````


---

## 🎯 Ising 模拟参数配置方法

### 1.1 在脚本中直接构造 Config

```python

# examples/inline_config_quick_start.py
"""
在脚本中直接构造 Config，然后用其中的 simulation 配置跑一次 REMC。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from ising_fss.utils.config import SimulationConfig, DataConfig, Config, validate_config
from ising_fss.simulation.remc_simulator import HybridREMCSimulator
from ising_fss.simulation.dispatcher import make_replica_seeds


def main():
    # ---- 1. 构造 Config ----
    sim_cfg = SimulationConfig(
        L=32,
        T_min=2.0,
        T_max=2.6,
        num_replicas=12,
        h_field=0.0,
        algorithm="metropolis",  # → 'metropolis_sweep'
        boundary="pbc",
        backend="cpu",
        equilibration=2000,
        production=8000,
        exchange_interval=5,
        sampling_interval=5,
    )
    data_cfg = DataConfig(
        L=32,
        T_range=(2.0, 2.6),
        n_T=12,
        n_configs=2000,
        output_dir="data/config_inline_demo",
        export_pytorch=False,
    )
    cfg = Config(simulation=sim_cfg, data=data_cfg)

    has_warning, warning_list = validate_config(cfg)
    for w in warning_list:
        print("[config warning]", w)

    # ---- 2. 构造模拟器 ----
    s = cfg.simulation
    replica_seeds = make_replica_seeds(master_seed=1234, n_replicas=s.num_replicas)

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
        save_dir=str(Path(data_cfg.output_dir) / "raw"),
        worker_id="inline_cfg",
    )

    print("Done. Raw REMC data written under", data_cfg.output_dir)


if __name__ == "__main__":
    main()

```
输出：
````
Done. Raw REMC data written under data/config_inline_demo
````


### 1.2. CPU 从 YAML 配置文件加载 Config (REMC)

```python
# examples/run_from_yaml.py

import os
from ising_fss.utils.config import load_config, validate_config
from ising_fss.simulation.remc_simulator import HybridREMCSimulator
#  from ising_fss.simulation.gpu_remc_simulator import GPU_REMC_Simulator
from ising_fss.simulation.dispatcher import make_replica_seeds


def main():
    # 1. 读取 YAML 配置并做一致性检查
    cfg = load_config("configs/config_L64.yaml")

    ok, warnings = validate_config(cfg)
    if not ok:
        for w in warnings:
            print("[config warning]", w)

    sim_cfg = cfg.simulation
    data_cfg = cfg.data

    # 2. 根据 backend 选择 CPU / GPU 版本的 REMC 模拟器
    backend = sim_cfg.backend.lower()
    SimCls = GPU_REMC_Simulator if backend == "gpu" else HybridREMCSimulator

    # 3. 生成显式 replica_seeds（Hybrid/GPU 两个类都要求显式种子）
    master_seed = sim_cfg.seed or 0   # 如果 YAML 里没写 seed，就用 0 或你喜欢的数
    replica_seeds = make_replica_seeds(master_seed, sim_cfg.num_replicas)

    # 4. 构造模拟器实例
    sim = SimCls(
        L=sim_cfg.L,
        T_min=sim_cfg.T_min,
        T_max=sim_cfg.T_max,
        num_replicas=sim_cfg.num_replicas,
        algorithm=sim_cfg.algorithm,                  # 已在 SimulationConfig 里归一化
        spacing=getattr(sim_cfg, "temp_spacing", "geom"),
        h=sim_cfg.h_field,
        replica_seeds=replica_seeds,                  # ★ 关键：显式传入
    )

    # 5. 运行 REMC
    outdir = "runs/L64_from_yaml"
    os.makedirs(outdir, exist_ok=True)

    thin = getattr(data_cfg, "sampling_interval", 1)  # 采样间隔放在 DataConfig 里

    sim.run(
        equilibration_steps=sim_cfg.equilibration,
        production_steps=sim_cfg.production,
        exchange_interval=sim_cfg.exchange_interval,
        thin=thin,
        save_lattices=True,
        save_dir=outdir,
        worker_id=f"{backend}_yaml_demo",
    )

    stats = sim.analyze(verbose=True)
    print("平均交换率:", stats["swap"]["rate"])


if __name__ == "__main__":
    main()

```

````
# config_L64.yaml
# L = 64, 在临界区域附近用 Metropolis + GPU 做 REMC，
# 同时在更大区间 [1.6, 3.2] 上生成机器学习数据集。

simulation:
  # 晶格尺寸
  L: 64

  # 这里的 T_min / T_max 主要是“物理参考窗口”（临界附近），不会直接用来铺点；
  # 真实的数据网格由 data.T_range / data.n_T 决定。
  T_min: 2.20
  T_max: 2.35

  # REMC 的副本数（温度数）；在 run_data_from_config.py 的 REMC 模式下，
  # 会被 data.n_T 覆盖（以 data.T_range 上的温度网格为准）。
  num_replicas: 16

  # 外场（本示例只做零外场）
  h_field: 0.0

  # 算法名称；会在 SimulationConfig 中归一化为 'metropolis_sweep'
  algorithm: "metropolis"

  # 边界条件：周期性边界 (Periodic Boundary Conditions)
  boundary: "pbc"

  # 后端：'cpu' | 'gpu' | 'auto'
  # 这里使用 GPU，要求你已安装 CuPy 且 gpu_remc_simulator 可用。
  # backend: "gpu"
  backend: "cpu"

  # 每条链的热化步数 / 采样步数上限（主要用于 GUI 或其它脚本时做参考）
  equilibration: 10000
  production: 20000

  # REMC 交换间隔（步数）
  exchange_interval: 10

  # 随机种子
  seed: 2025


data:
  # 希望数据集中所有构型的 L；推荐与 simulation.L 保持一致
  L: 64

  # 数据集要覆盖的温度范围（全局）
  T_range: [1.6, 3.2]

  # 在 T_range 上取多少个温度点
  n_T: 40

  # 每个温度期望得到多少个样本（大致值）
  n_configs: 1000

  # 数据生产专用的热化步数（优先于 simulation.equilibration）
  equilibration: 8192

  # 采样间隔（thin 的值）：每隔多少步保存一个构型
  sampling_interval: 8

  # 是否在一个 REMC 模拟中跨整个 T_range
  #   true  → REMC 模式：一个模拟覆盖所有 temps（温度网格）
  #   false → 单温度模式：每个 T 独立跑一个 num_replicas=1 的 MC
  use_remc: true

  # 外场扫描范围：本示例不做 h 扫描，因此设为 null
  h_range: null

  # 输出目录（可以是相对路径，run_data_from_config.py 会用它创建 tmp/、merged/、pytorch/）
  output_dir: "data/ising_L64"

  # 是否导出为 PyTorch 训练集
  export_pytorch: true

  # 导出数据类型：'uint8' (紧凑，适合图像类网络) 或 'float32'
  export_dtype: "uint8"

  # 训练/验证划分比例
  train_split: 0.8

  # 是否对导出的数据做归一化（例如把自旋映射到 {0,1} 或 [0,1]）
  normalize: true

````

输出：

````
[remc.analyze] sweep_index=30000 swap_rate=0.4615 total_attempts=45000 total_accepts=20769
平均交换率: 0.46153333333333335
````


---

### 1.3 使用 Config.from_args() + 命令行 --preset / --set / ENV 来驱动 REMC。

```python
# examples/run_sim_with_from_args.py
"""
使用 Config.from_args() + 命令行 --preset / --set / ENV 来驱动 REMC。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from ising_fss.utils.config import from_args, validate_config
from ising_fss.simulation.remc_simulator import HybridREMCSimulator
from ising_fss.simulation.dispatcher import make_replica_seeds


def main():
    cfg = from_args()  # 会解析 --preset / --config / --set / ENV 等
    has_warning, warning_list = validate_config(cfg)
    for w in warning_list:
        print("[config warning]", w)


    s = cfg.simulation
    replica_seeds = make_replica_seeds(master_seed=s.seed or 0, n_replicas=s.num_replicas)

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
        save_dir=str(Path(cfg.data.output_dir) / "raw_from_args"),
        worker_id="from_args",
    )

    print("Run finished. Output dir:", cfg.data.output_dir)


if __name__ == "__main__":
    main()

```
调用：
```bash
python -m run_sim_with_from_args \
  --preset publication \
  --set simulation.L=64 \
  --set simulation.algorithm=metropolis \
  --set simulation.backend=cpu \
  --set data.output_dir="data/from_args_demo"
```

---

### 1.4 将模拟数据导出成 PyTorch 友好数据集

```python
# examples/generate_dl_data.py
"""
从 Config 出发，一键生成用于 DL 的 HDF5 + PyTorch 数据集。

- 第一步：根据 Config 跑模拟（如果需要的话）
- 第二步：调用 ml.generate_dl_data.generate_from_hdf5 做导出
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from ising_fss.utils.config import from_args, validate_config
from ising_fss.simulation.remc_simulator import HybridREMCSimulator
from ising_fss.simulation.dispatcher import make_replica_seeds
from examples.ml.generate_dl_data import generate_from_hdf5  # type: ignore


def main():
    #  调用方法 1.1 在脚本中直接构造 Config        
    #  sim_cfg = SimulationConfig(
    #      L=32,
    #      T_min=2.0,
    #      T_max=2.6,
    #      num_replicas=12,
    #      h_field=0.0,
    #      algorithm="metropolis",  # → 'metropolis_sweep'
    #      boundary="pbc",
    #      backend="cpu",
    #      equilibration=2000,
    #      production=8000,
    #      exchange_interval=5,
    #      sampling_interval=5,
    #  )
    #  data_cfg = DataConfig(
    #      L=32,
    #      T_range=(2.0, 2.6),
    #      n_T=12,
    #      n_configs=2000,
    #      output_dir="data/config_inline_demo",
    #      export_pytorch=False,
    #  )
    #  cfg = Config(simulation=sim_cfg, data=data_cfg)
    #
    #  has_warning, warning_list = validate_config(cfg)
    #  for w in warning_list:
    #      print("[config warning]", w)
#
    cfg = from_args()
    warnings = validate_config(cfg)
    for w in warnings:
        print("[config warning]", w)

    s = cfg.simulation
    d = cfg.data
    out_root = Path(d.output_dir)
    raw_dir = out_root / "raw"

    # 只做一个简单逻辑：如果 raw_dir 下没有任何 .h5，就跑一次 REMC
    if not any(raw_dir.glob("*.h5")):
        raw_dir.mkdir(parents=True, exist_ok=True)
        replica_seeds = make_replica_seeds(master_seed=s.seed or 0, n_replicas=s.num_replicas)
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

    # 调用 ML 端导出
    generate_from_hdf5(
        raw_dir=raw_dir,
        out_dir=out_root / "pytorch",
        normalize=True,
        dtype="uint8",
    )


if __name__ == "__main__":
    main()

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
    #  调用方法 1.1 在脚本中直接构造 Config        
    #  sim_cfg = SimulationConfig(
    #      L=32,
    #      T_min=2.0,
    #      T_max=2.6,
    #      num_replicas=12,
    #      h_field=0.0,
    #      algorithm="metropolis",  # → 'metropolis_sweep'
    #      boundary="pbc",
    #      backend="cpu",
    #      equilibration=2000,
    #      production=8000,
    #      exchange_interval=5,
    #      sampling_interval=5,
    #  )
    #  data_cfg = DataConfig(
    #      L=32,
    #      T_range=(2.0, 2.6),
    #      n_T=12,
    #      n_configs=2000,
    #      output_dir="data/config_inline_demo",
    #      export_pytorch=False,
    #  )
    #  cfg = Config(simulation=sim_cfg, data=data_cfg)
    #
    #  has_warning, warning_list = validate_config(cfg)
    #  for w in warning_list:
    #      print("[config warning]", w)
#


    # 1.2 从命令行 / YAML 读取 Config
    cfg = from_args()

    has_problem, warning_list = validate_config(cfg)
    for w in warning_list:
        print("[config warning]", w)

    s = cfg.simulation
    d = cfg.data

    out_root = Path(d.output_dir)
    raw_dir = out_root / "raw"
    pt_dir = out_root / "pytorch"

    #  如果 raw_dir 下没有 .h5，就跑一次 REMC
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

    #   从 HDF5 导出 PyTorch 数据
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



```

 1.2 从命令行 / YAML 读取 Config
```bash
python generate_dl_data.py --config configs/config_L64.yaml     
```

1.3 使用 Config.from_args() + 命令行 --preset / --set / ENV 来驱动 REMC。
```bash
python -m run_sim_with_from_args \
  --preset publication \
  --set simulation.L=64 \
  --set simulation.algorithm=metropolis \
  --set simulation.backend=cpu \
  --set data.output_dir="data/from_args_demo"
```

## 模拟方法选择

### 2.1  CPU 模式下的 REMC 模拟

```python
# examples/cpu_remc_basic.py
"""
CPU REMC 基本示例：HybridREMCSimulator + make_replica_seeds
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from ising_fss.simulation.remc_simulator import HybridREMCSimulator
from ising_fss.simulation.dispatcher import make_replica_seeds


def main():
    L = 16
    T_min, T_max = 2.0, 2.6
    num_replicas = 8

    replica_seeds = make_replica_seeds(master_seed=2024, n_replicas=num_replicas)
    sim = HybridREMCSimulator(
        L=L,
        T_min=T_min,
        T_max=T_max,
        num_replicas=num_replicas,
        algorithm="metropolis_sweep",
        h=0.0,
        replica_seeds=replica_seeds,
    )

    sim.run(
        equilibration_steps=1000,
        production_steps=5000,
        exchange_interval=5,
        thin=5,
        save_lattices=False,
        save_dir="runs/cpu_basic",
        worker_id="cpu_basic",
    )
    stats = sim.analyze(verbose=False)
    print("Finished CPU REMC. #temps =", len(stats))


if __name__ == "__main__":
    main()

```

输出：
````
Finished CPU REMC. #temps = 13
````

### 2.2 GPU 模式下的 REMC

```python
# examples/gpu_remc_basic.py
"""
GPU REMC 基本示例：GPU_REMC_Simulator + 交换率 / 耗时
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from ising_fss.simulation.dispatcher import make_replica_seeds, gpu_available

try:
    from ising_fss.simulation.gpu_remc_simulator import GPU_REMC_Simulator
except ImportError:
    GPU_REMC_Simulator = None  # type: ignore


def main():
    if not gpu_available() or GPU_REMC_Simulator is None:
        print("❌ GPU/CuPy 不可用，本示例无法运行。")
        return

    L = 64
    T_min, T_max = 2.0, 2.6
    num_replicas = 32

    replica_seeds = make_replica_seeds(master_seed=2025, n_replicas=num_replicas)
    sim = GPU_REMC_Simulator(
        L=L,
        T_min=T_min,
        T_max=T_max,
        num_replicas=num_replicas,
        algorithm="metropolis",
        h=0.0,
        replica_seeds=replica_seeds,
    )

    t0 = time.time()
    sim.run(
        equilibration_steps=2000,
        production_steps=10000,
        exchange_interval=5,
        thin=20,
        save_lattices=False,
        save_dir="runs/gpu_basic",
        worker_id="gpu_basic",
    )
    dt = time.time() - t0
    res = sim.analyze(verbose=False)
    swap = res.get("swap", {})
    print(f"Finished GPU REMC in {dt:.2f}s, swap rate ≈ {swap.get('rate', 'N/A')}")


if __name__ == "__main__":
    main()

```

### 2.3 parallel 并行模式下的 Ising 模拟

```python
# examples/parallel_across_L.py
"""
parallel.across_L：多 L 并行 + checkpoint 恢复示例
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from ising_fss.simulation.parallel import across_L


def main():
    L_list = [16, 32, 64]
    out_ckpt = Path("runs/parallel_ckpt")
    out_ckpt.mkdir(parents=True, exist_ok=True)

    results = across_L(
        L_list=L_list,
        T_min=2.0,
        T_max=2.6,
        num_replicas=32,
        equilibration=2000,
        production=5000,
        algorithm="wolff",
        exchange_interval=5,
        thin=5,
        n_processes_per_L=1,
        checkpoint_dir=str(out_ckpt),
        checkpoint_final=True,
        resume_if_exists=True,
    )

    print("\nSummary:")
    for L, res in results.items():
        if isinstance(res, dict) and "error" in res:
            print(f" L={L}: ERROR -> {res['error']}")
        else:
            swap = res.get("swap", {})
            print(f" L={L}: swap rate ≈ {swap.get('rate', 'N/A')}")


if __name__ == "__main__":
    main()

```

输出：
````
[worker pid=33062] Starting L=16  seed=None replica_seeds_provided=True  h=0.0  checkpoint=ON
[worker pid=33060] Starting L=32  seed=None replica_seeds_provided=True  h=0.0  checkpoint=ON
[worker pid=33061] Starting L=64  seed=None replica_seeds_provided=True  h=0.0  checkpoint=ON
[worker pid=33062] L=16 已保存 checkpoint -> remc_L16_T2.000000-2.600000_R32_h0.000000_wolff_geom.ckpt.json
[worker pid=33062] L=16 completed
[worker pid=33060] L=32 已保存 checkpoint -> remc_L32_T2.000000-2.600000_R32_h0.000000_wolff_geom.ckpt.json
[worker pid=33060] L=32 completed
[worker pid=33061] L=64 已保存 checkpoint -> remc_L64_T2.000000-2.600000_R32_h0.000000_wolff_geom.ckpt.json
[worker pid=33061] L=64 completed

Summary:
 L=16: swap rate ≈ 0.9556912442396314
 L=32: swap rate ≈ 0.9604377880184332
 L=64: swap rate ≈ 0.9708755760368664
````

### 2.4 batch 并行模式下的 REMC
```python
# examples/batch_worker_remc.py
"""
直接在 Python 脚本中调用 batch_runner.main(argv) 启动多 worker REMC。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from ising_fss.simulation import batch_runner


def main():
    outdir = Path("runs/batch_worker_demo")
    outdir.mkdir(parents=True, exist_ok=True)

    argv = [
        "--mode", "run_workers",
        "--outdir", str(outdir),
        "--nworkers", "2",
        "--L", "32",
        "--T", "2.269",
        "--equil", "2000",
        "--prod", "5000",
        "--exchange_interval", "5",
        "--thin", "10",
        "--replicas", "16",
        "--algo", "metropolis_sweep",
        "--spacing", "geom",
        "--h", "0.0",
        "--save_lattices",
    ]
    batch_runner.main(argv)
    print("Workers finished. You can now run merge via 05_batch_demo_cli.py or CLI.")


if __name__ == "__main__":
    main()

```
输出：
````
[worker 1] save_dir -> runs/batch_worker_demo/tmp/worker_pid33089_1cab7f10_w1
[worker 0] save_dir -> runs/batch_worker_demo/tmp/worker_pid33088_09d2d812_w0
[worker 1] sim.run completed -> runs/batch_worker_demo/tmp/worker_pid33089_1cab7f10_w1
[worker 0] sim.run completed -> runs/batch_worker_demo/tmp/worker_pid33088_09d2d812_w0
All worker processes finished. You can now run --mode merge to combine results.
Workers finished. You can now run merge via 05_batch_demo_cli.py or CLI.
````

### 2.4.1 batch 模式的另一种启动方式
```python
# examples/batch_demo_cli.py
"""
展示几条推荐的 batch_runner 命令行。

本文件不直接跑，只是给用户 copy 粘贴用。
"""

EXAMPLE_RUN = r"""
# 启动 4 个 worker，在 L=64、T=2.269 附近进行 REMC 采样
python -m ising_fss.simulation.batch_runner \
  --mode run_workers \
  --outdir data/ising_L64_batch \
  --nworkers 4 \
  --L 64 \
  --T 2.269 \
  --equil 5000 \
  --prod 20000 \
  --exchange_interval 10 \
  --thin 10 \
  --replicas 32 \
  --algo metropolis_sweep \
  --spacing geom \
  --h 0.0 \
  --save_lattices
"""

EXAMPLE_MERGE = r"""
# 在同一个 outdir 下进行合并
python -m ising_fss.simulation.batch_runner \
  --mode merge \
  --outdir data/ising_L64_batch
"""

if __name__ == "__main__":
    print("==== batch_runner run_workers 示例 ====")
    print(EXAMPLE_RUN)
    print("\n==== batch_runner merge 示例 ====")
    print(EXAMPLE_MERGE)

```

### 2.5 dispatcher 模式下的单 REMC

```python
# examples/dispatcher_single_replica.py
"""
dispatcher.apply_move: 单副本一步更新示例
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from ising_fss.simulation import dispatcher


def main():
    L = 16
    beta = 1.0 / 2.269
    spins = np.random.choice([-1, 1], size=(L, L)).astype(np.int8)

    new_spins, info = dispatcher.apply_move(
        spins,
        beta,
        replica_seed=123,
        algo="metropolis_sweep",
        backend="auto",
    )

    print("Single replica update done.")
    print("Accepted moves:", info.get("accepted", "N/A"))


if __name__ == "__main__":
    main()

```

输出：

````
Single replica update done.
Accepted moves: 149
````

### 2.6 dispatcher 模式下的多 REMC

```python
# examples/dispatcher_multi_replicas.py
"""
dispatcher.apply_move_batch: 多副本批量更新示例
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from ising_fss.simulation import dispatcher


def main():
    R, L = 8, 16
    betas = [1.0 / 2.269] * R
    spins_batch = np.random.choice([-1, 1], size=(R, L, L)).astype(np.int8)

    replica_seeds = dispatcher.make_replica_seeds(master_seed=999, n_replicas=R)

    new_batch, meta = dispatcher.apply_move_batch(
        spins_batch,
        betas,
        replica_seeds=replica_seeds,
        algo="metropolis_sweep",
        backend="cpu",
        n_sweeps=10,
    )

    print("Batch update done.")
    print("meta keys:", meta.keys())


if __name__ == "__main__":
    main()

```

输出：
````
Batch update done.
meta keys: dict_keys(['per_replica'])
````

---

## 分析

### 3.1 从 REMC 输出目录 / HDF5 / NPZ 加载数据，做 E/M 及 FSS 统计量的作图。
```python
# examples/load_and_analyze.py
"""
从 REMC 输出目录 / HDF5 / NPZ 加载数据，做 E/M 及 FSS 统计量的作图。

主要使用场景：
  python load_and_analyze.py /path/to/remc_output_dir

其中 remc_simulator / GPU_REMC_Simulator 的输出目录里包含：
  - 若干 HDF5 格式的格点文件：
        <worker_prefix>__latt_T_2.350000_h0.000000.h5
        <worker_prefix>__latt_T_2.400000_h0.000000.h5
        ...
  - 对应的元数据 JSON：
        <worker_prefix>__metadata.json

本脚本会：
  1. 扫描目录中所有 HDF5，按 worker_prefix 分组；
  2. 对每个 worker：
     - 从所有 HDF5 中提取 E/M 序列，计算 <E>(T)、<m>(T)、C(T)、χ(T)、U(T)，并作图；
     - 若存在 worker_prefix__metadata.json，且其中包含 thermo_stats/swap 信息，
       则再做一张带误差条的 C/χ/U 图、以及交换率统计图。
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
            },
            ...
        }
      - "swap_summary" 或 "swap": {
            "rate": float,
            "attempts": [...],
            "accepts": [...],
            "pair_rates": [...]   # 若存在
        }

    返回 dict 或 None:
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
# 对单个 worker 的 HDF5 + JSON 进行分析和作图
# ------------------------------------------------------------------
def plot_worker_from_hdf5_group(worker_prefix: str,
                                files_to_process: List[Path],
                                meta_path: Optional[Path] = None,
                                out_prefix: Optional[Path] = None):
    """
    对某个 worker 的全部 HDF5 文件（不同 T）进行分析：

      1. 从所有 HDF5 里读出 configs，计算：
         E_mean(T)、M_mean(T)、C(T)、chi(T)、U(T);
      2. 若传入 meta_path 且其中有 thermo_stats / swap，则再从 JSON 中提取：
         C/chi/U 的 Bootstrap 估计及误差，交换率等信息；
      3. 生成若干 png 图：
         - <out_prefix>_obs.png      : E/M/C/chi (raw, 无误差)
         - <out_prefix>_binder.png   : Binder U(T) (raw)
         - <out_prefix>_thermo_meta.png : C/chi/U (来自 JSON, 带误差条, 若有)
         - <out_prefix>_swap.png     : swap 统计 (若有)

    参数：
      worker_prefix : worker 名字（前缀）
      files_to_process : 该 worker 所有温度的 HDF5 文件列表
      meta_path : 对应的 JSON 元数据路径（可为 None）
      out_prefix : 输出 png 文件的前缀（无扩展名）。若 None，则使用
                   files_to_process[0].with_suffix("") 作为前缀。
    """
    files_to_process = sorted(files_to_process,
                              key=lambda fp: _parse_worker_T_h_from_name(fp.name)[1]
                              if _parse_worker_T_h_from_name(fp.name) is not None
                              else 0.0)
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
        _, L, _ = configs.shape

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

        fig3.suptitle(f"Thermo observables from metadata (worker={worker_prefix})", fontsize=12)
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
                # 若没有 pair_rates，但有 attempts/accepts，也可以计算一下
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
                # 取 pair 中心温度作为横坐标，若长度匹配的话
                temps_mid = (temps_m[:-1] + temps_m[1:]) / 2.0
                if temps_mid.size == pair_rates.size:
                    ax4.plot(temps_mid, pair_rates, "o-", ms=3, label="pair swap rate")
                    ax4.set_xlabel("mid T of pair")
                else:
                    ax4.plot(np.arange(pair_rates.size), pair_rates, "o-", ms=3, label="pair swap rate")
                    ax4.set_xlabel("pair index")
            else:
                ax4.set_xlabel("pair index")

            ax4.axhline(rate_global, color="red", ls="--",
                        label=f"global rate={rate_global:.3f}")
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
# 主入口：给一个 remc_simulator 输出目录，自动识别并作图
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
        print(f"[warning] 目录 {dir_path} 下没有匹配模式 'xxx__latt_T_..._h....h5' 的 HDF5 文件。")
        return

    # 对每个 worker 分别作图
    for worker, files in groups.items():
        meta_path = dir_path / f"{worker}__metadata.json"
        print(f"[dir] worker='{worker}' 发现 {len(files)} 个温度文件，"
              f"metadata={'存在' if meta_path.is_file() else '不存在'}")

        # 输出前缀：在目录下生成 <worker>_remc_summary_*.png
        # 也可以直接用第一个 HDF5 的名字做前缀
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
        # 目录模式：对目录下所有 worker 自动作图
        analyze_remc_output_dir(path)
        return

    # 单文件模式：保持原来的逻辑
    if path.suffix.lower() == ".npz":
        plot_from_tnn_npz(path, out_png=path.with_suffix(".png"))
    else:
        # 单个 HDF5：当成只有一个 worker 的目录来用
        parsed = _parse_worker_T_h_from_name(path.name)
        if parsed is None:
            raise RuntimeError(
                f"单文件模式下，HDF5 文件名需符合 'xxx__latt_T_..._h....h5'，当前为 {path.name}"
            )
        worker, T0, h0 = parsed
        meta_path = path.parent / f"{worker}__metadata.json"
        plot_worker_from_hdf5_group(
            worker_prefix=worker,
            files_to_process=[path],
            meta_path=meta_path if meta_path.is_file() else None,
            out_prefix=path.with_suffix(""),
        )


if __name__ == "__main__":
    main()

```
使用方法：
a.针对整个 remc 输出目录
````
python examples/load_and_analyze.py /path/to/remc_output_dir
````
输出每个 worker 生成一组：
````

worker__remc_summary_obs.png

worker__remc_summary_binder.png

worker__remc_summary_thermo_meta.png（若 JSON 里有 thermo_stats）

worker__remc_summary_swap.png（若 JSON 里有 swap/swap_summary）
````
b.如果你只想分析某一个 HDF5：
````
python examples/load_and_analyze.py /path/to/remc_output_dir/worker__latt_T_2.350000_h0.000000.h5
````

c.tnn_L.npz：（适应gpu 的lattice_saved = False 情况下的元数据输出与记录）
````
python examples/load_and_analyze.py /path/to/tnn_L64.npz
````

### 3.2 小型演示：REMC → FSSAnalyzer → Tc / 临界指数 / 数据塌缩
```python
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

```
输出：

````
(base)  🔥 $ python remc_fss_pipeline_demo0.py 
======================================================================
物理版示例：REMC → FSSAnalyzer → Tc / γ/ν / 数据塌缩
======================================================================
参数概览：L_list=[16, 32, 64], T∈[2.1,2.5], replicas=16, equil=20000, prod=80000, thin=20

=== 运行 REMC (物理版): L=16, T∈[2.1, 2.5], replicas=16, algo=metropolis_sweep ===
  收到温度点数量: 16

=== 运行 REMC (物理版): L=32, T∈[2.1, 2.5], replicas=16, algo=metropolis_sweep ===
  收到温度点数量: 16

=== 运行 REMC (物理版): L=64, T∈[2.1, 2.5], replicas=16, algo=metropolis_sweep ===
  收到温度点数量: 16

=== results 预览 ===
L= 16 | #T = 16
L= 32 | #T = 16
L= 64 | #T = 16

=== 构建 FSSAnalyzer (物理版) ===
[INFO] estimate_Tc('U') 结果:
  Tc: 2.2616322349964766
  var: 1.1264680241881608e-05
  std: 0.0033562896540497824
  weights:
    - 0.5015754252200023
    - 1.3973103117372652
    - 0.8769157952135199
  pairs:
    - (16, 32)
    - (16, 64)
    - (32, 64)
  crossings:
    - L1=16, L2=32, Tc=2.268248, slope_diff=0.502, bracket=(2.2675146771037182, 2.268297455968689), method=bisection
    - L1=16, L2=64, Tc=2.261284, slope_diff=1.397, bracket=(2.261252446183953, 2.2620352250489235), method=bisection
    - L1=32, L2=64, Tc=2.258403, slope_diff=0.877, bracket=(2.2581213307240704, 2.258904109589041), method=bisection
[INFO] 估计 Tc ≈ 2.261632 (理论值 Tc≈2.269185)
exponents (from chi): {'Tc_used': 2.2616322349964766, 'gamma_over_nu': 1.8324530469170346, 'nu': 1.0, 'intercept': -0.9024085248605331, 'sizes_used': [16, 32, 64]}
[INFO] 识别到 gamma_over_nu ≈ 1.8325
[INFO] 理论值 γ/ν ≈ 1.75; 当前拟合得到 γ/ν ≈ 1.8325

=== chi 数据塌缩 (物理版) ===
data_collapse keys: ['observable', 'Tc', 'nu', 'exponent_ratio', 'curves', 'score', 'success', 16, 32, 64]
collapse score ≈ 0.00018497
（score 越小通常代表塌缩质量越好，仅供相对比较）
````

### 3.3 为张量网络 (TNN) / TNR 生成多 L 的热力学统计数据 (NPZ)。
```python
# examples/analysis/tnn_data_generation.py
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

```
输出：

````
=============
2025-12-02 16:55:12 | INFO | 实验开始: tnn_gen
2025-12-02 16:55:12 | INFO | 时间: 2025-12-02T16:55:12.001576
2025-12-02 16:55:12 | INFO | ======================================================================
2025-12-02 16:55:12 | INFO | L=[16, 32, 64], T=[2.0,2.6], n_T=32, algo=wolff
[worker pid=41377] Starting L=32  seed=None replica_seeds_provided=True  h=0.0  checkpoint=ON
[worker pid=41376] Starting L=16  seed=None replica_seeds_provided=True  h=0.0  checkpoint=ON
[worker pid=41375] Starting L=64  seed=None replica_seeds_provided=True  h=0.0  checkpoint=ON
[worker pid=41376] L=16 已保存 checkpoint -> remc_L16_T2.000000-2.600000_R32_h0.000000_wolff_geom.ckpt.json
[worker pid=41376] L=16 completed
[worker pid=41377] L=32 已保存 checkpoint -> remc_L32_T2.000000-2.600000_R32_h0.000000_wolff_geom.ckpt.json
[worker pid=41377] L=32 completed
[worker pid=41375] L=64 已保存 checkpoint -> remc_L64_T2.000000-2.600000_R32_h0.000000_wolff_geom.ckpt.json
[worker pid=41375] L=64 completed
2025-12-02 16:57:02 | INFO | 模拟完成，用时 110.1s
✓ 导出 L=16: data_tnn/npz/tnn_L16.npz
✓ 导出 L=32: data_tnn/npz/tnn_L32.npz
✓ 导出 L=64: data_tnn/npz/tnn_L64.npz
📊 概览图已保存: data_tnn/overview.png
````
### 3.4 检查单个 tnn_L*.npz 文件的内容，并画出简单曲线。
```python
# examples/analysis/check_tnn_file.py
"""
检查单个 tnn_L*.npz 文件的内容，并画出简单曲线。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("npz", help="tnn_L*.npz file")
    args = parser.parse_args()

    path = Path(args.npz)
    data = np.load(path)
    print("keys:", list(data.keys()))
    print("L =", data["L"])
    print("temperatures shape:", data["temperatures"].shape)

    T = data["temperatures"]
    E, M = data["E"], data["M"]

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(T, E, "o-")
    plt.xlabel("T")
    plt.ylabel("E")
    plt.subplot(1, 2, 2)
    plt.plot(T, M, "o-")
    plt.xlabel("T")
    plt.ylabel("M")
    plt.tight_layout()
    out_png = path.with_suffix(".check.png")
    plt.savefig(out_png, dpi=200)
    print("Saved preview to", out_png)


if __name__ == "__main__":
    main()

```
用法：
````
python check_tnn_file.py /Users//Python/ising-fss/examples/data_tnn/npz/tnn_L16.npz
````

输出：
````
keys: ['temperatures', 'L', 'E', 'M', 'C', 'chi', 'U', 'E_err', 'M_err', 'C_err', 'chi_err']
L = 16
temperatures shape: (32,)
Saved preview to /Users//Python/ising-fss/examples/data_tnn/npz/tnn_L16.check.png
````

### 3.5 .h5 to train_data
```python
# examples/ml/generate_dl_data.py
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

```
用法：
````

````

### 3.6 cpu_remc_fss_pipeline

```python
# examples/cpu_remc_fss_pipeline.py
"""
基于 CPU / HybridREMCSimulator 的 REMC → FSS 管线脚本（支持多进程并行不同 L）。

功能：
- 行为尽量模仿 gpu_large_scale_fss.py：
  * 支持多次运行同一个 outdir，自动在 raw_results.json 里“追加样本”；
  * 每次 run 之后都用 FSSAnalyzer 做一次 Tc / γ/ν / 数据塌缩分析；
  * 把 Binder U 的 crossing 信息写入 Tc_est.json。
- 区别：
  * 这里用的是 HybridREMCSimulator（CPU / 混合实现），而不是 GPU 版模拟器；
  * 支持通过 --nworkers 并行跑多个 L（每个 L 一个 worker 进程）。
"""

from __future__ import annotations

import sys
import json
import math
from pathlib import Path
from typing import Dict, Any

from multiprocessing import Pool

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
          这样与 GPU/CPU 版 analyze() 的逻辑保持一致的量纲；
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
                "E_samples": e_all,  # 保留为 float64 数组
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
                merged[key] = old_L[key]
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


# ---------- 给 multiprocessing.Pool 用的封装 ----------

def _run_one_L_wrapper(args_tuple):
    """
    给 multiprocessing.Pool 用的简单封装：
        输入: (L, outdir_str, args)
        输出: (L, res_new)
    """
    L, outdir_str, args = args_tuple
    outdir = Path(outdir_str)
    res_new = run_one_L(L, outdir, args)
    return L, res_new


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

    # 并行 worker 数：用于并行跑不同的 L
    parser.add_argument("--nworkers", type=int, default=1,
                        help="并行 worker 数量，用于并行跑不同的 L（默认 1，串行）。")

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
        f"replicas={args.num_replicas}, equil={args.equil_steps}, prod={args.prod_steps}, thin={args.thin}, "
        f"nworkers={args.nworkers}"
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

    L_list = list(args.L_list)
    tasks = [(int(L), str(outdir), args) for L in L_list]

    if args.nworkers is None or args.nworkers <= 1 or len(L_list) == 1:
        # 串行模式（和以前行为完全一致）
        for L in L_list:
            print(f"\n=== REMC for L={L} ===")
            res_new = run_one_L(int(L), outdir, args)

            L_key = str(L)
            if L_key in prev_all_raw:
                print(f"[L={L}] 🔄 与 raw_results.json 中旧样本进行合并（追加模式）")
                merged = merge_analyze_for_one_L(prev_all_raw[L_key], res_new, int(L))
            else:
                merged = res_new

            results_all_raw[L_key] = merged
    else:
        # 并行模式：不同的 L 分配给不同 worker
        print(f"\n=== 并行模式：nworkers={args.nworkers}, L_list={L_list} ===")
        with Pool(processes=args.nworkers) as pool:
            for L, res_new in pool.imap_unordered(_run_one_L_wrapper, tasks):
                print(f"\n=== REMC for L={L} 完成（来自 worker） ===")

                L_key = str(L)
                if L_key in prev_all_raw:
                    print(f"[L={L}] 🔄 与 raw_results.json 中旧样本进行合并（追加模式）")
                    merged = merge_analyze_for_one_L(prev_all_raw[L_key], res_new, int(L))
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

```

用法：
````
1）串行跑（行为和之前一样）：

```bash
python examples/pipelines/cpu_remc_large_scale_fss.py \
    --L_list 16 32 64 \
    --T_min 2.1 --T_max 2.5 \
    --num_replicas 16 \
    --equil_steps 20000 \
    --prod_steps 100000 \
    --thin 200 \
    --exchange_interval 5 \
    --outdir runs/cpu_fss_thin200
```

2）并行跑多个 L（比如 3 个 worker，同步跑 64/96/128）：

```bash
python examples/pipelines/cpu_remc_large_scale_fss.py \
    --L_list 64 96 128 \
    --T_min 2.1 --T_max 2.5 \
    --num_replicas 16 \
    --equil_steps 20000 \
    --prod_steps 600000 \
    --thin 200 \
    --exchange_interval 5 \
    --nworkers 3 \
    --outdir runs/cpu_L64_96_128_thin200_parallel
```

不加 `--auto_thin` 时，即为“固定 thin”模式；
如果后面你想试自适应 thin，只需在命令里加上 `--auto_thin` 即可。

````



### 4.1 从模拟结果到 FSS 分析的典型工作流

1. 选定若干系统尺寸：`L = 8, 16, 32, 64, ...`
2. 对每个 `L`：

   * 在临界点附近的一段温度区间 `[T_min, T_max]` 上运行 REMC
   * 将 `analyze()` 的输出（每个温度一个 dict 含 C / χ / U / n_samples）保存为 JSON / NPZ
3. 在单独的分析脚本中：

   * 读取所有 L 的结果，整理为结构化数据：

     ```python
     data[L][T]["C"], data[L][T]["chi"], data[L][T]["U"]
     ```
   * 实现 Binder 交叉点搜索 / 临界指数拟合 / 数据坍缩等

示意代码：

```python
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
    parser.add_argument("--L_list", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--T_min", type=float, default=2.0)
    parser.add_argument("--T_max", type=float, default=2.6)
    parser.add_argument("--replicas", type=int, default=16)
    parser.add_argument("--equil", type=int, default=5000)
    parser.add_argument("--prod", type=int, default=20000)
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
        exchange_interval=5,
        thin=5,
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


```
输出：
````
[worker pid=42956] Starting L=16  seed=None replica_seeds_provided=True  h=0.0  checkpoint=ON
[worker pid=42957] Starting L=32  seed=None replica_seeds_provided=True  h=0.0  checkpoint=ON
[worker pid=42956] L=16 已保存 checkpoint -> remc_L16_T2.000000-2.600000_R16_h0.000000_metropolis_sweep_geom.ckpt.json
[worker pid=42957] L=32 已保存 checkpoint -> remc_L32_T2.000000-2.600000_R16_h0.000000_metropolis_sweep_geom.ckpt.json
[worker pid=42956] L=16 completed
[worker pid=42957] L=32 completed
[INFO] raw results saved to runs/publication_fss/raw_results.json
[INFO] FSS-friendly results saved to runs/publication_fss/fss_results.json
[INFO] Tc estimate saved to runs/publication_fss/Tc_est.json
[FSSAnalyzer] extract_critical_exponents: insufficient per-point stderr for weighted fit; using unweighted LS.
[INFO] critical exponents (chi) saved to runs/publication_fss/exponents_chi.json
Publication run finished. Results under runs/publication_fss
````

### 4.2 使用 GPU REMC 对大 L 系统做 FSS 的骨架示例。
```python
# examples/pipelines/gpu_large_scale_fss.py
"""
使用 GPU REMC 对大 L 系统做 FSS 的骨架示例（带 auto_thin 与 checkpoint）。

功能概览：
- 对多个 L 运行 GPU REMC 模拟；
- 每个 L 单独建目录（例如 runs/gpu_large_scale_fss/L64/）；
- 每个 L 完成后自动保存 checkpoint（JSON + NPZ）；
- 可通过 --resume 从现有 checkpoint 续跑；
- 支持在命令行打开 auto_thin（由 GPU_REMC_Simulator.run 实现）；
- 输出：
    - raw_results.json      : 每个 L 的原始 analyze() 结果
    - Tc_est.json           : FSSAnalyzer 的临界温度估计
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

# CuPy 是可选的：没有 GPU 也不至于 import 崩掉
try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None

# 把项目根目录和 src 加进 sys.path
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from ising_fss.simulation.dispatcher import make_replica_seeds, gpu_available
from ising_fss.simulation.gpu_remc_simulator import GPU_REMC_Simulator  # noqa: E402
from ising_fss.analysis.fss_analyzer import FSSAnalyzer


# ---------- JSON 序列化 helper ----------
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
    if cp is not None:
        try:
            import cupy as _cp  # type: ignore
            if isinstance(o, _cp.ndarray):  # type: ignore[attr-defined]
                return _cp.asnumpy(o).tolist()
        except Exception:
            pass

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


# ---------- 将 GPU 原始结果转为 FSSAnalyzer 需要的结构 ----------
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

    只保留 key 形如 "T_..." 且 value 为 dict 的条目。
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
        out[T] = val
    return out


# ---------- 单个 L 的模拟（支持 resume + checkpoint） ----------
def run_one_L(L: int, args, outdir: Path) -> Dict[str, Any]:
    """
    跑单个 L 的 GPU REMC，返回 GPU 模拟器的原始 analyze() 结果：
        {
          "T_2.100000": {...},
          "T_2.225664": {...},
          "swap": {...},
          "field": 0.0,
          ...
        }

    - 每个 L 独立子目录： outdir / f"L{L}"
    - checkpoint 文件：   outdir / f"L{L}/gpu_L{L}_ckpt.json"
    """
    L_dir = outdir / f"L{L}"
    L_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = L_dir / f"gpu_L{L}_ckpt.json"

    # 温度范围和副本数从 args 中取，保持灵活
    T_min = float(args.T_min)
    T_max = float(args.T_max)
    num_replicas = int(args.num_replicas)

    # 显式 replica_seeds，确保可复现
    replica_seeds = make_replica_seeds(
        master_seed=L * 10,
        n_replicas=num_replicas,
    )

    # 构造 GPU 模拟器实例
    sim = GPU_REMC_Simulator(
        L=L,
        T_min=T_min,
        T_max=T_max,
        num_replicas=num_replicas,
        algorithm="metropolis",  # 内部会 normalize 成 metropolis_sweep
        h=0.0,
        replica_seeds=replica_seeds,
    )

    # ------- 是否从 checkpoint 续跑 -------
    equil_steps = int(args.equil_steps)
    prod_steps = int(args.prod_steps)

    if args.resume and ckpt_path.exists():
        print(f"[L={L}] 🔁 从 checkpoint 恢复：{ckpt_path}")
        try:
            info = sim.restore_from_checkpoint(str(ckpt_path))
            print(f"[L={L}] restore info: {info}")
        except Exception as exc:
            print(f"[L={L}] 恢复失败，将从头跑一遍：{exc}")
        else:
            # 续跑时通常不再做额外热化
            equil_steps = 0

    # ------- 正式运行 -------
    print(
        f"[L={L}] 运行参数: T∈[{T_min}, {T_max}], replicas={num_replicas}, "
        f"equil_steps={equil_steps}, prod_steps={prod_steps}, "
        f"exchange_interval={args.exchange_interval}, thin={args.thin}, "
        f"auto_thin={args.auto_thin}"
    )

    sim.run(
        equilibration_steps=equil_steps,
        production_steps=prod_steps,
        exchange_interval=int(args.exchange_interval),
        thin=int(args.thin),
        verbose=args.verbose,
        save_lattices=args.save_lattices,
        save_dir=str(L_dir),
        worker_id=f"gpu_L{L}",
        auto_thin=bool(args.auto_thin),
        thin_min=int(args.thin_min),
        thin_max=int(args.thin_max),
        tau_update_interval=args.tau_update_interval,
        tau_window=int(args.tau_window),
        unit_sanity_check=True,
    )

    # ------- 运行结束后立即写 checkpoint 方便续跑 -------
    try:
        sim.save_checkpoint(str(ckpt_path))
        print(f"[L={L}] ✅ checkpoint 已保存到 {ckpt_path}")
    except Exception as exc:
        print(f"[L={L}] ⚠️ 保存 checkpoint 失败：{exc}")

    # ------- 返回原始分析结果 -------
    res_raw = sim.analyze(verbose=False)
    return res_raw


# ---------- 主程序 ----------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="GPU REMC + FSS pipeline（支持 auto_thin 与 checkpoint）"
    )
    parser.add_argument("--L_list", type=int, nargs="+", default=[64, 96, 128],
                        help="系统尺寸列表，例如: --L_list 64 96 128")
    parser.add_argument("--outdir", default="runs/gpu_large_scale_fss",
                        help="输出目录（将自动创建子目录 L{L}）")

    # 物理/模拟参数
    parser.add_argument("--T_min", type=float, default=2.1,
                        help="温度下限")
    parser.add_argument("--T_max", type=float, default=2.5,
                        help="温度上限")
    parser.add_argument("--num_replicas", type=int, default=64,
                        help="副本数（温度槽数量）")

    parser.add_argument("--equil_steps", type=int, default=20000,
                        help="热化步数（sweeps）")
    parser.add_argument("--prod_steps", type=int, default=100000,
                        help="采样生产步数（sweeps）")
    parser.add_argument("--exchange_interval", type=int, default=10,
                        help="每多少 sweeps 做一次 replica 交换")
    parser.add_argument("--thin", type=int, default=50,
                        help="初始采样间隔 thin（auto_thin 关闭时固定使用）")

    # auto_thin 配置（由 GPU_REMC_Simulator.run 实现）
    parser.add_argument("--auto_thin", action="store_true",
                        help="开启 GPU 端自适应 thinning（默认关闭）")
    parser.add_argument("--thin_min", type=int, default=1,
                        help="auto_thin 时最小 thin")
    parser.add_argument("--thin_max", type=int, default=10000,
                        help="auto_thin 时最大 thin")
    parser.add_argument("--tau_update_interval", type=int, default=256,
                        help="auto_thin: 多久更新一次 τ_int (以 sweep 计)")
    parser.add_argument("--tau_window", type=int, default=2048,
                        help="auto_thin: 估计 τ_int 时使用的窗口长度")

    # 其它控制项
    parser.add_argument("--save_lattices", action="store_true",
                        help="是否把格点快照写入 HDF5（每个温度一个文件）")
    parser.add_argument("--resume", action="store_true",
                        help="若存在 checkpoint，则从 checkpoint 续跑（热化步数自动置 0）")
    parser.add_argument("--verbose", action="store_true",
                        help="打印一些进度信息")

    args = parser.parse_args()

    if not gpu_available():
        print("❌ GPU 不可用，本示例无法运行。")
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 保存“原始 GPU 输出”和“供 FSS 使用的整形结果”各一份
    results_all_raw: Dict[int, Dict[str, Any]] = {}
    results_all_fss: Dict[int, Dict[float, Dict[str, Any]]] = {}

    for L in args.L_list:
        print(f"\n=== 🚀 GPU REMC for L={L} ===")
        res_raw = run_one_L(L, args, outdir)
        results_all_raw[L] = res_raw
        results_all_fss[L] = to_fss_format(res_raw)

    # 注意：results_all_raw 里会包含 numpy/cupy 数组，必须用 json_default
    raw_path = outdir / "raw_results.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(results_all_raw, f, indent=2, default=json_default, ensure_ascii=False)
    print(f"\n[✓] 原始 GPU 结果已写入: {raw_path}")

    # 把“整形后”的 results_all_fss 喂给 FSSAnalyzer
    analyzer = FSSAnalyzer(results_all_fss)
    Tc_est = analyzer.estimate_Tc("U")

    tc_path = outdir / "Tc_est.json"
    with open(tc_path, "w", encoding="utf-8") as f:
        json.dump(Tc_est, f, indent=2, default=json_default, ensure_ascii=False)
    print(f"[✓] Tc 估计已写入: {tc_path}")

    print("\nDone. See", outdir)


if __name__ == "__main__":
    main()

```
输出：
````

````

### 4.3 多温度独立 Metropolis 采样（非 REMC），用于生成 ML 数据。
```python
# examples/pipelines/run_path_A_independent.py
"""
路径 A：多温度独立 Metropolis 采样（非 REMC），用于生成 ML 数据。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from ising_fss.core.algorithms import update_batch, spawn_replica_seeds  # 你已有的接口
from ising_fss.data.data_manager import save_ml_dataset  # 假设有类似函数


def simulate_independent(
    L: int,
    temps: List[float],
    n_configs_per_T: int,
    n_sweeps_per_sample: int,
    out_h5: Path,
):
    R = len(temps)
    spins_batch = np.random.choice([-1, 1], size=(R, L, L)).astype(np.int8)
    seeds = spawn_replica_seeds(master_seed=1234, n_replicas=R)

    records = []
    for i in range(n_configs_per_T):
        update_batch(
            spins_batch=spins_batch,
            beta=[1.0 / T for T in temps],
            replica_seeds=seeds,
            algo="metropolis_sweep",
            h=0.0,
            n_sweeps=n_sweeps_per_sample,
        )
        records.append(spins_batch.copy())

    configs = np.stack(records, axis=0)  # (n_configs, R, L, L)
    save_ml_dataset(configs=configs, temps=temps, out_path=str(out_h5))


def main():
    L = 32
    temps = np.linspace(1.6, 3.2, 40).tolist()
    simulate_independent(
        L=L,
        temps=temps,
        n_configs_per_T=1000,
        n_sweeps_per_sample=10,
        out_h5=Path("runs/pathA_independent_L32.h5"),
    )


if __name__ == "__main__":
    main()

```
使用方法：


```bash
# 最简单跑一遍（固定 thin=50，不开 auto_thin，带 checkpoint）
python gpu_large_scale_fss.py --L_list 64 96 128 --save_lattices

# 想要开启 auto_thin：
python gpu_large_scale_fss.py --L_list 64 96 128 --save_lattices --auto_thin --resume

# 模拟途中中断后想从 checkpoint 续跑（再加 50000 个 production sweeps）
python gpu_large_scale_fss.py --L_list 64 96 128 --resume --prod_steps 50000

python gpu_large_scale_fss.py \
    --L_list 64 96 128 \
    --auto_thin \
    --prod_steps 100000 \
    --save_lattices \
    --resume

python 42_gpu_large_scale_fss.py     --L_list 64 96 128     --auto_thin   --prod_steps 400000     --save_lattices     --resume

python gpu_large_scale_fss.py \
    --L_list 64 96 128 \
    --equil_steps 20000 \
    --prod_steps 100000 \
    --thin 50 \
    --auto_thin \
    --save_lattices \
    --resume

python gpu_large_scale_fss.py \
    --L_list 64 96 128 \
    --equil_steps 20000 \
    --prod_steps 100000 \
    --thin 50 \
    --auto_thin \
    --save_lattices
    --resume


python 42_gpu_large_scale_fss.py     --L_list 16     --num_replicas 16     --T_min 2.0 --T_max 3.0     --equil_steps 500     --prod_steps 1000     --thin 10     --exchange_interval 5     --outdir runs/test_mini     --verbose     --save_lattices --resume    


````


> 也可以基于这些数据实现更系统的 `FSSAnalyzer` 类，封装在 `ising_fss.analysis` 中。

---

## 🏗️ 项目结构

```
ising-fss/
├── src/ising_fss/
│   ├── core/                    # 核心算法（CPU/GPU）
│   │   ├── algorithms.py        # Metropolis/Wolff/SW（CPU + Numba JIT）
│   │   ├── gpu_algorithms.py    # GPU 加速（CuPy）
│   │   └── observables.py       # 物理量计算（能量、磁化等）
│   ├── simulation/              # 模拟器与调度
│   │   ├── remc_simulator.py    # CPU REMC 模拟器（Slot-bound RNG）
│   │   ├── gpu_remc_simulator.py# GPU REMC 模拟器
│   │   ├── dispatcher.py        # 后端统一调度（CPU/GPU/Auto）
│   │   ├── parallel.py          # 跨晶格尺寸并行任务（spawn-safe）
│   │   └── batch_runner.py      # 分布式生产任务启动器
│   ├── analysis/                # 统计分析与 FSS
│   │   ├── fss_analyzer.py      # FSS 主分析器（Tc/指数/坍缩）
│   │   ├── statistics.py        # 时间序列误差分析（τ_int/Bootstrap）
│   │   └── dl_tools.py          # PyTorch 数据工具
│   ├── data/                    # 数据管理
│   │   ├── data_manager.py      # 流式合并 + 原子化 I/O
│   │   └── config.py            # 配置管理（预设/验证/CLI）
│   └── utils/                   # 工具函数
│       ├── logger.py
│       └── config.py
├── tests/                       # 单元测试（pytest）
├── examples/                    # Jupyter 示例
└── docs/                        # Sphinx 文档
```


> 实际目录可能随开发演进略有调整，请以仓库当前结构为准。

---

## 🔬 核心模块简要说明

### `core.algorithms`

* 提供各种更新算法的统一接口：

  * `get_algorithm(name: str)` → 返回对应的更新函数
  * 更新函数签名约定：

    ```python
    def algo(lattice: np.ndarray, beta: float, rng: np.random.Generator, h: float):
        """
        返回:
            lattice_out: np.ndarray  # 更新后的格点，自旋 ∈ {-1, +1}
            meta: dict               # 额外信息（如簇大小, rng_consumed 等）
        """
    ```
* Metropolis：

  * 棋盘格分解（红黑更新），便于并行 / GPU 迁移
  * 每个 sweep 大约消耗 `L*L` 个 uniform RNG 调用
* Wolff / Swendsen–Wang：

  * 使用 Union–Find（DSU）管理簇
  * meta 字段中会记录 `cluster_size`、`num_clusters` 等信息（视实现而定）

### `core.gpu_algorithms`

* 依赖 CuPy，经优化适配 REMC 使用场景：

  * `metropolis_update_batch(spins, beta_list, ...)` 一次更新所有温度槽上的所有副本
  * 可选 `device_counters` / `replica_counters` 参数，用于记录 RNG 消耗
* 能量与磁化：

  * `device_energy(spins, h)`
  * （可选）`device_magnetization(spins)`

### `core.observables`

* 在 CPU 上计算单个或一批 lattice 的物理量：

  * `_observables_for_simulator(latt, h)` → `{"E", "M", "absM", "M2", "M4"}`
* REMC 中的 CPU / GPU 版本都使用 **统一的能量定义**：

  * 四邻居配对 + 1/2 因子（避免重复计数）
  * 加上外场项：`- h * Σ_i s_i`

### `simulation.remc_simulator.HybridREMCSimulator`

* 主要特性：

  * Slot-bound RNG：每个温度槽一个 `np.random.Generator`
  * 初始化用 `seed ^ 0xC2B2AE35` 的独立 RNG 生成初始构型，保证初始化与后续演化的随机流解耦
  * 支持 Metropolis / 簇算法（Wolff / SW）
  * 自适应 thin（可选）：根据在线估计的自相关时间自动调节采样间隔
* 重要方法：

  * `run(...)`：完成平衡 + 采样 + （可选）格点保存
  * `analyze(...)`：返回每个温度下的 `C`, `chi`, `U`, `n_samples` 等
  * `save_checkpoint(...)` / `restore_from_checkpoint(...)`：支持长时间运行的断点续算

### `simulation.gpu_remc_simulator.GPU_REMC_Simulator`

* 与 CPU 版保持语义一致的 GPU 版本：

  * 所有副本自旋构型驻留在 GPU 上
  * RE Metropolis 更新在 GPU 上向量化完成
  * 温度交换（swap）在 CPU 上完成，使用 host RNG 进行接受判据
* 输出接口与 CPU 版尽量统一：

  * `run(...)` / `analyze(...)`
  * 提供最终 lattice 列表 `final_lattices`（在 host 上）

### `analysis.statistics`

* 自相关时间与误差估计工具：

  * `estimate_block_len(series)`
  * `moving_block_bootstrap_error(series, func, ...)`
* REMC 的 `analyze()` 会调用此模块，对比热 / 磁化率给出 bootstrap 误差估计。


---

## 📚 更多示例

仓库的 `examples/` 目录建议包含


---

## 📖 引用

如果本工具包对您的研究或教学有帮助，请引用：

```bibtex
@software{ising_fss,
  title  = {Ising-FSS: A High-Performance Toolkit for Finite-Size Scaling Analysis},
  author = {Li},
  year   = {2025},
  url    = {https://github.com/liyongxin0123/Ising-FSS}
}
```

**相关物理文献：**

* L. Onsager, *Phys. Rev.* **65**, 117 (1944) – 2D Ising 模型精确解
* R. H. Swendsen, J.-S. Wang, *Phys. Rev. Lett.* **58**, 86 (1987) – Swendsen–Wang 簇算法
* U. Wolff, *Phys. Rev. Lett.* **62**, 361 (1989) – Wolff 单簇算法
* A. M. Ferrenberg, R. H. Swendsen, *Phys. Rev. Lett.* **61**, 2635 (1988) – 重加权与 FSS 分析

---

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源，欢迎在符合协议的条件下自由使用与修改。

---

## 🙏 致谢

* **NumPy / SciPy / Numba** 等科学计算生态
* **CuPy** 团队（提供易用的 GPU 数组计算接口）
* **h5py** 与 HDF5 生态（高性能数据存储）
* 所有在 Ising 模型与 FSS 理论方面做出贡献的研究者

---

## 📧 联系方式

* Issue: GitHub Issues（例如：`https://github.com/liyongxin0123/Ising-FSS/issues`）


---





