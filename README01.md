````markdown
# Ising FSS 参数速查表

> *本表主要覆盖我们当前实际会用到的参数，按“模拟 → 配置 → 并行 → FSS → DL 数据”顺序整理。*

---

## 1. CPU REMC 模拟器：`HybridREMCSimulator`

### 1.1 构造函数参数 `HybridREMCSimulator(...)`

```python
HybridREMCSimulator(
    L,
    T_min,
    T_max,
    num_replicas,
    algorithm="metropolis_sweep",
    spacing="geom",
    temperatures=None,
    h=0.0,
    replica_seeds=None,
    buffer_flush=64,
    record_swap_history=False,
    bitgen_steps_per_uniform=None,
)
````

| 参数名                        | 类型                  | 含义               | 常用/可选值                                                          | 备注                                            |
| -------------------------- | ------------------- | ---------------- | --------------------------------------------------------------- | --------------------------------------------- |
| `L`                        | int                 | 晶格线性尺寸，系统为 `L×L` | 8, 16, 32, 64, 96, 128…                                         | 越大越接近热极限，计算更慢                                 |
| `T_min`                    | float               | REMC 温度下界        | 如 2.0                                                           | 与临界区 2.269185 附近相关                            |
| `T_max`                    | float               | REMC 温度上界        | 如 2.5, 2.6                                                      | 通常覆盖 Tc 上下                                    |
| `num_replicas`             | int                 | 温度点/副本数          | 8, 12, 32, 48…                                                  | 越多温度越密，交换更平滑                                  |
| `temperatures`             | list[float] or None | **直接指定**温度网格     | 若不为 None：长度必须 = `num_replicas`                                  | 指定后会忽略 `T_min/T_max/spacing`                  |
| `spacing`                  | str                 | 自动生成温度的插点方式      | `"geom"`（默认），`"linear"`                                         | 只在 `temperatures is None` 时生效                 |
| `algorithm`                | str                 | 单副本更新算法          | `"metropolis"`、`"metropolis_sweep"`、`"wolff"`、`"swendsen_wang"` | 传 `"metropolis"` 会自动标准化为 `"metropolis_sweep"` |
| `h`                        | float               | 外场 (h)           | 常用 `0.0` 或小的非零值                                                 | `h ≠ 0` 时禁止簇算法（Wolff/SW）                      |
| `replica_seeds`            | Sequence[int]       | 每个温度槽的 RNG 种子    | 由 `make_replica_seeds(master_seed, n_replicas)` 生成              | **必须显式提供**                                    |
| `buffer_flush`             | int                 | 保存晶格到 HDF5 时缓冲大小 | 32, 64, 128…                                                    | 值越大 I/O 越少，但异常损失更多缓存                          |
| `record_swap_history`      | bool                | 是否记录每对温度的交换轨迹    | `False`（默认）或 `True`                                             | True 会多存一份 history，耗内存                        |
| `bitgen_steps_per_uniform` | int or None         | RNG 深度控制参数       | 一般留 `None`                                                      | 高级用，正常不用管                                     |

---

### 1.2 运行参数 `sim.run(...)`

```python
sim.run(
    equilibration_steps,
    production_steps,
    exchange_interval=1,
    thin=1,
    verbose=False,
    save_lattices=False,
    save_dir=None,
    worker_id=None,
    burn_in=None,
    auto_thin=True,
    thin_min=1,
    thin_max=10_000,
    tau_update_interval=None,
    tau_window=2048,
    unit_sanity_check=True,
)
```

#### 1.2.1 时间尺度相关

| 参数名                   | 含义                 | 单位                        | 示例                   | 备注                                |
| --------------------- | ------------------ | ------------------------- | -------------------- | --------------------------------- |
| `equilibration_steps` | 热化步骤数（不采样）         | sweep 数（每个 sweep 更新全晶格一次） | 2000, 5000, 20000…   | 越大越“充分”热化                         |
| `production_steps`    | 产线阶段 sweep 数（采样区间） | sweep                     | 8000, 20000, 100000… | 只在这段时间记录样本                        |
| `exchange_interval`   | 相邻温度之间尝试 swap 的间隔  | sweep                     | 1, 5, 10…            | 例：5 → 每 5 sweep 进行一次交换            |
| `thin`                | 采样间隔（采一次观测量/晶格）    | sweep                     | 1, 5, 10…            | 例：prod=8000, thin=5 → 1600 个样本/温度 |

> 产线上每个温度槽的样本数（理论上）≈ `production_steps / thin`。
> 实际存到 HDF5 的计数写在属性 `provenance.attrs["samples_written"]` 中。

#### 1.2.2 自适应采样相关

| 参数名                   | 含义                           | 常用值                 | 说明                     |
| --------------------- | ---------------------------- | ------------------- | ---------------------- |
| `auto_thin`           | 是否根据自相关时间自动调整采样间隔            | `True`（默认）或 `False` | `True` 时，thin 会在运行中被更新 |
| `thin_min`            | 自适应 thin 的下限                 | 1, 2…               | 例如 6τ_int → 但不会小于此值    |
| `thin_max`            | 自适应 thin 的上限                 | 1000, 10000…        | 防止 thin 被拉得太大          |
| `tau_update_interval` | 每隔多少个 production 步更新一次 τ_int | 默认 256              | 填 None 使用默认            |
| `tau_window`          | 用来估计 τ_int 的窗口长度             | 默认 2048             | 越大统计越稳                 |

#### 1.2.3 I/O & 诊断相关

| 参数名                 | 含义                 | 常用值                                  | 说明                         |   |       |   |                  |
| ------------------- | ------------------ | ------------------------------------ | -------------------------- | - | ----- | - | ---------------- |
| `verbose`           | 是否输出进度信息           | `False` 或 `True`                     | 大任务时建议 `True` 看进度          |   |       |   |                  |
| `save_lattices`     | 是否保存晶格配置到 HDF5     | `False` 或 `True`                     | 做 DL 数据/L 配置分析时需要 True     |   |       |   |                  |
| `save_dir`          | HDF5 输出目录          | 如 `"data/run_L32/raw"`               | `save_lattices=True` 时必须提供 |   |       |   |                  |
| `worker_id`         | 工作者标识，用在文件名中       | 如 `"inline_cfg"`, `"dl_from_config"` | 多任务/多进程时帮助区分文件             |   |       |   |                  |
| `burn_in`           | 热化步数记录到 provenance | 一般用 `equilibration_steps` 或 None     | 仅写元数据，不改变算法逻辑              |   |       |   |                  |
| `unit_sanity_check` | 单位/范围检查            | `True` 或 `False`                     | True 会检测 `                 | M | >1`或` | E | /N` 过大并写 warning |

---

## 2. 高层配置：`SimulationConfig` / `DataConfig` / `Config`

### 2.1 `SimulationConfig`

```python
SimulationConfig(
    L=32,
    T_min=2.0,
    T_max=2.6,
    num_replicas=12,
    h_field=0.0,
    algorithm="metropolis",
    boundary="pbc",
    backend="cpu",
    equilibration=2000,
    production=8000,
    exchange_interval=5,
    sampling_interval=5,
    seed=None,  # 可选
)
```

| 字段                  | 映射到 REMC 的参数                     | 常用/可选值                                                             | 说明                                                     |
| ------------------- | -------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------ |
| `L`                 | `L`                              | 8, 16, 32, 64…                                                     | 晶格尺寸                                                   |
| `T_min`             | `T_min`                          | 如 2.0                                                              |                                                        |
| `T_max`             | `T_max`                          | 如 2.6                                                              |                                                        |
| `num_replicas`      | `num_replicas`                   | 12, 32, 48…                                                        | 温度点数                                                   |
| `h_field`           | `h`                              | 0.0, 0.1…                                                          | 外场                                                     |
| `algorithm`         | `algorithm`                      | `"metropolis"`, `"metropolis_sweep"`, `"wolff"`, `"swendsen_wang"` | 会被标准化                                                  |
| `boundary`          | (目前只在配置层表示)                      | `"pbc"`                                                            | 周期边界；将来可以扩展                                            |
| `backend`           | 选择 CPU/GPU 实现                    | `"cpu"`, `"gpu"`, `"auto"`                                         | 由 dispatcher 决定用哪套算法                                   |
| `equilibration`     | `equilibration_steps`            | 2000, 5000…                                                        | 热化 sweep 数                                             |
| `production`        | `production_steps`               | 8000, 20000…                                                       | 产线 sweep 数                                             |
| `exchange_interval` | `exchange_interval`              | 1, 5, 10…                                                          | 交换间隔                                                   |
| `sampling_interval` | `thin`                           | 1, 5, 10…                                                          | 采样间隔                                                   |
| `seed`              | master seed 用于生成 `replica_seeds` | None 或 int                                                         | 通过 `make_replica_seeds(master_seed=seed or 0, ...)` 派生 |

---

### 2.2 `DataConfig`

```python
DataConfig(
    L=32,
    T_range=(2.0, 2.6),
    n_T=12,
    n_configs=2000,
    output_dir="data/config_inline_demo",
    export_pytorch=False,
)
```

| 字段               | 含义                           | 常用值                     | 说明                           |
| ---------------- | ---------------------------- | ----------------------- | ---------------------------- |
| `L`              | 数据目标晶格尺寸                     | 与 SimulationConfig.L 一致 | 用于自检                         |
| `T_range`        | (T_min, T_max)               | (2.0, 2.6) 等            | 目标温度区间                       |
| `n_T`            | 期望的温度点数量                     | 12, 32 等                | 用于检查/截取                      |
| `n_configs`      | 每个温度目标样本数                    | 1000, 2000…             | 导出时可以截取                      |
| `output_dir`     | 输出根目录                        | `"data/.../..."`        | raw HDF5 + pytorch 子目录都在这个下面 |
| `export_pytorch` | 是否在生成 HDF5 后同步导出 PyTorch 数据集 | `False` 或 `True`        | True 时调用 DL 导出工具             |

---

### 2.3 `Config` / `validate_config`

```python
cfg = Config(simulation=sim_cfg, data=data_cfg)
warnings = validate_config(cfg)
```

* `Config`：把 `simulation` + `data` 打包到一起。
* `validate_config(cfg)`：

  * 检查并给出 warning，例如：

    * `h_field != 0` 且 `algorithm` 是簇算法 → 警告/修正
    * `backend="gpu"` 但所选算法 GPU 不支持 → 回退到 cpu
    * L 是否满足某些算法/后端要求（比如 checkerboard 需要偶数 L）
  * 返回 `List[str]`，脚本里常见打印：

    * `[config warning] ...`

---

## 3. 跨尺寸并行：`across_L(...)`

用于 “publication 级” 生产，例如 `publication_run.py` 里：

```python
results = across_L(
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
```

| 参数名                 | 含义                          | 常用/可选值                            |
| ------------------- | --------------------------- | --------------------------------- |
| `L_list`            | 一次要跑的所有系统尺寸列表               | `[16, 32, 64, 96]` 等              |
| `T_min` / `T_max`   | REMC 温度区间                   | 与单 L 情况相同                         |
| `num_replicas`      | 每个 L 的温度点数                  | 32, 48…                           |
| `equilibration`     | 热化步数                        | 20000…                            |
| `production`        | 产线步数                        | 100000…                           |
| `algorithm`         | 更新算法                        | `"metropolis_sweep"`, `"wolff"` 等 |
| `exchange_interval` | 交换间隔（sweep）                 | 1, 5 等                            |
| `thin`              | 采样间隔（sweep）                 | 5, 10 等                           |
| `n_processes_per_L` | 每个 L 的并行进程数                 | 1, 2, 4…                          |
| `checkpoint_dir`    | checkpoint 保存目录             | `"runs/.../ckpt"`                 |
| `checkpoint_final`  | 是否在任务结束时强制保存一次最终 checkpoint | `True` 或 `False`                  |

返回值 `results` 的结构大致是：

```python
results[L][f"T_{T:.6f}"] = {
    "C": ...,
    "chi": ...,
    "U": ...,
    # 以及一些 metadata
}
```

---

## 4. FSS 分析：`FSSAnalyzer`

### 4.1 初始化

```python
analyzer = FSSAnalyzer(results, Tc_theory=None)
```

| 参数名         | 含义                       | 示例                                       |
| ----------- | ------------------------ | ---------------------------------------- |
| `results`   | 来自 REMC/`across_L` 的结果字典 | `Dict[int, Dict[float, Dict[str, Any]]]` |
| `Tc_theory` | 理论 Tc，做为参考或初始值           | 2.269185 等，可设为 None                      |

---

### 4.2 `estimate_Tc(...)`

```python
Tc_est = analyzer.estimate_Tc(
    observable="U",
    use_all_pairs=True,
    weight_by="slope",
)
```

| 参数名             | 含义                 | 可选值                                             |
| --------------- | ------------------ | ----------------------------------------------- |
| `observable`    | 用哪个观测量做 Binder 交叉  | 一般 `"U"`                                        |
| `use_all_pairs` | 是否使用所有 (L1,L2) 尺寸对 | `True` / `False`                                |
| `weight_by`     | 加权策略               | `"slope"`, `"var"`, `None`（依实现为准，一般用 `"slope"`） |

返回（典型）：

```python
{
  "Tc": float,
  "std": float,
  "pairs": [...],
  ...
}
```

---

### 4.3 `extract_critical_exponents(...)`

```python
exponents = analyzer.extract_critical_exponents(
    observable="chi",
    Tc_hint=Tc_val,   # 可选
    fit_nu=True,
)
```

| 参数名          | 含义           | 说明               |
| ------------ | ------------ | ---------------- |
| `observable` | 哪个观测量拟合临界指数  | 常用 `"chi"`、`"C"` |
| `Tc_hint`    | 提示/固定 Tc     | 不传可能使用内部估计或理论值   |
| `fit_nu`     | 是否同时拟合 (\nu) | `True` 或 `False` |

典型输出字段：

* `"Tc_used"`：用于拟合的 Tc
* `"gamma_over_nu"`：(\gamma/\nu)
* `"nu"`：(\nu)
* `"intercept"`：对数拟合截距
* `"sizes_used"`：参与拟合的 L 列表

---

### 4.4 数据塌缩：`data_collapse` / `optimize_collapse`

**单次评估：**

```python
collapse = analyzer.data_collapse(
    "chi",
    Tc=Tc_val,
    nu=1.0,
    exponent_ratio=gamma_over_nu,
)
```

| 参数名              | 含义         | 示例             |
| ---------------- | ---------- | -------------- |
| `observable`     | 坍缩的观测量     | `"chi"`        |
| `Tc`             | 使用的 Tc     | 2.269185 / 估计值 |
| `nu`             | 临界指数 (\nu) | 1.0 等          |
| `exponent_ratio` | 指数比，如 γ/ν  | 1.75 等         |

**网格搜索优化（如果实现了）：**

```python
collapse = analyzer.optimize_collapse(
    observable="chi",
    Tc_range=(2.2, 2.35, 31),
    nu_range=(0.8, 1.2, 21),
)
```

---

## 5. DL 数据脚本参数

### 5.1 简单预设脚本：`examples/generate_dl_data.py`

```bash
python examples/generate_dl_data.py \
  --mode quick \
  --out_dir data/pytorch_32 \
  --seed_base 1000 \
  --split_ratio 0.8 \
  --dtype float32 \
  --normalize
```

| 参数名             | 含义               | 可选值                                |                                      |
| --------------- | ---------------- | ---------------------------------- | ------------------------------------ |
| `--mode`        | 模式预设             | `"quick"`, `"standard"`, `"paper"` |                                      |
| `--out_dir`     | 输出目录             | 如 `data/pytorch_32`                |                                      |
| `--seed_base`   | 基础种子             | int                                | 构造 `replica_seeds = [seed_base + i]` |
| `--split_ratio` | train/val 划分比例   | 0.8 等                              |                                      |
| `--dtype`       | PyTorch 存储 dtype | `"float32"`, `"uint8"`             |                                      |
| `--normalize`   | 是否将 ±1 转为 [0,1]  | flag（给就 True，否则 False）             |                                      |

---

### 5.2 Config 驱动脚本：`examples/config/generate_dl_data.py`

```bash
python examples/config/generate_dl_data.py --config configs/config_L64.yaml
```

YAML 中主要字段就是前面的 `SimulationConfig` / `DataConfig`。脚本逻辑：

1. 用 `from_args()` 读入 `Config`（含 simulation+data）。
2. `validate_config(cfg)` 打印 warning。
3. 若 `raw_dir` 下没有 `.h5`，则根据 simulation 配置跑 REMC：

   * 用 `equilibration`/`production`/`exchange_interval`/`sampling_interval` 等。
4. 然后从 `raw_dir` 的 HDF5 生成 PyTorch 数据集到 `output_dir/pytorch/`。

---

> **建议用法：**
>
> * 做理论 / 方法测试：用 `demo_remc_fss_pipeline.py` + 小 L（8,16,32）+ 短 run（几百步）看 Tc/指数是否合理。
> * 做“物理版 demo”：用 `publication_run.py` / `across_L` + 大 L（32,64,96）+ 长 run，观察 Tc 和 γ/ν 接近理论值。
> * 做 DL 数据：用 `config/generate_dl_data.py` 或简单版 `generate_dl_data.py`，保证 HDF5 → PyTorch 一条龙贯通。

```
```

---
---

那我给你搭一个**真的能跑起来、又尽量“极简”的工作流**，专门为“插值论文里的 Ising 高维函数示例”服务。我们先只做**一个固定尺寸 L=32、一个临界附近温度 T≈Tc 的例子**，把 pipeline 全打通；以后想扩展到多温度 / FSS，再在这个基础上加就行。

---

## 总览：最小闭环的 5 步

1. **用 REMC 生成 Ising 自旋样本 ({\sigma})**（HDF5 存盘）
2. **从 HDF5 读出某个温度 T0 的样本，整理成训练集**
3. **用 TNN 拟合高维函数 (\Psi_\theta(\sigma) \approx \log P(\sigma;T_0) + \text{const})**
4. **用 TNN 做高维求和，算 (\langle H\rangle,\langle H^2\rangle,\langle M\rangle,\langle M^2\rangle)**
5. **和蒙特卡洛的真值（sim.analyze）对比，作为论文里的主实验图**

下面每一步都给一个“可以直接改的脚本骨架”。

---

## 第 1 步：用 REMC 生成样本（只为 TNN 提供数据）

**目标**：固定一个系统

* 维度：2D
* 线性尺寸：`L = 32`
* 外场：`h = 0`
* 温度：围绕 (T_c \simeq 2.269185)，例如 `T_min=2.0, T_max=2.6`，`num_replicas=12`（几何 spacing）。

我们只关心：**生成足够多的格点配置并写入 HDF5**。

```python
# scripts/gen_data_for_tnn.py
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
    L = 32
    h = 0.0
    T_min, T_max = 2.0, 2.6
    num_replicas = 12

    # 固定随机种子，方便复现
    replica_seeds = make_replica_seeds(master_seed=2025, n_replicas=num_replicas)

    sim = HybridREMCSimulator(
        L=L,
        T_min=T_min,
        T_max=T_max,
        num_replicas=num_replicas,
        algorithm="metropolis_sweep",  # 先用单自旋算法，最稳
        h=h,
        replica_seeds=replica_seeds,
    )

    sim.run(
        equilibration_steps=20_000,     # 先热化
        production_steps=100_000,       # 采样步
        exchange_interval=10,
        thin=20,                        # 每 20 步采一次
        auto_thin=False,                # 为了简单，先关闭自适应 thin
        save_lattices=True,             # ★ 关键：写 HDF5
        save_dir="runs/ising_L32_T_range",
        worker_id="L32_Tc_for_tnn",
    )

    # 顺手算一下传统物理量，留作对比用
    results = sim.analyze(verbose=True)
    print("Available T keys:", [k for k in results.keys() if k.startswith("T_")])

if __name__ == "__main__":
    main()
```

**大概会产生：**

* 目录：`runs/ising_L32_T_range/`

  * 若干 HDF5：`L32_Tc_for_tnn__latt_T_2.269185_h0.000000.h5` 等

    * dataset `"lattices"`：形状大致是 `(≈5000, 32, 32)`
    * group `"provenance"`：记录 `L`, `T`, `samples_written`, `thin` 等
  * `L32_Tc_for_tnn__metadata.json`：里面有每个温度的 `samples_per_temp`

**样本数估计：**

* `production_steps=100000`, `thin=20` → 每个温度大约 `100000/20 = 5000` 帧
* 对单温度的 TNN 来说，5000 是一个不错的起步规模。

---

## 第 2 步：从 HDF5 抽取某个温度 T0 的样本

我们选择**最接近临界温度**的那个 HDF5 文件，例如名字中 `T_2.269185`。

写一个小脚本 / 数据加载器：

```python
# scripts/load_ising_hdf5.py
import h5py
import numpy as np
from pathlib import Path

def load_lattices_for_T(h5_path: str, max_samples: int | None = None):
    with h5py.File(h5_path, "r") as f:
        latt = f["lattices"][...]   # shape (N, L, L), int8, values in {-1, +1}
    if max_samples is not None and latt.shape[0] > max_samples:
        idx = np.random.choice(latt.shape[0], size=max_samples, replace=False)
        latt = latt[idx]
    # 展平，并转为 float32
    N, L, _ = latt.shape
    x = latt.reshape(N, L * L).astype(np.float32)   # shape (N, 1024)
    return x

if __name__ == "__main__":
    base = Path("runs/ising_L32_T_range")
    # 你可以手动选文件，也可以写点逻辑按温度名匹配
    h5_file = next(base.glob("L32_Tc_for_tnn__latt_T_2.269185*_h0*.h5"))
    X = load_lattices_for_T(str(h5_file), max_samples=5000)
    print("Loaded samples:", X.shape)  # (N_samples, 1024)
```

接下来在 TNN 训练代码里，只要调用 `load_lattices_for_T(...)`，就能拿到一个 `N_samples × 1024` 的矩阵——这就是你的**高维自变量 (\sigma)**。

---

## 第 3 步：定义 TNN 的插值任务（高维 (\log P)）

在这个最小工作流里，我们先**只针对一个温度 (T_0)**：

* 真分布：
  [
  P(\sigma;T_0) = \frac{1}{Z(T_0)} e^{-\beta_0 H(\sigma)},\quad \beta_0 = 1/T_0
  ]
* TNN 模型：
  [
  P_\theta(\sigma) = \frac{1}{Z_\theta} \exp(\Psi_\theta(\sigma)),
  ]
  其中 (\Psi_\theta) 是你设计的 TNN 张量网络（可积结构）。

**训练目标：最大化似然（或最小化负对数似然）：**

[
\mathcal{L}(\theta) = -\frac{1}{K} \sum_{k=1}^K \log P_\theta(\sigma^{(k)})
= -\frac{1}{K} \sum_{k=1}^K \Psi_\theta(\sigma^{(k)}) + \log Z_\theta
]

* 关键点：

  * (\log Z_\theta = \log \sum_\sigma e^{\Psi_\theta(\sigma)})
  * 这一步就是用**你的 TNN 高维积分算法**来算；普通 FNN 做不到。

训练伪代码（不管你用 JAX / PyTorch / 自己的 C++）：

```python
# 伪代码，结构示意思想：
X = load_lattices_for_T(h5_file, max_samples=5000)  # (N, 1024)

tnn = TNNModel(L=32, rank=R, ...)   # 你的张量网络结构

for epoch in range(num_epochs):
    for batch in iterate_minibatches(X, batch_size):
        # 1. 计算 Psi_theta(σ) for 批量样本
        psi_batch = tnn.forward(batch)   # shape (batch_size,)

        # 2. 计算 log Z_theta，通过张量网络的精确求和
        logZ = tnn.log_partition_function()  # 标量

        # 3. 构造负 log-likelihood
        loss = -(psi_batch.mean() - logZ)

        # 4. 反向传播 + 优化
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**在插值论文里，这一步的“卖点”：**

> 你可以直接在高维空间 (\sigma \in {-1,+1}^{1024}) 上做最大似然训练，因为 TNN 允许你精确计算 (Z_\theta)。这属于“对高维 log 概率密度的插值”。

---

## 第 4 步：用 TNN 做高维求和，算热力学量

训练好之后，就进入“展示 TNN 优势”的环节：**用 TNN 代替真实 Boltzmann 分布做高维积分**。

### 4.1 用 TNN 分布定义期望

* 在 TNN 模型下，对任意函数 (F(\sigma))，
  期望为：
  [
  \langle F \rangle_\theta = \sum_{\sigma} F(\sigma), P_\theta(\sigma)
  ]
* 对我们来说，需要：

  * (H(\sigma))（Ising 能量）
  * (M(\sigma) = \sum_i \sigma_i)（总磁化）

因此：

[
\begin{aligned}
\langle H \rangle_\theta &= \sum_\sigma H(\sigma) P_\theta(\sigma), \
\langle H^2 \rangle_\theta &= \sum_\sigma H(\sigma)^2 P_\theta(\sigma), \
\langle M \rangle_\theta &= \sum_\sigma M(\sigma) P_\theta(\sigma), \
\langle M^2 \rangle_\theta &= \sum_\sigma M(\sigma)^2 P_\theta(\sigma).
\end{aligned}
]

这些都可以按你之前分析的方式展开成若干“乘积项”，用 TNN 的高维积分器算出来。

### 4.2 从期望到物理量

* 比热（定体）：
  [
  C_\theta = \frac{\beta_0^2}{N}\left(\langle H^2\rangle_\theta - \langle H\rangle_\theta^2\right)
  ]
* 磁化率：
  [
  \chi_\theta = \frac{\beta_0}{N}\left(\langle M^2\rangle_\theta - \langle M\rangle_\theta^2\right)
  ]

**在代码上，你只需要在 TNN 模块里提供几种“带权求和”接口**，然后一行公式转换成 (C_\theta, \chi_\theta)。

---

## 第 5 步：和 REMC 分析结果对比（作为论文图）

还记得第 1 步里调用的：

```python
results = sim.analyze(verbose=True)
```

它在温度 `T_2.269185` 那个条目里有：

* `results["T_2.269185"]["C"]`  → MC 比热
* `results["T_2.269185"]["chi"]` → MC 磁化率
* 还有 `U`（Binder 累积量）、误差条等

**对比方式（最小版）：**

* 选定 T0=2.269185、L=32，做一个表格或一个小图：

  * 行：`MC`, `TNN`
  * 列：`<E>/N`, `<M>/N`, `C`, `chi`
* 要求：TNN 预测值都落在 MC 误差条内，或者相对误差非常小。

稍微豪华一点，可以：

* 选 3 个温度：`T1 < Tc`, `T2 ≈ Tc`, `T3 > Tc`
* 对每个 T 单独训练一个 TNN，得到 `C_theta(Ti)`、`chi_theta(Ti)`，然后在图上画出：

  * MC 的点连成线
  * TNN 的点（带误差条）叠在同一张图

**这在论文里可以卖一个核心观点：**

> “我们提出的 TNN 插值框架，可以直接在高维自旋空间逼近 (\log P(\sigma;T))，并通过对 TNN 进行解析积分，精确恢复出 Ising 模型在临界附近的比热和磁化率等热力学导数。这表明我们的插值不仅在函数值层面准确，而且在高阶导数（物理量）层面同样可信。”

---

## 总结一下这个“最小工作流”

**一步话概括：**

1. **REMC 生成 HDF5 格点样本** → 高维自变量 (\sigma)
2. **从 HDF5 抽一个温度 T0 的样本** → 构建训练集
3. **用 TNN 拟合 (\log P(\sigma;T_0))**（通过最大似然 + 精确 (Z_\theta)）
4. **用 TNN 做高维积分** → (\langle H\rangle_\theta, \langle M\rangle_\theta, C_\theta, \chi_\theta)
5. **和 REMC 的 “真值” 对比** → 作为插值论文的核心示例图。

如果你愿意，下一步我可以帮你把**“TNN 侧”**也写成一个更具体的接口设计（比如 `TNNModel` 需要提供哪些函数：`forward`, `log_partition_function`, `expectation(F)` 等），这样你可以直接按这个接口实现 /改写你已有的 TNN 代码。


