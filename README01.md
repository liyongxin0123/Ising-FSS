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
