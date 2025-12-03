这是一份基于现有代码库（特别是 `dl_tools.py`, `config_io.py` 和 `gpu_algorithms.py`）重新编写的**深度学习训练指南**。

它修正了之前文档中虚构的模块名，使用了真实的 API，并重点突出了你代码中**确定性数据增强**和**物理感知损失函数**等独特优势。

-----

# 📚 深度学习训练指南 (Deep Learning Guide)

这份指南将引导你完成从 **Ising 构型生成** 到 **PyTorch 模型训练** 的完整工作流。本框架生成的“黄金数据”非常适合用于训练 VAE（识别相变）、CNN（分类相态）或 GNN（图神经网络）。

-----

## 🌟 核心优势

1.  **生产级数据管道**：支持从 TB 级 HDF5 数据流式清洗并导出为 PyTorch 友好的压缩格式 (`uint8` + 归一化)。
2.  **确定性增强 (Deterministic Augmentation)**：数据增强（旋转/翻转）与样本索引绑定。这意味着**训练过程完全可复现**，不会因为随机增强导致 Loss 抖动。
3.  **物理感知 (Physics-Aware)**：内置结构因子 $S(k)$ 和能量密度计算工具，可轻松构建包含物理约束的 Loss 函数。

-----

## 🛠️ 步骤 1：数据生产 (ETL Pipeline)

深度学习需要海量数据。我们使用两步走策略：**生成原始数据** -\> **清洗导出**。

### 1.1 生成原始数据 (Raw Data)

使用 `batch_runner` 在服务器上生成大规模数据（推荐使用 GPU 模式以获得最大吞吐量）。

```bash
# 示例：生成 L=32 的构型，覆盖临界区
# 结果将保存在 ./data_factory/raw/merged/final_ml_data.h5
python -m ising_fss.simulation.batch_runner \
    --mode run_workers \
    --nworkers 4 \
    --L 32 \
    --T 2.269 \
    --algo metropolis_sweep \
    --save_lattices \
    --outdir ./data_factory/raw

# 别忘了合并数据！
python -m ising_fss.simulation.batch_runner \
    --mode merge \
    --outdir ./data_factory/raw
```

### 1.2 导出训练集 (Export for PyTorch)

原始 HDF5 可能非常巨大且包含冗余信息。我们使用 `export_for_pytorch` 将其清洗、归一化并压缩。

**脚本：`prepare_data.py`**

```python
from ising_fss.data.config_io import load_configs_hdf5, export_for_pytorch

# 1. 加载原始数据 (Lazy Load，不占内存)
raw_path = "./data_factory/raw/merged/final_ml_data.h5"
dataset = load_configs_hdf5(raw_path, load_configs=False)

# 2. 导出清洗后的数据
# - 自动划分 80% 训练 / 20% 验证
# - 压缩为 uint8 (节省4倍空间)
# - 归一化到 [0, 1] (适合 VAE/CNN 输入)
export_for_pytorch(
    dataset,
    save_dir="./data_factory/pytorch_L32",
    split_ratio=0.8,
    normalize=True,  # 归一化
    dtype='uint8',   # 极致压缩
    verbose=True
)
```

-----

## 🧬 步骤 2：构建数据加载器 (Data Loading)

我们提供了 `dl_tools` 模块，它能自动处理 `uint8` 到 `float32` 的反向映射，并应用确定性增强。

**训练脚本片段：**

```python
import torch
from ising_fss.analysis.dl_tools import create_dataloaders_from_path, AugmentConfig

# 配置数据增强 (D4群: 旋转90度 + 翻转)
# 注意：这是确定性的！同一个样本每次被读取时，增强变换是固定的。
aug_cfg = AugmentConfig(
    enable=True, 
    rot90=True, 
    hflip=True, 
    vflip=True
)

# 一键创建 DataLoader
loaders = create_dataloaders_from_path(
    "./data_factory/pytorch_L32",
    batch_size=128,
    val_split=0.1,   # 在导出的数据基础上再切分（可选）
    num_workers=4,   # 多进程加载
    augment=aug_cfg, # 注入增强策略
    pin_memory=True  # GPU 加速
)

train_loader = loaders['train']
val_loader = loaders['val']

# 测试读取
batch = next(iter(train_loader))
x = batch['config']  # Shape: [128, 1, 32, 32], dtype: float32, range: [0, 1]
T = batch['temperature'] # 对应的温度标签
```

-----

## 🧠 步骤 3：模型训练示例 (VAE)

这里展示一个无监督学习相变的经典例子：使用 **变分自编码器 (VAE)** 学习 Ising 模型的潜在序参量。

### 3.1 定义模型

```python
import torch.nn as nn
import torch.nn.functional as F

class IsingVAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        # Encoder
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), # 32->16
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 16->8
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(64*8*8, latent_dim)
        self.fc_logvar = nn.Linear(64*8*8, latent_dim)
        
        # Decoder
        self.dec_fc = nn.Linear(latent_dim, 64*8*8)
        self.dec_conv = nn.Sequential(
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() # 输出 [0, 1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.enc_conv(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.dec_conv(z)
        return recon, mu, logvar
```

### 3.2 训练循环 (Physics-Informed)

我们可以利用 `dl_tools` 提供的物理工具来监控训练。

```python
from ising_fss.analysis.dl_tools import structure_factor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IsingVAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# 开始训练
model.train()
for epoch in range(10):
    total_loss = 0
    for batch in train_loader:
        x = batch['config'].to(device)
        
        optimizer.zero_grad()
        recon_x, mu, logvar = model(x)
        loss = loss_function(recon_x, x, mu, logvar)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader.dataset):.4f}")

    # --- 物理验证 ---
    # 检查重构图像的结构因子是否保留了物理特征
    with torch.no_grad():
        # 计算原始与重构的结构因子 S(k)
        sk_orig = structure_factor(x)
        sk_recon = structure_factor(recon_x)
        # ... 这里可以计算两者的差异作为物理指标 ...
```

-----

## 📊 步骤 4：可视化潜在空间

训练完成后，我们可以提取潜在空间变量 $z$，并观察它们如何随温度 $T$ 分布。这通常能直观地展示出“有序相”和“无序相”的分离。

```python
from ising_fss.visualization.plots import plot_latent_space
import numpy as np

model.eval()
zs = []
temps = []

with torch.no_grad():
    for batch in val_loader:
        x = batch['config'].to(device)
        t = batch['temperature']
        
        # 编码得到 mu (均值) 作为潜在表示
        h = model.enc_conv(x)
        z = model.fc_mu(h)
        
        zs.append(z.cpu().numpy())
        temps.append(t.numpy())

zs = np.concatenate(zs, axis=0)
temps = np.concatenate(temps, axis=0)

# 使用内置绘图工具
plot_latent_space(
    latent_codes=zs, 
    labels=temps, 
    label_type='temperature',
    save_path="vae_latent_space.png"
)
```

-----

## 🔍 高级功能：评估与指标

`dl_tools` 还提供了一些现成的评估函数，用于监督学习任务。

  * **分类任务** (预测 $T < T_c$ 或 $T > T_c$):

    ```python
    from ising_fss.analysis.dl_tools import evaluate_classification
    # 假设你有一个分类模型 classifier
    metrics = evaluate_classification(classifier, val_loader, device=device)
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    ```

  * **回归任务** (预测具体温度 $T$):

    ```python
    from ising_fss.analysis.dl_tools import evaluate_regression
    # 假设你有一个回归模型 regressor
    metrics = evaluate_regression(regressor, val_loader, target_key='temperature', device=device)
    print(f"MSE: {metrics['mse']:.4f}")
    ```
    
    
    
    可以，这份文档的结构是好的，但大量名字已经和你们现在的代码不一致（比如 `ising_config_saver`、`gpu_config_generator`、各种 “artifact”），而且现在项目已经整理成 `ising_fss` 包 + `dispatcher` + `gpu_algorithms` 那一套了。下面我给一份**重新对齐你们当前代码结构**的版本，你可以直接当作 `docs/config_data_summary.md` 或类似文件使用。

我会尽量避免写死不存在的类名函数名，只依赖你现在目录里真实存在的模块名：

* `ising_fss.core.{algorithms,gpu_algorithms,observables}`
* `ising_fss.simulation.{remc_simulator,gpu_remc_simulator,batch_runner,dispatcher}`
* `ising_fss.data.{config_io,data_manager}`
* `ising_fss.analysis.{fss_analyzer,dl_tools}`
* `examples/` 下面的脚本

---

````markdown
# 🎉 伊辛构型数据生成与深度学习工作流说明（基于当前 ising-fss 实现）

本说明文档面向两类用途：

1. **统计物理/FSS 分析**：重现二维 Ising 模型的临界行为与临界指数；
2. **深度学习/生成模型**：系统化地生成构型数据集，用于 VAE/CNN/GNN 等模型训练。

本文档基于当前 `ising-fss` 工程结构重新整理，所有模块名均对齐当前源码：

```text
ising-fss/
├── src/ising_fss/
│   ├── core/
│   │   ├── algorithms.py
│   │   ├── gpu_algorithms.py
│   │   └── observables.py
│   ├── analysis/
│   │   ├── dl_tools.py
│   │   ├── fss_analyzer.py
│   │   └── statistics.py
│   ├── simulation/
│   │   ├── batch_runner.py
│   │   ├── dispatcher.py
│   │   ├── gpu_remc_simulator.py
│   │   ├── parallel.py
│   │   └── remc_simulator.py
│   ├── data/
│   │   ├── config_io.py
│   │   └── data_manager.py
│   ├── visualization/
│   │   ├── plots.py
│   │   └── styles.py
│   └── utils/
│       ├── logger.py
│       └── config.py
└── examples/
````

---

## 🧩 整体结构概览

从功能角度看，当前项目提供了一条从**蒙特卡洛模拟 → 构型数据保存 → 物理分析 → 深度学习训练**的完整路径：

```text
Monte Carlo 模拟 (CPU/GPU)
          ↓
   构型与统计量采样
          ↓
  data_manager / config_io
          ↓
  FSS 分析 & 临界指数 (analysis)
          ↓
  构型数据 → 深度学习 (dl_tools)
```

* `core.*`：真正做自旋翻转的底层算法（Metropolis / GPU 版本等），含基本观测量计算。
* `simulation.*`：高层模拟器（REMC、GPU REMC）、批量任务调度（batch_runner）、CPU/GPU 分发（dispatcher）。
* `data.*`：HDF5 / 目录结构管理，负责把构型与统计量写进文件并读回。
* `analysis.*`：FSS、Binder 累积量、自相关与误差估计等物理分析工具，以及面向深度学习的辅助函数。
* `visualization.*`：统一的绘图和样式封装。
* `utils.*`：日志、配置加载等基础设施。

---

## 🧱 核心组件简介

### 1. Monte Carlo 更新内核（CPU & GPU）

* **`ising_fss.core.algorithms`**

  * 实现 CPU 版本的单步更新接口（例如 Metropolis sweep、簇算法等）。
  * 仅依赖 NumPy，适合无 GPU 环境或小体系。
  * 由 `simulation.dispatcher` 通过统一接口调用。

* **`ising_fss.core.gpu_algorithms`**

  * 实现 GPU 端的 Metropolis 批量更新：

    * 向量化的 `(R, L, L)` 批大小；
    * Philox/Generator 严格种子管理；
    * Checkerboard / Full sweep，支持外场 `h`；
    * 设备端积累 `accepts/attempts/E_sum/M_sum/rng_consumed`。
  * 由 `dispatcher` 在选择 GPU backend 时统一调度。

> **注意**：GPU 端当前仅支持 `metropolis_sweep` 类型算法，不支持 Wolff / Swendsen-Wang。

---

### 2. 模拟器与任务调度

* **`ising_fss.simulation.remc_simulator`**

  * 实现 CPU 端的 Replica-Exchange Monte Carlo 模拟器：

    * 多副本多温度 (`β` 网格)；
    * 定期副本交换；
    * 记录能量、磁化等统计量；
    * 由内部调用 `core.algorithms` 完成单步更新。

* **`ising_fss.simulation.gpu_remc_simulator`**

  * 对应的 GPU 版本 REMC：

    * 使用 `core.gpu_algorithms.metropolis_update_batch`；
    * 和 CPU 端保持尽量一致的物理语义（β 网格、交换策略等）；
    * 适合大规模构型数据生产。

* **`ising_fss.simulation.dispatcher`**

  * 统一的调度入口：

    * `apply_move(...)`：单副本更新（自动或显式选择 CPU/GPU 后端）；
    * `apply_move_batch(...)`：批量 `(R, L, L)` 更新；
    * 负责：

      * 算法名规范化；
      * 簇算法与外场 `h` 的物理合法性检查；
      * 自动选择 backend（`'auto' | 'cpu' | 'gpu'`）；
      * 严格的随机种子管理（`replica_seed` / `replica_seeds`）；
      * 可选 provenance 输出。

* **`ising_fss.simulation.batch_runner`**

  * 更高一层的“跑批工具”，主要职责：

    * 批量构造模拟任务（多个 `L`、多个 `β` 网格、多次独立 runs）；
    * 调用 `remc_simulator` / `gpu_remc_simulator` 执行；
    * 输出统一格式的结果（便于 `analysis` 与 `data_manager` 处理）。

---

### 3. 数据管理与构型 I/O

* **`ising_fss.data.data_manager`**

  * 负责与模拟结果、构型数据的高层打交道：

    * 统一文件命名与路径管理；
    * 将中间结果写入 HDF5 / 目录树；
    * 保存 provenance（参数、随机种子派生信息等）。

* **`ising_fss.data.config_io`**

  * 针对**构型数据**的具体读写函数：

    * 将 `(T, h)` 网格上的自旋构型数组写入 HDF5；
    * 还原为 NumPy 数组并附带必要元信息（温度、外场、L、采样间隔等）。

> 你可以把 `data_manager` 看作“谁管理文件 & 元数据”，`config_io` 看作“具体怎么把 array 写进 HDF5 / 读出来”。

---

### 4. 物理分析与深度学习支持

* **`ising_fss.analysis.fss_analyzer`**

  * 面向 FSS 的临界分析：

    * Binder 累积量交叉；
    * 不同 L 的缩放分析；
    * 临界温度/临界指数估计。

* **`ising_fss.analysis.statistics`**

  * 各类统计量与误差估计：

    * 分块平均；
    * 自相关时间估计；
    * 误差条计算；
    * 用于 FSS 与构型数据质量评估。

* **`ising_fss.analysis.dl_tools`**

  * 面向深度学习的辅助层：

    * 构型数据的标准化/重排；
    * （可选）PyTorch Dataset/ DataLoader 的封装；
    * 与 HDF5/NumPy 数组的接口。

---

## 📊 构型数据生成：与旧文档的映射关系

旧文档中提到的：

* `ising_config_saver` / `ising_config_saver.IsingConfigGenerator`
* `gpu_config_generator.GPUIsingConfigGenerator`
* `load_configs_hdf5(...)` 等

在当前代码结构中的对应关系大致如下：

| 旧描述/旧名字                          | 当前推荐用法/位置                                                            |
| -------------------------------- | -------------------------------------------------------------------- |
| CPU 构型生成器 `ising_config_saver`   | 使用 `simulation.remc_simulator` + `data_manager`/`config_io`          |
| GPU 构型生成器 `gpu_config_generator` | 使用 `simulation.gpu_remc_simulator` + `dispatcher` + `gpu_algorithms` |
| `load_configs_hdf5(...)`         | 在 `ising_fss.data.config_io` 中的读取函数（命名可能略有差异）                        |

在 **设计思路** 上是相同的：

1. 使用 REMC + Metropolis 更新产生构型；
2. 在热化之后，每隔固定 sweeps 采样自旋场；
3. 通过 `data_manager`/`config_io` 成批写入 HDF5；
4. 后续用深度学习工具直接消费这些 HDF5 文件。

---

## 🔁 典型工作流示例

> 下面代码是**示意用法**，具体函数名/参数请以源码或 `examples/` 为准。

### 场景 1：纯物理研究（只关心 Tc 与临界指数）

```python
from ising_fss.simulation import batch_runner
from ising_fss.analysis.fss_analyzer import FSSAnalyzer

# 1. 跑一批不同 L 的 REMC 模拟
results = batch_runner.run_remc_batch(
    L_list=[8, 12, 16, 24],
    T_min=2.0,
    T_max=2.5,
    n_T=32,
    backend="gpu",      # 或 "cpu"
    algo="metropolis_sweep",
    # 其它参数如：n_sweeps, equilibration, sampling_interval 等
)

# 2. 做 FSS 分析
analyzer = FSSAnalyzer(results)
Tc_est = analyzer.estimate_Tc()
exponents = analyzer.fit_critical_exponents()

print("Estimated Tc:", Tc_est)
```

**特点**：

* 不保存每个构型，只保留统计量；
* 文件体积小，适合做精细参数扫描。

---

### 场景 2：专注深度学习（大规模构型生成）

```python
from ising_fss.simulation.gpu_remc_simulator import GPUReplicaExchangeSimulator
from ising_fss.data import data_manager, config_io

# 1. 配置 REMC 模拟器
sim = GPUReplicaExchangeSimulator(
    L=32,
    betas=...,   # 对应 n_T 与温度范围
    h_values=...,   # 外场值列表（若使用）
    # 其它模拟相关参数...
)

# 2. 运行模拟并在采样点提取构型
results = sim.run_and_collect(
    equilibration_sweeps=8192,
    sampling_interval=8,
    n_configs_per_point=1024,
)

# 3. 使用 data_manager / config_io 保存成 HDF5
h5_path = "ising_L32_configs.h5"
config_io.save_configs_hdf5(h5_path, results)
```

之后用于深度学习：

```python
from ising_fss.data import config_io
from ising_fss.analysis import dl_tools

dataset = config_io.load_configs_hdf5("ising_L32_configs.h5")

# 利用 dl_tools 构造 PyTorch Dataset（示意）
torch_dataset = dl_tools.make_torch_dataset_from_configs(
    configs=dataset["configs"],
    temperatures=dataset["temperatures"],
    fields=dataset["fields"],
    # 可选：是否打乱、标准化等
)
```

---

### 场景 3：物理分析 + 深度学习混合（推荐）

```python
from ising_fss.simulation import batch_runner
from ising_fss.data import config_io
from ising_fss.analysis.fss_analyzer import FSSAnalyzer
from ising_fss.analysis import dl_tools

# 1. 用 batch_runner 跑一批 GPU REMC，并保存构型
h5_path = "ising_full_grid_L32.h5"
batch_runner.generate_and_save_configs(
    L=32,
    T_min=2.0, T_max=2.5, n_T=65,
    h_min=-0.5, h_max=0.5, n_h=65,
    n_configs=1024,
    backend="gpu",
    out_h5=h5_path,
)

# 2. 用 config_io 读取数据，提取统计量做 FSS
dataset = config_io.load_configs_hdf5(h5_path)
results = dl_tools.compute_observables_from_configs(dataset)
analyzer = FSSAnalyzer(results)
Tc_est = analyzer.estimate_Tc()

# 3. 同时用同一份数据做深度学习训练
vae_dataset = dl_tools.make_torch_dataset_from_configs(
    configs=dataset["configs"],
    temperatures=dataset["temperatures"],
    fields=dataset["fields"],
)
# → 交给 PyTorch 训练 VAE/CNN
```

---

## 📁 HDF5 数据格式约定（建议）

具体字段名以 `config_io` 实现为准，典型的约定可以是：

* `configs`: `int8` 数组，形状类似 `(..., L, L)` 或 `(n_h, n_T, n_sample, L, L)`
* `temperatures`: `float64`，存储所有 T 网格；
* `fields`: `float64`，存储所有 h 网格；
* `L`: 晶格线性尺寸；
* `equilibration_sweeps`: 热化 sweeps 数；
* `sampling_interval`: 采样间隔；
* 其它：如 `rng_seed_info`, `backend`, `algo` 等元数据可以通过 `data_manager` 的 provenance 功能写入。

你可以在 `config_io.py` 中进一步标准化这些键名，并在文档中列出一张完整表格。

---

## ⚙️ 性能与精度建议

1. **CPU vs GPU 选择**

   * 小体系 / 测试 / 教学：CPU 即可；
   * 论文级数据（例如 L=32、65×65×1024 构型）：强烈推荐 GPU。

2. **RNG 与复现**

   * 通过 `dispatcher`/`SeedSequence` 统一管理 `master_seed` 和 `replica_seeds`；
   * GPU 侧使用 Philox/Generator 显式种子，`rng_consumed` 以 CuPy 数组记录。

3. **外场 h 与簇算法**

   * `dispatcher` 已有防护：在 `h ≠ 0` 时禁止 Wolff / Swendsen-Wang；
   * GPU 端当前不支持簇算法，算法名应设为 `metropolis_sweep`。

4. **构型采样策略**

   * 热化：例如 `equilibration ≈ O(10^4)` sweeps；
   * 采样间隔：根据自相关时间设置（`statistics` 模块可估计），常见比如每 8–16 sweeps 取一次；
   * 每个 `(T, h)` 至少 10^2–10^3 个独立构型用于训练/验证。

---

## 📚 对比旧版说明的关键差异

* **不再使用**：

  * `ising_config_saver` / `gpu_config_generator` 等独立命名；
  * 各种 “artifact ID” 名字（这些更像是之前某环境的内部标识）。

* **统一转向当前结构**：

  * 模拟 → `simulation.remc_simulator` / `gpu_remc_simulator`；
  * 更新 → `core.algorithms` / `core.gpu_algorithms` + `simulation.dispatcher`；
  * I/O → `data.data_manager` + `data.config_io`；
  * 深度学习接口 → `analysis.dl_tools`。

* **优势**：

  * 与现有源码完全对齐；
  * CPU/GPU、物理分析/深度学习共用一套 Monte Carlo 底座；
  * 维护成本更低，接口更清晰。

---

## 🎯 建议的文档放置方式

* 本文可以命名为 `docs/config_data_summary.md` 或 `docs/ising_config_workflow.md`；
* 在 `README.md` 中加一行：

  > 需要大规模构型数据和深度学习训练工作流，请参见 `docs/config_data_summary.md`。

这样主 README 保持简洁，而本文件作为**专门的“构型+DL”说明**。

---

**到这里，旧文档中的概念（CPU/GPU 构型生成器、DL 管道）已经全部映射到当前代码结构之中。**
如果你愿意，我可以在下一步帮你把 `config_io.py` 的实际函数名/字段名梳理成一张表，放到本文件的“数据格式”章节里。

```
::contentReference[oaicite:0]{index=0}
```

    
    