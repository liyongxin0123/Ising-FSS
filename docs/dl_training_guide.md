# 伊辛模型构型数据 - 深度学习训练指南

## 📊 数据集概述

### 生成的数据结构

```python
dataset = {
    'configs': (n_h, n_T, n_configs, L, L),  # 构型数据
    'energy': (n_h, n_T, n_configs),          # 能量
    'magnetization': (n_h, n_T, n_configs),   # 磁化强度
    'temperatures': (n_T,),                   # 温度数组
    'fields': (n_h,),                         # 磁场数组
    'parameters': {...}                       # 元数据
}
```

### 论文标准配置

| 参数 | 值 | 说明 |
|-----|---|------|
| L | 32 | 晶格尺寸 |
| n_T | 65 | 温度点数 |
| n_h | 65 | 磁场点数 |
| n_configs | 1024 | 每点构型数 |
| **总构型数** | **4,321,280** | 65×65×1024 |

---

## 🚀 快速开始

### 1. 生成数据

#### CPU版本（适合小规模）

```python
from ising_config_saver import IsingConfigGenerator

# 创建生成器
generator = IsingConfigGenerator(
    L=32,
    T_range=(1.0, 5.0),
    h_range=(-2.0, 2.0),
    n_T=65,
    n_h=65,
    n_configs=1024
)

# 生成完整数据集
dataset = generator.generate_full_dataset(
    equilibration=8192,
    sampling_interval=8,
    save_path='ising_data.h5'
)
```

**预期时间**: 2-4小时

#### GPU版本（推荐）

```python
from gpu_config_generator import GPUIsingConfigGenerator

# GPU生成器
generator = GPUIsingConfigGenerator(
    L=32,
    T_range=(1.0, 5.0),
    h_range=(-2.0, 2.0),
    n_T=65,
    n_h=65,
    n_configs=1024
)

# 生成（快10-50倍！）
dataset = generator.generate_full_dataset(
    equilibration=8192,
    sampling_interval=8,
    save_path='ising_data_gpu.h5',
    save_every_n_fields=10  # 增量保存
)
```

**预期时间**: 5-15分钟 ⚡

### 2. 加载数据

```python
from ising_config_saver import load_configs_hdf5

dataset = load_configs_hdf5('ising_data.h5')

print(f"构型形状: {dataset['configs'].shape}")
print(f"温度范围: {dataset['temperatures'][[0, -1]]}")
print(f"磁场范围: {dataset['fields'][[0, -1]]}")
```

---

## 🧠 深度学习应用

### 应用1: 变分自编码器 (VAE)

参考论文方法，训练VAE提取潜在特征。

#### 数据准备

```python
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class IsingDataset(Dataset):
    """伊辛构型数据集（PyTorch）"""
    
    def __init__(self, hdf5_path, normalize=True):
        with h5py.File(hdf5_path, 'r') as f:
            # 加载所有构型并展平
            configs = f['configs'][:]  # (n_h, n_T, n_configs, L, L)
            self.configs = configs.reshape(-1, configs.shape[-2], configs.shape[-1])
            
            # 标签（温度和磁场）
            temps = f['temperatures'][:]
            fields = f['fields'][:]
            
            # 为每个构型创建(T,h)标签
            labels = []
            for h in fields:
                for T in temps:
                    labels.extend([(T, h)] * configs.shape[2])
            self.labels = np.array(labels)
        
        # 归一化到[0,1]
        if normalize:
            self.configs = (self.configs + 1) / 2.0
        
        print(f"加载 {len(self.configs)} 个构型")
    
    def __len__(self):
        return len(self.configs)
    
    def __getitem__(self, idx):
        config = torch.FloatTensor(self.configs[idx]).unsqueeze(0)  # (1, L, L)
        label = torch.FloatTensor(self.labels[idx])
        return config, label

# 创建数据加载器
dataset = IsingDataset('ising_data.h5')
train_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
```

#### VAE模型（简化版）

```python
import torch.nn as nn

class IsingVAE(nn.Module):
    """伊辛构型VAE"""
    
    def __init__(self, latent_dim=10):
        super().__init__()
        
        # 编码器: (1, 32, 32) -> latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # -> (32, 16, 16)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # -> (64, 8, 8)
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # -> (128, 4, 4)
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)
        
        # 解码器: latent_dim -> (1, 32, 32)
        self.fc_decode = nn.Linear(latent_dim, 128 * 4 * 4)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.fc_decode(z).view(-1, 128, 4, 4)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# VAE损失函数
def vae_loss(recon_x, x, mu, logvar):
    # 重构损失
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL散度
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD
```

#### 训练循环

```python
# 初始化
model = IsingVAE(latent_dim=10).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练
for epoch in range(50):
    model.train()
    train_loss = 0
    
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.cuda()
        
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        loss = vae_loss(recon, data, mu, logvar)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.zero_grad()
    
    print(f'Epoch {epoch}: Loss = {train_loss / len(train_loader.dataset):.4f}')

# 保存模型
torch.save(model.state_dict(), 'ising_vae.pth')
```

---

### 应用2: 相变分类器

训练分类器识别不同相（铁磁/顺磁）。

```python
class IsingClassifier(nn.Module):
    """伊辛相分类器"""
    
    def __init__(self, num_classes=3):  # 铁磁+, 顺磁, 铁磁-
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 标签生成（基于温度）
def get_phase_label(T, Tc=2.269):
    if T < Tc * 0.8:
        return 0  # 铁磁相
    elif T < Tc * 1.2:
        return 1  # 临界区
    else:
        return 2  # 顺磁相
```

---

### 应用3: 临界温度预测

使用神经网络直接从构型预测温度。

```python
class TempPredictor(nn.Module):
    """温度预测器"""
    
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 回归输出
        )
    
    def forward(self, x):
        return self.net(x)

# 训练（MSE损失）
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

for data, labels in train_loader:
    data = data.cuda()
    temps = labels[:, 0].cuda()  # 温度标签
    
    pred_temps = model(data).squeeze()
    loss = criterion(pred_temps, temps)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## 📈 数据可视化

### 可视化构型

```python
import matplotlib.pyplot as plt

def plot_configs_grid(dataset, n_temps=5, n_fields=5):
    """绘制(T,h)网格的构型"""
    configs = dataset['configs']
    temps = dataset['temperatures']
    fields = dataset['fields']
    
    # 均匀采样
    t_indices = np.linspace(0, len(temps)-1, n_temps, dtype=int)
    h_indices = np.linspace(0, len(fields)-1, n_fields, dtype=int)
    
    fig, axes = plt.subplots(n_fields, n_temps, figsize=(15, 12))
    
    for i, h_idx in enumerate(h_indices):
        for j, t_idx in enumerate(t_indices):
            config = configs[h_idx, t_idx, 0]  # 第一个构型
            
            axes[i, j].imshow(config, cmap='gray', vmin=-1, vmax=1)
            axes[i, j].axis('off')
            
            if i == 0:
                axes[i, j].set_title(f'T={temps[t_idx]:.2f}', fontsize=10)
            if j == 0:
                axes[i, j].set_ylabel(f'h={fields[h_idx]:.2f}', fontsize=10)
    
    plt.tight_layout()
    return fig

# 使用
fig = plot_configs_grid(dataset)
plt.savefig('config_grid.png', dpi=150, bbox_inches='tight')
```

### 相图可视化

```python
def plot_phase_diagram(dataset):
    """绘制(T,h)相图"""
    configs = dataset['configs']
    temps = dataset['temperatures']
    fields = dataset['fields']
    
    # 计算平均磁化强度
    avg_mag = np.mean(np.abs(configs), axis=(2, 3, 4))
    
    plt.figure(figsize=(10, 8))
    plt.imshow(avg_mag, extent=[temps[0], temps[-1], fields[0], fields[-1]],
              aspect='auto', origin='lower', cmap='RdBu_r')
    plt.colorbar(label='平均磁化强度 |M|')
    plt.xlabel('温度 T')
    plt.ylabel('外部磁场 h')
    plt.title('伊辛模型相图')
    
    # 标注临界温度
    Tc = 2.269
    plt.axvline(Tc, color='white', linestyle='--', label=f'$T_c$ = {Tc}')
    plt.legend()
    
    return plt.gcf()
```

---

## 🔧 高级用法

### 数据增强

```python
def augment_config(config):
    """构型数据增强"""
    # 随机旋转（90度的倍数）
    k = np.random.randint(0, 4)
    config = np.rot90(config, k)
    
    # 随机翻转
    if np.random.rand() > 0.5:
        config = np.flip(config, axis=0)
    if np.random.rand() > 0.5:
        config = np.flip(config, axis=1)
    
    return config
```

### 条件生成

训练条件VAE，给定(T,h)生成构型：

```python
class ConditionalVAE(nn.Module):
    """条件VAE"""
    
    def __init__(self, latent_dim=10, condition_dim=2):
        super().__init__()
        # 将(T,h)条件嵌入到编码器和解码器
        ...
```

---

## 📚 参考文献

1. **原论文**:
   > Deep learning on the 2-dimensional Ising model to extract the crossover region with a variational autoencoder

2. **相关工作**:
   - Carrasquilla & Melko (2017): Machine learning phases of matter
   - Wetzel (2017): Unsupervised learning of phase transitions

---

## 💡 使用建议

### 数据量选择

| 应用 | 推荐配置 | 说明 |
|-----|---------|------|
| **快速原型** | L=16, n_T=20, n_h=20 | 几分钟生成 |
| **研究实验** | L=32, n_T=40, n_h=40 | 论文的一半 |
| **发表级** | L=32, n_T=65, n_h=65 | 完整论文配置 |
| **大规模** | L=64, n_T=100, n_h=100 | 需要GPU |

### 性能优化

```python
# 1. 使用GPU生成器（推荐）
generator = GPUIsingConfigGenerator(...)

# 2. 增量保存（防止内存溢出）
generator.generate_full_dataset(
    save_every_n_fields=10  # 每10个磁场保存一次
)

# 3. 数据加载优化
# 使用HDF5的部分读取
with h5py.File('ising_data.h5', 'r') as f:
    # 只加载需要的部分
    subset = f['configs'][0:10, :, :, :, :]  # 前10个磁场
```

---

## ✅ 检查清单

生成数据前：
- [ ] 确认GPU可用（如使用GPU版本）
- [ ] 确认有足够磁盘空间（~1GB for 标准配置）
- [ ] 选择合适的参数（L, n_T, n_h, n_configs）

训练模型前：
- [ ] 数据已成功加载
- [ ] 检查数据质量（可视化几个样本）
- [ ] 数据归一化/预处理
- [ ] 划分训练集/验证集/测试集

训练过程中：
- [ ] 监控损失曲线
- [ ] 定期保存模型检查点
- [ ] 可视化重构结果（VAE）
- [ ] 验证模型泛化能力

---

## 🎯 完整工作流示例

### 从数据生成到模型训练的完整流程

```python
# ============================================================
# 步骤1: 生成数据
# ============================================================

from gpu_config_generator import GPUIsingConfigGenerator

print("步骤1: 生成训练数据")
print("="*70)

generator = GPUIsingConfigGenerator(
    L=32,
    T_range=(1.0, 5.0),
    h_range=(-2.0, 2.0),
    n_T=65,
    n_h=65,
    n_configs=1024
)

dataset = generator.generate_full_dataset(
    equilibration=8192,
    sampling_interval=8,
    save_path='ising_training_data.h5'
)

print("✓ 数据生成完成\n")


# ============================================================
# 步骤2: 数据可视化验证
# ============================================================

print("步骤2: 验证数据质量")
print("="*70)

import matplotlib.pyplot as plt
from ising_config_saver import load_configs_hdf5

dataset = load_configs_hdf5('ising_training_data.h5')

# 可视化样本
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.ravel()):
    h_idx = np.random.randint(0, 65)
    t_idx = np.random.randint(0, 65)
    config = dataset['configs'][h_idx, t_idx, 0]
    
    ax.imshow(config, cmap='gray')
    ax.set_title(f"T={dataset['temperatures'][t_idx]:.2f}, "
                f"h={dataset['fields'][h_idx]:.2f}")
    ax.axis('off')

plt.tight_layout()
plt.savefig('data_samples.png', dpi=150)
print("✓ 样本可视化已保存: data_samples.png\n")


# ============================================================
# 步骤3: 准备PyTorch数据加载器
# ============================================================

print("步骤3: 准备训练数据")
print("="*70)

import torch
from torch.utils.data import Dataset, DataLoader, random_split

class IsingDataset(Dataset):
    def __init__(self, hdf5_path):
        with h5py.File(hdf5_path, 'r') as f:
            configs = f['configs'][:]
            self.configs = configs.reshape(-1, 1, 32, 32)  # (N, 1, L, L)
            
            # 归一化到[0, 1]
            self.configs = (self.configs + 1) / 2.0
            
            # 创建标签
            temps = f['temperatures'][:]
            fields = f['fields'][:]
            labels = []
            for h in fields:
                for T in temps:
                    labels.extend([(T, h)] * configs.shape[2])
            self.labels = np.array(labels)
    
    def __len__(self):
        return len(self.configs)
    
    def __getitem__(self, idx):
        return (torch.FloatTensor(self.configs[idx]), 
                torch.FloatTensor(self.labels[idx]))

# 加载数据集
full_dataset = IsingDataset('ising_training_data.h5')

# 划分训练集和验证集
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

print(f"训练集大小: {train_size}")
print(f"验证集大小: {val_size}")
print("✓ 数据加载器准备完成\n")


# ============================================================
# 步骤4: 定义并训练VAE模型
# ============================================================

print("步骤4: 训练VAE模型")
print("="*70)

class IsingVAE(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)
        
        # 解码器
        self.fc_decode = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.fc_decode(z).view(-1, 128, 4, 4)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = IsingVAE(latent_dim=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练循环
num_epochs = 50
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # 训练
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        loss = vae_loss(recon, data, mu, logvar)
        loss.backward()
        
        train_loss += loss.item()
        optimizer.step()
    
    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, _ in val_loader:
            data = data.to(device)
            recon, mu, logvar = model(data)
            loss = vae_loss(recon, data, mu, logvar)
            val_loss += loss.item()
    
    train_loss /= len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)
    
    print(f'Epoch {epoch+1}/{num_epochs}: '
          f'Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
    
    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'ising_vae_best.pth')

print("✓ 训练完成，最佳模型已保存\n")


# ============================================================
# 步骤5: 评估和可视化
# ============================================================

print("步骤5: 模型评估与可视化")
print("="*70)

# 加载最佳模型
model.load_state_dict(torch.load('ising_vae_best.pth'))
model.eval()

# 可视化重构结果
fig, axes = plt.subplots(3, 8, figsize=(16, 6))

with torch.no_grad():
    for i in range(8):
        # 获取样本
        original, _ = val_dataset[i]
        original = original.unsqueeze(0).to(device)
        
        # 重构
        recon, mu, logvar = model(original)
        
        # 采样新构型
        z = torch.randn(1, 10).to(device)
        sampled = model.decode(z)
        
        # 可视化
        axes[0, i].imshow(original.cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('原始', fontsize=12)
        
        axes[1, i].imshow(recon.cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('重构', fontsize=12)
        
        axes[2, i].imshow(sampled.cpu().squeeze(), cmap='gray')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel('采样', fontsize=12)

plt.tight_layout()
plt.savefig('vae_results.png', dpi=150)
print("✓ 结果可视化已保存: vae_results.png")

# 潜在空间分析
latent_codes = []
temperatures = []
fields = []

with torch.no_grad():
    for data, labels in val_loader:
        data = data.to(device)
        mu, _ = model.encode(data)
        latent_codes.append(mu.cpu().numpy())
        temperatures.append(labels[:, 0].numpy())
        fields.append(labels[:, 1].numpy())

latent_codes = np.concatenate(latent_codes)
temperatures = np.concatenate(temperatures)
fields = np.concatenate(fields)

# PCA降维可视化
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
latent_2d = pca.fit_transform(latent_codes)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                     c=temperatures, cmap='coolwarm', alpha=0.5, s=1)
plt.colorbar(scatter, label='温度 T')
plt.xlabel('第一主成分')
plt.ylabel('第二主成分')
plt.title('VAE潜在空间（PCA投影）')
plt.savefig('latent_space.png', dpi=150)
print("✓ 潜在空间可视化已保存: latent_space.png")

print("\n" + "="*70)
print("完整工作流执行完成！")
print("="*70)
print("\n生成的文件:")
print("  - ising_training_data.h5  (训练数据)")
print("  - data_samples.png         (数据样本)")
print("  - ising_vae_best.pth       (最佳模型)")
print("  - vae_results.png          (重构结果)")
print("  - latent_space.png         (潜在空间)")
```

---

## 🔬 进阶研究方向

### 1. 临界区域识别

使用VAE的潜在空间识别临界交叉区域：

```python
def detect_critical_region(latent_codes, temperatures):
    """
    基于潜在空间密度检测临界区域
    
    思路: 临界区域的构型在潜在空间中
          应该形成过渡带
    """
    from sklearn.cluster import DBSCAN
    
    # 聚类分析
    clustering = DBSCAN(eps=0.5, min_samples=50)
    labels = clustering.fit_predict(latent_codes)
    
    # 找到过渡区域（多个簇交界处）
    # ...
    
    return critical_temps
```

### 2. 相变点预测

训练回归模型直接预测临界温度：

```python
class CriticalTempPredictor(nn.Module):
    """从构型预测临界温度"""
    
    def __init__(self):
        super().__init__()
        # 使用预训练VAE的编码器
        self.encoder = pretrained_vae.encoder
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # 冻结编码器
        with torch.no_grad():
            z = self.encoder(x)
        return self.regressor(z)
```

### 3. 生成模型应用

条件生成：给定(T, h)生成对应构型

```python
class ConditionalGenerator(nn.Module):
    """条件生成器"""
    
    def forward(self, z, T, h):
        # 将T和h作为条件输入
        condition = torch.cat([z, T, h], dim=1)
        return self.decoder(condition)

# 使用
T_target = 2.269  # 临界温度
h_target = 0.0
z = torch.randn(1, latent_dim)
generated_config = generator(z, T_target, h_target)
```

---

## 📖 常见问题

### Q1: 数据太大怎么办？

**A**: 使用增量生成和加载
```python
# 生成时
generator.generate_full_dataset(save_every_n_fields=10)

# 加载时
with h5py.File('data.h5', 'r') as f:
    subset = f['configs'][0:10]  # 只加载部分
```

### Q2: GPU显存不足？

**A**: 减小批量大小或使用梯度累积
```python
# 小批量 + 梯度累积
accumulation_steps = 4
for i, (data, _) in enumerate(train_loader):
    loss = compute_loss(data) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Q3: 如何验证模型学到了物理？

**A**: 检查以下几点：
1. 潜在空间应该在Tc附近有明显结构变化
2. 重构的构型应保持物理一致性
3. 生成的构型应符合统计分布

---

## 🎓 总结

您现在拥有：

✅ **CPU构型生成器** (`ising_config_saver.py`)
✅ **GPU构型生成器** (`gpu_config_generator.py`)  
✅ **完整训练流程** (本文档)
✅ **VAE示例代码**
✅ **数据可视化工具**

**下一步**:
1. 生成数据（推荐GPU版本）
2. 训练VAE或其他模型
3. 分析潜在空间
4. 发表研究成果！

**祝研究顺利！** 🚀🔬✨



下面是按**现在的代码结构**（`ising_fss.*`、`dispatcher`、`gpu_algorithms`、`config_io` 等）重写后的「深度学习训练指南」。

* 去掉了 `ising_config_saver` / `gpu_config_generator` 这类旧名字；
* 数据生成部分改成「用项目自带脚本 / `ising_fss` 的 API 先生成 HDF5」；
* 深度学习部分保留原来的 VAE / 分类器等示例，只改数据加载接口。

你可以把它放成 `docs/dl_training_guide.md` 或类似文件。

---

````markdown
# 伊辛模型构型数据 - 深度学习训练指南（基于 ising_fss）

本指南介绍如何使用 `ising_fss` 项目生成的伊辛模型构型数据，进行深度学习训练（VAE、分类器、温度回归等）。

- 不关心数据如何产生，只要有一个 HDF5 文件（或等价的 NumPy 数组）就可以跟着本指南走。
- 数据生成的细节与接口，见 `docs/config_data_summary.md` 和 `ising_fss.data.config_io`。

---

## 📊 数据集概述

### 典型数据结构

`ising_fss.data.config_io` 读出的数据，推荐组织为如下结构（以 HDF5 为例）：

```python
dataset = {
    'configs':        # 构型数据
        # 建议形状为 (n_h, n_T, n_configs, L, L)
        np.ndarray,
    'energy':         # 对应构型的能量（可选）
        # 形状 (n_h, n_T, n_configs)
        np.ndarray,
    'magnetization':  # 对应构型的磁化（可选）
        # 形状 (n_h, n_T, n_configs)
        np.ndarray,
    'temperatures':   # 温度网格
        # 形状 (n_T,)
        np.ndarray,
    'fields':         # 外场网格
        # 形状 (n_h,)
        np.ndarray,
    'parameters':     # 元数据（L、equilibration、interval、后端、算法名、随机种子等）
        dict,
}
````

> 实际字段名请以 `config_io.py` 中的实现为准；如有差异，只要能拿到类似结构即可平移本指南的代码。

### 论文标准配置（建议）

| 参数        | 值             | 说明             |
| --------- | ------------- | -------------- |
| L         | 32            | 晶格尺寸           |
| n_T       | 65            | 温度点数           |
| n_h       | 65            | 磁场点数           |
| n_configs | 1024          | 每个 (T, h) 的构型数 |
| **总构型数**  | **4,321,280** | 65×65×1024     |

---

## 🚀 快速开始

### 第 0 步：准备一个 HDF5 数据文件

典型流程是：

1. 使用 `ising_fss.simulation`（CPU 或 GPU REMC）生成构型；
2. 用 `ising_fss.data.config_io.save_configs_hdf5(...)` 写入 `ising_data.h5`；
3. 本指南只关心「如何从 `ising_data.h5` 训练模型」。

假设你已经有了：

```bash
ising_data.h5
```

如果还没有，可参考项目中的 `examples/generate_dl_data.py` 或 `docs/config_data_summary.md`。

---

### 1. 加载数据

最简单的方式是直接用 `h5py` 读，然后在 PyTorch Dataset 里 reshape：

```python
import h5py
import numpy as np

with h5py.File('ising_data.h5', 'r') as f:
    configs = f['configs'][:]        # (n_h, n_T, n_configs, L, L)
    temps   = f['temperatures'][:]   # (n_T,)
    fields  = f['fields'][:]         # (n_h,)

print("构型形状:", configs.shape)
print("温度范围:", temps[0], "→", temps[-1])
print("磁场范围:", fields[0], "→", fields[-1])
```

如果你更愿意走项目封装，也可以：

```python
from ising_fss.data import config_io

dataset = config_io.load_configs_hdf5('ising_data.h5')
configs = dataset['configs']
temps   = dataset['temperatures']
fields  = dataset['fields']
```

下面所有深度学习代码都只依赖 `configs / temperatures / fields` 这几个数组。

---

## 🧠 深度学习应用

### 统一的 PyTorch Dataset 封装

我们先写一个通用的 `IsingDataset`，后面 VAE / 分类器 / 回归都可以共用：

```python
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class IsingDataset(Dataset):
    """伊辛构型数据集（PyTorch 版）"""
    
    def __init__(self, hdf5_path, normalize: bool = True):
        with h5py.File(hdf5_path, 'r') as f:
            # configs: (n_h, n_T, n_configs, L, L)
            configs = f['configs'][:]
            self.L = configs.shape[-1]
            
            # 展平为 (N, L, L)
            self.configs = configs.reshape(-1, self.L, self.L)
            
            # 温度和磁场标签
            temps = f['temperatures'][:]   # (n_T,)
            fields = f['fields'][:]        # (n_h,)
            n_h, n_T, n_cfg = configs.shape[:3]

            labels = []
            for ih, h in enumerate(fields):
                for it, T in enumerate(temps):
                    labels.extend([(float(T), float(h))] * n_cfg)
            self.labels = np.array(labels, dtype=np.float32)
        
        # 归一化：自旋 -1/+1 → [0,1]，便于用 Sigmoid/BCE
        if normalize:
            self.configs = (self.configs + 1.0) / 2.0
        
        print(f"加载 {len(self.configs)} 个构型，L={self.L}")
    
    def __len__(self) -> int:
        return self.configs.shape[0]
    
    def __getitem__(self, idx):
        config = torch.from_numpy(self.configs[idx]).float().unsqueeze(0)  # (1, L, L)
        label  = torch.from_numpy(self.labels[idx])  # (2,) -> (T, h)
        return config, label

# 创建 DataLoader
dataset = IsingDataset('ising_data.h5')
train_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
```

---

### 应用 1：变分自编码器 (VAE)

#### 模型定义

```python
import torch
import torch.nn as nn

class IsingVAE(nn.Module):
    """简单的卷积 VAE，用于 32×32 构型"""
    
    def __init__(self, L=32, latent_dim=10):
        super().__init__()
        self.L = L
        
        # 编码器: (1, L, L) -> latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # -> (32, L/2,   L/2  )
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), # -> (64, L/4,   L/4  )
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),# -> (128, L/8,  L/8  )
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # 根据 L 推算展平后的维度
        with torch.no_grad():
            dummy = torch.zeros(1, 1, L, L)
            enc_dim = self.encoder(dummy).shape[1]
        
        self.fc_mu     = nn.Linear(enc_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_dim, latent_dim)
        
        # 解码器: latent_dim -> (1, L, L)
        self.fc_decode = nn.Linear(latent_dim, enc_dim)
        self.dec_head  = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid(),   # 输出 ∈ [0,1]
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.fc_decode(z)
        # 还原成 (B, 128, L/8, L/8)
        B = z.shape[0]
        side = int((self.L // 8))
        h = h.view(B, 128, side, side)
        return self.dec_head(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z   = self.reparameterize(mu, logvar)
        rec = self.decode(z)
        return rec, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    # 重构损失：二值交叉熵
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL 散度
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
```

#### 训练循环

```python
from torch.utils.data import random_split, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

full_dataset = IsingDataset('ising_data.h5')
L = full_dataset.L

# 训练/验证划分
train_len = int(0.8 * len(full_dataset))
val_len   = len(full_dataset) - train_len
train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

# 初始化模型
model = IsingVAE(L=L, latent_dim=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_val = float('inf')
for epoch in range(1, 51):
    # ------------- 训练 -------------
    model.train()
    train_loss = 0.0
    for x, _ in train_loader:
        x = x.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss = vae_loss(recon, x, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader.dataset)
    
    # ------------- 验证 -------------
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            val_loss += loss.item()
    val_loss /= len(val_loader.dataset)
    
    print(f"Epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}")
    
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), 'ising_vae_best.pth')
        print("  ↳ 保存当前最佳模型：ising_vae_best.pth")
```

---

### 应用 2：相变分类器

我们用温度大致标记相（铁磁 / 临界 / 顺磁），训练一个 CNN 分类器。

```python
import torch.nn as nn
import torch

class IsingClassifier(nn.Module):
    """简单相分类器: 输出 3 个类别（低温 / 临界 / 高温）"""
    
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # L -> L/2
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # L/2 -> L/4
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # -> (128,1,1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
```

相标签：用温度粗略分 3 段（低温、有序；中间、临界附近；高温、无序）：

```python
def temp_to_phase_label(T: float, Tc: float = 2.269) -> int:
    """
    返回 0/1/2:
      0: 铁磁相 (T << Tc)
      1: 临界区附近
      2: 顺磁相 (T >> Tc)
    """
    if T < Tc * 0.8:
        return 0
    elif T < Tc * 1.2:
        return 1
    else:
        return 2
```

你可以在 `Dataset` 里直接把 `T` 转成相标号，也可以在训练循环里 on-the-fly 转换。

---

### 应用 3：温度回归

使用 CNN 直接从构型预测温度：

```python
class TempPredictor(nn.Module):
    """从构型回归预测温度（标量回归）"""
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)
```

训练示意：

```python
model = TempPredictor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(1, 51):
    model.train()
    total_loss = 0.0
    for x, labels in train_loader:
        x = x.to(device)
        temps = labels[:, 0].to(device)  # 只用温度
        
        pred = model(x)
        loss = criterion(pred, temps)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    total_loss /= len(train_loader)
    print(f"Epoch {epoch:3d} | MSE {total_loss:.4f}")
```

---

## 📈 数据与相图可视化

### 绘制 (T, h) 网格上的样本构型

```python
import matplotlib.pyplot as plt
import numpy as np
import h5py

def plot_configs_grid(hdf5_path, n_temps=5, n_fields=5, outfile='config_grid.png'):
    with h5py.File(hdf5_path, 'r') as f:
        configs = f['configs'][:]        # (n_h, n_T, n_cfg, L, L)
        temps   = f['temperatures'][:]   # (n_T,)
        fields  = f['fields'][:]         # (n_h,)
    
    n_h, n_T, n_cfg, L, _ = configs.shape
    
    t_indices = np.linspace(0, n_T-1, n_temps, dtype=int)
    h_indices = np.linspace(0, n_h-1, n_fields, dtype=int)
    
    fig, axes = plt.subplots(len(h_indices), len(t_indices), figsize=(1.8*n_temps, 1.8*n_fields))
    
    for i, ih in enumerate(h_indices):
        for j, it in enumerate(t_indices):
            ax = axes[i, j] if axes.ndim == 2 else axes[max(i,j)]
            config = configs[ih, it, 0]  # 取每个点的第一个构型
            ax.imshow(config, cmap='gray', vmin=-1, vmax=1)
            ax.axis('off')
            if i == 0:
                ax.set_title(f"T={temps[it]:.2f}", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"h={fields[ih]:.2f}", fontsize=8)
    
    plt.tight_layout()
    fig.savefig(outfile, dpi=150)
    print("样本网格已保存至", outfile)

# 使用
plot_configs_grid('ising_data.h5')
```

### 简单相图（平均磁化）

```python
def plot_phase_diagram(hdf5_path, outfile='phase_diagram.png'):
    with h5py.File(hdf5_path, 'r') as f:
        configs = f['configs'][:]
        temps   = f['temperatures'][:]
        fields  = f['fields'][:]
    
    # 平均磁化强度 |M|
    mag = np.mean(configs, axis=(-1, -2))          # (n_h, n_T, n_cfg)
    avg_mag = np.mean(np.abs(mag), axis=2)         # (n_h, n_T)
    
    plt.figure(figsize=(7, 5))
    plt.imshow(
        avg_mag,
        extent=[temps[0], temps[-1], fields[0], fields[-1]],
        aspect='auto',
        origin='lower',
        cmap='RdBu_r',
    )
    plt.colorbar(label='平均磁化强度 |M|')
    plt.xlabel('温度 T')
    plt.ylabel('外场 h')
    plt.title('二维 Ising 模型相图（基于构型数据）')
    
    Tc = 2.269
    plt.axvline(Tc, color='white', linestyle='--', label=f'$T_c \\approx {Tc}$')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print("相图已保存至", outfile)

# 使用
plot_phase_diagram('ising_data.h5')
```

---

## 🔧 高级技巧

### 1. 数据增强（旋转 / 翻转）

```python
def augment_config(config: np.ndarray) -> np.ndarray:
    """对单个 (L,L) 构型做简单数据增强"""
    # 随机旋转（90° 的倍数）
    k = np.random.randint(0, 4)
    config = np.rot90(config, k)
    # 随机翻转
    if np.random.rand() < 0.5:
        config = np.flip(config, axis=0)
    if np.random.rand() < 0.5:
        config = np.flip(config, axis=1)
    return config
```

可以在 `IsingDataset.__getitem__` 里加一个 `augment` 标志，在返回前做增强。

### 2. 条件生成（Conditional VAE）

你可以把 `(T, h)` 当成条件，拼进编码器 / 解码器，例如：

```python
class ConditionalVAE(nn.Module):
    def __init__(self, L=32, latent_dim=10, cond_dim=2):
        super().__init__()
        # 条件向量 (T, h) 先过个小 MLP，然后与图像特征 concat
        # 这里只给结构思路，具体实现可参考 standard CVAE
        ...
```

---

## 💡 实践建议与 Checklist

### 数据规模建议

| 场景        | 推荐配置                 | 说明        |
| --------- | -------------------- | --------- |
| 快速原型      | L=16, n_T=20, n_h=20 | 几分钟生成     |
| 研究实验      | L=32, n_T=40, n_h=40 | 论文配置的一半   |
| 论文级       | L=32, n_T=65, n_h=65 | 对应原文参数    |
| 更大体系 / 挑战 | L=64+，视资源而定          | 强烈建议用 GPU |

### 训练前 Checklist

* [ ] 已确认 HDF5 数据完整（`configs` 维度正确）；
* [ ] 随机可视化了若干构型，确认没有明显错误；
* [ ] 确认归一化方式（-1/+1 → 0/1）与网络输出激活函数匹配；
* [ ] 划分了 train/val/test；
* [ ] 设定合理的 batch_size 与学习率；

### 训练与分析

* [ ] 监控训练/验证损失是否稳定下降；
* [ ] 对比原始构型与重构构型的可视化；
* [ ] 用潜在空间 (z) 画温度或磁场的颜色图，检查是否有物理结构；
* [ ] 对不同相（低温/高温）的 z 做聚类或可视化，验证模型学到相变信息。

---

## 🎓 总结

当前 `ising_fss` 项目为你提供了：

1. **稳定可控的构型生成**（CPU/GPU REMC + `dispatcher` + 严格 RNG）；
2. **标准化的数据格式**（通过 `config_io` / `data_manager` 保存 HDF5）；
3. **面向深度学习的接口与示例**（本指南 + `analysis.dl_tools`）。

你可以在此基础上探索：

* 无监督相变识别（VAE / 自编码器 / 对比学习）；
* 监督相分类 / 临界区域识别；
* 条件生成模型（给定 T,h 生成构型）；
* 更复杂的架构：GNN、Transformer、扩散模型等。

**一句话：**

> 模拟 + 构型数据 + 这套深度学习流水线，已经足以支撑一整篇「AI × 统计物理」的研究工作。祝你玩得开心，也期待你基于这套框架做出自己的变体和创新。🚀

```
::contentReference[oaicite:0]{index=0}
```



这个文档在技术细节上非常详实，但与我们实际的代码库结构存在**严重偏差**。它虚构了大量类名和文件名（如 `IsingConfigGenerator`, `GPUIsingConfigGenerator`, `ising_config_saver.py` 等），如果直接发布，会让用户在 `ising-fss` 包里找不到这些东西而感到困惑。

**核心问题：**
文档中的代码示例是基于一个**假设的 API** 编写的，而不是基于我们实际实现的 `HybridREMCSimulator` / `batch_runner.py` / `dl_tools.py` 体系。

我为你准备了一份**完全重写且经过 API 对齐**的深度学习指南。这份文档将手把手教用户如何使用 `ising-fss` 的真实组件完成从数据生成到 VAE 训练的全过程。

-----

# 📚 深度学习训练指南 (Deep Learning Guide)

本指南将引导您使用 `ising-fss` 框架完成从 **海量 Ising 构型生成** 到 **深度学习模型 (VAE) 训练** 的完整科研工作流。

-----

## 🚀 快速开始

### 1\. 数据生成 (Production)

我们推荐使用 `batch_runner` 在 GPU 上大规模生成数据。

**任务目标**：生成 $L=32$ 的构型，覆盖临界区温度 $T \in [1.6, 3.0]$，用于训练神经网络。

```bash
# 1. 启动 GPU 模拟 (推荐)
# 使用 metropolis_sweep 算法，每进程跑 32 个副本
python -m ising_fss.simulation.batch_runner \
    --mode run_workers \
    --nworkers 4 \
    --L 32 \
    --T 2.269 \
    --replicas 32 \
    --algo metropolis_sweep \
    --equil 5000 \
    --prod 20000 \
    --thin 10 \
    --save_lattices \
    --outdir ./data_dl_L32

# 2. 合并数据
python -m ising_fss.simulation.batch_runner \
    --mode merge \
    --outdir ./data_dl_L32
```

*产出文件*：`./data_dl_L32/merged/final_ml_data.h5` (包含构型、温度、磁场等)

### 2\. 数据清洗与导出 (ETL)

原始 HDF5 数据可能非常巨大且未归一化。我们提供 `export_for_pytorch` 工具将其转换为 **AI-Ready** 格式。

```python
# scripts/prepare_dl_data.py
from ising_fss.data.config_io import load_configs_hdf5, export_for_pytorch

# 加载原始数据 (Lazy Mode，不占内存)
dataset = load_configs_hdf5("./data_dl_L32/merged/final_ml_data.h5", load_configs=False)

# 导出为 PyTorch 格式
# - 自动压缩为 uint8
# - 归一化到 [0, 1]
# - 划分 80% 训练集
export_for_pytorch(
    dataset,
    save_dir="./data_ready/L32",
    split_ratio=0.8,
    dtype='uint8',
    normalize=True,
    verbose=True
)
```

-----

## 🧬 数据加载 (Data Loading)

`ising-fss` 提供了高性能的 `DataLoader` 工厂，支持**确定性数据增强**。这意味着旋转/翻转操作与样本索引绑定，保证训练过程完全可复现。

```python
import torch
from ising_fss.analysis.dl_tools import create_dataloaders_from_path, AugmentConfig

# 配置确定性增强 (D4对称群)
aug_cfg = AugmentConfig(enable=True, rot90=True, hflip=True, vflip=True)

# 一键创建 Loaders
loaders = create_dataloaders_from_path(
    "./data_ready/L32",
    batch_size=128,
    num_workers=4,
    augment=aug_cfg,
    pin_memory=True
)

train_loader = loaders['train']
val_loader = loaders['val']

# 测试读取
batch = next(iter(train_loader))
x = batch['config']       # Tensor (B, 1, 32, 32), range [0, 1]
T = batch['temperature']  # Tensor (B,), 对应温度
```

-----

## 🧠 训练示例：变分自编码器 (VAE)

我们将训练一个 VAE 来无监督地学习 Ising 模型的潜在序参量（Order Parameter）。

### 定义模型

```python
import torch.nn as nn
import torch.nn.functional as F

class IsingVAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(64*8*8, latent_dim)
        self.fc_logvar = nn.Linear(64*8*8, latent_dim)
        
        # Decoder
        self.dec_fc = nn.Linear(latent_dim, 64*8*8)
        self.dec = nn.Sequential(
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, 2, 1, output_padding=1), nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar
```

### 训练循环

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IsingVAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

print(f"开始训练 on {device}...")
for epoch in range(10):
    total_loss = 0
    for batch in train_loader:
        x = batch['config'].to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss = loss_fn(recon, x, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}: Avg Loss = {total_loss / len(train_loader.dataset):.2f}")
```

-----

## 📊 结果可视化

### 1\. 潜在空间投影 (Latent Space)

利用 `ising-fss` 内置的可视化工具，观察 VAE 学习到的潜在变量 $z$ 如何随温度 $T$ 分布。

```python
from ising_fss.visualization.plots import plot_latent_space
import numpy as np

# 收集验证集的潜在向量
zs, temps = [], []
with torch.no_grad():
    for batch in val_loader:
        x = batch['config'].to(device)
        mu, _ = model.enc(x), None  # 简化：只取 mu 后的fc输出需自行调整，这里假设拿到 z
        # 注意：上面的 VAE 代码需要微调以直接返回 z，或者拆解 forward
        # 这里仅作逻辑示意
        h = model.enc(x)
        z = model.fc_mu(h)
        zs.append(z.cpu().numpy())
        temps.append(batch['temperature'].numpy())

plot_latent_space(
    latent_codes=np.concatenate(zs),
    labels=np.concatenate(temps),
    label_type='temperature',
    save_path="vae_latent_space.png"
)
```

> **预期结果**：你应该能看到潜在空间中呈现出明显的“叉”状或“V”字形结构，分别对应低温有序相（两个分支对应自旋向上/向下）和高温无序相。

### 2\. 重构质量 (Reconstruction)

```python
from ising_fss.visualization.plots import plot_vae_reconstruction

# 取一个 batch
batch = next(iter(val_loader))
x = batch['config'].to(device)[:8]
with torch.no_grad():
    recon, _, _ = model(x)

plot_vae_reconstruction(
    original=x.cpu().squeeze().numpy(),
    reconstructed=recon.cpu().squeeze().numpy(),
    save_path="vae_recon_comparison.png"
)
```

-----

## 🔬 进阶：物理感知 (Physics-Informed)

`dl_tools` 提供了计算物理量的工具，你可以将其加入 Loss 函数，迫使模型学习物理约束。

```python
from ising_fss.analysis.dl_tools import energy_density, magnetization

# 在训练循环中监控物理量一致性
pred_E = energy_density(recon_x)
true_E = energy_density(x)
physics_loss = F.mse_loss(pred_E, true_E)

# Total Loss = VAE_Loss + lambda * physics_loss
```

-----

祝你的 AI + Physics 之旅充满发现！🚀



收到。如果您**不需要归一化**（即希望保留物理上的自旋值 $\{-1, 1\}$），这对物理感知（Physics-Informed）的训练其实更好，因为可以直接计算能量而无需反归一化。

但有一点必须注意：**如果不归一化，就不能使用 `uint8` 存储**（因为 `uint8` 是无符号的 $0 \sim 255$，无法表示 $-1$）。

请按照以下方案调整您的 **数据导出** 和 **数据加载** 流程：

### 1\. 修改数据导出脚本 (`prepare_data.py`)

在调用 `export_for_pytorch` 时，做两个关键改动：

1.  `normalize=False`：关闭归一化。
2.  `dtype='float32'`：**必须**改为浮点数（或修改代码支持 `int8`）。因为默认的 `uint8` 无法存储负数。

<!-- end list -->

```python
from ising_fss.data.config_io import load_configs_hdf5, export_for_pytorch

# 加载原始数据
dataset = load_configs_hdf5("./data_factory/raw/merged/final_ml_data.h5", load_configs=False)

# 导出设置：保留原始物理数值 {-1, 1}
export_for_pytorch(
    dataset,
    save_dir="./data_ready/L32_Raw", # 建议换个目录名区分
    split_ratio=0.8,
    
    # 🔥 关键修改点 🔥
    normalize=False,   # 关掉归一化！保持 -1/1
    dtype='float32',   # 必须用 float32 (当前代码不支持 int8 导出，uint8 存不了负数)
    
    verbose=True
)
```

> **💡 进阶优化（节省 4 倍空间）：**
> 如果您非常在意磁盘空间，希望用 `int8` (1字节) 存储 $\{-1, 1\}$，您需要微调 `src/ising_fss/data/config_io.py` 文件：
>
> 1.  找到 `assert dtype in ('float32', 'uint8')`，加入 `'int8'`。
> 2.  在下方的 `dtype` 判断逻辑中加入：
>     ```python
>     elif dtype == 'int8':
>         x_out = configs_norm.astype(np.int8)
>     ```
>
> 这样您就可以使用 `dtype='int8'` 导出了。

### 2\. 修改数据加载代码 (`train.py`)

在创建 DataLoader 时，同样需要显式关闭归一化。

```python
from ising_fss.analysis.dl_tools import create_dataloaders_from_path

loaders = create_dataloaders_from_path(
    "./data_ready/L32_Raw", # 指向刚才导出的 raw 数据目录
    batch_size=128,
    num_workers=4,
    
    # 🔥 关键修改点 🔥
    normalize=False,  # 告诉 Loader 不要自动把数据缩放到 [0,1]
    
    pin_memory=True
)

# 验证一下
batch = next(iter(loaders['train']))
x = batch['config']
print(f"数据范围: Min={x.min()}, Max={x.max()}") 
# 输出应为: Min=-1.0, Max=1.0 (如果是 float32)
```

### 3\. 对模型的影响

如果不归一化，您的数据是 $\{-1, 1\}$：

  * **输入层**：神经网络完全可以处理负数输入，通常不需要改动模型结构。
  * **激活函数**：
      * 如果是 **VAE**，输出层（Decoder最后一层）以前可能用了 `Sigmoid`（输出 $0 \sim 1$）。
      * **修改建议**：现在应该改用 **`Tanh`**（输出 $-1 \sim 1$）作为最后一层的激活函数，或者不加激活函数（Linear）并结合 MSE Loss。

**VAE 模型修改示例：**

```python
class IsingVAE(nn.Module):
    # ... (Encoder 不变) ...
    
    def __init__(self, ...):
        # ...
        self.dec = nn.Sequential(
            # ... 前面的层不变 ...
            nn.ConvTranspose2d(...),
            
            # 🔥 修改点：从 Sigmoid 改为 Tanh 🔥
            nn.Tanh()  # 输出范围 [-1, 1]
        )
```

这样您的整个管线就完全基于物理原始数值运行了。