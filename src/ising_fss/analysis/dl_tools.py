# -*- coding: utf-8 -*-
"""
    深度学习专用 Ising 数据工具集（PyTorch 完全兼容）

实现功能：
    - 支持 HDF5/NPZ 两种存储格式（5D 张量或 3D configs + labels）
    - 惰性加载（H5LazyDataset）+ 可 pickle（子进程安全）
    - DataLoader worker 自动使用独立 RNG（通过 torch.initial_seed()）
    - 内置数据增强：±1 ↔ 0/1 转换、随机翻转、旋转、对称群
    - 提供物理量计算：能量密度、最近邻关联、结构因子 S(q)

支持数据集类：
    - IsingNPZDataset
    - IsingH5Dataset
    - IsingDataset（通用包装）
"""
from __future__ import annotations

import os
import math
import random
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

__all__ = [
    "AugmentConfig",
    "IsingNPZDataset",
    "IsingH5Dataset",
    "IsingDataset",
    "load_ising_dataset",
    "create_dataloaders",
    "create_dataloaders_from_path",
    "compute_order_parameter",
    "energy_density",
    "nearest_neighbor_correlations",
    "structure_factor",
    "evaluate_classification",
    "evaluate_regression",
]

# ---------------------------------------------------------------------------
# Reproducible worker seeding (Python/NumPy only; do not touch global torch RNG)
# ---------------------------------------------------------------------------

def _seed_worker(worker_id: int):
    """
    Called by DataLoader worker processes.
    Use torch.initial_seed() (per-worker seed derived by PyTorch) to seed Python & NumPy.
    """
    seed = torch.initial_seed() % (2**32)
    random.seed(int(seed))
    np.random.seed(int(seed))


def _default_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    """Build a torch.Generator seeded once for DataLoader shuffling (optional)."""
    if seed is None:
        return None
    g = torch.Generator()
    g.manual_seed(int(seed) & ((1 << 63) - 1))
    return g


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_tensor_int8(x: np.ndarray) -> torch.Tensor:
    """Ensure (1,L,L) int8 spin tensor in {-1,+1}."""
    a = np.asarray(x)
    # Convert bool/uint8 0/1 -> -1/+1
    if a.dtype == np.bool_:
        a = a.astype(np.int8) * 2 - 1
    elif a.dtype == np.uint8:
        if a.size == 0:
            a = a.astype(np.int8)
        else:
            if np.max(a) <= 1 and np.min(a) >= 0:
                a = a.astype(np.int8) * 2 - 1
            else:
                a = a.astype(np.int8)
    elif a.dtype != np.int8 and a.dtype != np.int16 and a.dtype != np.int32:
        # try to map floats -> sign
        if np.issubdtype(a.dtype, np.floating):
            # if values in [-1,1] keep sign; else round to nearest int8
            if np.nanmin(a) >= -1.0 - 1e-6 and np.nanmax(a) <= 1.0 + 1e-6:
                a = np.sign(a).astype(np.int8)
                a[a == 0] = 1  # treat zeros as +1 to be conservative
            else:
                a = a.astype(np.int8)
        else:
            a = a.astype(np.int8)
    t = torch.from_numpy(a).to(torch.int8)
    if t.ndim == 2:
        t = t.unsqueeze(0)  # (1,L,L)
    return t


def _flatten_indices(shape: Tuple[int, ...]) -> np.ndarray:
    n = 1
    for s in shape:
        n *= int(s)
    return np.arange(n, dtype=np.int64)


def _per_sample_generator_from_index(idx: int) -> torch.Generator:
    """
    Derive a per-sample torch.Generator deterministically from torch.initial_seed() and sample index.
    Uses: seed_base XOR H(idx) (hash) to avoid simple collisions.
    Returns a Generator already manual_seed'ed.
    """
    base = int(torch.initial_seed()) & ((1 << 63) - 1)
    # hash index to 64-bit int deterministically
    h = hashlib.blake2b(str(idx).encode("utf-8"), digest_size=8).digest()
    idx_hash = int.from_bytes(h, "big") & ((1 << 63) - 1)
    seed = (base ^ idx_hash) & ((1 << 63) - 1)
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


# ---------------------------------------------------------------------------
# Data augmentation (D4 symmetries subset)
# ---------------------------------------------------------------------------

@dataclass
class AugmentConfig:
    enable: bool = True
    rot90: bool = True
    hflip: bool = True
    vflip: bool = True


def apply_augmentation(x: torch.Tensor, g: Optional[torch.Generator], cfg: AugmentConfig) -> torch.Tensor:
    """
    使用局部 generator 对单样本进行旋转/翻转增强（不会修改外部全局 RNG）。
    - x: (1,L,L) 或 (C,L,L) tensor
    - g: torch.Generator（可选）；若 None 则使用全局 RNG（不推荐用于严格可复现场景）
    返回新的 tensor（非原地修改保证）。
    """
    if not cfg.enable:
        return x
    y = x
    # rot90: 0..3
    if cfg.rot90:
        if g is not None:
            k = int(torch.randint(0, 4, (1,), generator=g).item())
        else:
            k = int(torch.randint(0, 4, (1,)).item())
        if k:
            y = torch.rot90(y, k, dims=(-2, -1))
    # horizontal flip
    if cfg.hflip:
        if g is not None:
            do = bool(int(torch.randint(0, 2, (1,), generator=g).item()))
        else:
            do = bool(int(torch.randint(0, 2, (1,)).item()))
        if do:
            y = torch.flip(y, dims=(-1,))
    # vertical flip
    if cfg.vflip:
        if g is not None:
            do = bool(int(torch.randint(0, 2, (1,), generator=g).item()))
        else:
            do = bool(int(torch.randint(0, 2, (1,)).item()))
        if do:
            y = torch.flip(y, dims=(-2,))
    return y


# ---------------------------------------------------------------------------
# NPZ dataset
# ---------------------------------------------------------------------------

class IsingNPZDataset(Dataset):
    """
    Lazy/mmap NPZ dataset.

    Keys supported:
    - configs: (N,L,L)  OR (n_h,n_T,n_cfg,L,L)
    - temperatures: optional; could be shape (N,), or (n_T,) or (n_h,n_T) etc.
    - fields: optional; could be shape (N,), or (n_h,) or similar.

    The class attempts best-effort resolution of temperature/field for each sample.
    """
    def __init__(self, npz_path: Union[str, os.PathLike], augment: Optional[AugmentConfig] = None):
        self.path = str(npz_path)
        self._npz = np.load(self.path, mmap_mode="r")
        if "configs" not in self._npz:
            raise ValueError("NPZ file missing 'configs' array.")
        cfg = self._npz["configs"]
        self._cfg = cfg
        self._shape = cfg.shape
        self._logical_shape = None  # (n_h, n_T, n_cfg) if 5D
        if cfg.ndim == 5:
            self._logical_shape = (int(cfg.shape[0]), int(cfg.shape[1]), int(cfg.shape[2]))
            self._L = int(cfg.shape[3])
            self._N = int(cfg.shape[0] * cfg.shape[1] * cfg.shape[2])
        elif cfg.ndim == 3:
            self._logical_shape = None
            self._L = int(cfg.shape[1])
            self._N = int(cfg.shape[0])
        else:
            raise ValueError("configs must be (N,L,L) or (n_h,n_T,n_cfg,L,L)")

        # optional metadata
        self._temps_raw = self._npz.get("temperatures", None)
        self._fields_raw = self._npz.get("fields", None)
        self.augment = augment or AugmentConfig(enable=False)

    def __len__(self) -> int:
        return self._N

    def _logical_indices(self, idx: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        if self._logical_shape is None:
            return None, None, int(idx)
        nh, nT, nC = self._logical_shape
        h_idx, t_idx, k = np.unravel_index(int(idx), (nh, nT, nC))
        return int(h_idx), int(t_idx), int(k)

    def _load_config(self, idx: int) -> np.ndarray:
        if self._logical_shape is None:
            return self._cfg[int(idx)]
        h_idx, t_idx, k = self._logical_indices(int(idx))
        return self._cfg[h_idx, t_idx, k]

    def _resolve_temperature(self, idx: int, t_idx: Optional[int]) -> Optional[float]:
        if self._temps_raw is None:
            return None
        temps = self._temps_raw
        # If temps is 1D:
        if getattr(temps, "ndim", 0) == 1:
            if self._logical_shape is None:
                # Expect temps.shape[0] == N OR temps.shape[0] == n_T
                if temps.shape[0] == self._N:
                    return float(temps[int(idx)])
                elif temps.shape[0] == 1:
                    return float(temps[0])
                else:
                    # ambiguous, fallback to nearest index mapping
                    return float(temps[int(min(int(idx), temps.shape[0]-1))])
            else:
                # logical: temps corresponds to n_T
                nT = self._logical_shape[1]
                if temps.shape[0] == nT:
                    return float(temps[int(t_idx)])
                elif temps.shape[0] == 1:
                    return float(temps[0])
                elif temps.shape[0] == self._N:
                    return float(temps[int(idx)])
                else:
                    return float(temps[int(min(int(t_idx) if t_idx is not None else 0, temps.shape[0]-1))])
        else:
            # multi-dim: try to index with t_idx/h_idx
            try:
                return float(np.asarray(temps).ravel()[int(idx)])
            except Exception:
                try:
                    return float(np.asarray(temps).ravel()[0])
                except Exception:
                    return None

    def _resolve_field(self, idx: int, h_idx: Optional[int]) -> Optional[float]:
        if self._fields_raw is None:
            return None
        fields = self._fields_raw
        if getattr(fields, "ndim", 0) == 1:
            if self._logical_shape is None:
                if fields.shape[0] == self._N:
                    return float(fields[int(idx)])
                elif fields.shape[0] == 1:
                    return float(fields[0])
                else:
                    return float(fields[int(min(int(idx), fields.shape[0]-1))])
            else:
                nH = self._logical_shape[0]
                if fields.shape[0] == nH:
                    return float(fields[int(h_idx)])
                elif fields.shape[0] == 1:
                    return float(fields[0])
                elif fields.shape[0] == self._N:
                    return float(fields[int(idx)])
                else:
                    return float(fields[int(min(int(h_idx) if h_idx is not None else 0, fields.shape[0]-1))])
        else:
            try:
                return float(np.asarray(fields).ravel()[int(idx)])
            except Exception:
                try:
                    return float(np.asarray(fields).ravel()[0])
                except Exception:
                    return None

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        h_idx, t_idx, _k = self._logical_indices(int(idx))
        x_np = self._load_config(int(idx))
        x = _to_tensor_int8(np.asarray(x_np))

        # per-sample generator deterministic derivation
        g = _per_sample_generator_from_index(int(idx))
        x = apply_augmentation(x, g, self.augment)

        out: Dict[str, torch.Tensor] = {"config": x}
        T = self._resolve_temperature(idx, t_idx)
        H = self._resolve_field(idx, h_idx)
        if T is not None:
            out["temperature"] = torch.tensor(float(T), dtype=torch.float32)
            # phase label as example
            out["phase"] = torch.tensor(1 if float(T) < 2.269185 else 0, dtype=torch.long)
        if H is not None:
            out["field"] = torch.tensor(float(H), dtype=torch.float32)
        return out


# ---------------------------------------------------------------------------
# HDF5 dataset
# ---------------------------------------------------------------------------

class IsingH5Dataset(Dataset):
    """
    HDF5 lazy dataset for generator output.

    Supports:
      - configs shape (n_h,n_T,n_cfg,L,L)  (5D)
      - configs shape (N,L,L) + datasets 'temperatures' (N,) and 'fields' (N,)  (3D + labels)
      - configs shape (N,L,L) + global 'temperatures'(n_T,) and 'fields'(n_h,) (best-effort mapping)

    index_map: optional flattened index mapping to select subset
    read_observables: if True, attempt to read 'energy' / 'magnetization' datasets.
    """
    def __init__(self, h5_path: Union[str, os.PathLike],
                 index_map: Optional[Sequence[int]] = None,
                 augment: Optional[AugmentConfig] = None,
                 read_observables: bool = False):
        self.path = str(h5_path)
        self._file = h5py.File(self.path, "r")
        if "configs" not in self._file:
            raise ValueError("HDF5 missing 'configs' dataset.")
        self._cfg = self._file["configs"]
        # detect layout
        if self._cfg.ndim == 5:
            self.layout = "5D"
            self.n_h, self.n_T, self.n_C, self.L, _ = (int(x) for x in self._cfg.shape)
            self._N = self.n_h * self.n_T * self.n_C
        elif self._cfg.ndim == 3:
            self.layout = "3D"
            self._N, self.L, _ = (int(x) for x in self._cfg.shape)
            # try to infer n_h/n_T from temperatures/fields if present — optional
            self.n_h = None
            self.n_T = None
        else:
            raise ValueError("configs in HDF5 must be (n_h,n_T,n_cfg,L,L) or (N,L,L).")

        # metadata datasets (best-effort)
        self._temps_ds = self._file.get("temperatures", None)
        self._fields_ds = self._file.get("fields", None)
        self._energy = self._file.get("energy", None) if read_observables else None
        self._mag = self._file.get("magnetization", None) if read_observables else None

        # index mapping (flat indices into N)
        if index_map is not None:
            self.index = np.asarray(index_map, dtype=np.int64)
        else:
            if self.layout == "5D":
                self.index = _flatten_indices((self.n_h, self.n_T, self.n_C))
            else:
                self.index = np.arange(self._N, dtype=np.int64)

        self.augment = augment or AugmentConfig(enable=False)

    def __len__(self) -> int:
        return int(self.index.shape[0])

    def _logical_indices(self, flat: int) -> Tuple[int, int, int]:
        if self.layout != "5D":
            raise RuntimeError("logical indices only for 5D layout")
        return tuple(int(x) for x in np.unravel_index(int(flat), (self.n_h, self.n_T, self.n_C)))

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        flat = int(self.index[int(i)])
        if self.layout == "5D":
            h_idx, t_idx, k = self._logical_indices(flat)
            x_np = self._cfg[h_idx, t_idx, k]
        else:  # 3D flat
            x_np = self._cfg[flat]
            # attempt to resolve t_idx/h_idx if temps/fields are multi-length
            t_idx = None
            h_idx = None
            k = flat

        x = _to_tensor_int8(np.asarray(x_np))

        # per-sample deterministic generator
        g = _per_sample_generator_from_index(int(flat))
        x = apply_augmentation(x, g, self.augment)

        out: Dict[str, torch.Tensor] = {"config": x}
        # temperature resolution
        if self._temps_ds is not None:
            try:
                temps = np.asarray(self._temps_ds)
                if temps.ndim == 1:
                    if self.layout == "5D":
                        # temps expected length n_T or N
                        if temps.shape[0] == self.n_T:
                            T = float(temps[int(t_idx)])
                        elif temps.shape[0] == self._N:
                            T = float(temps[int(flat)])
                        else:
                            T = float(temps[0])
                    else:
                        if temps.shape[0] == self._N:
                            T = float(temps[int(flat)])
                        else:
                            T = float(temps[0])
                else:
                    # fallback to first element
                    T = float(np.asarray(temps).ravel()[int(flat) % temps.size])
                out["temperature"] = torch.tensor(float(T), dtype=torch.float32)
                out["phase"] = torch.tensor(1 if float(T) < 2.269185 else 0, dtype=torch.long)
            except Exception:
                pass

        # field resolution
        if self._fields_ds is not None:
            try:
                fields = np.asarray(self._fields_ds)
                if fields.ndim == 1:
                    if self.layout == "5D" and fields.shape[0] == self.n_h:
                        H = float(fields[int(h_idx)])
                    elif fields.shape[0] == self._N:
                        H = float(fields[int(flat)])
                    else:
                        H = float(fields[0])
                else:
                    H = float(np.asarray(fields).ravel()[int(flat) % fields.size])
                out["field"] = torch.tensor(float(H), dtype=torch.float32)
            except Exception:
                pass

        # observables if present
        if self._energy is not None:
            try:
                if self.layout == "5D":
                    val = float(self._energy[int(h_idx), int(t_idx), int(k)])
                else:
                    val = float(self._energy[int(flat)])
                out["energy"] = torch.tensor(val, dtype=torch.float32)
            except Exception:
                pass
        if self._mag is not None:
            try:
                if self.layout == "5D":
                    val = float(self._mag[int(h_idx), int(t_idx), int(k)])
                else:
                    val = float(self._mag[int(flat)])
                out["magnetization"] = torch.tensor(val, dtype=torch.float32)
            except Exception:
                pass

        return out

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass

    def __del__(self):
        self.close()


# ---------------------------------------------------------------------------
# Unified ndarray wrapper
# ---------------------------------------------------------------------------

class _ArrayIsingDataset(Dataset):
    """从 ndarray 构造，shape=(N,L,L) 或 (n_h,n_T,n_cfg,L,L)，惰性展平。"""
    def __init__(self, arr: np.ndarray, augment: Optional[AugmentConfig] = None):
        arr = np.asarray(arr)
        if arr.ndim == 5:
            n_h, n_T, n_C, L, _ = arr.shape
            arr = arr.reshape(n_h * n_T * n_C, L, L)
        if arr.ndim != 3:
            raise ValueError(f"ndarray must be (N,L,L) or (n_h,n_T,n_cfg,L,L), got {arr.shape}")
        self.arr = arr
        self.N = int(arr.shape[0])
        self.augment = augment or AugmentConfig(enable=False)

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = _to_tensor_int8(np.asarray(self.arr[int(idx)]))
        g = _per_sample_generator_from_index(int(idx))
        x = apply_augmentation(x, g, self.augment)
        return {"config": x}


class IsingDataset(Dataset):
    """
    统一入口 Dataset：
      - source 为路径(.npz/.h5) -> 复用 IsingNPZDataset / IsingH5Dataset
      - source 为 ndarray -> 使用 _ArrayIsingDataset
      - normalize=True 时，将数据规范化到 [0,1]（适合 BCE/VAE）
    """
    def __init__(
        self,
        source: Union[str, os.PathLike, np.ndarray, Dataset],
        *,
        normalize: bool = False,
        augment: Optional[AugmentConfig] = None,
        read_observables: bool = False,
    ):
        self.normalize = bool(normalize)
        self.augment = augment or AugmentConfig(enable=False)

        if isinstance(source, (str, os.PathLike)):
            ds = load_ising_dataset(source, augment=self.augment, read_observables=read_observables)
        elif isinstance(source, np.ndarray):
            ds = _ArrayIsingDataset(source, augment=self.augment)
        elif isinstance(source, Dataset):
            ds = source
        else:
            raise TypeError("IsingDataset.source 必须是路径(.npz/.h5)或 numpy.ndarray 或 Dataset")
        self.base = ds

    def __len__(self) -> int:
        return len(self.base)

    def _normalize_tensor_to_0_1(self, x: torch.Tensor) -> torch.Tensor:
        """统一的归一化逻辑：先转 float32，再按 [-1,1] 或 [0,1] 或 min-max 缩放到 [0,1]。"""
        # ensure float tensor
        if not x.dtype.is_floating_point:
            x = x.to(torch.float32)
        else:
            x = x.clone().to(torch.float32)

        # compute min/max safely
        try:
            xmin = float(torch.nanmin(x))
            xmax = float(torch.nanmax(x))
        except Exception:
            xmin = 0.0
            xmax = 0.0

        # If already in [-1,1] (approx), map to [0,1]
        if xmin >= -1.0 - 1e-6 and xmax <= 1.0 + 1e-6:
            return (x + 1.0) * 0.5
        # If already in [0,1], keep
        if xmin >= 0.0 - 1e-12 and xmax <= 1.0 + 1e-12:
            return x
        # Else do min-max scaling
        denom = (xmax - xmin)
        if denom > 1e-12:
            return (x - xmin) / denom
        # Constant field or degenerate -> return zeros
        return torch.zeros_like(x, dtype=torch.float32)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        d = self.base[int(i)]  # {"config": int8 (1,L,L), ...}
        x = d["config"]
        # normalization if required
        if self.normalize:
            x = self._normalize_tensor_to_0_1(x)
        else:
            if not x.dtype.is_floating_point:
                x = x.to(torch.float32)

        out = dict(d)
        out["config"] = x
        return out


# ---------------------------------------------------------------------------
# Dataloader factory
# ---------------------------------------------------------------------------

def _build_loader(dataset: Dataset,
                  batch_size: int,
                  shuffle: bool,
                  num_workers: int,
                  seed: Optional[int],
                  pin_memory: Optional[bool] = None,
                  persistent_workers: Optional[bool] = None) -> DataLoader:
    g = _default_generator(seed)  # controls shuffle reproducibility
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if persistent_workers is None:
        persistent_workers = (num_workers > 0)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=g,
        persistent_workers=persistent_workers,
    )


def create_dataloaders(*args,
                       batch_size: int = 256,
                       val_split: float = 0.1,
                       shuffle: bool = True,
                       num_workers: int = 4,
                       seed: Optional[int] = 1234,
                       normalize: bool = False,
                       augment: Optional[AugmentConfig] = None,
                       read_observables: bool = False,
                       pin_memory: Optional[bool] = None,
                       persistent_workers: Optional[bool] = None) -> Dict[str, DataLoader]:
    """
    create_dataloaders 支持：
      1) create_dataloaders(dataset, val_split=..., ...)
      2) create_dataloaders(train_dataset, val_dataset, ...)
      3) create_dataloaders(train_source, val_source, ...) （source 为 path 或 ndarray）

    返回 {"train": train_loader, "val": val_loader}
    """
    # A) single Dataset -> split
    if len(args) == 1 and isinstance(args[0], Dataset):
        dataset: Dataset = args[0]
        N = len(dataset)
        # edge cases
        if N == 0:
            raise ValueError("Dataset is empty")
        idx = np.arange(N)
        rng = np.random.default_rng(int(seed) if seed is not None else None)
        if shuffle:
            rng.shuffle(idx)
        n_val = int(round(N * float(val_split))) if 0.0 < val_split < 1.0 else (N if val_split >= 1.0 else 0)
        val_idx = idx[:n_val].tolist()
        train_idx = idx[n_val:].tolist()
        # handle degenerate splits
        if len(train_idx) == 0:
            # ensure at least one train sample if possible
            if len(val_idx) > 1:
                train_idx = [val_idx.pop()]
        train = torch.utils.data.Subset(dataset, train_idx)
        val = torch.utils.data.Subset(dataset, val_idx)

        train_loader = _build_loader(train, batch_size, True, num_workers, seed,
                                     pin_memory=pin_memory, persistent_workers=persistent_workers)
        val_loader = _build_loader(val, batch_size, False, max(1, num_workers // 2), seed,
                                   pin_memory=pin_memory, persistent_workers=persistent_workers)
        return {"train": train_loader, "val": val_loader}

    # B) two sources/datasets
    if len(args) >= 2:
        a0, a1 = args[0], args[1]
        if isinstance(a0, Dataset) and isinstance(a1, Dataset):
            train_ds: Dataset = a0
            val_ds: Dataset = a1
        else:
            train_ds = IsingDataset(a0, normalize=normalize, augment=augment, read_observables=read_observables)
            val_ds = IsingDataset(a1, normalize=normalize, augment=augment, read_observables=read_observables)

        train_loader = _build_loader(train_ds, batch_size, True, num_workers, seed,
                                     pin_memory=pin_memory, persistent_workers=persistent_workers)
        val_loader = _build_loader(val_ds, batch_size, False, max(1, num_workers // 2), seed,
                                   pin_memory=pin_memory, persistent_workers=persistent_workers)
        return {"train": train_loader, "val": val_loader}

    raise TypeError(
        "create_dataloaders 的用法应为：\n"
        "  * create_dataloaders(dataset, val_split=..., ...)  或\n"
        "  * create_dataloaders(train_dataset, val_dataset, ...)  或\n"
        "  * create_dataloaders(train_source, val_source, batch_size=..., ...)"
    )


# ---------------------------------------------------------------------------
# Dataset factory by path
# ---------------------------------------------------------------------------

def load_ising_dataset(path: Union[str, os.PathLike],
                       augment: Optional[AugmentConfig] = None,
                       read_observables: bool = False,
                       index_map: Optional[Sequence[int]] = None) -> Dataset:
    """Auto-detect and open an Ising dataset from HDF5 or NPZ."""
    p = str(path)
    ext = Path(p).suffix.lower()
    if ext in (".h5", ".hdf5"):
        return IsingH5Dataset(p, index_map=index_map, augment=augment, read_observables=read_observables)
    elif ext == ".npz":
        return IsingNPZDataset(p, augment=augment)
    else:
        raise ValueError(f"Unsupported dataset extension: {ext}")


def create_dataloaders_from_path(path: Union[str, os.PathLike],
                                 batch_size: int = 256,
                                 val_split: float = 0.1,
                                 shuffle: bool = True,
                                 num_workers: int = 4,
                                 seed: Optional[int] = 1234,
                                 augment: Optional[AugmentConfig] = None,
                                 read_observables: bool = False) -> Dict[str, DataLoader]:
    """Convenience: load dataset by path then create train/val loaders."""
    ds = load_ising_dataset(path, augment=augment, read_observables=read_observables)
    return create_dataloaders(ds, batch_size=batch_size, val_split=val_split,
                              shuffle=shuffle, num_workers=num_workers, seed=seed)


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def compute_order_parameter(configs: torch.Tensor) -> torch.Tensor:
    """Mean spin per sample. configs: (N,1,L,L) int8/float -> (N,1) float32"""
    x = configs.to(torch.float32) if not configs.dtype.is_floating_point else configs
    return x.mean(dim=(-2, -1), keepdim=True)


def energy_density(configs: torch.Tensor, h: Optional[float] = None) -> torch.Tensor:
    """
    Bond energy per site and optional field term.
    Return shape: (N,1) float32
    """
    x = configs.to(torch.float32)
    right = torch.roll(x, shifts=-1, dims=-1)
    down = torch.roll(x, shifts=-1, dims=-2)
    e_bond = -(x * (right + down)).mean(dim=(-2, -1), keepdim=True)
    if h is None:
        return e_bond
    m = x.mean(dim=(-2, -1), keepdim=True)
    return e_bond - float(h) * m


def nearest_neighbor_correlations(configs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (corr_x, corr_y) per sample, mean over lattice. shapes (N,1)"""
    x = configs.to(torch.float32)
    right = torch.roll(x, shifts=-1, dims=-1)
    down = torch.roll(x, shifts=-1, dims=-2)
    corr_x = (x * right).mean(dim=(-2, -1), keepdim=True)
    corr_y = (x * down).mean(dim=(-2, -1), keepdim=True)
    return corr_x, corr_y


def structure_factor(configs: torch.Tensor, zero_mean: bool = True, radial_bins: Optional[int] = None
                    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute S(k) = |FFT(config)|^2 / (L*L). configs: (N,1,L,L).
    If radial_bins is not None returns (k_centers, S_radial) with shape:
      k_centers: (radial_bins,), S_radial: (N, radial_bins)
    """
    x = configs.to(torch.float32)
    if zero_mean:
        x = x - x.mean(dim=(-2, -1), keepdim=True)
    spec = torch.fft.fftn(x, dim=(-2, -1))
    S = (spec.real**2 + spec.imag**2) / (x.shape[-1] * x.shape[-2])  # shape (N,1,L,L)

    if radial_bins is None:
        return S

    # radial averaging on host numpy arrays
    N, C, L, _ = S.shape
    S_cpu = S.detach().cpu().numpy()  # (N,1,L,L)
    # build radius grid in integer wavevector units
    yy, xx = np.meshgrid(np.fft.fftfreq(L) * L, np.fft.fftfreq(L) * L, indexing="ij")
    rr = np.sqrt(xx**2 + yy**2).ravel()
    r_max = float(rr.max())
    bins = np.linspace(0.0, r_max + 1e-6, radial_bins + 1)
    S_rad = []
    k_centers = 0.5 * (bins[:-1] + bins[1:])
    for n in range(N):
        sn = S_cpu[n, 0].ravel()
        sums = np.zeros(radial_bins, dtype=np.float64)
        counts = np.zeros(radial_bins, dtype=np.int64)
        inds = np.digitize(rr, bins) - 1
        inds = np.clip(inds, 0, radial_bins - 1)
        np.add.at(sums, inds, sn)
        np.add.at(counts, inds, 1)
        S_rad.append(sums / np.maximum(1, counts))
    S_rad = torch.from_numpy(np.stack(S_rad, axis=0)).to(torch.float32)
    k_centers = torch.from_numpy(k_centers.astype(np.float32))
    return k_centers, S_rad


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_classification(model: torch.nn.Module, loader: DataLoader, device: Union[str, torch.device] = "cpu"
                           ) -> Dict[str, Any]:
    model.eval()
    device = torch.device(device)
    total = 0
    correct = 0
    all_preds: List[int] = []
    all_labels: List[int] = []
    for batch in loader:
        x = batch["config"].to(device).to(torch.float32)
        y = batch.get("phase", None)
        if y is None:
            continue
        y = y.to(device).view(-1)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(y.detach().cpu().tolist())
        correct += int((preds == y).sum())
        total += int(y.numel())
    acc = float(correct) / max(1, total)
    return {"accuracy": acc, "preds": np.array(all_preds, dtype=np.int64), "labels": np.array(all_labels, dtype=np.int64)}


@torch.no_grad()
def evaluate_regression(model: torch.nn.Module, loader: DataLoader, target_key: str = "temperature",
                        device: Union[str, torch.device] = "cpu") -> Dict[str, Any]:
    model.eval()
    device = torch.device(device)
    y_true: List[float] = []
    y_pred: List[float] = []
    for batch in loader:
        if target_key not in batch:
            continue
        x = batch["config"].to(device).to(torch.float32)
        y = batch[target_key].to(device).view(-1)
        pred = model(x).view(-1)
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    mse = float(np.mean((y_true - y_pred) ** 2)) if y_true.size > 0 else float("nan")
    mae = float(np.mean(np.abs(y_true - y_pred))) if y_true.size > 0 else float("nan")
    return {"mse": mse, "mae": mae, "y_true": y_true, "y_pred": y_pred}

