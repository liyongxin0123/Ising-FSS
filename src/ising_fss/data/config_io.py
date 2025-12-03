# -*- coding: utf-8 -*-
"""
Ising 构型数据集高效 I/O 工具（支持流式与惰性加载）

实现功能：
    - HDF5 / NPZ 的读写（支持压缩、chunks、libver）
    - H5LazyDataset：支持 pickle，子进程重开文件句柄
    - 流式批迭代器 batch_iterator（不占内存）
    - 自动验证数据集完整性（形状、温度、能量一致性）
    - 一键导出为 PyTorch 训练目录（train/val/test）
    - 合并 / 切分 / 验证 / 统计（尽量避免全量内存占用）
    - 兼容 3D / 5D 数据格式
"""
from __future__ import annotations

import json
import datetime
import math
from pathlib import Path
from typing import Dict, Any, Optional, Iterator, Tuple, List, Sequence, Union

import numpy as np
import h5py

__all__ = [
    'save_configs_hdf5',
    'load_configs_hdf5',
    'load_configs_lazy',
    'save_configs_npz',
    'load_configs_npz',
    'export_for_pytorch',
    'load_pytorch_dataset',
    'split_dataset',
    'augment_configs',
    'batch_iterator',
    'merge_datasets',
    'validate_dataset',
    'compute_dataset_statistics',
    'print_dataset_summary',
    'DatasetInfo',
    'H5LazyDataset',
]

# -----------------------------------------------------------------------------
# Dataset 元信息类（便于打印/诊断）
# -----------------------------------------------------------------------------
class DatasetInfo:
    """从 dataset dict 中尽力推断出关键信息，用于打印和诊断。"""
    def __init__(self, dataset: Dict[str, Any]):
        cfg = dataset.get('configs', None)
        self.total_size = None
        self.shape = None
        self.L = None
        self.n_h = None
        self.n_T = None
        self.n_configs = None
        self.T_range = (None, None)
        self.h_range = (None, None)

        if cfg is None:
            self.shape = dataset.get('configs_shape', None)
        else:
            # 支持 ndarray / h5py.Dataset / H5LazyDataset（具有 shape）
            try:
                if hasattr(cfg, 'shape') and not isinstance(cfg, np.ndarray):
                    shape = tuple(cfg.shape)
                else:
                    arr = np.asarray(cfg)
                    shape = arr.shape
                    try:
                        self.total_size = arr.nbytes
                    except Exception:
                        self.total_size = None
            except Exception:
                shape = None
            self.shape = shape

            if shape is not None:
                if len(shape) == 5:
                    n_h, n_T, n_c, L, _ = shape
                    self.L = int(L)
                    self.n_h = int(n_h)
                    self.n_T = int(n_T)
                    self.n_configs = int(n_c)
                elif len(shape) == 3:
                    N, L, _ = shape
                    self.L = int(L)
                    self.n_h = 1
                    self.n_T = 1
                    self.n_configs = int(N)

        temps = dataset.get('temperatures', None)
        fields = dataset.get('fields', None)
        if temps is not None:
            try:
                tarr = np.asarray(temps)
                if tarr.size:
                    self.T_range = (float(np.min(tarr)), float(np.max(tarr)))
                if self.n_T is None:
                    self.n_T = int(tarr.size)
            except Exception:
                pass
        if fields is not None:
            try:
                harr = np.asarray(fields)
                if harr.size:
                    self.h_range = (float(np.min(harr)), float(np.max(harr)))
                if self.n_h is None:
                    self.n_h = int(harr.size)
            except Exception:
                pass

        params = dataset.get('parameters', {}) or {}
        if 'n_configs' in params and self.n_configs is None:
            try:
                self.n_configs = int(params['n_configs'])
            except Exception:
                pass
        if 'L' in params and self.L is None:
            try:
                self.L = int(params['L'])
            except Exception:
                pass

    def __str__(self):
        shape = self.shape or '?'
        L = self.L or '?'
        total_mb = (self.total_size / 1e6) if (self.total_size is not None) else float('nan')
        return (
            f"Dataset Info:\n"
            f"  Shape: {shape}\n"
            f"  Lattice size: {L}×{L}\n"
            f"  Temperatures range: {self.T_range}\n"
            f"  Fields range: {self.h_range}\n"
            f"  Configs per point (approx): {self.n_configs}\n"
            f"  Total size (approx): {total_mb:.1f} MB\n"
        )

# -----------------------------------------------------------------------------
# HDF5 保存/加载
# -----------------------------------------------------------------------------
def _auto_h5_chunks(configs: np.ndarray) -> Optional[Tuple[int, ...]]:
    """基于真实 ndarray 形状选择合理 chunk（偏向按配置切片）"""
    if configs.ndim == 5:
        n_h, n_T, n_c, L, _ = configs.shape
        return (1, 1, min(16, max(1, n_c)), L, L)
    elif configs.ndim == 3:
        N, L, _ = configs.shape
        return (min(256, max(1, N)), L, L)
    else:
        return None

def _auto_h5_chunks_from_shape(shape: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
    """在未知实际 ndarray 的情况下，仅基于 shape 推断 chunks（避免整体物化惰性数据）"""
    if len(shape) == 5:
        n_h, n_T, n_c, L, _ = shape
        return (1, 1, min(16, max(1, int(n_c))), int(L), int(L))
    elif len(shape) == 3:
        N, L, _ = shape
        return (min(256, max(1, int(N))), int(L), int(L))
    return None

def _dataset_name_in_h5(f: h5py.File) -> Optional[str]:
    """返回文件中可作为配置数组的数据集名（优先 'configs'，否则 'lattices'），不存在则 None。"""
    if 'configs' in f:
        return 'configs'
    if 'lattices' in f:
        return 'lattices'
    return None


def save_configs_hdf5(dataset: Dict[str, Any],
                      filepath: Union[str, Path],
                      compression: Optional[str] = 'gzip',
                      compression_opts: Optional[int] = 4,
                      chunks: Optional[Union[Tuple[int, ...], str, None]] = 'auto',
                      libver: str = 'latest',
                      verbose: bool = True) -> None:
    """保存到 HDF5，支持自动 chunk、压缩、attrs 写入；对缺失字段做容错处理。"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    cfg = dataset.get('configs')
    if cfg is None:
        raise ValueError("dataset 必须包含 'configs' 字段")

    # 选择 chunks（惰性数据避免整体物化）
    try:
        if isinstance(chunks, str) and chunks == 'auto':
            if hasattr(cfg, 'shape') and not isinstance(cfg, np.ndarray):
                chunks_val = _auto_h5_chunks_from_shape(tuple(cfg.shape))
            else:
                chunks_val = _auto_h5_chunks(np.asarray(cfg))
        else:
            chunks_val = chunks
    except Exception:
        chunks_val = None

    if verbose:
        print(f"\n保存 HDF5 到: {filepath}")

    # h5py: lzf 不接受 compression_opts
    if compression == 'lzf':
        compression_opts = None

    # 写入：优先尝试流式拷贝（惰性数据），失败回退到一次性写入
    try:
        with h5py.File(filepath, 'w', libver=libver) as f:
            try:
                # 惰性数据：创建空 dset，分块拷贝，避免 OOM
                if hasattr(cfg, 'shape') and not isinstance(cfg, np.ndarray):
                    shape = tuple(cfg.shape)
                    # dtype 推断：优先 cfg.dtype；否则抽样一个切片；最后兜底 int8
                    dtype = getattr(cfg, 'dtype', None)
                    if dtype is None:
                        try:
                            if len(shape) == 5:
                                sample = np.array(cfg[0, 0, 0])
                            elif len(shape) == 3:
                                sample = np.array(cfg[0])
                            else:
                                raise RuntimeError("unsupported lazy shape")
                            dtype = sample.dtype
                        except Exception:
                            dtype = np.int8
                    dset = f.create_dataset(
                        'configs', shape=shape, dtype=dtype,
                        compression=compression, compression_opts=compression_opts,
                        chunks=chunks_val
                    )
                    # 分块策略：5D 按 n_cfg 维；3D 按样本维
                    if len(shape) == 5:
                        n_h, n_T, n_c, _, _ = shape
                        step = min(256, int(n_c))
                        for h in range(int(n_h)):
                            for t in range(int(n_T)):
                                for c0 in range(0, int(n_c), step):
                                    c1 = min(int(n_c), c0 + step)
                                    dset[h, t, c0:c1] = np.array(cfg[h, t, c0:c1])
                    elif len(shape) == 3:
                        N = int(shape[0]); step = min(4096, N)
                        for i0 in range(0, N, step):
                            i1 = min(N, i0 + step)
                            dset[i0:i1] = np.array(cfg[i0:i1])
                    else:
                        # 形状异常：退回一次性写入
                        raise RuntimeError("unsupported lazy shape for streaming copy")
                else:
                    # 内存数组：一次性写入
                    f.create_dataset('configs', data=np.asarray(cfg),
                                     compression=compression,
                                     compression_opts=compression_opts,
                                     chunks=chunks_val)
            except Exception:
                # fallback: 无压缩一次性写入
                f.create_dataset('configs', data=np.asarray(cfg))

            # optional datasets
            if 'energy' in dataset and dataset['energy'] is not None:
                try:
                    f.create_dataset('energy', data=np.asarray(dataset['energy']), compression=compression,
                                     compression_opts=compression_opts)
                except Exception:
                    f.create_dataset('energy', data=np.asarray(dataset['energy']))
            if 'magnetization' in dataset and dataset['magnetization'] is not None:
                try:
                    f.create_dataset('magnetization', data=np.asarray(dataset['magnetization']), compression=compression,
                                     compression_opts=compression_opts)
                except Exception:
                    f.create_dataset('magnetization', data=np.asarray(dataset['magnetization']))

            # temperatures/fields: only write if present
            if 'temperatures' in dataset and dataset['temperatures'] is not None:
                try:
                    f.create_dataset('temperatures', data=np.asarray(dataset['temperatures']))
                except Exception:
                    pass
            if 'fields' in dataset and dataset['fields'] is not None:
                try:
                    f.create_dataset('fields', data=np.asarray(dataset['fields']))
                except Exception:
                    pass

            # parameters as attrs (尽量转为原生类型)
            for k, v in (dataset.get('parameters') or {}).items():
                try:
                    if isinstance(v, (np.integer,)):
                        f.attrs[k] = int(v)
                    elif isinstance(v, (np.floating,)):
                        f.attrs[k] = float(v)
                    else:
                        f.attrs[k] = v
                except Exception:
                    f.attrs[k] = str(v)
            f.attrs['created_at'] = str(datetime.datetime.now())

            if verbose:
                try:
                    d = f['configs']
                    print(f"  实际压缩器: {d.compression}  块大小: {d.chunks}")
                except Exception:
                    pass

    except Exception as exc:
        # 退回到更保守的写法（无压缩）
        if verbose:
            print(f"⚠ HDF5 保存（带压缩/特殊参数）失败，尝试无压缩回退: {exc}")
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('configs', data=np.asarray(cfg))
            if 'energy' in dataset and dataset['energy'] is not None:
                f.create_dataset('energy', data=np.asarray(dataset['energy']))
            if 'magnetization' in dataset and dataset['magnetization'] is not None:
                f.create_dataset('magnetization', data=np.asarray(dataset['magnetization']))
            if 'temperatures' in dataset and dataset['temperatures'] is not None:
                try:
                    f.create_dataset('temperatures', data=np.asarray(dataset['temperatures']))
                except Exception:
                    pass
            if 'fields' in dataset and dataset['fields'] is not None:
                try:
                    f.create_dataset('fields', data=np.asarray(dataset['fields']))
                except Exception:
                    pass
            for k, v in (dataset.get('parameters') or {}).items():
                try:
                    f.attrs[k] = v
                except Exception:
                    f.attrs[k] = str(v)
            f.attrs['created_at'] = str(datetime.datetime.now())

    if verbose:
        try:
            file_size = filepath.stat().st_size / 1e6
        except Exception:
            file_size = float('nan')
        try:
            if hasattr(cfg, 'shape') and not isinstance(cfg, np.ndarray):
                dtype = np.dtype(getattr(cfg, 'dtype', np.int8))
                original_size = (int(np.prod(cfg.shape)) * dtype.itemsize) / 1e6
            else:
                original_size = np.asarray(cfg).nbytes / 1e6
        except Exception:
            original_size = float('nan')
        ratio = (original_size / file_size) if (file_size > 0 and not np.isnan(original_size)) else float('nan')
        print("✓ 保存成功")
        print(f"  文件大小: {file_size:.1f} MB  原始大小(估计): {original_size:.1f} MB  压缩比: {ratio:.2f}x")


def load_configs_hdf5(filepath: Union[str, Path],
                      load_configs: bool = True,
                      load_obs: bool = True,
                      verbose: bool = True) -> Dict[str, Any]:
    """从 HDF5 加载（可选择是否把 configs 载入内存）。"""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    if verbose:
        print(f"\n加载 HDF5: {filepath}")

    dataset: Dict[str, Any] = {}
    with h5py.File(filepath, 'r') as f:
        ds_name = _dataset_name_in_h5(f)
        if ds_name is None:
            raise ValueError("HDF5 不包含 'configs' 或 'lattices' 数据集")
        if load_configs:
            dataset['configs'] = f[ds_name][:]  # 读入内存
        else:
            dataset['configs'] = None
            dataset['configs_shape'] = tuple(f[ds_name].shape)
        dataset['source_dataset_name'] = ds_name

        if load_obs:
            if 'energy' in f:
                dataset['energy'] = f['energy'][:]
            else:
                dataset['energy'] = None
            if 'magnetization' in f:
                dataset['magnetization'] = f['magnetization'][:]
            else:
                dataset['magnetization'] = None

        if 'temperatures' in f:
            dataset['temperatures'] = f['temperatures'][:]
        else:
            dataset['temperatures'] = None

        if 'fields' in f:
            dataset['fields'] = f['fields'][:]
        else:
            dataset['fields'] = None

        # attrs -> parameters dict
        params = {}
        for k, v in f.attrs.items():
            try:
                params[k] = v
            except Exception:
                params[k] = str(v)
        dataset['parameters'] = params

    if verbose:
        if load_configs and dataset.get('configs') is not None:
            try:
                print(f"✓ 加载成功  形状: {dataset['configs'].shape}")
            except Exception:
                print("✓ 加载成功 (configs 已载入)")
        else:
            print(f"✓ 加载成功  形状(未载入): {dataset.get('configs_shape')}")

    return dataset

# -----------------------------------------------------------------------------
# 惰性 HDF5 访问器
# -----------------------------------------------------------------------------
class H5LazyDataset:
    """
    惰性 HDF5 访问器：
    - 仅保存 path；首次访问时打开文件（以只读方式）
    - 支持上下文管理；支持 pickle（__getstate__/__setstate__）在 worker 中重开
    - 提供 shape / dtype / __len__ / __getitem__ / slice 等便捷方法（返回 numpy array）
    """
    def __init__(self, filepath: Union[str, Path]):
        self.path = str(filepath)
        self._f: Optional[h5py.File] = None
        self._dset = None

    def _ensure_open(self):
        if self._f is None:
            try:
                # swmr 与 libver 参数在部分 h5py 版本上可能不支持
                try:
                    self._f = h5py.File(self.path, 'r', swmr=True, libver='latest')
                except TypeError:
                    self._f = h5py.File(self.path, 'r')
            except Exception as e:
                raise RuntimeError(f"打开 HDF5 失败: {e}")
            # 尝试定位 'configs'，回退 'lattices'
            name = _dataset_name_in_h5(self._f)
            if name is None:
                raise RuntimeError("HDF5 文件中未找到 'configs' 或 'lattices' 数据集")
            self._dset = self._f[name]

    def close(self):
        if self._f is not None:
            try:
                self._f.close()
            finally:
                self._f = None
                self._dset = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.close()

    def __getstate__(self):
        # pickle 时不要携带打开的文件句柄，子进程中会在需要时重开
        d = self.__dict__.copy()
        d['_f'] = None
        d['_dset'] = None
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._f = None
        self._dset = None

    @property
    def shape(self) -> Tuple[int, ...]:
        self._ensure_open()
        return tuple(self._dset.shape)

    @property
    def dtype(self):
        self._ensure_open()
        return getattr(self._dset, 'dtype', None)

    def __len__(self) -> int:
        self._ensure_open()
        # flatten 到第一个维度的总数（支持 3D 或 5D）
        shp = tuple(self._dset.shape)
        if len(shp) == 5:
            n_h, n_T, n_c, _, _ = shp
            return int(n_h * n_T * n_c)
        if len(shp) == 3:
            return int(shp[0])
        return int(shp[0])

    def _idx_to_trip(self, idx: int) -> Tuple[int, int, int]:
        self._ensure_open()
        shp = tuple(self._dset.shape)
        if len(shp) == 5:
            n_h, n_T, n_c, _, _ = shp
            idx = int(idx)
            h = idx // (n_T * n_c)
            rem = idx % (n_T * n_c)
            t = rem // n_c
            c = rem % n_c
            return h, t, c
        raise RuntimeError("not a 5D dataset")

    def get_config(self, h: int, t: int, c: int) -> np.ndarray:
        self._ensure_open()
        return np.array(self._dset[h, t, c])

    def get_point(self, h: int, t: int) -> np.ndarray:
        self._ensure_open()
        return np.array(self._dset[h, t])

    def slice(self, h_slice, t_slice, c_slice) -> np.ndarray:
        self._ensure_open()
        return np.array(self._dset[h_slice, t_slice, c_slice])

    def __getitem__(self, key):
        self._ensure_open()
        return np.array(self._dset[key])

def load_configs_lazy(filepath: Union[str, Path]) -> H5LazyDataset:
    """返回惰性访问器（适用于多进程 DataLoader 等场景）"""
    return H5LazyDataset(filepath)

# -----------------------------------------------------------------------------
# NPZ 保存/加载（同时写旁侧 parameters.json 以便可读）
# -----------------------------------------------------------------------------
def save_configs_npz(dataset: Dict[str, Any],
                     filepath: Union[str, Path],
                     compressed: bool = True,
                     verbose: bool = True) -> None:
    """保存为 NPZ，同时写一个 parameters.json（可读）"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"\n保存 NPZ: {filepath}")

    save_dict = {}
    for key in ('configs', 'temperatures', 'fields', 'energy', 'magnetization'):
        if key in dataset and dataset[key] is not None:
            save_dict[key] = np.asarray(dataset[key])

    # 写参数 JSON 旁文件
    params = dataset.get('parameters', {}) or {}
    try:
        (filepath.parent / (filepath.stem + '_parameters.json')).write_text(json.dumps(params, indent=2, ensure_ascii=False))
    except Exception:
        pass

    if compressed:
        np.savez_compressed(filepath, **save_dict)
    else:
        np.savez(filepath, **save_dict)

    if verbose:
        print("✓ 保存成功")


def load_configs_npz(filepath: Union[str, Path], verbose: bool = True) -> Dict[str, Any]:
    """加载 NPZ；兼容旁侧 parameters.json"""
    filepath = Path(filepath)
    if verbose:
        print(f"\n加载 NPZ: {filepath}")
    data = np.load(filepath, allow_pickle=False)
    dataset: Dict[str, Any] = {}
    # required keys handling with fallbacks
    for key in ('configs', 'temperatures', 'fields', 'energy', 'magnetization'):
        if key in data:
            dataset[key] = np.asarray(data[key])
        else:
            dataset[key] = None

    params_path = filepath.parent / (filepath.stem + '_parameters.json')
    if params_path.exists():
        try:
            dataset['parameters'] = json.loads(params_path.read_text(encoding='utf-8'))
        except Exception:
            dataset['parameters'] = {}
    else:
        dataset['parameters'] = {}

    if verbose:
        try:
            print(f"✓ 加载成功: {dataset['configs'].shape}")
        except Exception:
            print("✓ 加载成功")
    return dataset

# -----------------------------------------------------------------------------
# PyTorch 友好导出（更稳健的归一化与标签生成）
# -----------------------------------------------------------------------------
def export_for_pytorch(dataset: Dict[str, Any],
                       save_dir: Union[str, Path],
                       split_ratio: float = 0.8,
                       normalize: bool = True,
                       dtype: str = 'float32',   # 'float32' 或 'uint8'
                       seed: Optional[int] = 123,
                       verbose: bool = True) -> None:
    """
    导出为 PyTorch 友好格式（train/val npz + metadata.json）
    - 支持 5D (n_h,n_T,n_cfg,L,L) 或 3D (N,L,L)
    - normalize=True 启用智能归一化（signed/unit/minmax/constant）
    - dtype: 'float32' 或 'uint8'（uint8 存 0..255）
    注意：对非常大的 lazy dataset，若无法全部载入会尝试以流式读取（可能会比较慢）。
    """
    assert dtype in ('float32', 'uint8')
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n导出 PyTorch 格式数据...")

    cfg = dataset.get('configs')
    if cfg is None:
        raise ValueError("dataset must contain 'configs'")

    # 将数据展平为 (N, L, L) —— 对 lazy dataset 尝试流式读取以避免 OOM
    if hasattr(cfg, 'shape') and not isinstance(cfg, np.ndarray):
        # lazy HDF5 dataset
        shape = tuple(cfg.shape)
        if len(shape) == 5:
            n_h, n_T, n_c, L, _ = shape
            N = n_h * n_T * n_c
            configs_flat = np.empty((N, L, L), dtype=np.int8)
            idx = 0
            for h in range(n_h):
                for t in range(n_T):
                    for c in range(n_c):
                        configs_flat[idx] = np.array(cfg[h, t, c])
                        idx += 1
            temps_raw = np.asarray(dataset.get('temperatures', np.zeros(n_T)))
            fields_raw = np.asarray(dataset.get('fields', np.zeros(n_h)))
            temps_block = np.repeat(temps_raw, n_c)
            temperatures = np.tile(temps_block, n_h).astype(np.float32)
            fields_rep = np.repeat(fields_raw, n_T * n_c).astype(np.float32)
        elif len(shape) == 3:
            N, L, _ = shape
            configs_flat = np.empty((N, L, L), dtype=np.int8)
            for i in range(N):
                configs_flat[i] = np.array(cfg[i])
            temps_raw = np.asarray(dataset.get('temperatures', np.array([0.0])))
            fields_raw = np.asarray(dataset.get('fields', np.array([0.0])))
            if temps_raw.size == 1:
                temperatures = np.full((N,), float(temps_raw[0]), dtype=np.float32)
            elif temps_raw.size == N:
                temperatures = temps_raw.astype(np.float32)
            else:
                temperatures = np.full((N,), float(temps_raw.ravel()[0]), dtype=np.float32)
            if fields_raw.size == 1:
                fields_rep = np.full((N,), float(fields_raw[0]), dtype=np.float32)
            elif fields_raw.size == N:
                fields_rep = fields_raw.astype(np.float32)
            else:
                fields_rep = np.full((N,), float(fields_raw.ravel()[0]), dtype=np.float32)
        else:
            raise ValueError(f"不支持的 configs 维度: {len(shape)}")
    else:
        # 非 lazy：直接用 numpy
        arr = np.asarray(cfg)
        if arr.ndim == 5:
            n_h, n_T, n_c, L, _ = arr.shape
            N = n_h * n_T * n_c
            configs_flat = arr.reshape(N, L, L)
            temps_raw = np.asarray(dataset.get('temperatures', np.zeros(n_T)))
            fields_raw = np.asarray(dataset.get('fields', np.zeros(n_h)))
            temps_block = np.repeat(temps_raw, n_c)
            temperatures = np.tile(temps_block, n_h).astype(np.float32)
            fields_rep = np.repeat(fields_raw, n_T * n_c).astype(np.float32)
        elif arr.ndim == 3:
            N, L, _ = arr.shape
            configs_flat = arr
            temps_raw = np.asarray(dataset.get('temperatures', np.array([0.0])))
            fields_raw = np.asarray(dataset.get('fields', np.array([0.0])))
            if temps_raw.size == 1:
                temperatures = np.full((N,), float(temps_raw[0]), dtype=np.float32)
            elif temps_raw.size == N:
                temperatures = temps_raw.astype(np.float32)
            else:
                temperatures = np.full((N,), float(temps_raw.ravel()[0]), dtype=np.float32)
            if fields_raw.size == 1:
                fields_rep = np.full((N,), float(fields_raw[0]), dtype=np.float32)
            elif fields_raw.size == N:
                fields_rep = fields_raw.astype(np.float32)
            else:
                fields_rep = np.full((N,), float(fields_raw.ravel()[0]), dtype=np.float32)
        else:
            raise ValueError(f"不支持的 configs 维度: {arr.ndim}")

    # normalization: probe small subset first（稳健处理极端值）
    configs_float = configs_flat.astype(np.float32)
    norm_method = "none"
    norm_params: Dict[str, float] = {}
    if normalize:
        nprobe = min(256, max(1, configs_float.shape[0]))
        rng = np.random.default_rng(0)
        probe_idx = rng.choice(configs_float.shape[0], size=nprobe, replace=False)
        probe = configs_float[probe_idx]
        mn = float(np.nanmin(probe)); mx = float(np.nanmax(probe))
        if mn >= -1.0 - 1e-6 and mx <= 1.0 + 1e-6:
            configs_norm = (configs_float + 1.0) * 0.5
            norm_method = "signed"
            norm_params = {"scale": 0.5, "offset": 0.5}
        elif mn >= -1e-6 and mx <= 1.0 + 1e-6:
            configs_norm = configs_float
            norm_method = "unit"
            norm_params = {"scale": 1.0, "offset": 0.0}
        else:
            mn_full = float(np.nanmin(configs_float)); mx_full = float(np.nanmax(configs_float))
            if mx_full - mn_full > 1e-12:
                configs_norm = (configs_float - mn_full) / (mx_full - mn_full)
                norm_method = "minmax"
                norm_params = {"min": mn_full, "max": mx_full}
            else:
                configs_norm = np.zeros_like(configs_float)
                norm_method = "constant"
                norm_params = {"value": (float(configs_float.ravel()[0]) if configs_float.size else 0.0)}
    else:
        configs_norm = configs_float
        norm_method = "none"
        norm_params = {"scale": 1.0, "offset": 0.0}

    # dtype conversion
    if dtype == 'uint8':
        x_out = np.clip(np.round(configs_norm * 255.0), 0, 255).astype(np.uint8)
    else:
        x_out = configs_norm.astype(np.float32)

    # shuffle & split
    N = len(x_out)
    n_train = int(math.floor(N * float(split_ratio)))
    rng = np.random.default_rng(int(seed) if seed is not None else None)
    idx = np.arange(N)
    rng.shuffle(idx)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    # save npz
    np.savez_compressed(save_dir / 'train_data.npz',
                        configs=x_out[train_idx],
                        temperatures=temperatures[train_idx],
                        fields=fields_rep[train_idx])
    np.savez_compressed(save_dir / 'val_data.npz',
                        configs=x_out[val_idx],
                        temperatures=temperatures[val_idx],
                        fields=fields_rep[val_idx])

    # metadata
    metadata = {
        'L': int(configs_flat.shape[1]),
        'n_train': int(len(train_idx)),
        'n_val': int(len(val_idx)),
        'T_range': [float(np.min(temperatures)), float(np.max(temperatures))] if len(temperatures) else [None, None],
        'h_range': [float(np.min(fields_rep)), float(np.max(fields_rep))] if len(fields_rep) else [None, None],
        'normalized': bool(normalize),
        'norm_method': norm_method,
        'norm_params': norm_params,
        'dtype': dtype,
        'original_shape': list(getattr(dataset.get('configs'), 'shape', getattr(configs_flat, 'shape', None))),
        'created_at': str(datetime.datetime.now()),
    }
    try:
        with open(save_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    if verbose:
        print(f"✓ 导出完成  训练集: {len(train_idx)}  验证集: {len(val_idx)}")
        print(f"  存放目录: {save_dir}")

def load_pytorch_dataset(data_dir: Union[str, Path], split: str = 'train') -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """加载之前导出的 train/val npz 与 metadata"""
    data_dir = Path(data_dir)
    data = np.load(data_dir / f'{split}_data.npz', allow_pickle=False)
    metadata = {}
    try:
        with open(data_dir / 'metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception:
        metadata = {}
    return dict(data), metadata

# -----------------------------------------------------------------------------
# 切分 / 合并 / 增强 / 迭代
# -----------------------------------------------------------------------------
def split_dataset(dataset: Dict[str, Any],
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15,
                  test_ratio: float = 0.15,
                  seed: int = 42) -> Dict[str, np.ndarray]:
    """在每个 (h,T) 点上分层切分，返回 {'train','val','test'}（期望 5D 输入）。"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    rng = np.random.default_rng(seed)

    cfg = dataset.get('configs')
    arr = np.asarray(cfg)
    if arr.ndim != 5:
        raise ValueError("split_dataset 预期 (n_h,n_T,n_cfg,L,L) 形状。")
    n_h, n_T, n_c, L, _ = arr.shape

    train_list, val_list, test_list = [], [], []
    for h in range(n_h):
        for t in range(n_T):
            perm = rng.permutation(n_c)
            n_tr = int(math.floor(n_c * train_ratio))
            n_vl = int(math.floor(n_c * val_ratio))
            tr = perm[:n_tr]; vl = perm[n_tr:n_tr+n_vl]; te = perm[n_tr+n_vl:]
            if tr.size:
                train_list.append(arr[h, t, tr])
            if vl.size:
                val_list.append(arr[h, t, vl])
            if te.size:
                test_list.append(arr[h, t, te])

    out = {
        'train': np.concatenate(train_list, axis=0) if train_list else np.empty((0, L, L), dtype=arr.dtype),
        'val': np.concatenate(val_list, axis=0) if val_list else np.empty((0, L, L), dtype=arr.dtype),
        'test': np.concatenate(test_list, axis=0) if test_list else np.empty((0, L, L), dtype=arr.dtype),
    }
    return out

def _d4_transforms(x: np.ndarray) -> Iterator[np.ndarray]:
    """返回 D4 群的 8 种变换（旋转 × flip）"""
    for k in (0, 1, 2, 3):
        r = np.rot90(x, k=k, axes=(-2, -1))
        yield r
        yield np.flip(r, axis=-1)

def augment_configs(configs: np.ndarray,
                    rotations: bool = True,
                    flips: bool = True,
                    return_array: bool = True):
    """对 (N,L,L) 或 5D 展平到 N 后，做 D4 增强（返回 np.array 或 list）。"""
    x = np.asarray(configs)
    if x.ndim < 2:
        raise ValueError("configs 必须至少为二维数组，最后两维为 (L,L)。")
    if x.shape[-1] != x.shape[-2]:
        raise ValueError("最后两维必须为方阵 (L,L)。")
    L = x.shape[-1]
    if x.ndim > 2:
        N = int(np.prod(x.shape[:-2]))
        base = x.reshape(N, L, L)
    else:
        base = x
        N = base.shape[0]

    outs = []
    for k in (0, 1, 2, 3) if rotations else (0,):
        r = np.rot90(base, k=k, axes=(-2, -1))
        outs.append(r)
        if flips:
            outs.append(np.flip(r, axis=-1))
    if return_array:
        return np.concatenate(outs, axis=0)
    lst = []
    for arr in outs:
        for i in range(arr.shape[0]):
            lst.append(arr[i].copy())
    return lst

def _flatten_configs(cfg: np.ndarray) -> np.ndarray:
    if cfg.ndim == 5:
        n_h, n_T, n_c, L, _ = cfg.shape
        return cfg.reshape(n_h * n_T * n_c, L, L)
    elif cfg.ndim == 3:
        return cfg
    else:
        raise ValueError("期望 (n_h,n_T,n_cfg,L,L) 或 (N,L,L)")

def batch_iterator(dataset_or_array,
                   batch_size: int = 64,
                   shuffle: bool = True,
                   seed: Optional[int] = None,
                   flatten: bool = True) -> Iterator[Dict[str, Any]]:
    """
    通用批量迭代器（支持 numpy ndarray / dict(dataset) / h5py.Dataset / H5LazyDataset）
    - 对 lazy 数据按索引流式读取，避免一次性载入内存
    - 修复：惰性+flatten=False+5D 时不再依赖未定义的 idx_to_trip，按首维切片返回块
    """
    if isinstance(dataset_or_array, dict):
        cfg = dataset_or_array.get('configs')
    else:
        cfg = dataset_or_array

    is_lazy = hasattr(cfg, 'shape') and not isinstance(cfg, np.ndarray)
    shape = None
    if is_lazy:
        shape = tuple(cfg.shape)
        if flatten:
            if len(shape) == 5:
                n_h, n_T, n_c, L, _ = shape
                N = n_h * n_T * n_c

                def idx_to_trip(i):
                    i = int(i)
                    h = i // (n_T * n_c)
                    rem = i % (n_T * n_c)
                    t = rem // n_c
                    c = rem % n_c
                    return h, t, c
            elif len(shape) == 3:
                N = shape[0]

                def idx_to_trip(i):
                    return int(i), None, None
            else:
                raise ValueError("Unsupported lazy configs shape for batch_iterator.")
        else:
            # not flatten: treat first axes as batch dim
            N = shape[0]
    else:
        arr = np.asarray(cfg)
        if flatten:
            arr = _flatten_configs(arr)
        N = len(arr)

    idxs = np.arange(N)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idxs)

    for s in range(0, N, batch_size):
        sel = idxs[s:s+batch_size]
        if is_lazy:
            batch_list = []
            for i in sel:
                if flatten and len(shape) == 5:
                    # 展平模式：把 5D 映射到单个 (L,L) 样本
                    h, t, c = idx_to_trip(i)
                    batch_list.append(np.array(cfg[h, t, c]))
                else:
                    # 非展平：按第一维批处理
                    # 5D -> (n_T, n_c, L, L)，3D -> (L, L)
                    batch_list.append(np.array(cfg[int(i)]))
            batch_arr = np.stack(batch_list, axis=0)
        else:
            batch_arr = arr[sel]
        yield {'configs': batch_arr, 'indices': sel}

# -----------------------------------------------------------------------------
# 合并数据集（在相同 (h,T) 网格下，沿 n_cfg 维拼接）
# -----------------------------------------------------------------------------
def _assert_same_grid(ds_list: Sequence[Dict[str, Any]]):
    T0 = np.asarray(ds_list[0].get('temperatures', []))
    H0 = np.asarray(ds_list[0].get('fields', []))
    for ds in ds_list[1:]:
        if not (np.array_equal(np.asarray(ds.get('temperatures', [])), T0) and np.array_equal(np.asarray(ds.get('fields', [])), H0)):
            raise ValueError("温度/磁场网格不一致，无法直接合并；请先对齐。")

def _ensure_5d(arr: np.ndarray) -> np.ndarray:
    """如果传入 3D (N,L,L)，把它扩展为 (1,1,N,L,L) 以便按 (n_h,n_T,n_cfg,L,L) 统一处理。"""
    if arr.ndim == 3:
        N, L, _ = arr.shape
        return arr.reshape(1, 1, N, L, L)
    if arr.ndim == 5:
        return arr
    raise ValueError("期望 3D 或 5D 数组以用于合并")

def merge_datasets(datasets: List[Dict[str, Any]], save_path: Optional[Union[str, Path]] = None,
                   **h5_kwargs) -> Dict[str, Any]:
    """合并多个数据集（在相同 (h,T) 网格下，沿 n_cfg 维拼接）。"""
    if len(datasets) == 0:
        raise ValueError("空数据集列表。")
    _assert_same_grid(datasets)

    cfgs = []
    for ds in datasets:
        c = ds.get('configs')
        arr = np.asarray(c, dtype=np.int8)  # 统一 dtype，避免隐式 upcast
        arr5 = _ensure_5d(arr)  # 统一为 5D
        cfgs.append(arr5)

    first_L = cfgs[0].shape[-1]
    for c in cfgs:
        if c.shape[-1] != first_L:
            raise ValueError("configs 的 L 值不一致，无法合并。")

    merged_cfg = np.concatenate(cfgs, axis=2)  # axis=2 = n_cfg
    merged = {
        'configs': merged_cfg,
        'temperatures': datasets[0].get('temperatures'),
        'fields': datasets[0].get('fields'),
        'parameters': dict(datasets[0].get('parameters', {})),
    }

    # 观测量沿 axis=2 拼接：对每个数据集单独推断其 (nh, nT, nc) 再拼接
    # energy
    e_list = []
    for ds in datasets:
        e = ds.get('energy')
        if e is None:
            continue
        e = np.asarray(e)
        if e.ndim == 3:
            e_list.append(e)
        elif e.ndim == 2:
            e_list.append(e[None, ...])  # (1, n_T, n_cfg)
        else:
            nh, nT, nc, *_ = _ensure_5d(np.asarray(ds['configs'])).shape
            e_list.append(e.reshape(nh, nT, nc))
    if e_list:
        merged['energy'] = np.concatenate(e_list, axis=2)

    # magnetization
    m_list = []
    for ds in datasets:
        m = ds.get('magnetization')
        if m is None:
            continue
        m = np.asarray(m)
        if m.ndim == 3:
            m_list.append(m)
        elif m.ndim == 2:
            m_list.append(m[None, ...])
        else:
            nh, nT, nc, *_ = _ensure_5d(np.asarray(ds['configs'])).shape
            m_list.append(m.reshape(nh, nT, nc))
    if m_list:
        merged['magnetization'] = np.concatenate(m_list, axis=2)

    merged['parameters']['n_configs'] = int(merged_cfg.shape[2])

    if save_path is not None:
        save_configs_hdf5(merged, save_path, **h5_kwargs)
    return merged

# -----------------------------------------------------------------------------
# 验证 / 统计
# -----------------------------------------------------------------------------
def _check_spin_values(configs: np.ndarray, sample: int = 32) -> Optional[str]:
    """快速检查自旋是否属于 {-1, +1}（小数据全量，大数据抽样）。"""
    def ok(x):
        return np.all((x == -1) | (x == 1))
    arr = np.asarray(configs)
    if arr.size <= 1_000_000:
        if not ok(arr):
            uniq = np.unique(arr)
            return f"构型值不合法: {uniq}"
        return None
    rng = np.random.default_rng(0)
    if arr.ndim == 5:
        n_h, n_T, n_c, _, _ = arr.shape
        for _ in range(sample):
            h = int(rng.integers(0, n_h))
            t = int(rng.integers(0, n_T))
            c = int(rng.integers(0, n_c))
            x = arr[h, t, c]
            if not ok(x):
                return "构型值不合法（抽样检测未通过）"
    else:
        N = arr.shape[0]
        for _ in range(sample):
            i = int(rng.integers(0, N))
            x = arr[i]
            if not ok(x):
                return "构型值不合法（抽样检测未通过）"
    return None

def validate_dataset(dataset: Dict[str, Any], verbose: bool = False) -> Tuple[bool, List[str]]:
    """
    基本一致性检查（尽量高效，避免 OOM）：
      - 支持 5D 或 3D configs（或 lazy h5py.Dataset）
      - 检查形状、temps/fields 与 configs 的一致性（宽松）
      - 抽样检查 spin 值
    返回 (is_valid, issues)
    """
    issues: List[str] = []

    if 'configs' not in dataset:
        issues.append("缺少字段: configs")
        if verbose:
            for e in issues:
                print("[validate_dataset]", e)
        return False, issues

    cfg = dataset['configs']
    if hasattr(cfg, 'shape') and not isinstance(cfg, np.ndarray):
        shape = tuple(cfg.shape)
        ndim = len(shape)
    else:
        arr = np.asarray(cfg)
        shape = arr.shape
        ndim = arr.ndim

    if ndim == 5:
        n_h, n_T, n_cfg, L1, L2 = shape
    elif ndim == 3:
        N, L1, L2 = shape
        n_h, n_T, n_cfg = 1, 1, N
    else:
        issues.append(f"configs 维度应为 3 或 5，得到 {ndim}")
        if verbose:
            for e in issues:
                print("[validate_dataset]", e)
        return False, issues

    if L1 != L2:
        issues.append("configs 最后两维应为方阵 (L, L)")

    T = dataset.get('temperatures', None)
    H = dataset.get('fields', None)
    if T is not None:
        try:
            Tarr = np.asarray(T)
            if Tarr.ndim != 1:
                issues.append("temperatures 应为一维数组")
            else:
                if Tarr.size != n_T and n_T != 1:
                    issues.append(f"temperatures 长度 {Tarr.size} 与 configs 中 n_T={n_T} 不一致")
        except Exception:
            issues.append("无法解析 temperatures")
    if H is not None:
        try:
            Harr = np.asarray(H)
            if Harr.ndim != 1:
                issues.append("fields 应为一维数组")
            else:
                if Harr.size != n_h and n_h != 1:
                    issues.append(f"fields 长度 {Harr.size} 与 configs 中 n_h={n_h} 不一致")
        except Exception:
            issues.append("无法解析 fields")

    # energy / magnetization 形状（可选）
    def _check(name, expect_shape):
        if name in dataset and dataset[name] is not None:
            arr2 = np.asarray(dataset[name])
            if arr2.shape != expect_shape:
                issues.append(f"{name} 形状应为 {expect_shape}，得到 {arr2.shape}")
    _check("energy", (n_h, n_T, n_cfg))
    _check("magnetization", (n_h, n_T, n_cfg))

    params = dataset.get('parameters', {}) or {}
    if 'L' in params:
        try:
            if int(params['L']) != int(L1):
                issues.append(f"parameters.L={params['L']} 与 configs L={L1} 不一致")
        except Exception:
            issues.append("parameters.L 类型不可解析为 int")
    if 'n_configs' in params:
        try:
            if int(params['n_configs']) != int(n_cfg):
                issues.append(f"parameters.n_configs={params['n_configs']} 与 configs n_cfg={n_cfg} 不一致")
        except Exception:
            issues.append("parameters.n_configs 类型不可解析为 int")

    # spin value sampling check（对 lazy dataset 做抽样）
    try:
        if hasattr(cfg, 'shape') and not isinstance(cfg, np.ndarray):
            arr_shape = tuple(cfg.shape)
            rng = np.random.default_rng(0)
            if len(arr_shape) == 5:
                n_h0, n_T0, n_c0, _, _ = arr_shape
                for _ in range(32):
                    h = int(rng.integers(0, n_h0))
                    t = int(rng.integers(0, n_T0))
                    c = int(rng.integers(0, n_c0))
                    sample = np.array(cfg[h, t, c])
                    if not np.all((sample == -1) | (sample == 1)):
                        issues.append("构型值抽样检测不通过（非 {-1,+1}）")
                        break
            else:
                N0 = arr_shape[0]
                for _ in range(32):
                    i = int(rng.integers(0, N0))
                    sample = np.array(cfg[i])
                    if not np.all((sample == -1) | (sample == 1)):
                        issues.append("构型值抽样检测不通过（非 {-1,+1}）")
                        break
        else:
            msg = _check_spin_values(cfg)
            if msg:
                issues.append(msg)
    except Exception as e:
        issues.append(f"自旋值抽样检测失败: {e}")

    ok = len(issues) == 0
    if verbose:
        if ok:
            print("✓ 数据集验证通过")
            try:
                print(DatasetInfo(dataset))
            except Exception:
                pass
        else:
            for e in issues:
                print("[validate_dataset]", e)
    return ok, issues

def compute_dataset_statistics(dataset: Dict[str, Any], sample: int = 1024) -> Dict[str, Any]:
    """计算数据集统计量；对于大型数据，spin_up_ratio 用抽样估计。"""
    cfg = dataset.get('configs')
    arr_like = cfg
    if hasattr(arr_like, 'shape') and not isinstance(arr_like, np.ndarray):
        shape = tuple(arr_like.shape)
        if len(shape) == 5:
            n_h, n_T, n_c, L, _ = shape
            N = n_h * n_T * n_c
        else:
            N = shape[0]
        rng = np.random.default_rng(0)
        nprobe = min(sample, N)
        idx = rng.choice(N, size=nprobe, replace=False)
        up_count = 0
        total = 0
        for i in idx:
            if len(shape) == 5:
                h = i // (n_T * n_c)
                rem = i % (n_T * n_c)
                t = rem // n_c
                c = rem % n_c
                arr = np.array(arr_like[h, t, c])
            else:
                arr = np.array(arr_like[int(i)])
            up_count += int(np.sum(arr == 1))
            total += arr.size
        spin_up_ratio = float(up_count) / max(1, total)
        shape_repr = shape
        dtype = str(getattr(arr_like, 'dtype', 'unknown'))
        size_mb = None
    else:
        arr = np.asarray(arr_like)
        shape_repr = arr.shape
        dtype = str(arr.dtype)
        size_mb = float(arr.nbytes / 1e6)
        spin_up_ratio = float(np.mean(arr == 1))
    stats = {
        'configs': {
            'shape': tuple(shape_repr),
            'dtype': dtype,
            'size_mb': size_mb,
            'spin_up_ratio': spin_up_ratio,
        },
        'parameters': dataset.get('parameters', {}),
    }
    if 'energy' in dataset and dataset.get('energy') is not None:
        e = np.asarray(dataset['energy'])
        stats['energy'] = {'mean': float(np.mean(e)), 'std': float(np.std(e)),
                           'min': float(np.min(e)), 'max': float(np.max(e))}
    if 'magnetization' in dataset and dataset.get('magnetization') is not None:
        m = np.asarray(dataset['magnetization'])
        stats['magnetization'] = {'mean': float(np.mean(np.abs(m))), 'std': float(np.std(m)),
                                  'min': float(np.min(m)), 'max': float(np.max(m))}
    return stats

def print_dataset_summary(dataset: Dict[str, Any]) -> None:
    s = compute_dataset_statistics(dataset)
    print("\n" + "="*70)
    print("数据集摘要")
    print("="*70)
    print(f"\n构型信息:")
    print(f"  形状: {s['configs']['shape']}  dtype: {s['configs']['dtype']}")
    if s['configs']['size_mb'] is not None:
        print(f"  大小: {s['configs']['size_mb']:.1f} MB")
    print(f"  自旋向上比例 (估计): {s['configs']['spin_up_ratio']:.3f}")
    if 'energy' in s:
        print(f"\n能量分布:  均值 {s['energy']['mean']:.4f}  标准差 {s['energy']['std']:.4f}  "
              f"范围 [{s['energy']['min']:.4f}, {s['energy']['max']:.4f}]")
    if 'magnetization' in s:
        print(f"\n磁化强度分布:  平均|M| {s['magnetization']['mean']:.4f}  标准差 {s['magnetization']['std']:.4f}  "
              f"范围 [{s['magnetization']['min']:.4f}, {s['magnetization']['max']:.4f}]")
    print(f"\n参数:")
    for k, v in (s['parameters'] or {}).items():
        print(f"  {k}: {v}")
    print("="*70)

# -----------------------------------------------------------------------------
# 如果作为脚本运行，执行简单示例 / 测试
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print("config_io.py 修订版 演示")
    L = 16
    n_h, n_T, n_c = 2, 3, 5
    dummy = {
        'configs': np.random.choice([-1, 1], size=(n_h, n_T, n_c, L, L), dtype=np.int8),
        'energy':  np.random.randn(n_h, n_T, n_c).astype(np.float32),
        'magnetization': np.random.randn(n_h, n_T, n_c).astype(np.float32),
        'temperatures': np.linspace(2.0, 2.5, n_T).astype(np.float32),
        'fields': np.linspace(-1.0, 1.0, n_h).astype(np.float32),
        'parameters': {'L': L, 'n_configs': n_c, 'algorithm': 'metropolis'}
    }

    ok, issues = validate_dataset(dummy, verbose=True)
    assert ok, issues

    save_configs_hdf5(dummy, 'demo_data.h5', compression='lzf')
    ds = load_configs_hdf5('demo_data.h5', verbose=True)

    export_for_pytorch(dummy, 'pytorch_data', split_ratio=0.8, dtype='uint8', seed=7)
    tr, md = load_pytorch_dataset('pytorch_data', 'train')

    merged = merge_datasets([dummy, dummy.copy()])
    print_dataset_summary(merged)

