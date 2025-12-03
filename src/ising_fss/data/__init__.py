# -*- coding: utf-8 -*-
"""
数据管理层
==========

提供构型数据的 I/O、转换、合并与验证功能。

子模块
------
- config_io: HDF5/NPZ 读写与 PyTorch 导出
- data_manager: 数据合并与编排

示例
----
>>> from ising_fss.data import save_configs_hdf5, load_configs_hdf5
>>> dataset = {'configs': configs, 'temperatures': temps, 'fields': fields}
>>> save_configs_hdf5(dataset, 'output.h5', compression='gzip')
>>> loaded = load_configs_hdf5('output.h5')
"""

# ising_fss/data/__init__.py
from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["config_io", "data_manager"]

_lazy = {
    "config_io": ".config_io",
    "data_manager": ".data_manager",
}

def __getattr__(name: str):
    if name in _lazy:
        mod = import_module(_lazy[name], __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"{__name__} has no attribute {name!r}")

def __dir__():
    return sorted(list(__all__))

if TYPE_CHECKING:
    from . import config_io, data_manager 

#  # ============================================================================
#  # HDF5 I/O
#  # ============================================================================
#  from .config_io import (
#      # 保存/加载
#      save_configs_hdf5,
#      load_configs_hdf5,
#      load_configs_lazy,
#
#      # NPZ
#      save_configs_npz,
#      load_configs_npz,
#
#      # PyTorch 导出
#      export_for_pytorch,
#      load_pytorch_dataset,
#
#      # 数据集操作
#      split_dataset,
#      merge_datasets,
#      validate_dataset,
#      compute_dataset_statistics,
#      print_dataset_summary,
#
#      # 工具
#      augment_configs,
#      batch_iterator,
#
#      # 类
#      H5LazyDataset,
#      DatasetInfo,
#  )
#
#  # ============================================================================
#  # 数据管理与合并
#  # ============================================================================
#  from .data_manager import (
#      # 合并工具
#      merge_h5_files,
#      merge_h5_files_smart,
#      merge_npz_files,
#      merge_scalars_jsons,
#
#      # 原子写入
#      atomic_write_bytes,
#
#      # 锁机制
#      try_acquire_lock,
#      release_lock,
#
#      # 编排器
#      _orchestrate_worker_merge,
#  )
#
#  # ============================================================================
#  # 公开 API
#  # ============================================================================
#  __all__ = [
#      # HDF5
#      'save_configs_hdf5',
#      'load_configs_hdf5',
#      'load_configs_lazy',
#
#      # NPZ
#      'save_configs_npz',
#      'load_configs_npz',
#
#      # PyTorch
#      'export_for_pytorch',
#      'load_pytorch_dataset',
#
#      # 数据集操作
#      'split_dataset',
#      'merge_datasets',
#      'validate_dataset',
#      'compute_dataset_statistics',
#      'print_dataset_summary',
#      'augment_configs',
#      'batch_iterator',
#
#      # 类
#      'H5LazyDataset',
#      'DatasetInfo',
#
#      # 合并工具
#      'merge_h5_files',
#      'merge_h5_files_smart',
#      'merge_npz_files',
#      'merge_scalars_jsons',
#
#      # 底层工具
#      'atomic_write_bytes',
#      'try_acquire_lock',
#      'release_lock',
#      '_orchestrate_worker_merge',
#  ]
