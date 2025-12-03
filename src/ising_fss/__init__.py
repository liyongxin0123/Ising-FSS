# -*- coding: utf-8 -*-
"""
Ising Monte Carlo Finite-Size Scaling Framework
================================================

一个高性能的2D Ising模型模拟与有限尺寸标度分析工具包。

主要功能
--------
- 多算法蒙特卡洛模拟 (Metropolis/Wolff/Swendsen-Wang)
- CPU/GPU 自动调度与加速
- 副本交换蒙特卡洛 (REMC)
- 有限尺寸标度 (FSS) 分析
- 深度学习数据准备与工具
- 流式数据处理与管理

快速开始
--------
>>> import ising_fss as ifs
>>> sim = ifs.HybridREMCSimulator(L=32, T_min=2.0, T_max=2.6, num_replicas=8)
>>> sim.run(equilibration_steps=1000, production_steps=5000)
>>> results = sim.analyze()
>>> fss = ifs.FSSAnalyzer(results)
>>> Tc_est = fss.estimate_Tc()

模块组织
--------
- core: 核心算法与物理量计算
- simulation: 模拟器与并行调度
- data: 数据I/O与管理
- analysis: 统计分析与FSS工具
- visualization: 绘图与可视化
- utils: 日志与配置工具
"""

__version__ = "0.1.0"
__author__ = "Li"
__license__ = "MIT"

# ising_fss/__init__.py
from importlib import import_module, util as _import_util
from typing import TYPE_CHECKING

# ---- version ----
try:
    from importlib.metadata import version as _pkg_version  # py>=3.8
except Exception:
    _pkg_version = lambda _: "0.0.0"

try:
    __version__ = _pkg_version("ising_fss")
except Exception:
    __version__ = "0.1.0"

__all__ = [
    "core",
    "simulation",
    "data",
    "analysis",
    "visualization",
    "utils",
    "HAS_CUPY",
    "HAS_TORCH",
    "HAS_H5PY",
    "__version__",
]

def _has_module(name: str) -> bool:
    try:
        return _import_util.find_spec(name) is not None
    except Exception:
        return False

HAS_CUPY = _has_module("cupy")
HAS_TORCH = _has_module("torch")
HAS_H5PY  = _has_module("h5py")

_lazy_subpackages = {
    "core": ".core",
    "simulation": ".simulation",
    "data": ".data",
    "analysis": ".analysis",
    "visualization": ".visualization",
    "utils": ".utils",
}

def __getattr__(name: str):
    if name in _lazy_subpackages:
        mod = import_module(_lazy_subpackages[name], __name__)
        globals()[name] = mod  # cache
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(list(__all__))

if TYPE_CHECKING:  # for IDE/static type checkers only
    from . import core, simulation, data, analysis, visualization, utils  


#  # ============================================================================
#  # 核心算法层 (core)
#  # ============================================================================
#  from .core import (
#      # 单步更新
#      apply_move,
#      update_batch,
#
#      # 物理量计算
#      calculate_observables,
#      calculate_observables_batch,
#      calculate_binder_cumulant,
#      calculate_specific_heat_per_spin,
#      calculate_susceptibility_per_spin,
#
#      # 调度器
#      dispatch_move,
#      dispatch_move_batch,
#  )
#
#  # ============================================================================
#  # 模拟层 (simulation)
#  # ============================================================================
#  from .simulation import (
#      # REMC 模拟器
#      HybridREMCSimulator,
#      GPU_REMC_Simulator,
#
#      # 并行调度
#      across_L,
#      run_single_L_simulation,
#
#      # 批量任务
#      run_workers_demo,
#      single_writer_demo,
#  )
#
#  # ============================================================================
#  # 数据管理层 (data)
#  # ============================================================================
#  from .data import (
#      # HDF5 I/O
#      save_configs_hdf5,
#      load_configs_hdf5,
#      load_configs_lazy,
#
#      # NPZ I/O
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
#
#      # 数据管理
#      merge_h5_files,
#      merge_h5_files_smart,
#      merge_npz_files,
#      merge_scalars_jsons,
#
#      # 懒加载
#      H5LazyDataset,
#      DatasetInfo,
#  )
#
#  # ============================================================================
#  # 分析层 (analysis)
#  # ============================================================================
#  from .analysis import (
#      # FSS 分析
#      FSSAnalyzer,
#      PairCrossing,
#
#      # 统计工具
#      autocorrelation_time,
#      effective_sample_size,
#      estimate_error_with_autocorr,
#      blocking_analysis,
#      jackknife_error,
#      bootstrap_error,
#      moving_block_bootstrap_error,
#      estimate_block_len,
#
#      # 深度学习工具 (可选)
#      IsingDataset,
#      IsingNPZDataset,
#      IsingH5Dataset,
#      create_dataloaders,
#      create_dataloaders_from_path,
#      AugmentConfig,
#
#      # 物理量计算 (DL工具)
#      compute_order_parameter,
#      energy_density,
#      structure_factor,
#  )
#
#  # ============================================================================
#  # 可视化层 (visualization)
#  # ============================================================================
#  from .visualization import (
#      # 绘图函数
#      plot_lattice,
#      plot_observables_vs_T,
#      plot_binder_crossing,
#      plot_data_collapse,
#      plot_correlation_length,
#      plot_fss_curves,
#
#      # 样式
#      use_publication_style,
#      use_presentation_style,
#  )
#
#  # ============================================================================
#  # 工具层 (utils)
#  # ============================================================================
#  from .utils import (
#      # 日志
#      setup_logger,
#      get_logger,
#
#      # 配置
#      Config,
#      setup_directories,
#  )
#
#  # ============================================================================
#  # 公开 API (按模块分组)
#  # ============================================================================
#  __all__ = [
#      # 版本信息
#      '__version__',
#      '__author__',
#      '__license__',
#
#      # === 核心算法 ===
#      'apply_move',
#      'update_batch',
#      'calculate_observables',
#      'calculate_observables_batch',
#      'calculate_binder_cumulant',
#      'calculate_specific_heat_per_spin',
#      'calculate_susceptibility_per_spin',
#      'dispatch_move',
#      'dispatch_move_batch',
#
#      # === 模拟器 ===
#      'HybridREMCSimulator',
#      'GPU_REMC_Simulator',
#      'across_L',
#      'run_single_L_simulation',
#      'run_workers_demo',
#      'single_writer_demo',
#
#      # === 数据管理 ===
#      'save_configs_hdf5',
#      'load_configs_hdf5',
#      'load_configs_lazy',
#      'save_configs_npz',
#      'load_configs_npz',
#      'export_for_pytorch',
#      'load_pytorch_dataset',
#      'split_dataset',
#      'merge_datasets',
#      'validate_dataset',
#      'compute_dataset_statistics',
#      'merge_h5_files',
#      'merge_h5_files_smart',
#      'merge_npz_files',
#      'merge_scalars_jsons',
#      'H5LazyDataset',
#      'DatasetInfo',
#
#      # === 分析工具 ===
#      'FSSAnalyzer',
#      'PairCrossing',
#      'autocorrelation_time',
#      'effective_sample_size',
#      'estimate_error_with_autocorr',
#      'blocking_analysis',
#      'jackknife_error',
#      'bootstrap_error',
#      'moving_block_bootstrap_error',
#      'estimate_block_len',
#      'IsingDataset',
#      'IsingNPZDataset',
#      'IsingH5Dataset',
#      'create_dataloaders',
#      'create_dataloaders_from_path',
#      'AugmentConfig',
#      'compute_order_parameter',
#      'energy_density',
#      'structure_factor',
#
#      # === 可视化 ===
#      'plot_lattice',
#      'plot_observables_vs_T',
#      'plot_binder_crossing',
#      'plot_data_collapse',
#      'plot_correlation_length',
#      'plot_fss_curves',
#      'use_publication_style',
#      'use_presentation_style',
#
#      # === 工具 ===
#      'setup_logger',
#      'get_logger',
#      'Config',
#      'setup_directories',
#  ]
#
#  # ============================================================================
#  # 模块级别初始化
#  # ============================================================================
#  def _initialize():
#      """模块导入时的初始化工作"""
#      import warnings
#
#      # 检查可选依赖
#      _optional_deps = {
#          'cupy': GPU_REMC_Simulator is not None,
#          'torch': IsingDataset is not None,
#          'matplotlib': True,  # 延迟检查
#      }
#
#      # 设置默认日志（仅在用户未配置时）
#      import logging
#      if not logging.getLogger('ising_fss').handlers:
#          logging.getLogger('ising_fss').addHandler(logging.NullHandler())
#
#      return _optional_deps
#
#  # 执行初始化
#  _OPTIONAL_DEPS = _initialize()
#
#  # ============================================================================
#  # 便捷访问模块级信息
#  # ============================================================================
#  def get_info():
#      """返回包信息和可选依赖状态"""
#      return {
#          'version': __version__,
#          'author': __author__,
#          'license': __license__,
#          'optional_dependencies': _OPTIONAL_DEPS,
    #  }
