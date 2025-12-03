# -*- coding: utf-8 -*-
"""
分析层
======

提供统计分析、FSS 分析与深度学习工具。

子模块
------
- fss_analyzer: 有限尺寸标度分析
- statistics: 统计工具 (自相关/bootstrap/blocking)
- dl_tools: 深度学习数据集与工具

示例
----
>>> from ising_fss.analysis import FSSAnalyzer, autocorrelation_time
>>> fss = FSSAnalyzer(results)
>>> Tc_est = fss.estimate_Tc()
>>> tau = autocorrelation_time(energy_series)
"""

# ising_fss/analysis/__init__.py
from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["fss_analyzer", "statistics", "dl_tools"]

_lazy = {
    "fss_analyzer": ".fss_analyzer",
    "statistics": ".statistics",
    "dl_tools": ".dl_tools",  # needs torch/h5py (optional)
}

_dep_hints = {
    "dl_tools": "torch/h5py",
}

def __getattr__(name: str):
    if name in _lazy:
        try:
            mod = import_module(_lazy[name], __name__)
        except ModuleNotFoundError as e:
            hint = _dep_hints.get(name)
            if hint:
                raise ModuleNotFoundError(
                    f"`ising_fss.analysis.{name}` 依赖可选组件（例如 {hint}）。"
                    f"如需使用，请先安装对应依赖。"
                ) from e
            raise
        globals()[name] = mod
        return mod
    raise AttributeError(f"{__name__} has no attribute {name!r}")

def __dir__():
    return sorted(list(__all__))

if TYPE_CHECKING:
    from . import fss_analyzer, statistics, dl_tools 


#  # ============================================================================
#  # FSS 分析
#  # ============================================================================
#  from .fss_analyzer import (
#      FSSAnalyzer,
#      PairCrossing,
#  )
#
#  # ============================================================================
#  # 统计工具
#  # ============================================================================
#  from .statistics import (
#      # 自相关
#      autocorrelation_time,
#      effective_sample_size,
#      estimate_error_with_autocorr,
#
#      # 误差估计
#      blocking_analysis,
#      jackknife_error,
#      bootstrap_error,
#      moving_block_bootstrap_error,
#
#      # 工具
#      estimate_block_len,
#      windowed_average,
#  )
#
#  # ============================================================================
#  # 深度学习工具 (可选)
#  # ============================================================================
#  try:
#      from .dl_tools import (
#          # Dataset 类
#          IsingDataset,
#          IsingNPZDataset,
#          IsingH5Dataset,
#
#          # DataLoader 工厂
#          create_dataloaders,
#          create_dataloaders_from_path,
#          load_ising_dataset,
#
#          # 增强配置
#          AugmentConfig,
#
#          # 物理量计算
#          compute_order_parameter,
#          energy_density,
#          nearest_neighbor_correlations,
#          structure_factor,
#
#          # 评估工具
#          evaluate_classification,
#          evaluate_regression,
#      )
#      _HAS_DL_TOOLS = True
#  except ImportError:
#      # 占位符
#      IsingDataset = None
#      IsingNPZDataset = None
#      IsingH5Dataset = None
#      create_dataloaders = None
#      create_dataloaders_from_path = None
#      load_ising_dataset = None
#      AugmentConfig = None
#      compute_order_parameter = None
#      energy_density = None
#      nearest_neighbor_correlations = None
#      structure_factor = None
#      evaluate_classification = None
#      evaluate_regression = None
#      _HAS_DL_TOOLS = False
#
#  # ============================================================================
#  # 公开 API
#  # ============================================================================
#  __all__ = [
#      # FSS
#      'FSSAnalyzer',
#      'PairCrossing',
#
#      # 统计
#      'autocorrelation_time',
#      'effective_sample_size',
#      'estimate_error_with_autocorr',
#      'blocking_analysis',
#      'jackknife_error',
#      'bootstrap_error',
#      'moving_block_bootstrap_error',
#      'estimate_block_len',
#      'windowed_average',
#
#      # 深度学习
#      'IsingDataset',
#      'IsingNPZDataset',
#      'IsingH5Dataset',
#      'create_dataloaders',
#      'create_dataloaders_from_path',
#      'load_ising_dataset',
#      'AugmentConfig',
#      'compute_order_parameter',
#      'energy_density',
#      'nearest_neighbor_correlations',
#      'structure_factor',
#      'evaluate_classification',
#      'evaluate_regression',
#  ]
#
#  # ============================================================================
#  # 模块级别信息
#  # ============================================================================
#  def has_dl_support():
#      """检查深度学习工具是否可用"""
    #  return _HAS_DL_TOOLS
