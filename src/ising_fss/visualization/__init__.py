# -*- coding: utf-8 -*-
"""
可视化层
========

提供标准绘图函数与样式配置。

子模块
------
- plots: 绘图函数 (构型/观测量/FSS曲线)
- styles: Matplotlib 样式配置

示例
----
>>> from ising_fss.visualization import plot_lattice, use_publication_style
>>> use_publication_style()
>>> fig, ax = plot_lattice(config, title='Ising Configuration')
>>> fig.savefig('config.png', dpi=300)
"""

# ising_fss/visualization/__init__.py
from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["plots", "styles"]

_lazy = {
    "plots": ".plots",
    "styles": ".styles",
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
    from . import plots, styles
#
#  # ============================================================================
#  # 导入检查
#  # ============================================================================
#  try:
#      import matplotlib.pyplot as plt  # noqa: F401
#      _HAS_MATPLOTLIB = True
#  except Exception:
#      _HAS_MATPLOTLIB = False
#      import warnings
#
#      warnings.warn(
#          "Matplotlib not found. Visualization module will expose placeholder functions. "
#          "Install matplotlib to enable plotting: pip install matplotlib",
#          ImportWarning,
#      )
#
#  # ============================================================================
#  # 条件导入：当 matplotlib 可用时，导出真实实现；否则提供占位抛错函数
#  # ============================================================================
#  if _HAS_MATPLOTLIB:
#      from .plots import (
#          # 基础绘图
#          plot_lattice,
#          plot_observables_vs_T,
#          # FSS / 高阶绘图
#          plot_binder_crossing,
#          plot_data_collapse,
#          plot_correlation_length,
#          plot_fss_curves,
#          # 额外/诊断绘图
#          plot_magnetization_distribution,
#          plot_energy_histogram,
#          plot_autocorrelation,
#      )
#
#      from .styles import (
#          use_publication_style,
#          use_presentation_style,
#          use_default_style,
#      )
#  else:
#      # 占位符：在未安装 matplotlib 时，调用这些函数会直接报错，提示安装
#      def _not_available(*args, **kwargs):
#          raise ImportError(
#              "Matplotlib is required for visualization features. Install with: pip install matplotlib"
#          )
#
#      # 基础绘图
#      plot_lattice = _not_available
#      plot_observables_vs_T = _not_available
#
#      # FSS / 高阶绘图
#      plot_binder_crossing = _not_available
#      plot_data_collapse = _not_available
#      plot_correlation_length = _not_available
#      plot_fss_curves = _not_available
#
#      # 额外/诊断绘图
#      plot_magnetization_distribution = _not_available
#      plot_energy_histogram = _not_available
#      plot_autocorrelation = _not_available
#
#      # 样式切换
#      use_publication_style = _not_available
#      use_presentation_style = _not_available
#      use_default_style = _not_available
#
#  # ============================================================================
#  # 公开 API
#  # ============================================================================
#  __all__ = [
#      # 基础绘图
#      "plot_lattice",
#      "plot_observables_vs_T",
#      # FSS 绘图
#      "plot_binder_crossing",
#      "plot_data_collapse",
#      "plot_correlation_length",
#      "plot_fss_curves",
#      # 分布/诊断绘图
#      "plot_magnetization_distribution",
#      "plot_energy_histogram",
#      "plot_autocorrelation",
#      # 样式
#      "use_publication_style",
#      "use_presentation_style",
#      "use_default_style",
#  ]
#
#  # ============================================================================
#  # 模块级别信息
#  # ============================================================================
#  def has_matplotlib() -> bool:
#      """检查 Matplotlib 是否可用"""
#      return bool(_HAS_MATPLOTLIB)
#
