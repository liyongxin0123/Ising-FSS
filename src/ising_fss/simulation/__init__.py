# -*- coding: utf-8 -*-
"""
模拟层
======

提供 REMC 模拟器、并行调度与批量任务管理。

子模块
------
- remc_simulator: CPU REMC 主控
- gpu_remc_simulator: GPU REMC 主控
- parallel: 跨 L 并行调度
- batch_runner: 批量任务管理

示例
----
>>> from ising_fss.simulation import HybridREMCSimulator
>>> sim = HybridREMCSimulator(L=32, T_min=2.0, T_max=2.6, num_replicas=8,
...                             replica_seeds=[42+i for i in range(8)])
>>> sim.run(equilibration_steps=1000, production_steps=5000)
>>> results = sim.analyze()
"""

# ising_fss/simulation/__init__.py
from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["remc_simulator", "gpu_remc_simulator", "parallel", "batch_runner"]

_lazy = {
    "remc_simulator": ".remc_simulator",
    "gpu_remc_simulator": ".gpu_remc_simulator",  # needs cupy (indirectly via core.gpu_algorithms)
    "parallel": ".parallel",
    "batch_runner": ".batch_runner",
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
    from . import remc_simulator, gpu_remc_simulator, parallel, batch_runner


#  # ============================================================================
#  # CPU REMC 模拟器
#  # ============================================================================
#  from .remc_simulator import HybridREMCSimulator
#
#  # ============================================================================
#  # GPU REMC 模拟器 (可选)
#  # ============================================================================
#  try:
#      from .gpu_remc_simulator import GPU_REMC_Simulator
#      _HAS_GPU_SIMULATOR = True
#  except ImportError:
#      GPU_REMC_Simulator = None
#      _HAS_GPU_SIMULATOR = False
#
#  # ============================================================================
#  # 并行调度
#  # ============================================================================
#  from .parallel import (
#      across_L,
#      run_single_L_simulation,
#  )
#
#  # ============================================================================
#  # 批量任务管理
#  # ============================================================================
#  from .batch_runner import (
#      run_workers_demo,
#      single_writer_demo,
#      run_worker_process,
#      make_unique_save_dir,
#  )
#
#  # ============================================================================
#  # 公开 API
#  # ============================================================================
#  __all__ = [
#      # REMC 模拟器
#      'HybridREMCSimulator',
#      'GPU_REMC_Simulator',
#
#      # 并行调度
#      'across_L',
#      'run_single_L_simulation',
#
#      # 批量任务
#      'run_workers_demo',
#      'single_writer_demo',
#      'run_worker_process',
#      'make_unique_save_dir',
#  ]
#
#  # ============================================================================
#  # 便捷函数
#  # ============================================================================
#  def create_simulator(L, T_min, T_max, num_replicas, use_gpu='auto', **kwargs):
#      """
#      工厂函数：根据可用性自动选择 CPU/GPU 模拟器
#
#      Parameters
#      ----------
#      L : int
#          格子大小
#      T_min, T_max : float
#          温度范围
#      num_replicas : int
#          副本数
#      use_gpu : str or bool
#          'auto' / 'cpu' / 'gpu' / True / False
#      **kwargs : dict
#          传递给模拟器的其他参数
#
#      Returns
#      -------
#      simulator : HybridREMCSimulator or GPU_REMC_Simulator
#      """
#      if use_gpu == 'auto':
#          use_gpu = _HAS_GPU_SIMULATOR
#      elif isinstance(use_gpu, str):
#          use_gpu = (use_gpu.lower() == 'gpu')
#
#      if use_gpu and _HAS_GPU_SIMULATOR:
#          return GPU_REMC_Simulator(L, T_min, T_max, num_replicas, **kwargs)
#      else:
#          return HybridREMCSimulator(L, T_min, T_max, num_replicas, **kwargs)
#
#
#  __all__.append('create_simulator')
