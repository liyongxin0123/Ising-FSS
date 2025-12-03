# -*- coding: utf-8 -*-
"""
核心算法模块
============

提供 2D Ising 模型的蒙特卡洛更新算法与物理量计算。

子模块
------
- algorithms: CPU 实现 (Metropolis/Wolff/Swendsen-Wang)
- gpu_algorithms: GPU 加速实现 (CuPy)
- dispatcher: 统一调度器 (CPU/GPU 自动选择)
- observables: 物理量计算 (能量/磁化/Binder累积量等)

示例
----
>>> from ising_fss.core import apply_move, calculate_observables
>>> lattice = np.random.choice([-1, 1], size=(32, 32))
>>> new_lattice, info = apply_move(lattice, 'metropolis_sweep', beta=0.44, replica_seed=42)
>>> obs = calculate_observables(new_lattice, h=0.0)
>>> print(obs['E_per_spin'], obs['m'])
"""


# ising_fss/core/__init__.py
from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["algorithms", "gpu_algorithms", "dispatcher", "observables"]

_lazy = {
    "algorithms": ".algorithms",
    "gpu_algorithms": ".gpu_algorithms",  # needs cupy
    "dispatcher": ".dispatcher",
    "observables": ".observables",
}

_dep_hints = {
    "gpu_algorithms": "cupy",
}

def __getattr__(name: str):
    if name in _lazy:
        try:
            mod = import_module(_lazy[name], __name__)
        except ModuleNotFoundError as e:
            hint = _dep_hints.get(name)
            if hint and (hint in str(e) or hint in getattr(e, "name", "")):
                raise ModuleNotFoundError(
                    f"`ising_fss.core.{name}` 需要可选依赖 `{hint}`。请先安装：pip install {hint}"
                ) from e
            raise
        globals()[name] = mod
        return mod
    raise AttributeError(f"{__name__} has no attribute {name!r}")

def __dir__():
    return sorted(list(__all__))

if TYPE_CHECKING:
    from . import algorithms, gpu_algorithms, dispatcher, observables  

#  # ============================================================================
#  # CPU 算法 (algorithms.py)
#  # ============================================================================
#  from .algorithms import (
#      apply_move,
#      update_batch,
#      get_algorithm,
#      normalize_algo_name,
#      MoveInfo,
#  )
#
#  # ============================================================================
#  # 物理量计算 (observables.py)
#  # ============================================================================
#  from .observables import (
#      # 单构型
#      calculate_observables,
#
#      # 批量
#      calculate_observables_batch,
#
#      # 统计量
#      calculate_binder_cumulant,
#      calculate_specific_heat_per_spin,
#      calculate_susceptibility_per_spin,
#
#      # 内部工具（高级用户可能需要）
#      _energy_total_numpy,
#      _observables_for_simulator,
#  )
#
#  # ============================================================================
#  # GPU 算法 (gpu_algorithms.py) - 可选
#  # ============================================================================
#  try:
#      from .gpu_algorithms import (
#          metropolis_update_batch,
#          init_device_counters,
#          get_and_reset_counters,
#          device_energy,
#          device_magnetization,
#      )
#      _HAS_GPU = True
#  except ImportError:
#      metropolis_update_batch = None
#      init_device_counters = None
#      get_and_reset_counters = None
#      device_energy = None
#      device_magnetization = None
#      _HAS_GPU = False
#
#  # ============================================================================
#  # 统一调度器 (dispatcher.py)
#  # ============================================================================
#  from .dispatcher import (
#      apply_move as dispatch_move,
#      apply_move_batch as dispatch_move_batch,
#      gpu_available,
#      normalize_algo_name as dispatch_normalize_algo,
#  )
#
#  # ============================================================================
#  # 公开 API
#  # ============================================================================
#  __all__ = [
#      # CPU 算法
#      'apply_move',
#      'update_batch',
#      'get_algorithm',
#      'normalize_algo_name',
#      'MoveInfo',
#
#      # 物理量
#      'calculate_observables',
#      'calculate_observables_batch',
#      'calculate_binder_cumulant',
#      'calculate_specific_heat_per_spin',
#      'calculate_susceptibility_per_spin',
#      '_energy_total_numpy',
#      '_observables_for_simulator',
#
#      # GPU 算法 (可选)
#      'metropolis_update_batch',
#      'init_device_counters',
#      'get_and_reset_counters',
#      'device_energy',
#      'device_magnetization',
#
#      # 调度器
#      'dispatch_move',
#      'dispatch_move_batch',
#      'gpu_available',
#      'dispatch_normalize_algo',
#  ]
#
#  # ============================================================================
#  # 模块级别信息
#  # ============================================================================
#  def has_gpu_support():
#      """检查是否有 GPU 支持"""
    #  return _HAS_GPU and gpu_available()
