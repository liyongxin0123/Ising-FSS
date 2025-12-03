# -*- coding: utf-8 -*-
"""
工具层
======

提供日志配置与全局配置管理。

子模块
------
- logger: 日志工具
- config: 全局配置

示例
----
>>> from ising_fss.utils import setup_logger, Config
>>> logger = setup_logger('my_simulation', level='INFO')
>>> Config.setup_directories()
"""


# ising_fss/utils/__init__.py
from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["logger", "config"]

_lazy = {
    "logger": ".logger",
    "config": ".config",
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
    from . import logger, config 

#  # ============================================================================
#  # 日志工具
#  # ============================================================================
#  from .logger import (
#      setup_logger,
#      get_logger,
#  )
#
#  # ============================================================================
#  # 全局配置
#  # ============================================================================
#  from .config import (
#      Config,
#      setup_directories,
#  )
#
#  # ============================================================================
#  # 公开 API
#  # ============================================================================
#  __all__ = [
#      # 日志
#      "setup_logger",
#      "get_logger",
#      # 配置
#      "Config",
#      "setup_directories",
#  ]
#
