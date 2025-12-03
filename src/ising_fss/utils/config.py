# -*- coding: utf-8 -*-
"""
统一配置管理系统（支持预设、分层覆盖、严格物理一致性检查）

实现功能：
    - 自动将算法名标准化（"metro" → "metropolis_sweep"）
    - 兼容同义输入：'metropolis'/'metro' → 'metropolis_sweep'；'sw' → 'swendsen_wang' 等
    - 硬性约束：
         h ≠ 0 时禁止 Wolff/SW
         Metropolis + PBC 要求 L 为偶数
         GPU 后端仅支持 metropolis_sweep
    - 输出路径自动绑定项目根目录
    - 完整验证函数 validate_config() 返回 warnings 列表

注意：
1) 物理与实现一致性硬约束：
   - PBC + Metropolis 棋盘 ⇒ L 必须为偶数（否则立刻抛错）
   - 存在外场 (h_field ≠ 0 或 data.h_range 含非零) ⇒ 仅允许 'metropolis_sweep'
   - GPU 后端仅支持 'metropolis_sweep'；簇算法仅 CPU
2) REMC 一致性：data.use_remc=True ⇒ simulation.num_replicas ≥ 2；簇算法+REMC 发出提示
3) ENV/CLI/YAML 合并，优先级：默认/预设 < 文件 < 环境变量 < CLI --set
"""

from __future__ import annotations

import os
import sys
import json
import ast
import copy
from dataclasses import dataclass, asdict, field, replace
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

# 可选 YAML
try:
    import yaml
except Exception:
    yaml = None  # type: ignore

__all__ = [
    'Config', 'SimulationConfig', 'DataConfig', 'TrainingConfig',
    'load_config', 'save_config', 'get_preset_config',
    'load_from_env', 'merge_configs', 'validate_config', 'from_args'
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
_FLOAT_TOL = 1e-12
_SUPPORTED_ALGOS = ('metropolis_sweep', 'wolff', 'swendsen_wang')
_SUPPORTED_BACKENDS = ('cpu', 'gpu', 'auto')

def _to_serializable(obj: Any):
    """将对象递归转换为 JSON/YAML 友好格式。"""
    try:
        import numpy as _np
    except Exception:
        _np = None

    if isinstance(obj, tuple):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    if _np is not None and isinstance(obj, _np.generic):
        try:
            return obj.item()
        except Exception:
            return float(obj)
    return obj

def _deep_merge(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
    """将 d2 深度合并到 d1（原地修改 d1 并返回它）。"""
    for k, v in (d2 or {}).items():
        if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
            _deep_merge(d1[k], v)
        else:
            d1[k] = v
    return d1

def _set_by_path(d: Dict[str, Any], path: List[str], value: Any):
    """按照 path（list）在嵌套 dict 中设置 value。"""
    cur = d
    for key in path[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[path[-1]] = value

def _parse_env_value(s: str):
    """将环境变量字符串解析为 Python 值（literal_eval 优先，兼容 true/false/none）。"""
    if s is None:
        return None
    try:
        return ast.literal_eval(s)
    except Exception:
        sl = s.strip()
        sl_l = sl.lower()
        if sl_l == 'true':
            return True
        if sl_l == 'false':
            return False
        if sl_l in ('none', 'null'):
            return None
        try:
            if '.' in sl:
                return float(sl)
            return int(sl)
        except Exception:
            return sl

def _normalize_algo(name: str) -> str:
    """将各种同义算法名归一化到实现键。"""
    s = str(name).strip().lower().replace('-', '_').replace(' ', '_')
    if s in ('metro', 'metropolis', 'metropolissweep', 'metropolis_sweep'):
        return 'metropolis_sweep'
    if 'wolff' in s:
        return 'wolff'
    if ('swendsen' in s) or (s in ('sw', 'swendsen_wang', 'swendsenwang', 'sw_cluster')):
        return 'swendsen_wang'
    return s

def _is_pbc(boundary: str) -> bool:
    return str(boundary).strip().lower() in ('pbc', 'periodic', 'periodic_bc', 'periodicbc')

def _range_contains_nonzero(h_range: Optional[Tuple[float, float]]) -> bool:
    if h_range is None:
        return False
    try:
        h0, h1 = float(h_range[0]), float(h_range[1])
    except Exception:
        return True
    return not (abs(h0) <= _FLOAT_TOL and abs(h1) <= _FLOAT_TOL)

# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------
@dataclass
class SimulationConfig:
    # 晶格与温度
    L: int = 32
    T_min: float = 2.0
    T_max: float = 2.5
    num_replicas: int = 16

    # 外场（单值；若扫描外场用 DataConfig.h_range）
    h_field: float = 0.0

    # 算法与采样
    algorithm: str = 'wolff'      # 'metropolis_sweep' | 'wolff' | 'swendsen_wang'（同义词会归一化）
    boundary: str = 'pbc'         # 'pbc' | 'open'
    equilibration: int = 5000
    production: int = 10000
    exchange_interval: int = 10   # REMC 的交换周期（步）
    sampling_interval: int = 1    # 落盘/下游采样抽稀

    # 资源
    backend: str = 'cpu'          # 'cpu' | 'gpu' | 'auto'
    n_processes: Optional[int] = None
    device_id: int = 0

    # 随机种子
    seed: Optional[int] = None

    def __post_init__(self):
        # 基础数值校验
        if not (isinstance(self.L, int) and self.L > 0):
            raise ValueError(f"L must be a positive integer, got {self.L}")
        if not (float(self.T_min) < float(self.T_max)):
            raise ValueError(f"T_min must be < T_max (got {self.T_min} >= {self.T_max})")
        for name in ('num_replicas', 'equilibration', 'production', 'exchange_interval', 'sampling_interval'):
            v = getattr(self, name)
            if not (isinstance(v, int) and v >= 0):
                raise ValueError(f"{name} must be a non-negative integer (got {v})")
        if self.backend not in _SUPPORTED_BACKENDS:
            raise ValueError(f"backend must be one of {_SUPPORTED_BACKENDS}, got {self.backend}")
        if self.n_processes is not None and (not isinstance(self.n_processes, int) or self.n_processes <= 0):
            raise ValueError("n_processes must be a positive int or None")
        if not (isinstance(self.device_id, int) and self.device_id >= 0):
            raise ValueError("device_id must be a non-negative int")

        # 归一化算法名并白名单检查
        algo_norm = _normalize_algo(self.algorithm)
        self.algorithm = algo_norm
        if self.algorithm not in _SUPPORTED_ALGOS:
            raise ValueError(f"Unknown algorithm: {self.algorithm!r}. "
                             f"Use one of {_SUPPORTED_ALGOS} (synonyms accepted).")

        # 外场一致性：h ≠ 0 ⇒ 仅允许 Metropolis（棋盘 sweep）
        try:
            h_nonzero = abs(float(self.h_field)) > _FLOAT_TOL
        except Exception:
            raise ValueError("h_field must be numeric")
        if h_nonzero and self.algorithm != 'metropolis_sweep':
            raise ValueError("Detected non-zero h_field but algorithm is not 'metropolis_sweep'. "
                             "With external field, only Metropolis (checkerboard sweep) is allowed.")

        # PBC + 棋盘 Metropolis ⇒ L 必须为偶数（我们没有非棋盘 Metropolis 版本）
        if _is_pbc(self.boundary) and self.algorithm == 'metropolis_sweep' and (self.L % 2 == 1):
            raise ValueError("PBC + checkerboard Metropolis requires an EVEN L. "
                             f"Got L={self.L}. Please choose L even (e.g., 32/64/128).")

        # 后端能力约束：GPU 只支持 Metropolis；簇算法仅 CPU
        if self.backend == 'gpu' and self.algorithm != 'metropolis_sweep':
            raise ValueError("GPU backend currently supports only 'metropolis_sweep'. "
                             f"Got backend='gpu', algorithm='{self.algorithm}'.")
        if self.backend == 'auto' and self.algorithm in ('wolff', 'swendsen_wang'):
            # 自动降级为 CPU（不抛错，便于易用）
            self.backend = 'cpu'

        # 一些合理性约束/建议
        if self.num_replicas < 2 and self.exchange_interval > 0:
            # 非 REMC 情况 exchange_interval 会被忽略；这里不强制，仅提示由 validate_config 给出
            pass

@dataclass
class DataConfig:
    L: int = 32
    T_range: Tuple[float, float] = (1.5, 3.0)
    h_range: Optional[Tuple[float, float]] = None
    n_T: int = 50
    n_h: int = 1
    n_configs: int = 1024

    # MC 细节（可与 SimulationConfig 不同，用于批量生产）
    equilibration: int = 8192
    sampling_interval: int = 8
    use_remc: bool = False

    # 输出
    output_dir: str = 'data'
    format: str = 'hdf5'              # 'hdf5' | 'npz'
    compression: bool = True

    # 深度学习导出
    export_pytorch: bool = True
    train_split: float = 0.8
    normalize: bool = True
    export_dtype: str = 'uint8'       # 'float32' | 'uint8'

    def __post_init__(self):
        if not (isinstance(self.L, int) and self.L > 0):
            raise ValueError("data.L must be a positive integer")
        if not (isinstance(self.T_range, (list, tuple)) and len(self.T_range) == 2
                and float(self.T_range[0]) < float(self.T_range[1])):
            raise ValueError(f"T_range must be a (min, max) tuple, got {self.T_range}")
        if self.h_range is not None:
            if not (isinstance(self.h_range, (list, tuple)) and len(self.h_range) == 2
                    and float(self.h_range[0]) <= float(self.h_range[1])):
                raise ValueError(f"h_range must be (h_min, h_max) or None, got {self.h_range}")
        for name in ('n_T', 'n_h', 'n_configs', 'equilibration', 'sampling_interval'):
            v = getattr(self, name)
            if not (isinstance(v, int) and v >= 0):
                raise ValueError(f"{name} must be a non-negative integer")
        if self.format not in ('hdf5', 'npz'):
            raise ValueError("format must be 'hdf5' or 'npz'")
        if not (0.0 < float(self.train_split) < 1.0):
            raise ValueError("train_split must be in (0, 1)")
        if self.export_dtype not in ('float32', 'uint8'):
            raise ValueError("export_dtype must be 'float32' or 'uint8'")

@dataclass
class TrainingConfig:
    model_type: str = 'vae'  # 'vae' | 'cnn' | 'resnet'
    latent_dim: int = 16
    hidden_dims: List[int] = field(default_factory=lambda: [32, 64, 128])

    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    beta: float = 1.0

    optimizer: str = 'adam'  # 'adam' | 'sgd' | 'adamw'
    scheduler: Optional[str] = None  # None | 'step' | 'cosine' | 'plateau'

    augmentation: bool = True

    device: str = 'cuda'
    num_workers: int = 4  # 若需兼容 DataLoader 单进程，可允许 0

    output_dir: str = 'results'
    save_interval: int = 10
    log_interval: int = 10

    def __post_init__(self):
        if self.model_type not in ('vae', 'cnn', 'resnet'):
            raise ValueError("model_type must be one of 'vae', 'cnn', 'resnet'")
        for name in ('latent_dim', 'batch_size', 'epochs', 'num_workers', 'save_interval', 'log_interval'):
            v = getattr(self, name)
            if not (isinstance(v, int) and v >= 0):
                raise ValueError(f"{name} must be a non-negative integer")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if self.optimizer not in ('adam', 'sgd', 'adamw'):
            raise ValueError("optimizer must be one of 'adam', 'sgd', 'adamw'")
        if self.scheduler not in (None, 'step', 'cosine', 'plateau'):
            raise ValueError("scheduler must be None or one of 'step', 'cosine', 'plateau'")

@dataclass
class Config:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    project_name: str = 'ising_fss'
    verbose: bool = True
    debug: bool = False
    version: int = 2  # 升级版本号

    def to_dict(self) -> Dict[str, Any]:
        return {
            'simulation': asdict(self.simulation),
            'data': asdict(self.data),
            'training': asdict(self.training),
            'project_name': self.project_name,
            'verbose': self.verbose,
            'debug': self.debug,
            'version': self.version,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Config':
        d = d or {}
        sim_d = d.get('simulation', {}) or {}
        data_d = d.get('data', {}) or {}
        train_d = d.get('training', {}) or {}
        sim = SimulationConfig(**sim_d)
        data = DataConfig(**data_d)
        train = TrainingConfig(**train_d)
        return cls(
            simulation=sim,
            data=data,
            training=train,
            project_name=d.get('project_name', 'ising_fss'),
            verbose=bool(d.get('verbose', True)),
            debug=bool(d.get('debug', False)),
            version=int(d.get('version', 2)),
        )

    def add_path_root(self, root: str | Path) -> 'Config':
        """将相对输出路径绑定到项目根（返回新 Config，不修改原对象）。"""
        if root is None:
            return self
        root_p = Path(os.path.expandvars(os.path.expanduser(str(root)))).resolve()

        def _bind(p):
            if p is None:
                return None
            pp = Path(os.path.expandvars(os.path.expanduser(str(p))))
            if pp.is_absolute():
                return str(pp.resolve())
            return str((root_p / pp).resolve())

        new_data = replace(self.data, output_dir=_bind(self.data.output_dir))
        new_train = replace(self.training, output_dir=_bind(self.training.output_dir))
        return replace(self, data=new_data, training=new_train)

# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------
def load_config(filepath: str) -> Config:
    """从 YAML 或 JSON 文件加载配置并返回 Config 对象。"""
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    suf = p.suffix.lower()
    if suf in ('.yaml', '.yml'):
        if yaml is None:
            raise RuntimeError("PyYAML not installed; cannot load YAML config")
        with open(p, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
    elif suf == '.json':
        with open(p, 'r', encoding='utf-8') as f:
            cfg = json.load(f) or {}
    else:
        raise ValueError(f"Unsupported config file extension: {suf}")
    return Config.from_dict(cfg)

def save_config(config: Config, filepath: str, format: Optional[str] = None):
    """将 Config 保存为 YAML 或 JSON。默认根据后缀判断格式。"""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = _to_serializable(config.to_dict())
    fmt = format
    if fmt is None:
        suf = p.suffix.lower()
        if suf in ('.yaml', '.yml'):
            fmt = 'yaml'
        elif suf == '.json':
            fmt = 'json'
        else:
            fmt = 'yaml'

    if fmt == 'yaml':
        if yaml is None:
            raise RuntimeError("PyYAML not installed; cannot save YAML")
        with open(p, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    elif fmt == 'json':
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    print(f"✓ Config saved: {p}")

# -----------------------------------------------------------------------------
# Presets
# -----------------------------------------------------------------------------
def get_preset_config(name: str) -> Config:
    """返回内置预设配置的副本（deepcopy）。"""
    presets: Dict[str, Config] = {
        'quick': Config(
            simulation=SimulationConfig(
                L=16, T_min=2.0, T_max=2.5, num_replicas=8,
                h_field=0.0, algorithm='wolff', equilibration=1000, production=2000
            ),
            data=DataConfig(L=16, n_T=10, n_configs=100, equilibration=1000)
        ),
        'standard': Config(
            simulation=SimulationConfig(
                L=32, T_min=2.15, T_max=2.40, num_replicas=16,
                h_field=0.0, algorithm='wolff', equilibration=5000, production=10000
            ),
            data=DataConfig(L=32, n_T=40, n_configs=512, equilibration=4096)
        ),
        'publication': Config(
            simulation=SimulationConfig(
                L=64, T_min=2.20, T_max=2.35, num_replicas=20,
                h_field=0.0, algorithm='metropolis_sweep',
                equilibration=10000, production=20000
            ),
            data=DataConfig(
                L=64, T_range=(1.0, 5.0), h_range=(-2.0, 2.0),
                n_T=65, n_h=65, n_configs=1024, equilibration=8192, export_dtype='uint8'
            )
        ),
        'dl_training': Config(
            data=DataConfig(
                L=32, T_range=(1.5, 3.5), n_T=50, n_configs=1000,
                export_pytorch=True, export_dtype='uint8'
            ),
            training=TrainingConfig(
                latent_dim=32, batch_size=128, epochs=100, learning_rate=1e-3
            )
        ),
    }
    if name not in presets:
        raise ValueError(f"Unknown preset: {name}. Available: {list(presets.keys())}")
    return copy.deepcopy(presets[name])

# -----------------------------------------------------------------------------
# Environment variables (nested via sep, e.g., ISING__simulation__L=64)
# -----------------------------------------------------------------------------
def load_from_env(prefix: str = 'ISING', sep: str = '__') -> Dict[str, Any]:
    """
    从环境变量读取以 prefix 开头、用 sep 分层的键，返回嵌套 dict。
    例： ISING__simulation__L=64  → {'simulation': {'L': 64}}
    """
    out: Dict[str, Any] = {}
    pfx = prefix + sep
    for k, v in os.environ.items():
        if not k.startswith(pfx):
            continue
        tail = k[len(pfx):]
        if not tail:
            continue
        parts = [p for p in tail.split(sep) if p]
        if not parts:
            continue
        parsed = _parse_env_value(v)
        # 注：不强制改 key 大小写；建议外部使用小写字段名与 dataclass 匹配
        _set_by_path(out, parts, parsed)
    return out

# -----------------------------------------------------------------------------
# Merge & validate
# -----------------------------------------------------------------------------
def merge_configs(base: Config, override: Dict[str, Any]) -> Config:
    """将 override（nested dict）深度合并到 base Config 的字典表示上，并返回新的 Config。"""
    base_dict = base.to_dict()
    _deep_merge(base_dict, override or {})
    return Config.from_dict(base_dict)

def validate_config(cfg: Config) -> Tuple[bool, List[str]]:
    """
    跨块一致性检查（仅返回 issues，不抛错；构造阶段的硬约束已在 __post_init__ 完成）。
    """
    issues: List[str] = []

    # L 一致性
    if cfg.data.L != cfg.simulation.L:
        issues.append(f"data.L ({cfg.data.L}) != simulation.L ({cfg.simulation.L}) -- 推荐保持一致")

    # 训练设备与仿真后端匹配
    if cfg.simulation.backend == 'gpu' and str(cfg.training.device).lower().startswith('cpu'):
        issues.append("simulation.backend='gpu' but training.device looks like CPU")

    # 外场扫描 ⇒ 仅 Metropolis
    has_field_scan = _range_contains_nonzero(cfg.data.h_range)
    try:
        sim_h_nonzero = abs(float(cfg.simulation.h_field)) > _FLOAT_TOL
    except Exception:
        sim_h_nonzero = True
    if (sim_h_nonzero or has_field_scan) and cfg.simulation.algorithm != 'metropolis_sweep':
        issues.append("Detected external field (simulation.h_field ≠ 0 or data.h_range contains non-zero). "
                      "请将 simulation.algorithm 设为 'metropolis_sweep'。")

    # REMC 合理性
    if cfg.data.use_remc and cfg.simulation.num_replicas < 2:
        issues.append("data.use_remc=True 但 simulation.num_replicas < 2 -- REMC 至少需要 2 个副本")
    if cfg.data.use_remc and cfg.simulation.algorithm in ('wolff', 'swendsen_wang'):
        issues.append("REMC 通常与 Metropolis 配合。簇算法 + REMC 可能无效或未实现。")

    ok = len(issues) == 0
    return ok, issues

# -----------------------------------------------------------------------------
# CLI helpers
# -----------------------------------------------------------------------------
def _parse_cli_overrides(kv_list: List[str]) -> Dict[str, Any]:
    """
    解析 --set key=value（点分路径）列表，返回 nested dict。
    例：--set simulation.L=64 → {'simulation': {'L': 64}}
    """
    out: Dict[str, Any] = {}
    for kv in (kv_list or []):
        if '=' not in kv:
            raise ValueError(f"--set expects key=value pairs, got: {kv}")
        key, val = kv.split('=', 1)
        path = [p.strip() for p in key.split('.') if p.strip()]
        if not path:
            continue
        parsed = _parse_env_value(val)
        _set_by_path(out, path, parsed)
    return out

def from_args(args: Optional[List[str]] = None, env_prefix: str = 'ISING') -> Config:
    """
    从命令行加载并合并配置（优先级从低到高）:
      默认/预设 <- 文件 (--config) <- 环境变量 (--env-prefix) <- CLI --set
    支持参数:
      --preset NAME
      --config FILE
      --env-prefix PREFIX
      --set k=v   (可重复)
      --root PATH (绑定输出目录到 PATH)
    """
    import argparse
    ap = argparse.ArgumentParser(description="Load & merge configuration")
    ap.add_argument('--preset', type=str,
                    choices=['quick', 'standard', 'publication', 'dl_training'],
                    help='preset name')
    ap.add_argument('--config', type=str, help='config file (yaml|json)')
    ap.add_argument('--env-prefix', type=str, default=env_prefix, help='environment variable prefix (default ISING)')
    ap.add_argument('--set', dest='sets', action='append', default=[], help='override key=value (dot notation, can repeat)')
    ap.add_argument('--root', type=str, default=None, help='project root to bind output dirs')
    ns = ap.parse_args(args=args)

    # base config: preset 或默认
    cfg = get_preset_config(ns.preset) if ns.preset else Config()

    # 文件覆盖
    if ns.config:
        try:
            file_cfg = load_config(ns.config).to_dict()
            cfg = merge_configs(cfg, file_cfg)
        except Exception as e:
            print(f"⚠ Failed to load config file {ns.config}: {e}", file=sys.stderr)

    # 环境变量覆盖
    env_over = load_from_env(prefix=ns.env_prefix)
    if env_over:
        cfg = merge_configs(cfg, env_over)

    # CLI --set（最高优先）
    cli_over = _parse_cli_overrides(ns.sets)
    if cli_over:
        cfg = merge_configs(cfg, cli_over)

    # 绑定输出路径到 root
    if ns.root:
        cfg = cfg.add_path_root(ns.root)

    # 最终验证：仅打印 warnings
    ok, issues = validate_config(cfg)
    if not ok:
        print("⚠ Config validation warnings:", file=sys.stderr)
        for it in issues:
            print("  -", it, file=sys.stderr)
    return cfg

# -----------------------------------------------------------------------------
# Module quick demo / self-test
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print("Config module demo")
    cfg = get_preset_config('quick')
    ok, issues = validate_config(cfg)
    print("Validation:", "OK" if ok else "Issues found")
    for i in issues:
        print(" -", i)
    example_path = 'config_example.yaml'
    try:
        save_config(cfg, example_path)
        loaded = load_config(example_path)
        print("Loaded simulation.L =", loaded.simulation.L)
    except Exception as e:
        print("I/O demo failed:", e)
    merged = from_args(['--preset', 'standard', '--set', 'simulation.L=48',
                        '--set', 'simulation.algorithm=metropolis',
                        '--set', 'training.batch_size=256'])
    print("Merged algorithm =", merged.simulation.algorithm,
          "simulation.L =", merged.simulation.L,
          "training.batch_size =", merged.training.batch_size)

