# -*- coding: utf-8 -*-
"""
    统一调度器（dispatcher）——用于将 Ising 模型更新任务派发到 CPU/GPU 后端。

实现功能：
    - 算法名标准化（"metro"、"Metropolis"、"MH" → "metropolis_sweep"）
    - 物理硬约束检查：
        • h ≠ 0 时禁止使用 Wolff / Swendsen-Wang（破坏各态遍历）
        • GPU 后端仅支持 ``metropolis_sweep``
        • Metropolis + 周期边界要求晶格尺寸 L 为偶数
    - 后端自动选择：'auto'（优先 GPU）→ 'gpu' → 'cpu'（带 warning 回退）
    - 完全基于显式 ``replica_seeds``，拒绝任何隐式随机状态
    - 返回完整的 provenance 信息（RNG 类型、消耗量、接受率）
    - 随机性来源：**仅接受显式 replica_seed / replica_seeds**（整数或整数序列）
    - 支持 strict 模式：
       strict=True 时，若不支持的组合直接抛出异常；
       strict=False 时，会尝试回退到 CPU 并 warnings.warn。
"""
from typing import Tuple, Any, Dict, Sequence, Optional, List, Union
import warnings
import logging
import json
import os
import inspect

import numpy as np
from numpy.random import SeedSequence

logger = logging.getLogger(__name__)

# ---------------------------
# 全局常量
# ---------------------------
CLUSTER_ALGOS = {'wolff', 'swendsen_wang'}
H_TOL = 1e-12

SUPPORTED_ALGOS = {
    'cpu': {'metropolis_sweep', 'wolff', 'swendsen_wang'},
    'gpu': {'metropolis_sweep'},
}

PREFERRED_RNG = "philox"

# GPU 可用性缓存（避免重复导入副作用）
_GPU_AVAILABLE_CACHE: Optional[bool] = None

# ---------------------------
# 算法名标准化
# ---------------------------
def normalize_algo_name(name: str) -> str:
    """
    将算法名统一为小写并把连字符/空格替换为下划线。
    例如： "Swendsen-Wang" -> "swendsen_wang"
    """
    return str(name).strip().lower().replace('-', '_').replace(' ', '_')

# ---------------------------
# GPU 可用性检测（带缓存）
# ---------------------------
def gpu_available(force_refresh: bool = False) -> bool:
    """
    检测 GPU 后端是否可用并缓存结果。

    逻辑：
      - 尝试导入 gpu_algorithms 模块；
      - 若模块存在并提供 gpu_available() 函数，则调用并返回其布尔值；
      - 若导入失败或内部异常，返回 False（保守策略）。
    参数：
      - force_refresh: 若为 True 则强制重新检测（忽略缓存）。
    """
    global _GPU_AVAILABLE_CACHE
    if _GPU_AVAILABLE_CACHE is not None and not force_refresh:
        return _GPU_AVAILABLE_CACHE

    try:
        from ..core import gpu_algorithms as gpu_mod
    except Exception as e:
        logger.debug("gpu_algorithms 导入失败: %s", e)
        _GPU_AVAILABLE_CACHE = False
        return False

    if hasattr(gpu_mod, "gpu_available"):
        try:
            _GPU_AVAILABLE_CACHE = bool(gpu_mod.gpu_available())
            return _GPU_AVAILABLE_CACHE
        except Exception as e:
            logger.warning("gpu_algorithms.gpu_available() 抛出异常: %s，视为 GPU 不可用。", e)
            _GPU_AVAILABLE_CACHE = False
            return False

    logger.debug("gpu_algorithms 模块存在但未提供 gpu_available()，视为不可用。")
    _GPU_AVAILABLE_CACHE = False
    return False

# ---------------------------
# JSON / provenance 辅助
# ---------------------------
def _make_json_serializable(v):
    """
    递归地把 numpy scalars/arrays, set, frozenset 等转换成 Python 基本类型以便 JSON 序列化。
    其它对象尝试 json.dumps，失败则使用 str(v) 作为回退。
    """
    # 基本类型直接返回
    if v is None or isinstance(v, (str, bool, int, float)):
        return v
    # numpy scalars
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    # numpy arrays
    if isinstance(v, np.ndarray):
        try:
            return v.tolist()
        except Exception:
            # flatten then convert each element
            return [_make_json_serializable(x) for x in v.flatten().tolist()]
    # set-like -> 有序 List（保证可重复性）
    if isinstance(v, (set, frozenset)):
        return [_make_json_serializable(x) for x in sorted(list(v), key=lambda x: str(x))]
    # list/tuple
    if isinstance(v, (list, tuple)):
        return [_make_json_serializable(x) for x in v]
    # dict
    if isinstance(v, dict):
        return {str(k): _make_json_serializable(val) for k, val in v.items()}
    # fallback：尝试 json.dumps
    try:
        json.dumps(v)
        return v
    except Exception:
        return str(v)

def write_provenance(out_base: str, prov_obj: Dict):
    """
    尝试调用 data_manager.save_provenance(path, obj)；若不可用则回退到写入本地 <out_base>.provenance.json。
    out_base 可以是路径或基名（例如 output.h5）。
    """
    serial = _make_json_serializable(prov_obj)
    try:
        from ..data import data_manager as dm
        #  dm = importlib.import_module("../data/data_manager", package=__package__)
        if hasattr(dm, "save_provenance"):
            try:
                # 优先期望 new interface: save_provenance(path, obj)
                dm.save_provenance(out_base, serial)
                return
            except TypeError:
                # 兼容老接口：save_provenance(obj)
                try:
                    dm.save_provenance(serial)
                    return
                except Exception as e:
                    logger.warning("data_manager.save_provenance 兼容调用失败: %s", e)
            except Exception as e:
                logger.warning("data_manager.save_provenance 调用失败: %s", e)
    except Exception:
        # data_manager 不可用 -> 后续回退写本地文件
        pass

    # 回退：写入本地 JSON 文件
    try:
        bn = out_base or "provenance"
        if isinstance(bn, str) and bn.endswith(".h5"):
            bn = bn[:-3]
        p = str(bn) + ".provenance.json"
        dirname = os.path.dirname(p)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(serial, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning("write_provenance: 写入本地 JSON 失败: %s", e)

# ---------------------------
# 随机种子派生（SeedSequence）
# ---------------------------
def make_replica_seeds(master_seed: int, n_replicas: int) -> List[int]:
    """
    使用 numpy.random.SeedSequence.spawn 从 master_seed 派生 n_replicas 个子种子。
    返回值为截断至 64 位的整数列表（取低 64 位），以便跨平台一致性。

    另外：这个函数保证在派生过程中收集每个 child 的生成状态（seed_info），
    便于把 provenance 写成可重建形式（即可以在恢复时重建 SeedSequence）。
    """
    if master_seed is None:
        raise ValueError("make_replica_seeds 需要提供 master_seed 参数。")
    ss_root = SeedSequence(int(master_seed))
    children = ss_root.spawn(int(n_replicas))
    replica_seeds: List[int] = []
    # Note: seed_info collects small integer arrays that allow reconstruction if needed
    # e.g., for provenance we may store the generate_state(4) output per child.
    for child in children:
        try:
            # generate_state(4) 返回 ndarray（32-bit ints），取前 1-4 个作为 entropy 表示
            st = child.generate_state(4)
            # choose a canonical representation: combine 4x32 bits -> 128-bit conceptual value,
            # but we store low 64 bits for use as a numeric seed and keep the full ints for provenance.
            low64 = int(st[0]) & 0xFFFFFFFFFFFFFFFF
            replica_seeds.append(low64)
        except Exception:
            # 最后保底：使用 entropy 或 hash
            try:
                ent = getattr(child, "entropy", None)
                if ent is not None:
                    # entropy 可能是数组或标量
                    if hasattr(ent, "__len__"):
                        low64 = int(int(ent[0]) & 0xFFFFFFFFFFFFFFFF)
                    else:
                        low64 = int(int(ent) & 0xFFFFFFFFFFFFFFFF)
                else:
                    low64 = (int(master_seed) ^ 0x9e3779b97f4a7c15) & 0xFFFFFFFFFFFFFFFF
            except Exception:
                low64 = (int(master_seed) ^ 0x9e3779b97f4a7c15) & 0xFFFFFFFFFFFFFFFF
            replica_seeds.append(low64)
    return replica_seeds

def make_replica_seed_info(master_seed: int, n_replicas: int) -> Dict[str, Any]:
    """
    返回用于 provenance 的详细 seed info：
      {
         "master_seed": ...,
         "derivation": "SeedSequence.spawn",
         "children": [ {"state_ints": [...], "low64": ...}, ... ]
      }
    便于写入 meta 并在恢复时重建 SeedSequence（若需要）。
    """
    ss_root = SeedSequence(int(master_seed))
    children = ss_root.spawn(int(n_replicas))
    children_info = []
    for child in children:
        try:
            st = child.generate_state(4)
            children_info.append({
                "state_ints": [int(x) for x in st.tolist()],
                "low64": int(int(st[0]) & 0xFFFFFFFFFFFFFFFF)
            })
        except Exception:
            ent = getattr(child, "entropy", None)
            if ent is None:
                ent_repr = None
            else:
                try:
                    ent_repr = list(ent)
                except Exception:
                    ent_repr = str(ent)
            children_info.append({
                "state_ints": None,
                "entropy": ent_repr,
                "low64": None
            })
    return {
        "master_seed": int(master_seed),
        "derivation": "SeedSequence.spawn",
        "children": children_info
    }

# ---------------------------
# BitGenerator 工厂
# ---------------------------
def get_bitgen_factory(name: Optional[str] = None):
    """
    返回 (canonical_name, BitGeneratorClass) 或 (None, None) 如果不可用。
    canonical_name 使用小写标准名，例如 'philox','pcg64','sfc64'。
    """
    nm = (name or PREFERRED_RNG).strip().lower()
    if nm == "philox":
        try:
            from numpy.random import Philox
            return "philox", Philox
        except Exception as e:
            logger.debug("Philox import failed: %s", e)
            return None, None
    if nm in ("pcg64", "pcg64dxsm"):
        try:
            from numpy.random import PCG64
            return "pcg64", PCG64
        except Exception:
            try:
                from numpy.random import PCG64DXSM  # type: ignore
                return "pcg64dxsm", PCG64DXSM
            except Exception as e2:
                logger.debug("PCG64 / PCG64DXSM import failed: %s", e2)
                return None, None
    if nm in ("sfc64", "sfc64_"):
        try:
            from numpy.random import SFC64
            return "sfc64", SFC64
        except Exception as e:
            logger.debug("SFC64 import failed: %s", e)
            return None, None
    return None, None

# ---------------------------
# provenance builder
# ---------------------------
def make_provenance_for_job(job_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    构造任务级别的 provenance 字典（包含 job_info、numpy 版本、支持算法等）。
    如果 job_info 中包含 master_seed / n_replicas，则会补充 replica seed info。
    """
    prov: Dict[str, Any] = {"dispatcher": {}}
    prov["dispatcher"]["job_info"] = _make_json_serializable(job_info)
    prov["dispatcher"]["numpy_version"] = np.__version__
    try:
        prov["dispatcher"]["supported_algos"] = {k: sorted(list(v)) for k, v in SUPPORTED_ALGOS.items()}
    except Exception:
        prov["dispatcher"]["supported_algos"] = list(SUPPORTED_ALGOS.keys())
    prov["dispatcher"]["rng_algo_recommendation"] = job_info.get("rng_recommendation", PREFERRED_RNG)

    # 如果 job_info 指定 master_seed 和 n_replicas，则把派生细节也写入 provenance
    try:
        master = job_info.get("master_seed", None)
        nrep = job_info.get("n_replicas", None)
        if master is not None and nrep is not None:
            prov["dispatcher"]["replica_seed_info"] = make_replica_seed_info(master, int(nrep))
    except Exception as e:
        logger.debug("构造 replica_seed_info 失败: %s", e)

    return prov

# ---------------------------
# 辅助：判定序列/标量
# ---------------------------
def _is_sequence_like(x) -> bool:
    """
    判定是否为序列（list/tuple/numpy array），但排除字符串/bytes 与 numpy 标量.
    Treat 0-d numpy arrays (shape==()) as scalar-like (not sequence).
    """
    if x is None:
        return False
    if isinstance(x, (str, bytes)):
        return False
    # numpy scalar (np.float64, np.int64 etc.) should be treated as scalar
    if isinstance(x, (np.generic, )):
        return False
    # numpy ndarray: treat 0-d as scalar, others as sequence
    if isinstance(x, np.ndarray):
        if x.shape == ():
            return False
        return True
    return isinstance(x, (list, tuple))

def _is_scalar_like(x) -> bool:
    """
    判定是否为标量（python int/float 或 numpy 0-d array / numpy scalar）。
    """
    if x is None:
        return False
    if isinstance(x, (str, bytes)):
        return False
    if isinstance(x, (int, float, np.floating, np.integer)):
        return True
    if isinstance(x, np.ndarray) and x.shape == ():
        return True
    # numpy scalar types
    if isinstance(x, np.generic):
        return True
    return False

# ---------------------------
# helper: normalize MoveInfo / meta to dict
# ---------------------------
def _meta_to_dict(m):
    """
    将可能是 dataclass (MoveInfo) 或 dict 转为普通 dict（字段名->值），以便上层聚合/JSON化。
    """
    if m is None:
        return {}
    if isinstance(m, dict):
        return dict(m)
    # dataclass-like or object with __dict__
    try:
        return dict(vars(m))
    except Exception:
        # try to introspect public attributes
        d = {}
        for k in dir(m):
            if k.startswith("_"):
                continue
        try:
            for k in [a for a in dir(m) if not a.startswith("_")]:
                v = getattr(m, k)
                if not inspect.ismethod(v) and not inspect.isfunction(v):
                    d[k] = v
            return d
        except Exception:
            return {"info": str(m)}

# ---------------------------
# apply_move：单步调度接口（兼容两种签名）
# ---------------------------
def apply_move(*args, **kwargs) -> Tuple[Any, Dict]:
    """
    统一 apply_move 接口，兼容：
      - apply_move(spins, beta=..., replica_seed=..., algo=..., backend=..., ...)
      - apply_move(beta, spins, replica_seed=..., algo=..., backend=..., ...)

    关键 keyword 参数： replica_seed, h, backend, strict, out_h5
    返回： (spins_out, info_dict)
    """
    # defaults
    algo = kwargs.pop("algo", 'metropolis_sweep')
    backend = kwargs.pop("backend", 'auto')
    strict = bool(kwargs.pop("strict", True))
    out_h5 = kwargs.pop("out_h5", None)
    h = float(kwargs.pop("h", 0.0))
    replica_seed = kwargs.pop("replica_seed", None)

    if len(args) == 0:
        raise TypeError("apply_move requires at least spins and beta (either as positional or keyword args).")

    # 解析位置参数风格（更稳健）：判断首参是标量还是 lattice
    first = args[0]

    if _is_scalar_like(first):
        # new-style: (beta, spins, ...)
        beta = first
        if len(args) >= 2:
            spins = args[1]
        else:
            raise TypeError("apply_move: spins must be provided as second positional argument for new-style call.")
    else:
        # old-style: (spins, beta, ...)
        spins = first
        if "beta" in kwargs:
            beta = kwargs.pop("beta")
        else:
            if len(args) >= 2:
                beta = args[1]
            else:
                raise TypeError("apply_move: beta must be provided (keyword or second positional argument).")

    # 对单步调用，beta 必须是标量（不能是序列）
    if _is_sequence_like(beta):
        raise TypeError("apply_move: beta must be a scalar for single-step call; for batched calls use apply_move_batch.")
    try:
        beta = float(beta)
    except Exception as e:
        raise TypeError(f"apply_move: cannot convert beta to float: {e}") from e

    algo_norm = normalize_algo_name(algo)

    # 物理检查：簇算法在非零外场下不可用
    if algo_norm in CLUSTER_ALGOS and (h is not None) and (abs(h) > H_TOL):
        raise ValueError(f"Cluster algorithm '{algo}' is invalid with non-zero external field h={h}.")

    # 后端选择
    if backend == 'auto':
        backend = 'gpu' if gpu_available() else 'cpu'
    if backend not in SUPPORTED_ALGOS:
        raise ValueError(f"Unknown backend '{backend}'")

    # 算法支持检查
    if algo_norm not in SUPPORTED_ALGOS[backend]:
        msg = f"Algorithm '{algo_norm}' is not implemented on backend '{backend}'."
        if strict:
            raise NotImplementedError(msg)
        else:
            warnings.warn(msg + " Attempting fallback to cpu.", UserWarning)
            backend = 'cpu'

    # provenance 写入：把本次调用的关键信息写到 provenance 中（非必需）
    if out_h5:
        try:
            # build minimal job info to include in provenance
            job_info = {
                "replica_seeds": [replica_seed] if replica_seed is not None else None,
                "device": backend,
                "algo": algo_norm,
                "master_seed": kwargs.get("master_seed", None),
                "n_replicas": kwargs.get("n_replicas", None)
            }
            prov = make_provenance_for_job(job_info)
            write_provenance(out_h5, prov)
        except Exception:
            logger.debug("Failed to write per-call provenance (非致命).")

    # Dispatch
    if backend == 'gpu':
        # GPU path: batch-only on GPU module -> wrap single lattice into batch
        try:
            from ..core import gpu_algorithms as gpu_mod
        except Exception as e:
            if strict:
                raise RuntimeError(f"Failed to import gpu_algorithms: {e}") from e
            else:
                warnings.warn(f"Failed to import gpu_algorithms ({e}), falling back to cpu backend", UserWarning)
                backend = 'cpu'

    if backend == 'gpu':
        from ..core import gpu_algorithms as gpu_mod  # type: ignore
        # GPU side expects batch interface; adapt single lattice to batch of size 1
        spins_arr = np.asarray(spins, dtype=np.int8)
        batch = spins_arr.reshape((1,) + spins_arr.shape)
        # gpu expects replica_seeds as sequence
        if replica_seed is None:
            raise ValueError("apply_move (gpu): 'replica_seed' must be provided for deterministic RNG on GPU.")
        rv, info = None, None
        # prefer gpu apply_move_batch if available
        if hasattr(gpu_mod, "apply_move_batch"):
            rv = gpu_mod.apply_move_batch(
                batch,
                beta=beta,
                replica_seeds=[int(replica_seed)],
                algo=algo_norm,
                h=h,
                n_sweeps=1,
            )
            # gpu.apply_move_batch expected to return (batch_out, info)
            if isinstance(rv, tuple) and len(rv) == 2:
                out_batch, info = rv
            else:
                out_batch = rv
                info = {}
            out_latt = np.asarray(out_batch[0], dtype=np.int8)
            return out_latt, _make_json_serializable(info)
        else:
            # fallback to known gpu function name
            if hasattr(gpu_mod, "metropolis_update_batch"):
                out_batch, devc = gpu_mod.metropolis_update_batch(
                    batch,
                    beta,
                    n_sweeps=1,
                    replica_seeds=[int(replica_seed)],
                    h=h,
                )
                out_latt = np.asarray(out_batch[0], dtype=np.int8)
                # device counters -> meta
                meta = {"device_counters": _make_json_serializable(devc)}
                return out_latt, meta
            raise NotImplementedError("GPU backend lacks a single-step/batch update API.")

    # CPU path
    try:
        from ..core import algorithms as cpu_mod  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Failed to import algorithms module: {e}") from e

    # build seed->generator helper (prefer cpu_mod._seed_to_generator if present)
    seed_to_generator = getattr(cpu_mod, "_seed_to_generator", None)
    if seed_to_generator is None:
        import numpy as _np

        def seed_to_generator(s):
            if s is None:
                raise ValueError("replica_seed must not be None")
            # keep compatibility: truncate to 32 bits
            return _np.random.default_rng(int(int(s) & 0xFFFFFFFF))

    # detect if cpu_mod.apply_move supports 'rng' parameter
    supports_rng = False
    try:
        sig = inspect.signature(cpu_mod.apply_move)
        supports_rng = "rng" in sig.parameters
    except Exception:
        # fallback: we'll attempt a dynamic call later
        supports_rng = False

    # Single-lattice call: construct rng once and call cpu_mod.apply_move accordingly
    if replica_seed is None:
        raise ValueError("apply_move: 'replica_seed' must be provided for deterministic RNG on CPU backend.")
    seed_i = int(replica_seed)
    rng = seed_to_generator(seed_i)

    # Call preferred form
    try:
        if supports_rng:
            latt_out, info_obj = cpu_mod.apply_move(
                np.asarray(spins, dtype=np.int8),
                algo=algo_norm,
                beta=beta,
                rng=rng,
                h=h,
            )
        else:
            # legacy: pass replica_seed
            if not supports_rng:
                warnings.warn(
                    "cpu_mod.apply_move does not declare an 'rng' parameter; calling with 'replica_seed' for compatibility. "
                    "Consider updating cpu_mod.apply_move to accept an RNG generator.",
                    UserWarning,
                )
            latt_out, info_obj = cpu_mod.apply_move(
                np.asarray(spins, dtype=np.int8),
                algo=algo_norm,
                beta=beta,
                replica_seed=seed_i,
                h=h,
            )
    except TypeError as e:
        # defensive: if signature detection was wrong, fallback to replica_seed call
        logger.debug("apply_move: calling cpu_mod.apply_move with rng raised TypeError: %s", e)
        warnings.warn("apply_move: falling back to calling cpu_mod.apply_move with replica_seed due to TypeError.", UserWarning)
        latt_out, info_obj = cpu_mod.apply_move(
            np.asarray(spins, dtype=np.int8),
            algo=algo_norm,
            beta=beta,
            replica_seed=seed_i,
            h=h,
        )

    # normalize info to dict
    info_dict = _meta_to_dict(info_obj)
    return np.asarray(latt_out, dtype=np.int8), _make_json_serializable(info_dict)


# ---------------------------
# apply_move_batch：批量接口（兼容两种签名）
# ---------------------------
def apply_move_batch(*args, **kwargs) -> Tuple[Any, Dict]:
    """
    批量接口，支持两种常见调用方式：
    - apply_move_batch(spins_batch, beta=..., replica_seeds=..., ...)
    - apply_move_batch(beta, spins_batch, replica_seeds=..., ...)

    spins_batch shape: (R, L, L)
    replica_seeds: sequence of length R
    """
    algo = kwargs.pop("algo", 'metropolis_sweep')
    backend = kwargs.pop("backend", 'auto')
    strict = bool(kwargs.pop("strict", True))
    n_sweeps = int(kwargs.pop("n_sweeps", 1))
    replica_seeds = kwargs.pop("replica_seeds", None)
    h = float(kwargs.pop("h", 0.0))

    # 解析位置参数
    if len(args) == 0:
        raise TypeError("apply_move_batch requires at least spins_batch and beta (either positional or keyword).")

    first = args[0]
    if _is_scalar_like(first):
        # new-style: (beta, spins_batch, ...)
        beta = first
        if len(args) >= 2:
            spins_batch = args[1]
        else:
            raise TypeError("apply_move_batch: spins_batch must be provided as second positional argument in new-style call.")
    else:
        # old-style: (spins_batch, beta=...)
        spins_batch = first
        if "beta" in kwargs:
            beta = kwargs.pop("beta")
        else:
            if len(args) >= 2:
                beta = args[1]
            else:
                raise TypeError("apply_move_batch: beta must be provided (keyword or second positional arg).")

    algo_norm = normalize_algo_name(algo)

    # 物理检查：簇算法在非零外场下不可用
    if algo_norm in CLUSTER_ALGOS and (h is not None) and (abs(h) > H_TOL):
        raise ValueError(f"Cluster algorithm '{algo}' is invalid with non-zero external field h={h}.")

    # 后端选择
    if backend == 'auto':
        backend = 'gpu' if gpu_available() else 'cpu'
    if backend not in SUPPORTED_ALGOS:
        raise ValueError(f"Unknown backend '{backend}'")

    # 算法支持检查
    if algo_norm not in SUPPORTED_ALGOS[backend]:
        msg = f"Algorithm '{algo_norm}' is not implemented on backend '{backend}'."
        if strict:
            raise NotImplementedError(msg)
        else:
            warnings.warn(msg + " Attempting fallback to cpu.", UserWarning)
            backend = 'cpu'

    # ---------------- GPU 路径 ----------------
    if backend == 'gpu':
        try:
            from ..core import gpu_algorithms as gpu_mod  # type: ignore
        except Exception as e:
            if strict:
                raise RuntimeError(f"Failed to import gpu_algorithms: {e}") from e
            else:
                warnings.warn(
                    f"Failed to import gpu_algorithms ({e}), falling back to cpu backend",
                    UserWarning,
                )
                backend = 'cpu'

    if backend == 'gpu':
        from ..core import gpu_algorithms as gpu_mod  # type: ignore

        if not hasattr(gpu_mod, "metropolis_update_batch"):
            raise NotImplementedError(
                "gpu_algorithms lacks metropolis_update_batch for batch updates."
            )

        # 这里只调用实际存在的 metropolis_update_batch，
        # 按 gpu_algorithms.metropolis_update_batch 的真实签名传参。
        return gpu_mod.metropolis_update_batch(
            spins=spins_batch,
            beta=beta,
            n_sweeps=n_sweeps,
            replica_seeds=replica_seeds,
            device_counters=kwargs.get("device_counters", None),
            checkerboard=kwargs.get("checkerboard", True),
            legacy_metropolis=False,
            h=h,
            # 其余参数使用 GPU 端默认值；如需以后从 dispatcher 控制，可以再加：
            # vectorized_rng=kwargs.get("vectorized_rng", False),
            # rng_chunk_replicas=kwargs.get("rng_chunk_replicas", None),
            # sweep_start=kwargs.get("sweep_start", 0),
            # precision=kwargs.get("precision", "float32"),
        )

    # ---------------- CPU 路径 ----------------
    try:
        from ..core import algorithms as cpu_mod  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Failed to import algorithms module: {e}") from e

    # 若 cpu module 提供批量 API，则优先使用
    if hasattr(cpu_mod, "update_batch"):
        return cpu_mod.update_batch(
            spins_batch,
            beta,
            replica_seeds=replica_seeds,
            algo=algo_norm,
            h=h,
            n_sweeps=n_sweeps,
            **kwargs,
        )

    # -------- CPU fallback：逐副本循环 --------
    arr = np.asarray(spins_batch)
    if arr.ndim < 3:
        raise ValueError("apply_move_batch: spins_batch must be array with shape (R, L, L).")
    R = arr.shape[0]

    # 提前校验 replica_seeds
    if replica_seeds is None:
        raise ValueError("apply_move_batch: 'replica_seeds' must be provided for CPU-fallback batch processing.")
    replica_seeds_seq = list(replica_seeds)
    if len(replica_seeds_seq) != R:
        raise ValueError(
            f"apply_move_batch: replica_seeds length {len(replica_seeds_seq)} "
            f"does not match number of replicas R={R}."
        )

    out = np.empty_like(arr, dtype=np.int8)
    per_meta: List[Dict[str, Any]] = []

    # 解析 beta：可能是序列或标量
    if _is_sequence_like(beta):
        beta_arr = np.asarray(beta)
        # allow broadcasting scalar-like sequences of length 1
        if beta_arr.ndim != 1 or (beta_arr.shape[0] != R and beta_arr.size != 1):
            raise ValueError(
                "apply_move_batch: if beta is sequence it must have length == number of replicas (R) "
                "or be length-1 for broadcasting."
            )
        if beta_arr.size == 1:
            beta_list = [float(beta_arr.item())] * R
        else:
            beta_list = [float(x) for x in beta_arr.tolist()]
    else:
        beta_list = [float(beta)] * R

    # seed->Generator helper (prefer cpu_mod._seed_to_generator)
    seed_to_generator = getattr(cpu_mod, "_seed_to_generator", None)
    if seed_to_generator is None:
        import numpy as _np

        def seed_to_generator(s):
            if s is None:
                raise ValueError("replica_seed must not be None")
            return _np.random.default_rng(int(int(s) & 0xFFFFFFFF))

    # detect if cpu_mod.apply_move accepts rng
    supports_rng = False
    try:
        sig = inspect.signature(cpu_mod.apply_move)
        supports_rng = "rng" in sig.parameters
    except Exception:
        supports_rng = False

    # prebuild rngs (one per replica)
    rngs = []
    seeds_int = []
    for r in range(R):
        try:
            seed = int(replica_seeds_seq[r])
        except Exception:
            raise ValueError("apply_move_batch: replica_seeds must be a sequence of integers length R.")
        seeds_int.append(seed)
        try:
            rngs.append(seed_to_generator(seed))
        except Exception as e:
            raise RuntimeError(f"Failed to construct RNG for replica {r} from seed {seed}: {e}") from e

    # per-replica loop
    for r in range(R):
        latt = arr[r].astype(np.int8).copy()
        meta_acc: Dict[str, Any] = {}
        total_rng_consumed = 0
        seed = seeds_int[r]
        rng = rngs[r]

        if supports_rng:
            # call apply_move n_sweeps times reusing same Generator
            for _ in range(max(1, int(n_sweeps))):
                latt_out, info_obj = cpu_mod.apply_move(
                    latt,
                    algo=algo_norm,
                    beta=beta_list[r],
                    rng=rng,
                    h=h,
                )
                latt = np.asarray(latt_out, dtype=np.int8)
                info = _meta_to_dict(info_obj)
                # aggregate numeric fields
                for k, v in info.items():
                    if isinstance(v, (int, float)):
                        meta_acc[k] = meta_acc.get(k, 0) + v
                if "rng_consumed" in info:
                    try:
                        total_rng_consumed += int(info["rng_consumed"])
                    except Exception:
                        pass
        else:
            # fallback to legacy replica_seed calls (compatibility)
            warnings.warn(
                "cpu_mod.apply_move does not accept 'rng' parameter; falling back to calling with 'replica_seed' for each replica. "
                "This will reconstruct RNG from seed on each call and is less desirable for reproducible streams. "
                "Consider updating cpu_mod.apply_move signature.",
                UserWarning,
            )
            for _ in range(max(1, int(n_sweeps))):
                latt_out, info_obj = cpu_mod.apply_move(
                    latt,
                    algo=algo_norm,
                    beta=beta_list[r],
                    replica_seed=seed,
                    h=h,
                )
                latt = np.asarray(latt_out, dtype=np.int8)
                info = _meta_to_dict(info_obj)
                for k, v in info.items():
                    if isinstance(v, (int, float)):
                        meta_acc[k] = meta_acc.get(k, 0) + v
                if "rng_consumed" in info:
                    try:
                        total_rng_consumed += int(info["rng_consumed"])
                    except Exception:
                        pass

        out[r] = latt
        if total_rng_consumed > 0:
            meta_acc["rng_consumed"] = int(total_rng_consumed)
        per_meta.append(meta_acc)

    info = {"per_replica": per_meta}
    return out, _make_json_serializable(info)

