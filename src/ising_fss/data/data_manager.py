# -*- coding: utf-8 -*-
"""
分布式模拟产物流式合并引擎

实现功能：
    - 流式写入：不将整个数据集载入内存
    - 原子文件替换 + fsync 保证崩溃安全
    - 自动合并 worker 临时文件（tmp/worker_xxx/）
    - 支持 HDF5（推荐）与 NPZ 双格式输出
    - 生成完整 manifest.json + summary.json 便于审计
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# optional imports (在模块顶层尝试导入以便早期报错)
try:
    import h5py
except Exception:
    h5py = None
try:
    import numpy as np
except Exception:
    np = None
try:
    import fcntl
except Exception:
    fcntl = None

# Logger: 不在模块导入时执行 basicConfig（避免污染上层应用）
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# -----------------------------------------------------------------------------
# 原子写与简单跨进程锁
# -----------------------------------------------------------------------------
def atomic_write_bytes(path: Union[str, Path], data: bytes) -> None:
    """
    原子写字节数据：在同一目录创建临时文件，fsync 后使用 os.replace 原子替换目标文件。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dirp = str(path.parent)
    with tempfile.NamedTemporaryFile(mode="wb", dir=dirp, delete=False, suffix=".tmp") as tf:
        tmpname = tf.name
        tf.write(data)
        try:
            tf.flush()
            os.fsync(tf.fileno())
        except Exception:
            logger.debug("fsync not available for %s", tmpname, exc_info=True)
    try:
        os.replace(tmpname, str(path))
    except Exception:
        try:
            if os.path.exists(str(path)):
                os.remove(str(path))
            os.replace(tmpname, str(path))
        except Exception:
            try:
                os.remove(tmpname)
            except Exception:
                pass
            raise


def try_acquire_lock(lockfile: Union[str, Path], timeout: float = 30.0, poll: float = 0.1):
    """
    尝试获取全局文件锁。优先使用 fcntl.flock（Unix），否则退化为创建 lockdir。
    返回 lock handle（file object 或 lockdir Path）。
    超时将抛 TimeoutError。
    """
    lockfile = Path(lockfile)
    lockfile.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()

    if fcntl is not None:
        f = open(str(lockfile), "a+")
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return f
            except BlockingIOError:
                if (time.time() - start) > timeout:
                    try:
                        f.close()
                    except Exception:
                        pass
                    raise TimeoutError(f"Timeout acquiring lock {lockfile}")
                time.sleep(poll)
    else:
        lockdir = lockfile.with_suffix(".lockdir")
        while True:
            try:
                os.mkdir(str(lockdir))
                try:
                    with open(lockdir / "pid", "w") as fp:
                        fp.write(str(os.getpid()))
                except Exception:
                    logger.debug("failed to write pid in lockdir %s", lockdir, exc_info=True)
                return lockdir
            except FileExistsError:
                if (time.time() - start) > timeout:
                    raise TimeoutError(f"Timeout acquiring lockdir {lockdir}")
                time.sleep(poll)


def release_lock(lock_handle) -> None:
    """
    释放由 try_acquire_lock 返回的锁。
    """
    try:
        if hasattr(lock_handle, "fileno"):
            if fcntl is not None:
                try:
                    fcntl.flock(lock_handle, fcntl.LOCK_UN)
                except Exception:
                    logger.debug("failed to unlock fcntl handle", exc_info=True)
            try:
                lock_handle.close()
            except Exception:
                logger.debug("failed to close file handle", exc_info=True)
        else:
            try:
                pidfile = Path(lock_handle) / "pid"
                if pidfile.exists():
                    try:
                        pidfile.unlink()
                    except Exception:
                        logger.debug("failed to unlink pidfile", exc_info=True)
                os.rmdir(str(lock_handle))
            except Exception:
                try:
                    shutil.rmtree(str(lock_handle))
                except Exception:
                    logger.warning("failed to remove lockdir %s", lock_handle, exc_info=True)
    except Exception:
        logger.debug("error while releasing lock", exc_info=True)


# -----------------------------------------------------------------------------
# 类型转换 / scalars/arrays 拆分
# -----------------------------------------------------------------------------
def _to_builtin(obj: Any) -> Any:
    try:
        import numpy as _np
    except Exception:
        _np = None

    if obj is None:
        return None
    if isinstance(obj, (str, bool, int, float)):
        return obj
    if _np is not None and isinstance(obj, _np.generic):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    if _np is not None and isinstance(obj, _np.ndarray):
        try:
            return obj.tolist()
        except Exception:
            return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    try:
        return str(obj)
    except Exception:
        return None


def _split_scalars_and_arrays(d: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    scalars: Dict[str, Any] = {}
    arrays: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        try:
            if np is not None and isinstance(v, np.ndarray):
                arrays[k] = v
            else:
                scalars[k] = _to_builtin(v)
        except Exception:
            scalars[k] = _to_builtin(str(v))
    return scalars, arrays


# -----------------------------------------------------------------------------
# 保存：pickle / sharded
# -----------------------------------------------------------------------------
def _save_pickle(save_dir: Union[str, Path], all_results: Dict, compressed: bool = True, compresslevel: int = 5) -> None:
    import pickle
    import gzip

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    outp = save_dir / ("results.pkl.gz" if compressed else "results.pkl")
    data = pickle.dumps(all_results, protocol=pickle.HIGHEST_PROTOCOL)
    if compressed:
        buf = gzip.compress(data, compresslevel=compresslevel)
    else:
        buf = data
    atomic_write_bytes(outp, buf)


def _save_sharded(save_dir: Union[str, Path], all_results: Dict, npz_compressed: bool = True) -> None:
    """
    Sharded 保存：scalars 写 scalars.json, arrays 写为 per-prefix 的 .npz（或若失败写 .npy）。
    arrays_index 的 key 统一使用 "prefix/array_name"。
    """
    save_dir = Path(save_dir)
    tmp_dir = save_dir / "sharded"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    scalars_global: Dict[str, Any] = {}
    arrays_index: Dict[str, Any] = {}

    for key_L, vL in (all_results or {}).items():
        if isinstance(vL, dict):
            for key_T, record in vL.items():
                prefix = f"L={key_L}/T={key_T}"
                rec = record if isinstance(record, dict) else {"value": record}
                scalars, arrays = _split_scalars_and_arrays(rec)
                scalars_global[prefix] = scalars
                if arrays:
                    npz_path = tmp_dir / f"{prefix.replace('/', '__')}.npz"
                    try:
                        if np is None:
                            for aname, arr in arrays.items():
                                p = tmp_dir / f"{prefix.replace('/', '__')}_{aname}.txt"
                                p.write_text(str(arr))
                                arrays_index[f"{prefix}/{aname}"] = {"file": str(p.relative_to(save_dir)), "members": [aname]}
                        else:
                            if npz_compressed:
                                np.savez_compressed(str(npz_path), **arrays)
                            else:
                                np.savez(str(npz_path), **arrays)
                            arrays_index.update({f"{prefix}/{aname}": {"file": str(npz_path.relative_to(save_dir)), "members": list(arrays.keys())} for aname in arrays.keys()})
                    except Exception:
                        for aname, arr in arrays.items():
                            p = tmp_dir / f"{prefix.replace('/', '__')}_{aname}.npy"
                            try:
                                if np is not None:
                                    np.save(str(p), arr)
                                    arrays_index[f"{prefix}/{aname}"] = {"file": str(p.relative_to(save_dir)), "members": [aname]}
                                else:
                                    p.write_text(str(arr))
                                    arrays_index[f"{prefix}/{aname}"] = {"file": str(p.relative_to(save_dir)), "members": [aname]}
                            except Exception:
                                arrays_index[f"{prefix}/{aname}"] = {"file": str(p.relative_to(save_dir)), "members": [aname]}
        else:
            prefix = f"key={key_L}"
            rec = vL if isinstance(vL, dict) else {"value": vL}
            scalars, arrays = _split_scalars_and_arrays(rec)
            scalars_global[prefix] = scalars
            if arrays:
                npz_path = tmp_dir / f"{prefix.replace('/', '__')}.npz"
                try:
                    if np is None:
                        for aname, arr in arrays.items():
                            p = tmp_dir / f"{prefix.replace('/', '__')}_{aname}.txt"
                            p.write_text(str(arr))
                            arrays_index[f"{prefix}/{aname}"] = {"file": str(p.relative_to(save_dir)), "members": [aname]}
                    else:
                        if npz_compressed:
                            np.savez_compressed(str(npz_path), **arrays)
                        else:
                            np.savez(str(npz_path), **arrays)
                        arrays_index.update({f"{prefix}/{aname}": {"file": str(npz_path.relative_to(save_dir)), "members": list(arrays.keys())} for aname in arrays.keys()})
                except Exception:
                    for aname, arr in arrays.items():
                        p = tmp_dir / f"{prefix.replace('/', '__')}_{aname}.npy"
                        try:
                            if np is not None:
                                np.save(str(p), arr)
                                arrays_index[f"{prefix}/{aname}"] = {"file": str(p.relative_to(save_dir)), "members": [aname]}
                            else:
                                p.write_text(str(arr))
                                arrays_index[f"{prefix}/{aname}"] = {"file": str(p.relative_to(save_dir)), "members": [aname]}
                        except Exception:
                            arrays_index[f"{prefix}/{aname}"] = {"file": str(p.relative_to(save_dir)), "members": [aname]}

    scalars_path = save_dir / "scalars.json"
    atomic_write_bytes(scalars_path, json.dumps(scalars_global, indent=2, ensure_ascii=False).encode("utf-8"))
    arrays_path = save_dir / "arrays_index.json"
    atomic_write_bytes(arrays_path, json.dumps(arrays_index, indent=2, ensure_ascii=False).encode("utf-8"))


# -----------------------------------------------------------------------------
# 合并工具：scalars/npz/h5（保持原样 / 兼容）
# -----------------------------------------------------------------------------
def merge_scalars_jsons(worker_json_paths: Iterable[str], out_json_path: Union[str, Path], prefer: str = "max_samples") -> List[str]:
    merged: Dict[str, Any] = {}
    merged_from: List[str] = []

    for p in (worker_json_paths or []):
        pth = Path(p)
        if not pth.exists():
            continue
        try:
            rec = json.loads(pth.read_text(encoding="utf-8"))
        except Exception:
            logger.debug("failed to parse json %s", pth, exc_info=True)
            continue
        merged_from.append(str(pth))
        if isinstance(rec, dict) and any((isinstance(k, str) and (k.startswith("L=") or k.startswith("T=") or k.startswith("key="))) for k in rec.keys()):
            for key, val in rec.items():
                if key not in merged:
                    merged[key] = val
                else:
                    if prefer == "max_samples" and isinstance(val, dict) and isinstance(merged[key], dict):
                        if val.get("samples", 0) > merged[key].get("samples", 0):
                            merged[key] = val
                    else:
                        merged[key] = val
        else:
            merged[pth.stem] = rec

    out_json_path = Path(out_json_path)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(out_json_path, json.dumps(merged, indent=2, ensure_ascii=False).encode("utf-8"))
    return merged_from


def merge_npz_files(worker_npz_paths: Iterable[str], out_npz_path: Union[str, Path]) -> List[str]:
    """
    修正后的 merge_npz_files：确保 tmp 有 .npz 后缀，np.savez* 写入 tmpname，随后原子替换。
    """
    if np is None:
        raise RuntimeError("numpy required for merge_npz_files")
    merged: Dict[str, np.ndarray] = {}
    merged_from: List[str] = []
    for p in (worker_npz_paths or []):
        pth = Path(p)
        if not pth.exists():
            continue
        merged_from.append(str(pth))
        try:
            data = dict(np.load(str(pth), allow_pickle=True))
        except Exception:
            try:
                arr = np.load(str(pth), allow_pickle=True)
                data = {pth.stem: arr}
            except Exception:
                logger.debug("failed to load npz/npy %s", pth, exc_info=True)
                continue
        for k, v in data.items():
            arr = np.asarray(v)
            if k not in merged:
                merged[k] = arr
            else:
                try:
                    merged[k] = np.concatenate([merged[k], arr], axis=0)
                except Exception:
                    logger.warning("falling back to flatten concat for key %s from %s", k, pth)
                    merged[k] = np.concatenate([np.atleast_1d(merged[k]).ravel(), arr.ravel()], axis=0)

    out_npz_path = Path(out_npz_path)
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    # 使用带后缀的临时文件，保证 numpy 不会改变路径语义
    with tempfile.NamedTemporaryFile(dir=str(out_npz_path.parent), suffix=".npz", delete=False) as tf:
        tmpname = tf.name
    try:
        try:
            np.savez_compressed(tmpname, **merged)
        except Exception:
            np.savez(tmpname, **merged)
        os.replace(tmpname, str(out_npz_path))
    except Exception:
        try:
            if os.path.exists(tmpname):
                os.remove(tmpname)
        except Exception:
            pass
        logger.exception("Failed to write merged npz to %s", out_npz_path)
        raise
    return merged_from


def merge_h5_files(worker_h5_paths: Iterable[str], out_h5_path: Union[str, Path], manifest: Optional[Dict] = None) -> List[str]:
    if h5py is None:
        raise RuntimeError("h5py required for merge_h5_files")

    out_h5_path = Path(out_h5_path)
    out_h5_path.parent.mkdir(parents=True, exist_ok=True)
    merged_from: List[str] = []

    lock = None
    try:
        lock = try_acquire_lock(out_h5_path.with_suffix(".lock"), timeout=30.0)
    except Exception:
        lock = None
        logger.warning("Could not acquire lock for merge_h5_files on %s", out_h5_path)

    try:
        with h5py.File(str(out_h5_path), "a") as hf_out:
            workers_grp = hf_out.require_group("workers")
            for p in (worker_h5_paths or []):
                pth = Path(p)
                if not pth.exists():
                    continue
                merged_from.append(str(pth))
                try:
                    with h5py.File(str(pth), "r") as hf_in:
                        wname = pth.stem
                        tgt = workers_grp.require_group(wname)
                        def _copy_group(src, dst):
                            for k, v in src.attrs.items():
                                try:
                                    dst.attrs[k] = v
                                except Exception:
                                    dst.attrs[k] = str(v)
                            for name, item in src.items():
                                if isinstance(item, h5py.Dataset):
                                    if name in dst:
                                        del dst[name]
                                    try:
                                        # 对于可能较大的 dataset，优先使用 chunked copy
                                        _copy_dataset_chunked(item, dst, name)
                                    except Exception:
                                        try:
                                            dst.create_dataset(name, data=item[...])
                                        except Exception:
                                            try:
                                                dst.create_dataset(name, data=np.asarray(item[...]))
                                            except Exception:
                                                dst.create_dataset(name, data=str(item[...]))
                                    try:
                                        for ak, av in item.attrs.items():
                                            try:
                                                dst[name].attrs[ak] = av
                                            except Exception:
                                                dst[name].attrs[ak] = str(av)
                                    except Exception:
                                        pass
                                else:
                                    sub = dst.require_group(name)
                                    _copy_group(item, sub)
                        _copy_group(hf_in, tgt)
                except Exception:
                    logger.exception("failed to copy worker h5 %s", pth)
                    continue
    finally:
        if manifest is None:
            manifest = {}
        manifest.setdefault("merged_from", merged_from)
        manifest.setdefault("merged_at", datetime.utcnow().isoformat() + "Z")
        manifest_path = out_h5_path.with_suffix(".manifest.json")
        try:
            atomic_write_bytes(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"))
        except Exception:
            try:
                manifest_path.write_text(json.dumps(manifest))
            except Exception:
                logger.debug("failed to write manifest", exc_info=True)
        try:
            if lock is not None:
                release_lock(lock)
        except Exception:
            logger.debug("failed to release lock", exc_info=True)
    return merged_from


# -----------------------------------------------------------------------------
# 辅助：分块复制辅助函数
# -----------------------------------------------------------------------------
def _copy_dataset_chunked(src_ds: "h5py.Dataset", dst_group: "h5py.Group", name: str, chunk_bytes: int = 64 * 1024 * 1024):
    """
    将 src_ds 以分块方式复制到 dst_group[name] 中（dst_group 中事前不存在该 name）。
    """
    try:
        shape = src_ds.shape
        dtype = src_ds.dtype
    except Exception:
        try:
            arr = np.asarray(src_ds[...])
            shape = arr.shape
            dtype = arr.dtype
        except Exception:
            dst_group.create_dataset(name, data=str(src_ds))
            return

    try:
        dst = dst_group.create_dataset(name, shape=shape, dtype=dtype, chunks=True)
    except Exception:
        dst = dst_group.create_dataset(name, shape=shape, dtype=dtype)

    if len(shape) >= 2:
        bytes_per_entry = int(np.prod(shape[1:]) * np.dtype(dtype).itemsize)
    else:
        bytes_per_entry = int(np.dtype(dtype).itemsize)

    slices_per_chunk = max(1, int(max(1, chunk_bytes) // max(1, bytes_per_entry)))
    total = int(shape[0]) if len(shape) >= 1 else 1
    s = 0
    while s < total:
        e = min(total, s + slices_per_chunk)
        try:
            data = src_ds[s:e]
            dst[s:e] = data
        except Exception:
            for i in range(s, e):
                try:
                    dst[i] = src_ds[i]
                except Exception:
                    try:
                        dst[i] = np.asarray(src_ds[i])
                    except Exception:
                        dst[i] = 0
        s = e


# -----------------------------------------------------------------------------
# 智能合并 HDF5（尝试 axis=0 拼接兼容 datasets）
# -----------------------------------------------------------------------------
def merge_h5_files_smart(worker_h5_paths: Iterable[str], out_h5_path: Union[str, Path], manifest: Optional[Dict] = None, lock_timeout: float = 30.0) -> List[str]:
    """
    更智能的合并：只读取小样本判断兼容性并流式拼接；失败回退并记录。
    """
    if h5py is None or np is None:
        raise RuntimeError("h5py and numpy required for merge_h5_files_smart")

    MAX_CHUNK_BYTES = 64 * 1024 * 1024
    SAMPLE_ROWS = 4

    out_h5_path = Path(out_h5_path)
    out_h5_path.parent.mkdir(parents=True, exist_ok=True)
    merged_from: List[str] = []
    failed_sources: List[Dict[str, Any]] = []

    lock = None
    try:
        lock = try_acquire_lock(out_h5_path.with_suffix(".lock"), timeout=lock_timeout)
    except Exception:
        lock = None
        logger.warning("Could not acquire lock for merge_h5_files_smart on %s", out_h5_path)

    try:
        merge_targets: Dict[str, List[Tuple[Path, Optional[np.ndarray], Dict[str, Any], Optional[Tuple[int, ...]]]]] = {}
        for p in (worker_h5_paths or []):
            pth = Path(p)
            if not pth.exists():
                logger.warning("worker h5 missing: %s", pth)
                continue
            merged_from.append(str(pth))
            try:
                with h5py.File(str(pth), "r") as hf_in:
                    def _walk(g, prefix=""):
                        for name, item in g.items():
                            full = f"{prefix}/{name}" if prefix else name
                            if isinstance(item, h5py.Dataset):
                                sample = None
                                shape = None
                                try:
                                    shape = item.shape
                                    if len(shape) >= 1 and shape[0] > 0:
                                        nread = min(SAMPLE_ROWS, shape[0])
                                        try:
                                            sample = item[0:nread]
                                        except Exception:
                                            sample = None
                                    else:
                                        try:
                                            sample = item[()]
                                        except Exception:
                                            sample = None
                                except Exception:
                                    sample = None
                                    shape = None
                                merge_targets.setdefault(full, []).append((pth, sample, dict(item.attrs), shape))
                            else:
                                _walk(item, full)
                    _walk(hf_in)
            except Exception:
                logger.exception("failed to read h5 %s", pth)
                continue

        with h5py.File(str(out_h5_path), "a") as hf_out:
            for path, items in merge_targets.items():
                parts = path.split("/")
                if len(items) == 1:
                    src_file, sample, attrs, shape = items[0]
                    dst = hf_out.require_group("workers").require_group(src_file.stem)
                    cur = dst
                    for part in parts[:-1]:
                        cur = cur.require_group(part)
                    name = parts[-1]
                    if name in cur:
                        del cur[name]
                    try:
                        if shape is not None:
                            with h5py.File(str(src_file), "r") as fh_src:
                                ds = fh_src
                                for part in parts:
                                    ds = ds[part]
                                _copy_dataset_chunked(ds, cur, name)
                        else:
                            if sample is not None:
                                cur.create_dataset(name, data=np.asarray(sample))
                            else:
                                cur.create_dataset(name, data=[])
                    except Exception:
                        try:
                            cur.create_dataset(name, data=np.asarray(sample) if sample is not None else "")
                        except Exception:
                            cur.create_dataset(name, data=str(sample))
                    try:
                        for ak, av in attrs.items():
                            try:
                                cur[name].attrs[ak] = av
                            except Exception:
                                cur[name].attrs[ak] = str(av)
                    except Exception:
                        logger.debug("failed to copy attrs for %s", name, exc_info=True)
                    continue

                shapes = [it[3] for it in items]
                candidate_shapes = [s for s in shapes if s is not None and len(s) >= 1]
                if not candidate_shapes:
                    logger.warning("No shape metadata for %s; falling back to per-worker copy", path)
                    for src_file, sample, attrs, shape in items:
                        dst = hf_out.require_group("workers").require_group(src_file.stem)
                        cur = dst
                        for part in parts[:-1]:
                            cur = cur.require_group(part)
                        name = parts[-1]
                        if name in cur:
                            del cur[name]
                        try:
                            if shape is not None:
                                with h5py.File(str(src_file), "r") as fh_src:
                                    ds = fh_src
                                    for part in parts:
                                        ds = ds[part]
                                    _copy_dataset_chunked(ds, cur, name)
                            else:
                                cur.create_dataset(name, data=np.asarray(sample) if sample is not None else "")
                        except Exception:
                            try:
                                cur.create_dataset(name, data=str(sample))
                            except Exception:
                                logger.exception("failed to write fallback worker dataset %s from %s", name, src_file)
                    continue

                ref_shape = candidate_shapes[0]
                trailing = tuple(ref_shape[1:]) if len(ref_shape) > 1 else ()
                compatible = True
                for s in candidate_shapes:
                    if len(s) != len(ref_shape):
                        compatible = False
                        break
                    if len(s) > 1 and tuple(s[1:]) != trailing:
                        compatible = False
                        break

                if not compatible:
                    logger.warning("Incompatible shapes for %s; falling back to per-worker copy", path)
                    for src_file, sample, attrs, shape in items:
                        dst = hf_out.require_group("workers").require_group(src_file.stem)
                        cur = dst
                        for part in parts[:-1]:
                            cur = cur.require_group(part)
                        name = parts[-1]
                        if name in cur:
                            del cur[name]
                        try:
                            if shape is not None:
                                with h5py.File(str(src_file), "r") as fh_src:
                                    ds = fh_src
                                    for part in parts:
                                        ds = ds[part]
                                    _copy_dataset_chunked(ds, cur, name)
                            else:
                                cur.create_dataset(name, data=np.asarray(sample) if sample is not None else "")
                        except Exception:
                            try:
                                cur.create_dataset(name, data=str(sample))
                            except Exception:
                                logger.exception("failed to write fallback worker dataset %s from %s", name, src_file)
                    continue

                total_N = 0
                per_source_N: List[int] = []
                source_shapes: List[Optional[Tuple[int, ...]]] = []
                for src_file, sample, attrs, shape in items:
                    if shape is not None:
                        n_src = int(shape[0])
                        per_source_N.append(n_src)
                        source_shapes.append(shape)
                        total_N += n_src
                    else:
                        try:
                            with h5py.File(str(src_file), "r") as fh_src:
                                ds = fh_src
                                for part in parts:
                                    ds = ds[part]
                                sshape = ds.shape
                                n_src = int(sshape[0])
                                per_source_N.append(n_src)
                                source_shapes.append(sshape)
                                total_N += n_src
                        except Exception:
                            logger.exception("failed to inspect dataset shape for %s in %s", path, src_file)
                            per_source_N.append(0)
                            source_shapes.append(None)

                if total_N == 0:
                    logger.warning("Total length zero for %s; skipping", path)
                    continue

                # pick first non-None source shape as reference
                ref_src_shape = next((s for s in source_shapes if s is not None), None)
                if ref_src_shape is None:
                    full_shape = (total_N,)
                else:
                    if len(ref_src_shape) >= 2:
                        full_shape = (total_N,) + tuple(ref_src_shape[1:])
                    elif len(ref_src_shape) == 1:
                        full_shape = (total_N,)
                    else:
                        full_shape = (total_N,)

                dtype = None
                for src_file, sample, attrs, shape in items:
                    if sample is not None:
                        try:
                            sd = np.asarray(sample).dtype
                            dtype = sd
                            break
                        except Exception:
                            continue
                if dtype is None:
                    for src_file, sample, attrs, shape in items:
                        try:
                            with h5py.File(str(src_file), "r") as fh_src:
                                ds = fh_src
                                for part in parts:
                                    ds = ds[part]
                                dtype = ds.dtype
                                break
                        except Exception:
                            continue
                if dtype is None:
                    dtype = np.float32

                tgt_grp = hf_out.require_group("entries").require_group("merged")
                cur = tgt_grp
                for part in parts[:-1]:
                    cur = cur.require_group(part)
                name = parts[-1]
                if name in cur:
                    del cur[name]
                try:
                    cur.create_dataset(name, shape=full_shape, dtype=dtype, chunks=True, compression="gzip", compression_opts=4)
                except Exception:
                    cur.create_dataset(name, shape=full_shape, dtype=dtype, chunks=True)

                failed_for_dataset: List[Dict[str, Any]] = []

                offset = 0
                for idx, (src_file, sample, attrs, shape) in enumerate(items):
                    n_src = per_source_N[idx]
                    if n_src == 0:
                        offset += 0
                        continue

                    success = False
                    try:
                        with h5py.File(str(src_file), "r") as fh_src:
                            ds = fh_src
                            for part in parts:
                                ds = ds[part]
                            item_bytes = int(np.prod(ds.shape[1:]) * np.dtype(ds.dtype).itemsize) if len(ds.shape) > 1 else int(np.dtype(ds.dtype).itemsize)
                            slices_per_chunk = max(1, int(MAX_CHUNK_BYTES // max(1, item_bytes)))
                            s = 0
                            while s < n_src:
                                e = min(n_src, s + slices_per_chunk)
                                chunk = ds[s:e]
                                cur[name][offset + s: offset + e] = chunk
                                s = e
                        success = True
                    except Exception as e:
                        logger.exception("failed to stream-copy dataset %s from %s: %s", path, src_file, e)
                        rec = {"source": str(src_file), "dataset": path, "reason": str(e)}
                        failed_for_dataset.append(rec)
                        failed_sources.append(rec)

                        try:
                            dst = hf_out.require_group("workers").require_group(src_file.stem)
                            subcur = dst
                            for part in parts[:-1]:
                                subcur = subcur.require_group(part)
                            subname = parts[-1]
                            if subname in subcur:
                                del subcur[subname]
                            with h5py.File(str(src_file), "r") as fh_src2:
                                ds2 = fh_src2
                                for part in parts:
                                    ds2 = ds2[part]
                                _copy_dataset_chunked(ds2, subcur, subname)
                        except Exception:
                            try:
                                subcur.create_dataset(subname, data=np.asarray(sample) if sample is not None else "")
                            except Exception:
                                try:
                                    subcur.create_dataset(subname, data=str(sample))
                                except Exception:
                                    logger.exception("failed fallback write for %s from %s", path, src_file)

                        try:
                            placeholder_shape = (n_src,) + tuple(full_shape[1:]) if len(full_shape) > 1 else (n_src,)
                            try:
                                if np.issubdtype(cur[name].dtype, np.floating):
                                    cur[name][offset:offset + n_src] = np.full(placeholder_shape, np.nan, dtype=cur[name].dtype)
                                else:
                                    cur[name][offset:offset + n_src] = np.zeros(placeholder_shape, dtype=cur[name].dtype)
                            except Exception:
                                for i in range(n_src):
                                    try:
                                        cur[name][offset + i] = 0
                                    except Exception:
                                        try:
                                            cur[name][offset + i] = np.array(0, dtype=cur[name].dtype)
                                        except Exception:
                                            logger.debug("failed to fill single placeholder at %s[%d]", name, offset + i, exc_info=True)
                        except Exception:
                            logger.exception("failed to fill placeholder for failed source %s", src_file)
                    finally:
                        offset += n_src

                try:
                    cur[name].attrs["merged_from"] = json.dumps([str(it[0]) for it in items], ensure_ascii=False)
                except Exception:
                    try:
                        cur[name].attrs["merged_from"] = str([str(it[0]) for it in items])
                    except Exception:
                        logger.debug("failed to write merged_from attr for %s", name, exc_info=True)

                if failed_for_dataset:
                    try:
                        cur[name].attrs["failed_sources"] = json.dumps(failed_for_dataset, ensure_ascii=False)
                    except Exception:
                        cur[name].attrs["failed_sources"] = str(failed_for_dataset)

    finally:
        if manifest is None:
            manifest = {}
        manifest.setdefault("merged_from", merged_from)
        manifest.setdefault("merged_at", datetime.utcnow().isoformat() + "Z")
        if failed_sources:
            manifest["failed_sources"] = failed_sources
        manifest_path = out_h5_path.with_suffix(".manifest.json")
        try:
            atomic_write_bytes(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"))
        except Exception:
            try:
                manifest_path.write_text(json.dumps(manifest))
            except Exception:
                logger.debug("failed to write manifest", exc_info=True)
        try:
            if lock is not None:
                release_lock(lock)
        except Exception:
            logger.exception("failed to release merge_h5_files_smart lock")
    return merged_from


# -----------------------------------------------------------------------------
# 辅助：分片读取 worker 元数据（不加载整个 dataset）
# -----------------------------------------------------------------------------
def _read_worker_meta_and_h5(h5path: Path, scalars_path: Optional[Path] = None) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    读取 worker h5 的元数据和少量样本（避免加载完整 dataset）。
    返回 (meta, datasets_info) 或 (None, None) 表示失败。

    datasets_info 结构示例：
      {
        "configs": {"shape": (N,L,L), "dtype": dtype, "sample": np.ndarray or None, "exists": True},
        "magnetizations": {"shape": (N,), "dtype": dtype, "exists": True/False},
        "energies": {"shape": (N,), "dtype": dtype, "exists": True/False}
      }
    """
    meta: Dict[str, Any] = {}
    datasets: Dict[str, Any] = {}
    try:
        with h5py.File(str(h5path), "r") as fh:
            # provenance attrs
            if "provenance" in fh:
                try:
                    for k, v in fh["provenance"].attrs.items():
                        meta[k] = _to_builtin(v)
                except Exception:
                    pass
            for k, v in fh.attrs.items():
                if k not in meta:
                    meta[k] = _to_builtin(v)

            # configs or lattices: do NOT read full array; only inspect shape/dtype and small sample
            cfg_name = None
            if "configs" in fh:
                cfg_name = "configs"
            elif "lattices" in fh:
                cfg_name = "lattices"

            if cfg_name is not None and cfg_name in fh:
                try:
                    ds = fh[cfg_name]
                    sshape = ds.shape
                    sdtype = ds.dtype
                    sample = None
                    if len(sshape) >= 1 and sshape[0] > 0:
                        nread = min(4, int(sshape[0]))
                        try:
                            sample = ds[0:nread]
                        except Exception:
                            sample = None
                    datasets["configs"] = {"exists": True, "shape": tuple(sshape), "dtype": sdtype, "sample": sample, "dataset_name": cfg_name}
                except Exception:
                    datasets["configs"] = {"exists": True, "shape": None, "dtype": None, "sample": None, "dataset_name": cfg_name}
            else:
                datasets["configs"] = {"exists": False}

            # magnetizations
            if "magnetizations" in fh:
                try:
                    ds = fh["magnetizations"]
                    datasets["magnetizations"] = {"exists": True, "shape": tuple(ds.shape), "dtype": ds.dtype}
                except Exception:
                    datasets["magnetizations"] = {"exists": True, "shape": None, "dtype": None}
            else:
                datasets["magnetizations"] = {"exists": False}

            # energies
            if "energies" in fh:
                try:
                    ds = fh["energies"]
                    datasets["energies"] = {"exists": True, "shape": tuple(ds.shape), "dtype": ds.dtype}
                except Exception:
                    datasets["energies"] = {"exists": True, "shape": None, "dtype": None}
            else:
                datasets["energies"] = {"exists": False}

        # try to supplement meta from scalars.json if present
        if scalars_path is not None and scalars_path.exists():
            try:
                rec = json.loads(scalars_path.read_text(encoding="utf-8"))
                if isinstance(rec, dict):
                    for k, v in rec.items():
                        if k not in meta:
                            meta[k] = v
            except Exception:
                logger.debug("failed to read scalars json %s", scalars_path, exc_info=True)
        return meta, datasets
    except Exception:
        logger.exception("failed to open/parse worker h5 %s", h5path)
        return None, None


# -----------------------------------------------------------------------------
# 辅助：获取当前 git short sha（容错）
# -----------------------------------------------------------------------------
def safe_get_git_sha(default: str = "unknown") -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        return sha
    except Exception:
        return default


# -----------------------------------------------------------------------------
# Orchestrator（流式写入 final HDF5，避免 OOM）
# -----------------------------------------------------------------------------
def _orchestrate_worker_merge(output_base_dir: Union[str, Path],
                              merge_policy: str = "prefer_max_samples",
                              tmp_subdir: str = "tmp",
                              merged_subdir: str = "merged") -> Path:
    """
    扫描 output_base_dir/tmp 下的 worker 输出，并把它们合并到 output_base_dir/merged。
    关键改进：对大型 configs 使用流式写入（预分配 target dataset，再分块从每个 source 写入），避免 OOM。
    """
    if np is None or h5py is None:
        raise RuntimeError("numpy and h5py are required for orchestration/merge")

    MAX_CHUNK_BYTES = 64 * 1024 * 1024  # chunking target for streaming
    SAMPLE_ROWS = 4

    base = Path(output_base_dir)
    tmpdir = base / tmp_subdir
    merged_dir = base / merged_subdir
    merged_dir.mkdir(parents=True, exist_ok=True)

    # collect worker dirs and any root .h5 files
    worker_dirs = [p for p in tmpdir.iterdir() if p.is_dir()] if tmpdir.exists() else []
    root_h5s = list(tmpdir.glob("*.h5")) if tmpdir.exists() else []

    if not worker_dirs and not root_h5s:
        manifest = {"merged_from": [], "merged_at": datetime.utcnow().isoformat() + "Z", "note": "no worker outputs found"}
        manifest_path = merged_dir / "manifest.json"
        atomic_write_bytes(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"))
        summary_path = merged_dir / "summary.json"
        atomic_write_bytes(summary_path, json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"))
        return summary_path

    # We'll build a list of source descriptors for configs and labels.
    # descriptor: {
    #   "src_path": Path,
    #   "dataset": "configs" or "lattices",
    #   "n": int,
    #   "T": float or None,
    #   "h": float or None,
    #   "has_mags": bool,
    #   "has_enes": bool,
    #   "shape": tuple or None,
    #   "dtype": dtype or None,
    #   "sample": small ndarray or None
    # }
    sources: List[Dict[str, Any]] = []
    provenance_records: List[Dict[str, Any]] = []
    merged_sources: List[str] = []
    total_skipped = 0
    total_samples = 0

    # helper to infer T/h from name
    import re
    def _infer_from_name(s: str):
        out = {}
        mT = re.search(r"[Tt]=?([+-]?\d+(\.\d+)?)", s)
        if mT:
            try:
                out["T"] = float(mT.group(1))
            except Exception:
                pass
        mh = re.search(r"[hH]=?([+-]?\d+(\.\d+)?)", s)
        if mh:
            try:
                out["h"] = float(mh.group(1))
            except Exception:
                pass
        return out

    # scan worker dirs
    for wd in sorted(worker_dirs):
        h5_path = wd / "result.h5"
        if not h5_path.exists():
            h5_list = list(wd.rglob("*.h5"))
            h5_path = h5_list[0] if h5_list else None
        scalars_path = wd / "scalars.json"
        if h5_path is None or not Path(h5_path).exists():
            total_skipped += 1
            provenance_records.append({"worker_dir": str(wd), "reason": "no_h5_found"})
            continue

        meta, datasets = _read_worker_meta_and_h5(Path(h5_path), scalars_path if scalars_path.exists() else None)
        if meta is None or datasets is None:
            total_skipped += 1
            provenance_records.append({"worker_dir": str(wd), "worker_h5": str(h5_path), "reason": "invalid_h5_or_missing_configs"})
            continue

        # ensure we have configs dataset metadata
        cfg_info = datasets.get("configs", {})
        if not cfg_info or not cfg_info.get("exists", False):
            total_skipped += 1
            provenance_records.append({"worker_dir": str(wd), "worker_h5": str(h5_path), "reason": "missing_configs_metadata"})
            continue

        # infer T/h from meta or name if missing
        if ("T" not in meta) or ("h" not in meta):
            inferred = {}
            inferred.update(_infer_from_name(wd.name))
            inferred.update(_infer_from_name(Path(h5_path).stem))
            for k, v in inferred.items():
                if k not in meta:
                    meta[k] = v

        if ("T" not in meta) or ("h" not in meta):
            total_skipped += 1
            provenance_records.append({"worker_dir": str(wd), "worker_h5": str(h5_path), "reason": "missing_T_or_h", "meta_keys": list(meta.keys())})
            continue

        # determine Nw from shape if available
        sshape = cfg_info.get("shape")
        if sshape and len(sshape) >= 1:
            Nw = int(sshape[0])
        else:
            # fallback: open file and inspect
            try:
                with h5py.File(str(h5_path), "r") as fh:
                    dsn = cfg_info.get("dataset_name", "configs")
                    ds = fh[dsn]
                    Nw = int(ds.shape[0])
                    # try to update dtype/shape if missing
                    if not sshape:
                        cfg_info["shape"] = tuple(ds.shape)
                        cfg_info["dtype"] = ds.dtype
                        try:
                            cfg_info["sample"] = ds[0:min(SAMPLE_ROWS, Nw)]
                        except Exception:
                            cfg_info["sample"] = None
            except Exception:
                total_skipped += 1
                provenance_records.append({"worker_dir": str(wd), "worker_h5": str(h5_path), "reason": "cannot_inspect_configs_shape"})
                continue

        # append source descriptor
        desc = {
            "src_path": Path(h5_path),
            "dataset": cfg_info.get("dataset_name", "configs"),
            "n": int(Nw),
            "T": float(meta.get("T")) if meta.get("T") is not None else None,
            "h": float(meta.get("h")) if meta.get("h") is not None else None,
            "has_mags": bool(datasets.get("magnetizations", {}).get("exists", False)),
            "has_enes": bool(datasets.get("energies", {}).get("exists", False)),
            "shape": cfg_info.get("shape"),
            "dtype": cfg_info.get("dtype"),
            "sample": cfg_info.get("sample"),
            "worker_dir": str(wd),
            "worker_h5": str(h5_path),
        }
        sources.append(desc)
        merged_sources.append(str(h5_path))
        provenance_records.append({
            "worker_dir": str(wd),
            "worker_h5": str(h5_path),
            "n_samples": int(Nw),
            "T": float(meta.get("T")) if meta.get("T") is not None else None,
            "h": float(meta.get("h")) if meta.get("h") is not None else None,
            "meta_keys": list(meta.keys())
        })
        total_samples += int(Nw)

    # scan root h5s similarly
    for h5f in sorted(root_h5s):
        meta, datasets = _read_worker_meta_and_h5(Path(h5f), None)
        if meta is None or datasets is None:
            total_skipped += 1
            provenance_records.append({"worker_file": str(h5f), "reason": "invalid_h5"})
            continue
        cfg_info = datasets.get("configs", {})
        if not cfg_info or not cfg_info.get("exists", False):
            total_skipped += 1
            provenance_records.append({"worker_file": str(h5f), "reason": "missing_configs"})
            continue

        if ("T" not in meta) or ("h" not in meta):
            inferred = {}
            inferred.update(_infer_from_name(Path(h5f).stem))
            for k, v in inferred.items():
                if k not in meta:
                    meta[k] = v

        if ("T" not in meta) or ("h" not in meta):
            total_skipped += 1
            provenance_records.append({"worker_file": str(h5f), "reason": "missing_T_or_h", "meta_keys": list(meta.keys())})
            continue

        sshape = cfg_info.get("shape")
        if sshape and len(sshape) >= 1:
            Nw = int(sshape[0])
        else:
            try:
                with h5py.File(str(h5f), "r") as fh:
                    dsn = cfg_info.get("dataset_name", "configs")
                    ds = fh[dsn]
                    Nw = int(ds.shape[0])
                    if not sshape:
                        cfg_info["shape"] = tuple(ds.shape)
                        cfg_info["dtype"] = ds.dtype
                        try:
                            cfg_info["sample"] = ds[0:min(SAMPLE_ROWS, Nw)]
                        except Exception:
                            cfg_info["sample"] = None
            except Exception:
                total_skipped += 1
                provenance_records.append({"worker_file": str(h5f), "reason": "cannot_inspect_configs_shape"})
                continue

        desc = {
            "src_path": Path(h5f),
            "dataset": cfg_info.get("dataset_name", "configs"),
            "n": int(Nw),
            "T": float(meta.get("T")) if meta.get("T") is not None else None,
            "h": float(meta.get("h")) if meta.get("h") is not None else None,
            "has_mags": bool(datasets.get("magnetizations", {}).get("exists", False)),
            "has_enes": bool(datasets.get("energies", {}).get("exists", False)),
            "shape": cfg_info.get("shape"),
            "dtype": cfg_info.get("dtype"),
            "sample": cfg_info.get("sample"),
            "worker_file": str(h5f),
        }
        sources.append(desc)
        merged_sources.append(str(h5f))
        provenance_records.append({
            "worker_file": str(h5f),
            "n_samples": int(Nw),
            "T": float(meta.get("T")) if meta.get("T") is not None else None,
            "h": float(meta.get("h")) if meta.get("h") is not None else None,
            "meta_keys": list(meta.keys())
        })
        total_samples += int(Nw)

    # if no samples, write summary and return
    if total_samples == 0:
        summary = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "git_sha": safe_get_git_sha(),
            "total_workers": len(worker_dirs) + len(root_h5s),
            "workers_used": len(provenance_records),
            "total_samples": int(total_samples),
            "total_skipped_workers": int(total_skipped),
            "provenance_records": provenance_records,
            "schema": "ml_flat_3d_with_labels_v1",
            "note": "no valid samples found"
        }
        out_json = merged_dir / "summary.json"
        atomic_write_bytes(out_json, json.dumps(summary, indent=2, ensure_ascii=False).encode("utf-8"))
        return out_json

    # Determine configs trailing shape and dtype using first source that provides shape
    trailing_shape = None
    dtype = None
    first_src_example = None
    for s in sources:
        if s.get("shape") is not None:
            sh = s["shape"]
            if len(sh) >= 2:
                trailing_shape = tuple(sh[1:])
            elif len(sh) == 1:
                trailing_shape = ()
            else:
                trailing_shape = ()
            dtype = s.get("dtype", None)
            first_src_example = s
            break
    if trailing_shape is None:
        # fallback: try to inspect first source
        for s in sources:
            try:
                with h5py.File(str(s["src_path"]), "r") as fh:
                    ds = fh[s["dataset"]]
                    sh = ds.shape
                    if len(sh) >= 2:
                        trailing_shape = tuple(sh[1:])
                    elif len(sh) == 1:
                        trailing_shape = ()
                    else:
                        trailing_shape = ()
                    dtype = ds.dtype
                    first_src_example = s
                    break
            except Exception:
                continue

    if trailing_shape is None:
        # if still unknown, abort
        summary = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "git_sha": safe_get_git_sha(),
            "error": "cannot_determine_configs_shape",
            "provenance_records": provenance_records,
        }
        out_json = merged_dir / "summary.json"
        atomic_write_bytes(out_json, json.dumps(summary, indent=2, ensure_ascii=False).encode("utf-8"))
        return out_json

    # compute final shapes
    if trailing_shape:
        configs_full_shape = (total_samples,) + tuple(trailing_shape)
    else:
        configs_full_shape = (total_samples,)

    # create tmp h5 and stream-copy into it
    out_h5 = merged_dir / "final_ml_data.h5"
    tmp_h5 = Path(str(out_h5) + ".tmp")
    out_h5.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "git_sha": safe_get_git_sha(),
        "total_workers": len(worker_dirs) + len(root_h5s),
        "workers_used": len(provenance_records),
        "total_samples": int(total_samples),
        "total_skipped_workers": int(total_skipped),
        "provenance_records": provenance_records,
        "schema": "ml_flat_3d_with_labels_v1",
    }

    failed_sources: List[Dict[str, Any]] = []

    try:
        with h5py.File(str(tmp_h5), "w", libver="latest") as fh:
            # create configs dataset
            try:
                fh.create_dataset("configs", shape=configs_full_shape, dtype=dtype or np.float32, chunks=True, compression="gzip", compression_opts=4)
            except Exception:
                fh.create_dataset("configs", shape=configs_full_shape, dtype=dtype or np.float32, chunks=True)

            # create temperatures and fields datasets (1D)
            try:
                fh.create_dataset("temperatures", shape=(total_samples,), dtype=np.float32, compression="gzip")
            except Exception:
                fh.create_dataset("temperatures", shape=(total_samples,), dtype=np.float32)
            try:
                fh.create_dataset("fields", shape=(total_samples,), dtype=np.float32, compression="gzip")
            except Exception:
                fh.create_dataset("fields", shape=(total_samples,), dtype=np.float32)

            # optionally create magnetizations/energies (float32)
            any_mags = any(s.get("has_mags", False) for s in sources)
            any_enes = any(s.get("has_enes", False) for s in sources)
            if any_mags:
                try:
                    fh.create_dataset("magnetizations", shape=(total_samples,), dtype=np.float32, compression="gzip")
                except Exception:
                    fh.create_dataset("magnetizations", shape=(total_samples,), dtype=np.float32)
            if any_enes:
                try:
                    fh.create_dataset("energies", shape=(total_samples,), dtype=np.float32, compression="gzip")
                except Exception:
                    fh.create_dataset("energies", shape=(total_samples,), dtype=np.float32)

            # stream-copy each source into target
            offset = 0
            for s in sources:
                n = int(s["n"])
                if n == 0:
                    continue

                src_path = s["src_path"]
                dsname = s["dataset"]
                T_val = s.get("T", float("nan"))
                h_val = s.get("h", float("nan"))
                try:
                    with h5py.File(str(src_path), "r") as shf:
                        src_ds = shf[dsname]
                        # prepare chunking parameters
                        if len(src_ds.shape) > 1:
                            bytes_per_entry = int(np.prod(src_ds.shape[1:]) * np.dtype(src_ds.dtype).itemsize)
                        else:
                            bytes_per_entry = int(np.dtype(src_ds.dtype).itemsize)
                        slices_per_chunk = max(1, int(MAX_CHUNK_BYTES // max(1, bytes_per_entry)))
                        s_i = 0
                        while s_i < n:
                            e_i = min(n, s_i + slices_per_chunk)
                            chunk = src_ds[s_i:e_i]
                            fh["configs"][offset + s_i: offset + e_i] = chunk
                            s_i = e_i
                except Exception as e:
                    logger.exception("failed to stream-copy configs from %s: %s", src_path, e)
                    failed_sources.append({"source": str(src_path), "dataset": dsname, "reason": str(e)})
                    # fallback: write placeholder NaNs/zeros for this slice
                    try:
                        if np.issubdtype(fh["configs"].dtype, np.floating):
                            fh["configs"][offset: offset + n] = np.full((n,) + fh["configs"].shape[1:], np.nan, dtype=fh["configs"].dtype)
                        else:
                            fh["configs"][offset: offset + n] = np.zeros((n,) + fh["configs"].shape[1:], dtype=fh["configs"].dtype)
                    except Exception:
                        # best-effort elementwise
                        for ii in range(n):
                            try:
                                fh["configs"][offset + ii] = 0
                            except Exception:
                                logger.debug("failed to fill placeholder element at configs[%d]", offset + ii, exc_info=True)

                # write temperatures and fields slices (scalar repeated per sample)
                try:
                    fh["temperatures"][offset: offset + n] = np.full((n,), float(T_val), dtype=np.float32)
                except Exception:
                    # elementwise fallback
                    for ii in range(n):
                        try:
                            fh["temperatures"][offset + ii] = float(T_val)
                        except Exception:
                            fh["temperatures"][offset + ii] = np.float32(np.nan)
                try:
                    fh["fields"][offset: offset + n] = np.full((n,), float(h_val), dtype=np.float32)
                except Exception:
                    for ii in range(n):
                        try:
                            fh["fields"][offset + ii] = float(h_val)
                        except Exception:
                            fh["fields"][offset + ii] = np.float32(0.0)

                # magnetizations
                if any_mags:
                    if s.get("has_mags", False):
                        try:
                            with h5py.File(str(src_path), "r") as shf:
                                md = shf["magnetizations"]
                                # chunked copy
                                if len(md.shape) > 1:
                                    m_bytes_per_entry = int(np.prod(md.shape[1:]) * np.dtype(md.dtype).itemsize)
                                else:
                                    m_bytes_per_entry = int(np.dtype(md.dtype).itemsize)
                                m_slices = max(1, int(MAX_CHUNK_BYTES // max(1, m_bytes_per_entry)))
                                mi = 0
                                while mi < n:
                                    me = min(n, mi + m_slices)
                                    mch = md[mi:me]
                                    fh["magnetizations"][offset + mi: offset + me] = mch
                                    mi = me
                        except Exception:
                            logger.exception("failed to copy magnetizations from %s", src_path)
                            fh["magnetizations"][offset: offset + n] = np.full((n,), np.nan, dtype=np.float32)
                    else:
                        # fill NaNs
                        fh["magnetizations"][offset: offset + n] = np.full((n,), np.nan, dtype=np.float32)

                # energies
                if any_enes:
                    if s.get("has_enes", False):
                        try:
                            with h5py.File(str(src_path), "r") as shf:
                                ed = shf["energies"]
                                if len(ed.shape) > 1:
                                    e_bytes_per_entry = int(np.prod(ed.shape[1:]) * np.dtype(ed.dtype).itemsize)
                                else:
                                    e_bytes_per_entry = int(np.dtype(ed.dtype).itemsize)
                                e_slices = max(1, int(MAX_CHUNK_BYTES // max(1, e_bytes_per_entry)))
                                ei = 0
                                while ei < n:
                                    ee = min(n, ei + e_slices)
                                    ech = ed[ei:ee]
                                    fh["energies"][offset + ei: offset + ee] = ech
                                    ei = ee
                        except Exception:
                            logger.exception("failed to copy energies from %s", src_path)
                            fh["energies"][offset: offset + n] = np.full((n,), np.nan, dtype=np.float32)
                    else:
                        fh["energies"][offset: offset + n] = np.full((n,), np.nan, dtype=np.float32)

                offset += n

            # record provenance attrs in group
            prov = fh.create_group("provenance")
            for k, v in summary.items():
                if k == "provenance_records":
                    continue
                try:
                    prov.attrs[k] = _to_builtin(v)
                except Exception:
                    prov.attrs[k] = str(v)
            try:
                prov.attrs["records_json"] = json.dumps(provenance_records, default=str, ensure_ascii=False)
            except Exception:
                prov.attrs["records_json"] = str(provenance_records)
            try:
                prov.attrs["merged_from"] = json.dumps(merged_sources, ensure_ascii=False)
            except Exception:
                prov.attrs["merged_from"] = str(merged_sources)
            prov.attrs["created_at"] = str(datetime.utcnow().isoformat() + "Z")
            if failed_sources:
                try:
                    prov.attrs["failed_sources"] = json.dumps(failed_sources, ensure_ascii=False)
                except Exception:
                    prov.attrs["failed_sources"] = str(failed_sources)

        # atomic replace tmp -> final
        os.replace(str(tmp_h5), str(out_h5))
    except Exception as e:
        try:
            if tmp_h5.exists():
                tmp_h5.unlink()
        except Exception:
            pass
        summary["error"] = f"h5_write_failed_streaming: {e}"
        out_json = merged_dir / "summary.json"
        atomic_write_bytes(out_json, json.dumps(summary, indent=2, ensure_ascii=False).encode("utf-8"))
        return out_json

    # write summary + manifest
    out_json = merged_dir / "summary.json"
    atomic_write_bytes(out_json, json.dumps(summary, indent=2, ensure_ascii=False).encode("utf-8"))

    manifest = {
        "merged_from": merged_sources,
        "merged_at": datetime.utcnow().isoformat() + "Z",
    }
    manifest_path = merged_dir / "manifest.json"
    atomic_write_bytes(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"))

    return out_h5


# -----------------------------------------------------------------------------
# 简易 CLI / 调试入口
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="data_manager utilities (merge worker outputs).")
    parser.add_argument("--outdir", type=str, default="out", help="output base dir (contains tmp/ and merged/)")
    parser.add_argument("--action", choices=["merge", "list"], default="merge")
    args = parser.parse_args()

    if args.action == "merge":
        print("Merging outputs in", args.outdir)
        result = _orchestrate_worker_merge(args.outdir)
        print("Result path:", result)
    else:
        base = Path(args.outdir)
        tmp = base / "tmp"
        if tmp.exists():
            print("Found tmp files:")
            for p in tmp.rglob("*"):
                print("-", p)
        else:
            print("No tmp dir found under", base)

