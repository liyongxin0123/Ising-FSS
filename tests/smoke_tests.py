# smoke_tests.py
# Lightweight smoke tests for quick local verification of many modules.
# Run with: python smoke_tests.py
import os
import sys
import tempfile
import json
import shutil
import time
from pathlib import Path
import numpy as np

# try import modules created/edited earlier
from ising_fss.data import config_io as cio
from ising_fss.analysis import statistics as stats
from ising_fss.core import observables as obs
from ising_fss.analysis import dl_tools as dl
from ising_fss.utils import logger as lg
from ising_fss.simulation import batch_runner as br
from ising_fss.simulation import parallel as pr


def main():
    print("=== SMOKE TESTS START ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="smoke_"))
    print("tmpdir:", tmpdir)

    try:
        # 1) config_io basic save/load/export
        print("\n-- config_io basic tests --")
        ds = {
            "configs": np.random.choice([-1, 1], size=(1, 2, 3, 8, 8)).astype(np.int8),
            "temperatures": np.array([1.8, 2.2], dtype=np.float32),
            "fields": np.array([-0.1, 0.1], dtype=np.float32),
            "energy": np.random.randn(1, 2, 3).astype(np.float32),
            "magnetization": np.random.randn(1, 2, 3).astype(np.float32),
            "parameters": {"L": 8, "n_configs": 3},
        }
        h5path = tmpdir / "test_data.h5"
        cio.save_configs_hdf5(ds, h5path, compression="gzip", verbose=True)
        loaded = cio.load_configs_hdf5(h5path, load_configs=False, verbose=True)
        print("loaded shape (deferred):", loaded.get("configs_shape"))

        # export for pytorch
        ptdir = tmpdir / "pytorch_export"
        cio.export_for_pytorch(ds, ptdir, split_ratio=0.7, normalize=True, dtype="uint8", seed=42, verbose=True)
        data_train, metadata = cio.load_pytorch_dataset(ptdir, "train")
        print("pytorch metadata keys:", list(metadata.keys())[:5])

        # 2) statistics basic tests
        print("\n-- statistics basic tests --")
        series = np.random.randn(2000)
        tau = stats.autocorrelation_time(series)
        ess = stats.effective_sample_size(series, tau)
        err, tau2 = stats.estimate_error_with_autocorr(series)
        print(f"tau={tau:.3f} ess={ess:.1f} err={err:.4f}")

        block_err, curve = stats.blocking_analysis(series, return_curve=True)
        print("blocking err:", block_err)

        # jackknife on mean
        theta, stderr = stats.jackknife_error(series, func=np.mean, block_len=50)
        print("jackknife mean stderr:", stderr)

        # 3) observables tests
        print("\n-- observables tests --")
        L = 8
        lat = np.random.choice([-1, 1], size=(L, L)).astype(np.int8)
        o = obs.calculate_observables(lat, h=0.0)
        print("observables:", {k: float(v) for k, v in o.items()})

        # batch
        batch = np.stack([lat, -lat], axis=0)
        Earr, Marr = obs.calculate_observables_batch(batch, h=0.0)
        print("batch E shape:", Earr.shape, "M shape:", Marr.shape)

        # binder, cv, chi
        m_series = np.random.randn(200)
        U = obs.calculate_binder_cumulant(m_series)
        print("Binder cumulant:", U)

        # 4) dl_tools: dataset/dataloader smoke (if torch installed)
        print("\n-- dl_tools smoke --")
        try:
            import torch
            arr = np.random.choice([-1, 1], size=(20, 8, 8)).astype(np.int8)
            ds_obj = dl.IsingDataset(arr, normalize=True)
            loaders = dl.create_dataloaders(ds_obj, batch_size=4, val_split=0.2, seed=123, num_workers=0)
            b = next(iter(loaders["train"]))
            print("dl_tools batch keys:", list(b.keys()))
        except Exception as e:
            print("dl_tools torch-related smoke skipped or failed:", e)

        # 5) logger basic
        print("\n-- logger smoke --")
        logger = lg.setup_logger("smoke_logger", level=20, log_file=str(tmpdir / "smoke.log"), use_color=False, mp_safe=False)
        logger.info("This is a smoke info message")
        exp = lg.ExperimentLogger("smoke_experiment", output_dir=str(tmpdir), level=20, mp_safe=False)
        exp.log_config({"smoke": True, "tmp": str(tmpdir)})
        exp.log_metric("accuracy", 0.5, step=1)
        exp.finish()

        # 6) batch_runner demo (runs demo workers that write to unique dirs)
        print("\n-- batch_runner demo (light) --")
        try:
            br.run_workers_demo(str(tmpdir / "batch_demo"), nworkers=2, tasks_per_worker=1)
            print("batch_runner demo produced:", list((tmpdir / "batch_demo").glob("**/*"))[:5])
        except Exception as e:
            print("batch_runner demo failed (ok if remc_simulator absent):", e)

        # 7) parallel across_L quick run (will likely fail if simulator not installed; ensure graceful)
        print("\n-- parallel across_L dry-run (no simulator expected) --")
        try:
            res = pr.across_L(L_list=[8], T_min=1.5, T_max=3.0, num_replicas=4,
                              equilibration=10, production=10, algorithm="wolff", seed=None, pool_size=1)
            print("parallel returned keys:", list(res.keys()))
        except Exception as e:
            print("parallel across_L could not run (expected if no simulator):", e)

        print("\nSMOKE TESTS OK (if no uncaught exceptions)")

    finally:
        # cleanup - comment out if you want to inspect files
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

    print("=== SMOKE TESTS END ===")


if __name__ == "__main__":
    import multiprocessing as mp
    # 统一强制使用 spawn（必须在创建任何进程/进程池之前设置）
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # 已经在交互环境/外层设置过启动方式，忽略
        pass
    mp.freeze_support()  # Windows/macOS 友好
    main()

