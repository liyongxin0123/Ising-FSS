# tests/test_gpu_remc_simulator.py
import os
import json
import numpy as np
import tempfile
import types
import cupy as cp
import pytest

# 假定模块路径为 mypkg.simulator.gpu_remc_simulator (按你项目结构调整 import)
from mypkg.simulator import gpu_remc_simulator as grs

# --- 简单 mock 下游 gpu_algorithms（用于 unit tests，无需真实 GPU 算法） ---
def make_mock_ga():
    ga = types.SimpleNamespace()
    # metropolis_update_batch: 直接返回原 spins 并返回 device_counters, meta
    def metropolis_update_batch(spins, beta, **kwargs):
        R = int(spins.shape[0])
        # device_counters 提供基本字段，允许 numpy/cupy
        device_counters = {
            "accepts": cp.zeros((R,), dtype=cp.int64),
            "attempts": cp.zeros((R,), dtype=cp.int64),
            "E_sum": cp.zeros((R,), dtype=cp.float64),
            "M_sum": cp.zeros((R,), dtype=cp.float64),
        }
        # 假设每副本消耗 rng_per_sweep = spins.shape[1]*spins.shape[2]
        rng_consumed = int(spins.shape[1] * spins.shape[2])
        meta = {"rng_consumed": [rng_consumed] * R, "replica_counters": [0]*R, "rng_model": "mock_philox_slot_bound"}
        return spins, (device_counters, meta)
    ga.metropolis_update_batch = metropolis_update_batch
    ga.init_device_counters = lambda R: {
        "accepts": cp.zeros((R,), dtype=cp.int64),
        "attempts": cp.zeros((R,), dtype=cp.int64),
        "E_sum": cp.zeros((R,), dtype=cp.float64),
        "M_sum": cp.zeros((R,), dtype=cp.float64),
    }
    ga.device_energy = lambda spins, h=0.0: cp.asarray([float(0.0) for _ in range(int(spins.shape[0]))])
    ga.device_magnetization = lambda spins: cp.asarray([0.0 for _ in range(int(spins.shape[0]))])
    # Optional RNG state helpers
    ga.get_device_rng_state = lambda: {"mock": "state"}
    ga.set_device_rng_state = lambda s: None
    return ga

@pytest.fixture(autouse=True)
def patch_ga(monkeypatch):
    mock = make_mock_ga()
    # 把模拟的 ga 注入到模块
    monkeypatch.setattr(grs, "ga", mock)
    yield

def test_save_and_restore_checkpoint(tmp_path):
    # small simulator
    sim = grs.GPU_REMC_Simulator(L=4, T_min=1.0, T_max=2.0, num_replicas=2, replica_seeds=[42,43])
    # ensure device spins initialized
    sim._ensure_device_spins()
    # do a snapshot and save checkpoint
    cp_path = str(tmp_path / "chk.json")
    sim.save_checkpoint(cp_path)
    assert os.path.exists(cp_path)
    assert os.path.exists(cp_path + ".npz")
    # mutate rng counters and spins then restore
    sim.replica_counters = [99, 99]
    notes = sim.restore_from_checkpoint(cp_path)
    # after restore, replica_counters should be overwritten by checkpoint (or at least seeds checked)
    assert "sweep_index" in notes["restored"]
    # verify device spins got restored
    assert sim._spins_host.shape == (2, sim.L, sim.L)

def test_parse_meta_and_replica_counters_update():
    sim = grs.GPU_REMC_Simulator(L=4, T_min=1.0, T_max=2.0, num_replicas=2, replica_seeds=[1,2])
    # prepare a device_counters,meta tuple simulating gpu_algorithms return
    dc = {"accepts": np.array([1,2]), "attempts": np.array([10,10]), "E_sum": np.array([0.0,0.0]), "M_sum": np.array([0.0,0.0])}
    meta = {"replica_counters": [5, 6], "rng_consumed": [16, 16], "rng_model": "mock"}
    sim._parse_ga_meta_and_update_counters((dc, meta))
    assert sim.replica_counters[0] >= 5 and sim.replica_counters[1] >= 6

def test_run_smoke_and_analyze():
    sim = grs.GPU_REMC_Simulator(L=4, T_min=1.0, T_max=2.0, num_replicas=2, replica_seeds=[11,22])
    # run with tiny steps (uses mocked ga)
    sim.run(equilibration_steps=1, production_steps=2, exchange_interval=1, thin=1)
    res = sim.analyze()
    assert "swap" in res
    assert "T_1.000000" in res or any(k.startswith("T_") for k in res.keys())

