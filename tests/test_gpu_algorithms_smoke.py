# tests/test_gpu_algorithms_smoke.py
import numpy as np
import pytest
import cupy as cp
from ising_fss.core.gpu_algorithms import _make_cupy_generator, metropolis_update_batch, init_device_counters

# small helpers
def random_spins(R, L, seed=123):
    rng = np.random.default_rng(seed)
    arr = rng.choice([-1, 1], size=(R, L, L)).astype(np.int8)
    return arr

@pytest.mark.skipif(not hasattr(cp.random, "default_rng"), reason="CuPy RNG not available")
def test_philox_smoke():
    # will raise if generator cannot be constructed
    for seed in [0, 1, 12345, 2**31-1]:
        g = _make_cupy_generator(seed)
        assert g is not None
        # draw some numbers
        a = g.random((2,2), dtype=cp.float32)
        assert a.shape == (2,2)

@pytest.mark.parametrize("chunk", [1, 2, 4])
def test_chunk_consistency(chunk):
    R, L = 8, 16
    spins0 = random_spins(R, L, seed=42)
    seeds = [100 + i for i in range(R)]
    beta = 0.44
    n_sweeps = 2

    # run reference with chunk = 1
    s_ref, counters_ref = metropolis_update_batch(spins0.copy(), beta,
                                                 n_sweeps=n_sweeps,
                                                 replica_seeds=seeds,
                                                 rng_chunk_replicas=1,
                                                 precision='float64',
                                                 vectorized_rng=False)
    s_ref_np = cp.asnumpy(s_ref)

    # run test with given chunk
    s_test, _ = metropolis_update_batch(spins0.copy(), beta,
                                       n_sweeps=n_sweeps,
                                       replica_seeds=seeds,
                                       rng_chunk_replicas=chunk,
                                       precision='float64',
                                       vectorized_rng=False)
    s_test_np = cp.asnumpy(s_test)

    # expect exact equality for float64 deterministic path (slot_bound)
    assert np.array_equal(s_ref_np, s_test_np), "chunked result differs from reference"

def test_mask_assignment_and_attempts():
    R, L = 4, 7  # test odd/even sizes in CI combos
    spins0 = random_spins(R, L, seed=7)
    seeds = [11 + i for i in range(R)]
    beta = 0.5
    device_counters = init_device_counters(R)

    s_after, counters = metropolis_update_batch(spins0.copy(), beta,
                                               n_sweeps=1,
                                               replica_seeds=seeds,
                                               device_counters=device_counters,
                                               rng_chunk_replicas=2,
                                               precision='float32',
                                               vectorized_rng=True,
                                               checkerboard=True)

    # attempts must be non-zero positive integers
    attempts = counters["attempts"]
    assert isinstance(attempts, cp.ndarray) or isinstance(attempts, list)
    # convert if cupy array
    attempts_np = cp.asnumpy(attempts) if isinstance(attempts, cp.ndarray) else np.array(attempts)
    assert np.all(attempts_np > 0)

    # accepts must be <= attempts for each replica
    accepts_np = cp.asnumpy(counters["accepts"]) if isinstance(counters["accepts"], cp.ndarray) else np.array(counters["accepts"])
    assert np.all(accepts_np <= attempts_np)

def test_basic_run_no_crash():
    R, L = 2, 8
    spins0 = random_spins(R, L, seed=1)
    seeds = [1, 2]
    beta = [0.4, 0.5]
    s_out, counters = metropolis_update_batch(spins0.copy(), beta,
                                            n_sweeps=1,
                                            replica_seeds=seeds,
                                            rng_chunk_replicas=1,
                                            precision='float32',
                                            vectorized_rng=False)
    assert s_out.shape == (R, L, L)


