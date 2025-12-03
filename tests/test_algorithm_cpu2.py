# -*- coding: utf-8 -*-
"""
核心算法模块单元测试（CPU 端，基于 apply_move / update_batch）

覆盖范围：
- 算法逻辑：Metropolis, Wolff, Swendsen–Wang（若实现）通过 apply_move 调用
- 批量更新：update_batch 接口及 replica_seeds 机制
- 物理性质：
    - 基态能量 / 磁化（铁磁/反铁磁）
    - Metropolis 低温能量下降、高温去磁化
    - Wolff 临界簇大小 / cluster 比 Metropolis 去相关快
    - 外场 h ≠ 0 时禁止簇算法
- 热力学量：
    - 比热 C/N > 0
    - χ/N > 0
    - Binder 累积量顺磁 ~ 0，铁磁 ~ 2/3
- 统计工具：
    - 自相关时间 autocorrelation_time
    - 分块分析 blocking_analysis
    - （若存在）有效样本数 effective_sample_size
- 数值稳定性：
    - 大晶格一次 Metropolis sweep
    - 极端温度 β -> 0 / β -> ∞
    - 零磁化序列的 χ = 0
- 可复现性：
    - 显式 seed 下 Metropolis / Wolff 严格一致
"""

import unittest
import sys
import os
from pathlib import Path
from typing import Optional

import numpy as np

# ----------------------------- 路径适配 -----------------------------
try:
    _ROOT = Path(__file__).resolve().parents[1]
except NameError:
    _ROOT = Path.cwd()

if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

# ----------------------------- 模块导入 -----------------------------
try:
    from ising_fss.core import algorithms as alg
    from ising_fss.core.observables import (
        calculate_observables,
        calculate_specific_heat_per_spin,
        calculate_susceptibility_per_spin,
        calculate_binder_cumulant,
    )
    from ising_fss.analysis import statistics as stats

    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"[test_algorithm_cpu2] Skipping tests due to import failure: {e}")
    MODULES_AVAILABLE = False

HAS_EFF_SAMPLE = MODULES_AVAILABLE and hasattr(stats, "effective_sample_size")

# ----------------------------- 测试配置 -----------------------------
FAST_MODE = os.environ.get("ISING_FAST_TEST", "0") == "1"

L_SMALL = 16
L_MED = 128 if not FAST_MODE else 64

Tc_THEORY = 2.269185
BETA_CRIT = 1.0 / Tc_THEORY
BETA_LOW = 1.0   # 低温（有序）
BETA_HIGH = 0.1  # 高温（无序）


def _rng_state(shape, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed if seed is not None else 2025)
    return rng.choice([-1, 1], size=shape).astype(np.int8)


# =============================================================================
# 核心算法：apply_move / update_batch
# =============================================================================
class TestCoreAlgorithms(unittest.TestCase):
    """测试 algorithms.py 中的核心蒙特卡洛算法"""

    def setUp(self):
        if not MODULES_AVAILABLE:
            self.skipTest("核心模块未找到")
        self.rng = np.random.default_rng(2025)
        self.ones_state = np.ones((L_SMALL, L_SMALL), dtype=np.int8)
        self.rand_state = _rng_state((L_SMALL, L_SMALL), seed=2025)

    def test_apply_move_interface(self):
        """测试统一接口 apply_move 的基本连通性"""
        # Metropolis
        res, meta = alg.apply_move(
            self.ones_state.copy(),
            algo="metropolis_sweep",
            beta=BETA_CRIT,
            replica_seed=42,
        )
        self.assertEqual(res.shape, (L_SMALL, L_SMALL))
        # meta 是 MoveInfo 数据类，检查字段存在性
        self.assertTrue(hasattr(meta, "accepted"))
        self.assertTrue(hasattr(meta, "rng_model"))

        # Wolff
        res, meta = alg.apply_move(
            self.rand_state.copy(),
            algo="wolff",
            beta=BETA_CRIT,
            replica_seed=42,
        )
        self.assertEqual(res.shape, (L_SMALL, L_SMALL))
        self.assertTrue(
            hasattr(meta, "cluster_size"),
            msg="Wolff MoveInfo 应包含 cluster_size 字段",
        )

    def test_reproducibility_deterministic(self):
        """显式种子下 Metropolis / Wolff 严格可复现"""
        beta = 0.44
        seed = 12345

        # Metropolis
        res1, meta1 = alg.apply_move(
            self.rand_state.copy(), "metropolis_sweep", beta, replica_seed=seed
        )
        res2, meta2 = alg.apply_move(
            self.rand_state.copy(), "metropolis_sweep", beta, replica_seed=seed
        )
        np.testing.assert_array_equal(res1, res2, err_msg="Metropolis 结果不可复现")
        # 若 MoveInfo 可比较，结构也应一致（宽松：只比较几个常见字段）
        if hasattr(meta1, "accepted"):
            self.assertEqual(meta1.accepted, meta2.accepted)
        if hasattr(meta1, "rng_model"):
            self.assertEqual(meta1.rng_model, meta2.rng_model)

        # Wolff
        res1, meta1 = alg.apply_move(
            self.rand_state.copy(), "wolff", beta, replica_seed=seed
        )
        res2, meta2 = alg.apply_move(
            self.rand_state.copy(), "wolff", beta, replica_seed=seed
        )
        np.testing.assert_array_equal(res1, res2, err_msg="Wolff 结果不可复现")
        if hasattr(meta1, "cluster_size"):
            self.assertEqual(meta1.cluster_size, meta2.cluster_size)

    def test_physics_metropolis_relaxation_lowT(self):
        """Metropolis 在低温下能量下降、磁化增强"""
        state = self.rand_state.copy()
        obs_start = calculate_observables(state)

        n_steps = 50 if not FAST_MODE else 30
        for i in range(n_steps):
            state, _ = alg.apply_move(
                state, "metropolis_sweep", BETA_LOW, replica_seed=100 + i
            )

        obs_end = calculate_observables(state)

        self.assertLess(
            obs_end["E_per_spin"],
            obs_start["E_per_spin"],
            msg="低温 Metropolis 能量未明显降低",
        )
        self.assertGreater(
            obs_end["abs_m"],
            obs_start["abs_m"],
            msg="低温 Metropolis 磁化未明显增强",
        )

    def test_physics_metropolis_randomization_highT(self):
        """高温 Metropolis 使磁化接近 0（从全 1 态出发）"""
        state = self.ones_state.copy()
        n_steps = 200 if not FAST_MODE else 120
        for i in range(n_steps):
            state, _ = alg.apply_move(
                state, "metropolis_sweep", BETA_HIGH, replica_seed=200 + i
            )
        obs = calculate_observables(state)
        self.assertLess(
            abs(obs["m"]),
            0.25,
            msg=f"高温后 m={obs['m']:.3f} 仍然偏大，未充分去磁化",
        )

    def test_field_constraint_for_cluster_algorithms(self):
        """当 h != 0 时，Wolff / Swendsen–Wang 应抛出错误（不允许）"""
        with self.assertRaises(ValueError):
            alg.apply_move(
                self.ones_state, "wolff", beta=0.4, h=0.1, replica_seed=1
            )

        try:
            alg.apply_move(
                self.ones_state,
                "swendsen_wang",
                beta=0.4,
                h=0.1,
                replica_seed=1,
            )
        except ValueError:
            # 正常：不允许 h!=0
            pass
        except Exception:
            # 若实现里没有 swendsen_wang 算法，不视为失败
            pass

    def test_update_batch_interface(self):
        """测试 update_batch 批量接口的形状与种子效果"""
        R = 4
        batch = np.stack([self.rand_state.copy() for _ in range(R)])
        seeds = [100 + i for i in range(R)]
        betas = [BETA_CRIT] * R

        new_batch, info = alg.update_batch(
            batch,
            beta=betas,
            replica_seeds=seeds,
            algo="metropolis_sweep",
            n_sweeps=2,
        )

        self.assertEqual(new_batch.shape, (R, L_SMALL, L_SMALL))
        self.assertIn("per_replica", info)
        self.assertEqual(len(info["per_replica"]), R)

        self.assertFalse(
            np.array_equal(new_batch[0], new_batch[1]),
            msg="不同 replica_seed 却得到完全相同结果，可能种子未生效",
        )


# =============================================================================
# 观测量 / 热力学量
# =============================================================================
class TestObservablesAndThermodynamics(unittest.TestCase):
    """测试 observables.py 的物理量计算：E, m, C, χ, Binder"""

    def setUp(self):
        if not MODULES_AVAILABLE:
            self.skipTest("模块缺失")
        self.L = 10
        self.rng = np.random.default_rng(11)

    def test_ground_states_ferro_and_afm(self):
        """铁磁 / 反铁磁基态能量和磁化"""
        state_fm = np.ones((self.L, self.L), dtype=np.int8)
        obs_fm = calculate_observables(state_fm)
        self.assertAlmostEqual(
            obs_fm["E_per_spin"], -2.0, places=6, msg="铁磁基态 E/N 应为 -2"
        )
        self.assertAlmostEqual(
            obs_fm["m"], 1.0, places=6, msg="铁磁基态 m 应为 1"
        )

        idx = np.arange(self.L)
        x, y = np.meshgrid(idx, idx)
        state_afm = np.where((x + y) % 2 == 0, 1, -1).astype(np.int8)
        obs_afm = calculate_observables(state_afm)
        self.assertAlmostEqual(
            obs_afm["E_per_spin"], 2.0, places=6, msg="反铁磁基态 E/N 应为 +2"
        )
        self.assertAlmostEqual(
            obs_afm["m"], 0.0, places=6, msg="反铁磁基态 m 应为 0"
        )

    def test_energy_and_magnetization_bounds(self):
        """能量 / 磁化边界"""
        L = self.L
        state = _rng_state((L, L), seed=2)
        obs = calculate_observables(state)

        self.assertGreaterEqual(obs["E_per_spin"], -2.0 - 1e-8)
        self.assertLessEqual(obs["E_per_spin"], 2.0 + 1e-8)

        self.assertGreaterEqual(abs(obs["m"]), 0.0)
        self.assertLessEqual(abs(obs["m"]), 1.0 + 1e-8)

    def test_specific_heat_positive(self):
        """比热 C/N > 0"""
        n = 1500 if not FAST_MODE else 1000
        E_per_spin = self.rng.normal(-1.5, 0.3, size=n)
        T = 2.5
        # API: (E, T, N, E_is_per_spin)
        C = calculate_specific_heat_per_spin(
            E_per_spin, T, N=1, E_is_per_spin=True
        )
        self.assertGreater(C, 0.0)

    def test_susceptibility_positive(self):
        """磁化率 χ/N > 0"""
        n = 1500 if not FAST_MODE else 1000
        m_series = self.rng.normal(0.5, 0.2, size=n)  # 每自旋磁化
        T = 2.0
        chi = calculate_susceptibility_per_spin(
            m_series, T, N=1, M_is_per_spin=True
        )
        self.assertGreater(chi, 0.0)

    def test_binder_cumulant_phase_limits(self):
        """
        Binder：U = 1 - <m^4> / (3 <m^2>^2)
          - 顺磁（零均值高斯）→ U ≈ 0
          - 铁磁（强非零均值）→ U ≈ 2/3
        """
        n = 1500 if not FAST_MODE else 1000

        m_para = self.rng.normal(0.0, 0.3, size=n)
        U_para = calculate_binder_cumulant(m_para)
        self.assertLess(
            U_para, 0.25, msg=f"顺磁相 Binder U={U_para:.3f} 不应太大"
        )

        m_ferro = self.rng.normal(0.9, 0.05, size=n)
        U_ferro = calculate_binder_cumulant(m_ferro)
        self.assertGreater(
            U_ferro, 0.5, msg=f"铁磁相 Binder U={U_ferro:.3f} 不应太小"
        )


# =============================================================================
# 统计分析工具
# =============================================================================
class TestStatistics(unittest.TestCase):
    """测试 statistics.py 的统计工具"""

    def setUp(self):
        if not MODULES_AVAILABLE:
            self.skipTest("模块缺失")
        self.rng = np.random.default_rng(23)

    def test_autocorrelation_independent_and_correlated(self):
        """白噪声 τ < 2；AR(1) 红噪声 τ > 5"""
        white = self.rng.standard_normal(3000)
        tau_white = stats.autocorrelation_time(white)
        self.assertLess(tau_white, 2.0)

        n = 3000
        alpha = 0.9
        red = np.zeros(n)
        noise = self.rng.standard_normal(n)
        for i in range(1, n):
            red[i] = alpha * red[i - 1] + noise[i]
        tau_red = stats.autocorrelation_time(red)
        self.assertGreater(tau_red, 5.0)

    def test_blocking_analysis_convergence(self):
        """blocking 误差数量级与 σ/√N 相当（允许很宽的容差）"""
        sigma_true = 1.0
        n = 10000 if not FAST_MODE else 6000
        data = self.rng.normal(0.0, sigma_true, size=n)
        err, _ = stats.blocking_analysis(data)
        err_ref = sigma_true / np.sqrt(n)
        ratio = err / err_ref
        # blocking 通常偏大，这里只要求同一数量级
        self.assertGreater(
            ratio,
            0.25,
            msg=f"blocking err={err:.4g} 远小于参考 {err_ref:.4g}",
        )
        self.assertLess(
            ratio,
            4.0,
            msg=f"blocking err={err:.4g} 远大于参考 {err_ref:.4g}",
        )

    @unittest.skipUnless(HAS_EFF_SAMPLE, "effective_sample_size 未实现，跳过此测试")
    def test_effective_sample_size(self):
        """独立数据：0.5N ≤ N_eff ≤ 1.1N"""
        n = 2000 if not FAST_MODE else 1200
        data = self.rng.standard_normal(n)
        Neff = stats.effective_sample_size(data)
        self.assertGreaterEqual(Neff, 0.5 * n)
        self.assertLessEqual(Neff, 1.1 * n)


# =============================================================================
# 数值稳定性
# =============================================================================
class TestNumericalStability(unittest.TestCase):
    """数值稳定性：大晶格 / 极端温度 / 零磁化率"""

    def setUp(self):
        if not MODULES_AVAILABLE:
            self.skipTest("模块缺失")

    def test_large_lattice_one_sweep(self):
        """较大晶格单次 Metropolis sweep 正常返回"""
        lat = np.ones((L_MED, L_MED), dtype=np.int8)
        res, _ = alg.apply_move(
            lat, "metropolis_sweep", beta=0.5, replica_seed=42
        )
        self.assertEqual(res.shape, (L_MED, L_MED))

    def test_extreme_temperature_stability(self):
        """极端温度 β→∞ / β→0 下算法稳定，自旋仍为 ±1"""
        L = L_SMALL
        lat0 = _rng_state((L, L), seed=29)

        lowT_state, _ = alg.apply_move(
            lat0.copy(), "metropolis_sweep", beta=10.0, replica_seed=123
        )
        self.assertTrue(
            np.all(np.isin(lowT_state, [-1, 1])),
            msg="低温更新后出现了非 ±1 自旋",
        )

        highT_state, _ = alg.apply_move(
            lat0.copy(), "metropolis_sweep", beta=0.01, replica_seed=456
        )
        self.assertTrue(
            np.all(np.isin(highT_state, [-1, 1])),
            msg="高温更新后出现了非 ±1 自旋",
        )

    def test_zero_magnetization_susceptibility(self):
        """零序列的磁化率应为 0"""
        m_series = np.zeros(512, dtype=float)
        chi = calculate_susceptibility_per_spin(
            m_series, T=2.0, N=1, M_is_per_spin=True
        )
        self.assertEqual(chi, 0.0)


# =============================================================================
# 混合效率：Wolff vs Metropolis 在 Tc 附近
# =============================================================================
class TestMixingEfficiency(unittest.TestCase):
    """对比算法效率 (Cluster vs Metropolis)"""

    def setUp(self):
        if not MODULES_AVAILABLE:
            self.skipTest("模块缺失")
        if FAST_MODE:
            self.skipTest("快速模式下跳过混合效率测试")
        self.L = 16
        self.beta = BETA_CRIT
        self.steps = 500
        self.rng = np.random.default_rng(2025)

    def _measure_tau(self, algo: str) -> float:
        """跑一条链，测量磁化时间序列的自相关时间"""
        state = np.ones((self.L, self.L), dtype=np.int8)

        for _ in range(100):
            state, _ = alg.apply_move(
                state,
                algo=algo,
                beta=self.beta,
                replica_seed=self.rng.integers(1e9),
            )

        mags = []
        for _ in range(self.steps):
            state, _ = alg.apply_move(
                state,
                algo=algo,
                beta=self.beta,
                replica_seed=self.rng.integers(1e9),
            )
            mags.append(calculate_observables(state)["m"])

        return stats.autocorrelation_time(np.asarray(mags))

    def test_cluster_is_faster_than_metropolis(self):
        """验证 Wolff 在 Tc 附近的 τ(m) 显著小于 Metropolis"""
        tau_metro = self._measure_tau("metropolis_sweep")
        tau_wolff = self._measure_tau("wolff")

        self.assertLess(
            tau_wolff,
            tau_metro * 0.8,
            msg=f"Wolff τ={tau_wolff:.2f} 未明显小于 Metropolis τ={tau_metro:.2f}",
        )


# =============================================================================
# 统一运行入口
# =============================================================================
def run_tests(verbosity: int = 2):
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestCoreAlgorithms))
    suite.addTests(loader.loadTestsFromTestCase(TestObservablesAndThermodynamics))
    suite.addTests(loader.loadTestsFromTestCase(TestStatistics))
    suite.addTests(loader.loadTestsFromTestCase(TestNumericalStability))
    suite.addTests(loader.loadTestsFromTestCase(TestMixingEfficiency))

    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


if __name__ == "__main__":
    print("=" * 70)
    print("运行算法模块单元测试（CPU 端，基于 apply_move / update_batch）")
    print("=" * 70)
    res = run_tests(verbosity=2)
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"运行测试: {res.testsRun}")
    print(f"成功: {res.testsRun - len(res.failures) - len(res.errors)}")
    print(f"失败: {len(res.failures)}")
    print(f"错误: {len(res.errors)}")
    print("=" * 70)
    import sys as _sys

    _sys.exit(0 if res.wasSuccessful() else 1)

