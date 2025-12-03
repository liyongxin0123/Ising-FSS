# -*- coding: utf-8 -*-
"""
核心算法模块单元测试 (Modernized)

覆盖范围：
- 算法逻辑：Metropolis, Wolff, Swendsen-Wang 通过 apply_move 接口调用
- 批量更新：update_batch 接口及 replica_seeds 机制
- 物理性质：能量边界、磁化强度、临界慢化对比
- 统计工具：自相关时间、分块分析
- 可复现性：基于显式种子的严格一致性检查

适配说明：
- 使用 apply_move 替代了旧的直接函数调用
- 适配了新的 (lattice, info) 返回值签名（info 为 MoveInfo 数据类）
"""

import unittest
import sys
import os
import numpy as np
from pathlib import Path

# ----------------------------- 路径适配 -----------------------------
# 确保能导入 src
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
    print(f"Skipping tests due to import failure: {e}")
    MODULES_AVAILABLE = False

# ----------------------------- 测试配置 -----------------------------
FAST_MODE = os.environ.get("ISING_FAST_TEST", "0") == "1"
L_SMALL = 16
Tc_THEORY = 2.269185
BETA_CRIT = 1.0 / Tc_THEORY
BETA_LOW = 1.0   # 低温 (有序)
BETA_HIGH = 0.1  # 高温 (无序)


class TestCoreAlgorithms(unittest.TestCase):
    """测试 algorithms.py 中的核心蒙特卡洛算法"""

    def setUp(self):
        if not MODULES_AVAILABLE:
            self.skipTest("核心模块未找到")
        self.rng = np.random.default_rng(2025)
        self.ones_state = np.ones((L_SMALL, L_SMALL), dtype=np.int8)
        self.rand_state = self.rng.choice([-1, 1], size=(L_SMALL, L_SMALL)).astype(np.int8)

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

        # meta 是 MoveInfo 数据类，这里只检查“公共字段”
        # rng_model 是新接口里比较核心的信息来源
        self.assertTrue(hasattr(meta, "rng_model"))
        # accepted 是常见字段，但不是强制；若存在就检查类型
        if hasattr(meta, "accepted"):
            self.assertIsInstance(meta.accepted, (int, np.integer))

        # Wolff
        res, meta = alg.apply_move(
            self.rand_state.copy(),
            algo="wolff",
            beta=BETA_CRIT,
            replica_seed=42,
        )

        # Wolff 情况下应至少包含簇大小信息
        self.assertTrue(hasattr(meta, "cluster_size"))
        self.assertIsInstance(meta.cluster_size, (int, np.integer))

    def test_reproducibility_deterministic(self):
        """测试显式种子下的严格可复现性"""
        beta = 0.44
        seed = 12345

        # Metropolis：格点配置必须完全一致
        res1, meta1 = alg.apply_move(
            self.rand_state.copy(), "metropolis_sweep", beta, replica_seed=seed
        )
        res2, meta2 = alg.apply_move(
            self.rand_state.copy(), "metropolis_sweep", beta, replica_seed=seed
        )
        np.testing.assert_array_equal(res1, res2, err_msg="Metropolis 结果不可复现")

        # 若 MoveInfo 里有 accepted 字段，也要求一致；没有就忽略
        if hasattr(meta1, "accepted") and hasattr(meta2, "accepted"):
            self.assertEqual(meta1.accepted, meta2.accepted)

        # Wolff：配置和簇大小都应可复现
        res1, meta1 = alg.apply_move(
            self.rand_state.copy(), "wolff", beta, replica_seed=seed
        )
        res2, meta2 = alg.apply_move(
            self.rand_state.copy(), "wolff", beta, replica_seed=seed
        )
        np.testing.assert_array_equal(res1, res2, err_msg="Wolff 结果不可复现")
        self.assertTrue(hasattr(meta1, "cluster_size"))
        self.assertTrue(hasattr(meta2, "cluster_size"))
        self.assertEqual(meta1.cluster_size, meta2.cluster_size)

    def test_physics_metropolis_relaxation(self):
        """测试 Metropolis 在低温下能否降低能量"""
        # 从随机态开始，在低温下演化
        state = self.rand_state.copy()
        obs_start = calculate_observables(state)

        # 演化 50 步
        for i in range(50):
            state, _ = alg.apply_move(
                state, "metropolis_sweep", BETA_LOW, replica_seed=100 + i
            )

        obs_end = calculate_observables(state)

        # 能量应显著降低 (变得更负)
        self.assertLess(obs_end["E_per_spin"], obs_start["E_per_spin"])
        # 磁化强度应增大 (走向有序)
        self.assertGreater(obs_end["abs_m"], obs_start["abs_m"])

    def test_field_constraint(self):
        """测试当 h!=0 时，Cluster 算法是否被禁止"""
        with self.assertRaises(ValueError):
            alg.apply_move(
                self.ones_state,
                "wolff",
                beta=0.4,
                h=0.1,
                replica_seed=1,
            )

        with self.assertRaises(ValueError):
            alg.apply_move(
                self.ones_state,
                "swendsen_wang",
                beta=0.4,
                h=0.1,
                replica_seed=1,
            )

    def test_batch_update(self):
        """测试 update_batch 批量接口"""
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
        # update_batch 这里返回的 info 仍然是 dict，因此原来的断言可以保留
        self.assertIn("per_replica", info)
        self.assertEqual(len(info["per_replica"]), R)
        # 确保不同种子导致不同结果
        self.assertFalse(np.array_equal(new_batch[0], new_batch[1]))


class TestObservables(unittest.TestCase):
    """测试 observables.py 的物理量计算"""

    def setUp(self):
        if not MODULES_AVAILABLE:
            self.skipTest("模块缺失")
        self.L = 10

    def test_ground_states(self):
        """测试基态能量和磁化"""
        # 1. 铁磁基态 (全 1)
        # 能量 = -J * (2N) bonds (PBC下每个点2个键) -> E/N = -2
        state_fm = np.ones((self.L, self.L), dtype=np.int8)
        obs = calculate_observables(state_fm)
        self.assertAlmostEqual(obs["E_per_spin"], -2.0)
        self.assertAlmostEqual(obs["m"], 1.0)

        # 2. 反铁磁基态 (棋盘格)
        # 能量 = +J * (2N) -> E/N = +2
        idx = np.arange(self.L)
        x, y = np.meshgrid(idx, idx)
        state_afm = np.where((x + y) % 2 == 0, 1, -1).astype(np.int8)
        obs = calculate_observables(state_afm)
        self.assertAlmostEqual(obs["E_per_spin"], 2.0)
        self.assertAlmostEqual(obs["m"], 0.0)


class TestStatistics(unittest.TestCase):
    """测试 statistics.py 的统计工具"""

    def setUp(self):
        if not MODULES_AVAILABLE:
            self.skipTest("模块缺失")
        self.rng = np.random.default_rng(999)

    def test_autocorrelation_time(self):
        """测试自相关时间计算"""
        # 1. 白噪声 (tau ~ 1)
        white = self.rng.standard_normal(10000)
        tau = stats.autocorrelation_time(white)
        self.assertLess(tau, 1.5)

        # 2. 红噪声 (AR1 process, tau > 1)
        # x_t = 0.9 * x_{t-1} + noise
        red = np.zeros(10000)
        for i in range(1, 10000):
            red[i] = 0.9 * red[i - 1] + self.rng.standard_normal()
        tau_red = stats.autocorrelation_time(red)
        self.assertGreater(tau_red, 4.0)

    def test_blocking_analysis(self):
        """测试阻塞分析 (Blocking Analysis)"""
        # 构造一组已知误差的数据
        # mean=10, std=2, N=400 -> stderr_expected = 2/20 = 0.1
        data = self.rng.normal(10.0, 2.0, size=1000)
        err, _ = stats.blocking_analysis(data)
        # 允许一定统计波动
        self.assertTrue(0.05 < err < 0.15)


class TestMixingEfficiency(unittest.TestCase):
    """对比算法效率 (Cluster vs Metropolis)"""

    def setUp(self):
        if not MODULES_AVAILABLE:
            self.skipTest("模块缺失")
        if FAST_MODE:
            self.skipTest("快速模式跳过混合效率测试")
        self.L = 16
        self.beta = BETA_CRIT
        self.steps = 500
        self.rng = np.random.default_rng(2025)

    def _measure_tau(self, algo: str) -> float:
        # 跑一条链，测量磁化的自相关时间
        state = np.ones((self.L, self.L), dtype=np.int8)
        mags = []
        # 热化
        for _ in range(100):
            state, _ = alg.apply_move(
                state, algo, self.beta, replica_seed=self.rng.integers(1e9)
            )

        # 采样
        for _ in range(self.steps):
            state, _ = alg.apply_move(
                state, algo, self.beta, replica_seed=self.rng.integers(1e9)
            )
            mags.append(calculate_observables(state)["m"])

        return stats.autocorrelation_time(np.array(mags))

    def test_cluster_is_faster(self):
        """验证 Wolff 算法在临界点附近的自相关时间显著小于 Metropolis"""
        tau_metro = self._measure_tau("metropolis_sweep")
        tau_wolff = self._measure_tau("wolff")

        # Wolff 应该快得多 (tau 更小)
        # 注意：由于 steps 较短，统计可能有波动，这里放宽条件
        self.assertLess(
            tau_wolff,
            tau_metro * 0.8,
            f"Wolff ({tau_wolff:.2f}) 应比 Metropolis ({tau_metro:.2f}) 更快去相关",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

