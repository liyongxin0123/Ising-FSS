"""
启动方式 #2：GPU 版 REMC —— GPU_REMC_Simulator.run(...)
- 显式传入 replica_seeds；通常在 CPU 标定 thin 后固定到 GPU。
"""
import numpy as np

def make_replica_seeds(n: int, master_seed: int = 2025):
    ss = np.random.SeedSequence(master_seed)
    # 使用子 SeedSequence 生成彼此独立的 32-bit 种子
    return [int(cs.generate_state(1, dtype=np.uint32)[0]) for cs in ss.spawn(n)]

if __name__ == "__main__":
    try:
        from ising_fss.simulation.gpu_remc_simulator import GPU_REMC_Simulator
    except Exception as e:
        print("GPU simulator not available:", e)
    else:
        L = 64
        num_replicas = 24
        replica_seeds = make_replica_seeds(num_replicas, master_seed=2025)

        sim = GPU_REMC_Simulator(
            L=L,
            T_min=1.8, T_max=3.2, num_replicas=num_replicas,
            algorithm="metropolis_sweep",
            h=0.0,
            replica_seeds=replica_seeds,
        )
        sim.run(
            equilibration_steps=1000,
            production_steps=6000,
            exchange_interval=1,
            thin=10,              # 建议先用 CPU 标定 τ_int 后确定
            save_lattices=True,
        )
        print("Done. Check outputs:", getattr(sim, "checkpoint", None) or getattr(sim, "results", None))

