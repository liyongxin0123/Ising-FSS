# examples/run_from_yaml.py

import os
from ising_fss.utils.config import load_config, validate_config
from ising_fss.simulation.remc_simulator import HybridREMCSimulator
#  from ising_fss.simulation.gpu_remc_simulator import GPU_REMC_Simulator
from ising_fss.simulation.dispatcher import make_replica_seeds


def main():
    # 1. 读取 YAML 配置并做一致性检查
    cfg = load_config("configs/config_L64.yaml")

    ok, warnings = validate_config(cfg)
    if not ok:
        for w in warnings:
            print("[config warning]", w)

    sim_cfg = cfg.simulation
    data_cfg = cfg.data

    # 2. 根据 backend 选择 CPU / GPU 版本的 REMC 模拟器
    backend = sim_cfg.backend.lower()
    SimCls = GPU_REMC_Simulator if backend == "gpu" else HybridREMCSimulator

    # 3. 生成显式 replica_seeds（Hybrid/GPU 两个类都要求显式种子）
    master_seed = sim_cfg.seed or 0   # 如果 YAML 里没写 seed，就用 0 或你喜欢的数
    replica_seeds = make_replica_seeds(master_seed, sim_cfg.num_replicas)

    # 4. 构造模拟器实例
    sim = SimCls(
        L=sim_cfg.L,
        T_min=sim_cfg.T_min,
        T_max=sim_cfg.T_max,
        num_replicas=sim_cfg.num_replicas,
        algorithm=sim_cfg.algorithm,                  # 已在 SimulationConfig 里归一化
        spacing=getattr(sim_cfg, "temp_spacing", "geom"),
        h=sim_cfg.h_field,
        replica_seeds=replica_seeds,                  # ★ 关键：显式传入
    )

    # 5. 运行 REMC
    outdir = "runs/L64_from_yaml"
    os.makedirs(outdir, exist_ok=True)

    thin = getattr(data_cfg, "sampling_interval", 1)  # 采样间隔放在 DataConfig 里

    sim.run(
        equilibration_steps=sim_cfg.equilibration,
        production_steps=sim_cfg.production,
        exchange_interval=sim_cfg.exchange_interval,
        thin=thin,
        save_lattices=True,
        save_dir=outdir,
        worker_id=f"{backend}_yaml_demo",
    )

    stats = sim.analyze(verbose=True)
    print("平均交换率:", stats["swap"]["rate"])


if __name__ == "__main__":
    main()

