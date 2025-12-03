# run_path_B_remc.py  —— 方式 B：通过 import 调用 batch_runner.main(argv)
from pathlib import Path
from ising_fss.simulation import batch_runner

# ===== 你可以改动的参数 =====
OUTPUT_DIR = "data/ising_L64_remc"   # 结果目录
L          = 64                      # ⚠ PBC + Metropolis 需偶数 L
ALGO       = "metropolis"            # GPU 仅支持 Metropolis（内部会归一化为 'metropolis_sweep'）
H_FIELD    = 0.0

# 想覆盖的温区（batch_runner 用中心温度来生成温标）
T_MIN, T_MAX   = 2.0, 2.6
T_CENTER       = 0.5 * (T_MIN + T_MAX)   # 这里作为中心温度传给 --T
N_REPLICAS     = 24                       # REMC 副本数
SPACING        = "geom"                   # 'geom' | 'linear'

# MC 步数与抽稀
EQUIL          = 8192                     # 预热 sweeps
PROD           = 20000                    # 生产 sweeps（注意：batch_runner 是按 sweeps，不是“n_configs”）
THIN           = 8
EXCHANGE_INTERVAL = 10
NWORKERS       = 4
SEED           = 12345                    # 可选；None 则不传

# 是否保存格点快照
SAVE_LATTICES  = True

def _algo_name(a: str) -> str:
    a = a.lower()
    return "metropolis_sweep" if a in ("metropolis", "metro", "metropolis_sweep") else a

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    argv = [
        "--mode", "run_workers",
        "--outdir", OUTPUT_DIR,
        "--nworkers", str(NWORKERS),
        "--L", str(L),
        "--T", f"{T_CENTER:.6f}",                 # 中心温度；温标由 batch_runner 依据 replicas/spacing 生成
        "--equil", str(EQUIL),
        "--prod", str(PROD),
        "--exchange_interval", str(EXCHANGE_INTERVAL),
        "--thin", str(THIN),
        "--replicas", str(N_REPLICAS),            # REMC 关键：>1
        "--algo", _algo_name(ALGO),
        "--spacing", SPACING,
        "--h", str(H_FIELD),
    ]
    if SEED is not None:
        argv += ["--seed", str(SEED)]
    if SAVE_LATTICES:
        argv += ["--save_lattices"]

    # 调用包内入口
    batch_runner.main(argv)

    # ===== 用 dl_tools 读回数据做一个快速检查（可选）=====
    try:
        from ising_fss.analysis.dl_tools import IsingH5Dataset
        train_path = Path(OUTPUT_DIR) / "train.h5"
        if train_path.exists():
            ds = IsingH5Dataset(str(train_path), split="train")
            x0, y0 = ds[0]
            print(f"[OK] Loaded {len(ds)} samples from {train_path}")
            print(" sample[0] shape:", getattr(x0, "shape", None), "label:", y0)
        else:
            print(f"[INFO] {train_path} not found; check batch_runner's output schema for your version.")
    except Exception as e:
        print("[WARN] Read-back via IsingH5Dataset failed:", e)

if __name__ == "__main__":
    # 多进程/REMC 场景务必加这一行保护
    main()

