# run_path_A_independent.py
from pathlib import Path
from ising_fss.simulation import batch_runner

# ===== 可改参数 =====
BASE_OUTDIR = Path("data/ising_L64_wolff_independent")  # 基础输出目录
L           = 64
ALGO        = "wolff"       # 'wolff' | 'swendsen_wang' | 'metropolis' (-> 'metropolis_sweep')
H_FIELD     = 0.0
EQUIL       = 4096          # 预热 sweep 数
PROD        = 10000         # 生产 sweep 数（batch_runner 的 CLI 用 prod，不是 n_configs）
THIN        = 8
NWORKERS    = 4

# 单温示例
T_SINGLE    = 2.27

# 多温独立采样示例（彼此不交换）
T_MIN, T_MAX = 1.6, 3.2
N_T          = 5             # 演示值；实际可以 40

def _algo_name(a: str) -> str:
    a = a.lower()
    return "metropolis_sweep" if a in ("metropolis", "metro", "metropolis_sweep") else a

def run_one_temperature(T: float, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    argv = [
        "--mode", "run_workers",          # 按 batch_runner 的 usage
        "--outdir", str(outdir),
        "--nworkers", str(NWORKERS),
        "--L", str(L),
        "--T", f"{T:.6f}",
        "--equil", str(EQUIL),
        "--prod", str(PROD),
        "--thin", str(THIN),
        "--replicas", "1",                # 非 REMC：1 个副本
        "--algo", _algo_name(ALGO),
        "--spacing", "geom",              # 此处对非 REMC无实质影响，但参数需要
        "--h", str(H_FIELD),
        "--save_lattices",                # 如需格点快照/落盘
    ]
    batch_runner.main(argv)

def linspace(a: float, b: float, n: int):
    if n < 2:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]

if __name__ == "__main__":
    # ---- (A) 单温运行 ----
    out_single = BASE_OUTDIR / f"T_{T_SINGLE:.4f}"
    run_one_temperature(T_SINGLE, out_single)

    # ---- (B) 多温独立采样（循环调用，每温度各自目录）----
    for T in linspace(T_MIN, T_MAX, N_T):
        out_T = BASE_OUTDIR / f"T_{T:.4f}"
        run_one_temperature(T, out_T)

    # ===== 读回一个样本做快速检查（可选）=====
    try:
        from ising_fss.analysis.dl_tools import IsingH5Dataset
        # 注意：上面每个温度一个子目录；这里随便挑一个目录检查
        any_train = next((p / "train.h5" for p in (BASE_OUTDIR.iterdir())
                          if (p.is_dir() and (p / "train.h5").exists())), None)
        if any_train:
            ds = IsingH5Dataset(str(any_train), split="train")
            x0, y0 = ds[0]
            print(f"[OK] Loaded {len(ds)} samples from {any_train}")
            print(" sample[0] shape:", getattr(x0, "shape", None), "label:", y0)
        else:
            print("[INFO] No train.h5 found yet under", BASE_OUTDIR)
    except Exception as e:
        print("[WARN] Could not import/read via dl_tools.IsingH5Dataset:", e)

