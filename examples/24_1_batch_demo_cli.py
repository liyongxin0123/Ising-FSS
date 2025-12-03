# examples/batch_demo_cli.py
"""
展示几条推荐的 batch_runner 命令行。

本文件不直接跑，只是给用户 copy 粘贴用。
"""

EXAMPLE_RUN = r"""
# 启动 4 个 worker，在 L=64、T=2.269 附近进行 REMC 采样
python -m ising_fss.simulation.batch_runner \
  --mode run_workers \
  --outdir data/ising_L64_batch \
  --nworkers 4 \
  --L 64 \
  --T 2.269 \
  --equil 5000 \
  --prod 20000 \
  --exchange_interval 10 \
  --thin 10 \
  --replicas 32 \
  --algo metropolis_sweep \
  --spacing geom \
  --h 0.0 \
  --save_lattices
"""

EXAMPLE_MERGE = r"""
# 在同一个 outdir 下进行合并
python -m ising_fss.simulation.batch_runner \
  --mode merge \
  --outdir data/ising_L64_batch
"""

if __name__ == "__main__":
    print("==== batch_runner run_workers 示例 ====")
    print(EXAMPLE_RUN)
    print("\n==== batch_runner merge 示例 ====")
    print(EXAMPLE_MERGE)

