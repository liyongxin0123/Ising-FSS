#!/usr/bin/env bash
set -e

# 一旦收到 Ctrl-C（SIGINT），就 kill 当前进程组里的所有子进程
trap 'echo "收到 Ctrl-C，中止所有 REMC..."; kill 0; exit 1' INT

mkdir -p logs

# -------- L = 32 --------
python ./36_cpu_remc_fss_pipeline.py \
  --L_list 32 \
  --T_min 2.1 --T_max 2.5 \
  --num_replicas 64 \
  --equil_steps 20000 \
  --prod_steps 40000 \
  --thin 20 \
  --save_lattices \
  --exchange_interval 10 \
  --outdir runs/cpu_L32_thin20 \
  > logs/L32_thin20.log 2>&1 &

# -------- L = 64 --------
python ./36_cpu_remc_fss_pipeline.py \
  --L_list 64 \
  --T_min 2.1 --T_max 2.5 \
  --num_replicas 64 \
  --equil_steps 20000 \
  --prod_steps 180000 \
  --thin 90 \
  --save_lattices \
  --exchange_interval 10 \
  --outdir runs/cpu_L64_thin90 \
  > logs/L64_thin90.log 2>&1 &

# -------- L = 96 --------
python ./36_cpu_remc_fss_pipeline.py \
  --L_list 96 \
  --T_min 2.1 --T_max 2.5 \
  --num_replicas 64 \
  --equil_steps 20000 \
  --prod_steps 800000 \
  --thin 400 \
  --save_lattices \
  --exchange_interval 10 \
  --outdir runs/cpu_L96_thin400 \
  > logs/L96_thin400.log 2>&1 &

# -------- L = 128 --------
python ./36_cpu_remc_fss_pipeline.py \
  --L_list 128 \
  --T_min 2.1 --T_max 2.5 \
  --num_replicas 64 \
  --equil_steps 20000 \
  --prod_steps 3600000 \
  --thin 1800 \
  --save_lattices \
  --exchange_interval 10 \
  --outdir runs/cpu_L128_thin1800 \
  > logs/L128_thin1800.log 2>&1 &


# 等所有 REMC 进程结束
wait
echo "所有 L 的 REMC 都完成了。开始统一 FSS 分析..."

python ./fss_from_multiple_cpu_raw.py


