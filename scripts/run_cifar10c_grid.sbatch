#!/usr/bin/env bash
#SBATCH --job-name=pcp_c10c
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.out

PY="${PY:-python}"
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
mkdir -p build logs caches
CORRS=(gaussian_noise shot_noise defocus_blur gaussian_blur contrast brightness fog jpeg_compression)

i=0
for c in "${CORRS[@]}"; do
  for s in 1 2 3 4 5; do
    g=$(( i % 8 ))
    CUDA_VISIBLE_DEVICES=$g "$PY" scripts/build_cifar10c_clip_cache.py \
      --corruption "$c" --severity "$s" --device cuda \
      --output "build/cifar10c_${c}_s${s}.npz" > "logs/build_cifar10c_${c}_s${s}.log" 2>&1 &
    i=$(( i + 1 ))
    (( i % 8 == 0 )) && wait
  done
done
wait
echo "built: $(ls build/cifar10c_*_s*.npz 2>/dev/null | wc -l)/40"

grep -H "top-1" logs/build_cifar10c_*_s*.log | sort

for c in "${CORRS[@]}"; do
  for s in 1 2 3 4 5; do
    "$PY" -m pcpbench.datasets.convert \
      --in "build/cifar10c_${c}_s${s}.npz" \
      --out "caches/cifar10c_${c}_s${s}.npz" \
      --name "cifar10c_${c}_s${s}" --shift-family corruption 2>&1 | tail -1
  done
done
echo "converted: $(ls caches/cifar10c_*_s*.npz 2>/dev/null | wc -l)/40"
echo "BUILD DONE (leaderboard is a SEPARATE CPU job: sbatch scripts/run_grid.sbatch)"
