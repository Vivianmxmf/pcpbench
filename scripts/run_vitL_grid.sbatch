#!/usr/bin/env bash
#SBATCH --job-name=pcp_vitL
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.out

PY="${PY:-python}"
NGPU="${SLURM_GPUS_ON_NODE:-4}"
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
mkdir -p build logs caches
CORRS=(gaussian_noise shot_noise defocus_blur gaussian_blur contrast brightness fog jpeg_compression)
MODEL=ViT-L-14
PRE=laion2b_s32b_b82k

"$PY" -c "import open_clip; open_clip.create_model_and_transforms('$MODEL', pretrained='$PRE'); print('ViT-L weights cached')"

i=0
for DS in cifar10c cifar100c; do
  B=scripts/build_${DS}_clip_cache.py
  for c in "${CORRS[@]}"; do
    for s in 1 2 3 4 5; do
      out="build/${DS}_vitL_${c}_s${s}.npz"
      if [ -s "$out" ]; then echo "skip $out (exists)"; continue; fi
      g=$(( i % NGPU ))
      CUDA_VISIBLE_DEVICES=$g "$PY" "$B" --corruption "$c" --severity "$s" \
        --clip-model "$MODEL" --pretrained "$PRE" --device cuda \
        --output "$out" > "logs/build_${DS}_vitL_${c}_s${s}.log" 2>&1 &
      i=$(( i + 1 ))
      (( i % NGPU == 0 )) && wait
    done
  done
done
wait
echo "built: $(ls build/*_vitL_*_s*.npz 2>/dev/null | wc -l)/80"

grep -H "top-1" logs/build_*_vitL_*_s*.log | sort

for DS in cifar10c cifar100c; do
  for c in "${CORRS[@]}"; do
    for s in 1 2 3 4 5; do
      "$PY" -m pcpbench.datasets.convert \
        --in "build/${DS}_vitL_${c}_s${s}.npz" \
        --out "caches/${DS}_vitL_${c}_s${s}.npz" \
        --name "${DS}_vitL_${c}_s${s}" --shift-family corruption 2>&1 | tail -1
    done
  done
done
echo "converted: $(ls caches/*_vitL_*_s*.npz 2>/dev/null | wc -l)/80"
echo "VITL BUILD DONE (run 'bash scripts/run_grid.sh' once, after BOTH jobs finish)"
