#!/usr/bin/env bash
#SBATCH --job-name=pcp_inet
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.out

set -uo pipefail

PY="${PY:-python}"
cd "${SLURM_SUBMIT_DIR:-.}" || { echo "ERROR: could not cd to ${SLURM_SUBMIT_DIR:-.}" >&2; exit 1; }
mkdir -p build logs caches

NGPU="${SLURM_GPUS_ON_NODE:-4}"

MAXJOBS="${MAXJOBS:-2}"

echo "=== HF auth preflight: probing gated ILSVRC/imagenet-1k dataset ==="
if ! "$PY" - <<'PYEOF'
import sys
try:
    from datasets import load_dataset
    load_dataset("ILSVRC/imagenet-1k", split="validation[:2]", cache_dir="data/imagenet")
except Exception as e:
    print(f"HF AUTH PREFLIGHT FAILED: {e}", file=sys.stderr)
    sys.exit(1)
print("HF AUTH PREFLIGHT OK: ILSVRC/imagenet-1k reachable.")
PYEOF
then
  echo "=================================================================================" >&2
  echo "ERROR: could not load the gated ILSVRC/imagenet-1k dataset on HuggingFace." >&2
  echo "Refusing to burn the 8h GPU allocation without dataset access. Most likely a" >&2
  echo "missing HF login/grant, but also check network access, the 'datasets' package" >&2
  echo "install, and available disk space. Fix BEFORE resubmitting this job:" >&2
  echo "  1. huggingface-cli login   (or export HF_TOKEN / HUGGING_FACE_HUB_TOKEN)" >&2
  echo "  2. request access and wait for approval at:" >&2
  echo "     https://huggingface.co/datasets/ILSVRC/imagenet-1k" >&2
  echo "=================================================================================" >&2
  exit 1
fi

BACKBONES=(
  "ViT-B-32|laion2b_s34b_b79k|vitB32"
  "ViT-L-14|laion2b_s32b_b82k|vitL14"
)

for bb in "${BACKBONES[@]}"; do
  IFS='|' read -r MODEL PRE SLUG <<< "$bb"
  "$PY" -c "import open_clip; open_clip.create_model_and_transforms('$MODEL', pretrained='$PRE'); print('$SLUG weights cached')"
done

MANIFEST=(
  "ViT-B-32|laion2b_s34b_b79k|vitB32|gaussian_noise|3"
  "ViT-B-32|laion2b_s34b_b79k|vitB32|defocus_blur|3"
  "ViT-B-32|laion2b_s34b_b79k|vitB32|contrast|3"
  "ViT-B-32|laion2b_s34b_b79k|vitB32|fog|3"
  "ViT-L-14|laion2b_s32b_b82k|vitL14|gaussian_noise|3"
  "ViT-L-14|laion2b_s32b_b82k|vitL14|defocus_blur|3"
  "ViT-L-14|laion2b_s32b_b82k|vitL14|contrast|3"
  "ViT-L-14|laion2b_s32b_b82k|vitL14|fog|3"
  "ViT-B-32|laion2b_s34b_b79k|vitB32|gaussian_noise|1"
  "ViT-B-32|laion2b_s34b_b79k|vitB32|gaussian_noise|5"
)

launched=0
for cell in "${MANIFEST[@]}"; do
  IFS='|' read -r MODEL PRE SLUG c s <<< "$cell"
  out="build/imagenet1k_${SLUG}_${c}_s${s}.npz"
  if [ -s "$out" ]; then echo "skip $out (exists)"; continue; fi
  g=$(( launched % NGPU ))
  CUDA_VISIBLE_DEVICES=$g "$PY" scripts/build_imagenetc_clip_cache.py \
    --root data/imagenet --num-classes 1000 --corruption "$c" --severity "$s" \
    --clip-model "$MODEL" --pretrained "$PRE" --device cuda --batch 256 \
    --output "$out" > "logs/build_imagenet1k_${SLUG}_${c}_s${s}.log" 2>&1 &
  launched=$(( launched + 1 ))
  (( launched % MAXJOBS == 0 )) && wait
done
wait

FAILED=()
VALID=()
for i in "${!MANIFEST[@]}"; do
  cell="${MANIFEST[$i]}"
  IFS='|' read -r MODEL PRE SLUG c s <<< "$cell"
  out="build/imagenet1k_${SLUG}_${c}_s${s}.npz"
  if [ ! -s "$out" ]; then
    FAILED+=("$SLUG/$c/s$s (missing/empty build)")
    continue
  fi
  if ! "$PY" - "$out" <<'PYEOF'
import sys
import numpy as np
z = np.load(sys.argv[1])
sl, tl = z["source_logits"], z["target_logits"]
sy, ty = z["source_labels"], z["target_labels"]
assert sl.shape[1] == 1000 and tl.shape[1] == 1000, ("K", sl.shape[1], tl.shape[1])
assert sl.shape[0] == 50000 and tl.shape[0] == 50000, ("N_logits", sl.shape[0], tl.shape[0])
assert sy.shape[0] == 50000 and ty.shape[0] == 50000, ("N_labels", sy.shape[0], ty.shape[0])
assert np.isfinite(sl).all() and np.isfinite(tl).all(), "non-finite logits"
assert int(sy.min()) >= 0 and int(sy.max()) < 1000, ("src label range", int(sy.min()), int(sy.max()))
assert int(ty.min()) >= 0 and int(ty.max()) < 1000, ("tgt label range", int(ty.min()), int(ty.max()))
PYEOF
  then
    FAILED+=("$SLUG/$c/s$s (failed integrity check: K/N/finite/labels)")
    continue
  fi
  VALID+=("$out")
done
echo "built+validated: ${#VALID[@]}/10"

grep -H "top-1" logs/build_imagenet1k_*.log | sort

for f in ${VALID[@]+"${VALID[@]}"}; do
  stem="$(basename "$f" .npz)"
  "$PY" -m pcpbench.datasets.convert \
    --in "$f" --out "caches/${stem}.npz" \
    --name "$stem" --shift-family corruption 2>&1 | tail -1
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    FAILED+=("$stem (convert failed)")
  fi
done
converted=$(ls caches/imagenet1k_*.npz 2>/dev/null | wc -l)
echo "converted: ${converted}/10"

if [ "${#FAILED[@]}" -gt 0 ] || [ "$converted" -ne 10 ]; then
  echo "=================================================================================" >&2
  echo "IMAGENET BUILD INCOMPLETE: ${#FAILED[@]} failed cell(s), ${converted}/10 converted." >&2
  echo "Failed cells:" >&2
  printf '  - %s\n' ${FAILED[@]+"${FAILED[@]}"} >&2
  echo "=================================================================================" >&2
  exit 1
fi

echo "IMAGENET BUILD DONE: all 10 K=1000 caches valid."
echo "Leaderboard is a SEPARATE CPU job, run AFTER this one finishes:"
echo "  sbatch scripts/run_grid.sbatch"
