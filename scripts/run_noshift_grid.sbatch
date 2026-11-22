#!/usr/bin/env bash
#SBATCH --job-name=pcp_noshift_grid
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.out

export PY="${PY:-python}"
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
mkdir -p outputs/noshift logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

FORCE="${FORCE:-0}"
FAILURES=()

run_one() {
    local cache="$1"
    local stem
    stem="$(basename "$cache" .npz)"
    local out="outputs/noshift/${stem}.json"
    if [ -f "$out" ] && [ "$FORCE" != "1" ]; then
        echo "SKIP: ${stem} (output ${out} already exists; set FORCE=1 to rerun)"
        return 0
    fi
    if "$PY" scripts/no_shift_reference.py \
        --cache "$cache" \
        --alpha 0.10 \
        --seeds 10 \
        --eval-frac 0.5 \
        --out "$out" \
        > "logs/noshift_${stem}.log" 2>&1; then
        echo "OK:   ${stem}"
    else
        echo "FAIL: ${stem} (see logs/noshift_${stem}.log)"
        FAILURES+=("${stem}")
    fi
}

shopt -s nullglob
found=0
for cache in caches/*.npz; do
    run_one "$cache"
    found=1
done
shopt -u nullglob
if [ "$found" -eq 0 ]; then
    echo "No caches found under caches/ -- convert or build some first (see docs/DATASHEET.md)."
fi

if [ "${#FAILURES[@]}" -gt 0 ]; then
    echo "NO-SHIFT GRID COMPLETED WITH ${#FAILURES[@]} FAILED CACHE(S):"
    printf '  - %s\n' "${FAILURES[@]}"
    exit 1
fi
echo "NO-SHIFT GRID COMPLETED: all present caches succeeded."
