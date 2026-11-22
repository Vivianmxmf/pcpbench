#!/usr/bin/env bash
set -euo pipefail

FAILURES=()

PY="${PY:-python}"

CACHES="${CACHES:-caches}"
FORCE="${FORCE:-0}"

mkdir -p outputs/analysis logs

run_one() {
    local cache="$1"
    local stem
    stem="$(basename "$cache" .npz)"
    local name="${stem#cifar100c_}"
    local out="outputs/analysis/controlled_k_sweep_${name}.json"
    if [ -f "$out" ] && [ "$FORCE" != "1" ]; then
        echo "SKIP: ${name} (output ${out} already exists; set FORCE=1 to rerun)"
        return 0
    fi
    if "$PY" scripts/controlled_k_sweep.py \
        --cache "$cache" \
        --k-values 10,25,50,100 \
        --subset-seeds 5 \
        --n-cal-per-class 50 \
        --alpha 0.10 \
        --eval-frac 0.5 \
        --out "$out" \
        2>&1 | tee "logs/ksweep_${name}.log"; then
        :
    else
        echo "FAIL: ${name} (see logs/ksweep_${name}.log)"
        FAILURES+=("${name}")
    fi
}

shopt -s nullglob
found=0
for cache in "${CACHES}"/cifar100c_*_s*.npz; do
    run_one "$cache"
    found=1
done
shopt -u nullglob
if [ "$found" -eq 0 ]; then
    echo "No caches matching ${CACHES}/cifar100c_*_s*.npz -- convert or build some first (see docs/DATASHEET.md)."
fi

if [ "${#FAILURES[@]}" -gt 0 ]; then
    echo "K-SWEEP COMPLETED WITH ${#FAILURES[@]} FAILED CACHE(S):"
    printf '  - %s\n' "${FAILURES[@]}"
    exit 1
fi
echo "K-SWEEP COMPLETED: all present caches succeeded."
