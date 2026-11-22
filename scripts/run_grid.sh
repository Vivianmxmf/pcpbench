#!/usr/bin/env bash
set -euo pipefail

FAILURES=()

PY="${PY:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

CACHES="${CACHES:-caches}"

SCORE="${SCORE:-lac}"
if [ "$SCORE" = "lac" ]; then
    SUFFIX=""
    OUTDIR="outputs"
    LEADERBOARD="leaderboard.md"
else
    SUFFIX="_${SCORE}"
    OUTDIR="outputs/${SCORE}"
    LEADERBOARD="leaderboard_${SCORE}.md"
fi

read -r -a METHOD_LIST <<< "${METHODS:-split mondrian weighted ppi acp oracle clustered}"

mkdir -p "$OUTDIR" logs

run_if() {
    local cache="$1"
    local name="$2"
    if [ ! -f "$cache" ]; then
        echo "SKIP: $name (cache $cache not found)"
        return 0
    fi
    for m in "${METHOD_LIST[@]}"; do
        if [ "$m" = "acp" ] && [ "$SCORE" != "lac" ]; then
            echo "SKIP: ${name}/acp (acp ignores SCORE=${SCORE})"
            continue
        fi
        local tag="${name}_${m}${SUFFIX}"
        if "$PY" -m pcpbench.cli run \
            --dataset "$cache" \
            --method "$m" \
            --alpha 0.10 \
            --seeds 10 \
            --score "$SCORE" \
            --out "${OUTDIR}/${tag}.json" \
            2>&1 | tee "logs/${tag}.log"; then
            :
        else
            echo "FAIL: ${name}/${m}${SUFFIX} (see logs/${tag}.log)"
            FAILURES+=("${name}/${m}${SUFFIX}")
        fi
    done
}

shopt -s nullglob
found=0
for cache in "${CACHES}"/*.npz; do
    run_if "$cache" "$(basename "$cache" .npz)"
    found=1
done
shopt -u nullglob
if [ "$found" -eq 0 ]; then
    echo "No caches found under ${CACHES}/ -- convert or build some first (see docs/DATASHEET.md)."
    exit 1
fi

"$PY" -m pcpbench.leaderboard --in "$OUTDIR" --out "$LEADERBOARD"
echo "leaderboard written to $LEADERBOARD"

if [ "${#FAILURES[@]}" -gt 0 ]; then
    echo "GRID COMPLETED WITH ${#FAILURES[@]} FAILED CELL(S):"
    printf '  - %s\n' "${FAILURES[@]}"
    exit 1
fi
echo "GRID COMPLETED: all present cells succeeded."
