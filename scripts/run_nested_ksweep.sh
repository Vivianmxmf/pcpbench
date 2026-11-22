#!/usr/bin/env bash
set -euo pipefail

PY="${PY:?set PY to the absolute path of the interpreter}"
CACHES="${CACHES:-caches}"
SEEDS="${SEEDS:-30}"
JOBS="${JOBS:-1}"
FORCE="${FORCE:-0}"
STAMP="${STAMP:-unknown}"
OUT_DIR="outputs/nested_ksweep"
NOSHIFT_BASES="contrast_s3 vitL_contrast_s3"
NOSHIFT_VIEW_SEEDS="0 1 2 3 4"
GRID_RE='^cifar100c_(vitL_)?(gaussian_noise|shot_noise|defocus_blur|gaussian_blur|contrast|brightness|fog|jpeg_compression)_s[1-5]\.npz$'
export PY CACHES SEEDS FORCE STAMP

sha_of() {
    if command -v sha256sum > /dev/null; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

mark_want() {
    printf '%s seeds=%s sha256=%s' "$STAMP" "$SEEDS" "$(sha_of "$R_OUT")"
}

resolve() {
    local kind="$1" cache="$2" view_seed="$3"
    local name backbone
    name="$(basename "$cache" .npz)"
    name="${name#cifar100c_}"
    if [ "$kind" = "noshift" ]; then
        case "$name" in
            vitL_*) backbone="vitL" ;;
            *) backbone="vitB" ;;
        esac
        R_OUT="${OUT_DIR}/noshift_k_sweep_${backbone}_v${view_seed}.json.gz"
        R_LABEL="noshift_${backbone}_v${view_seed}"
    else
        R_OUT="${OUT_DIR}/nested_k_sweep_${name}.json.gz"
        R_LABEL="${name}"
    fi
    R_LOG="logs/nested_ksweep_${R_LABEL}.log"
    R_MARK="logs/nested_ksweep_${R_LABEL}.written_by"
}

run_one() {
    local kind="$1" cache="$2" view_seed="$3"
    local view_args="" want have=""
    resolve "$kind" "$cache" "$view_seed"
    if [ ! -f "$cache" ]; then
        echo "FAIL: ${R_LABEL} (cache ${cache} not found)"
        return 1
    fi
    if [ -f "$R_OUT" ] && [ "$FORCE" != "1" ]; then
        want="$(mark_want)"
        if [ -f "$R_MARK" ]; then
            have="$(cat "$R_MARK")"
        fi
        if [ "$have" = "$want" ]; then
            echo "SKIP: ${R_LABEL} (output ${R_OUT} already written by ${STAMP} with ${SEEDS} seeds)"
            return 0
        fi
        echo "RERUN: ${R_LABEL} (the output on disk is not one that ${STAMP} with ${SEEDS} seeds wrote; mark: '${have:-none}')"
    fi
    if [ "$kind" = "noshift" ]; then
        view_args="--view no_shift --view-seed ${view_seed}"
    fi
    # shellcheck disable=SC2086
    if "$PY" scripts/controlled_k_sweep.py \
        --cache "$cache" \
        --k-values 10,25,50,100 \
        --subset-seeds "$SEEDS" \
        --n-cal-per-class 50 \
        --alpha 0.10 \
        --eval-frac 0.5 \
        --subset-design nested \
        --stamp "$STAMP" \
        $view_args \
        --out "$R_OUT" \
        > "$R_LOG" 2>&1; then
        printf '%s\n' "$(mark_want)" > "$R_MARK"
        echo "DONE: ${R_LABEL}"
    else
        echo "FAIL: ${R_LABEL} (see ${R_LOG})"
        return 1
    fi
}

mkdir -p "$OUT_DIR" logs

if [ "${1:-}" = "--one" ]; then
    run_one "$2" "$3" "$4"
    exit $?
fi

TASKS=""
nl=$'\n'
found=0
shopt -s nullglob
for cache in "${CACHES}"/cifar100c_*_s*.npz; do
    if [[ "$(basename "$cache")" =~ $GRID_RE ]]; then
        TASKS="${TASKS}shift ${cache} -${nl}"
        found=1
    else
        echo "IGNORE: ${cache} (not a cache of the 8 by 5 severity grid)"
    fi
done
shopt -u nullglob
if [ "$found" -eq 0 ]; then
    echo "No grid caches matching ${CACHES}/cifar100c_*_s*.npz. Convert or build some first (see docs/DATASHEET.md)."
    exit 1
fi
for base in $NOSHIFT_BASES; do
    for v in $NOSHIFT_VIEW_SEEDS; do
        TASKS="${TASKS}noshift ${CACHES}/cifar100c_${base}.npz ${v}${nl}"
    done
done

printf '%s' "$TASKS" | xargs -P "$JOBS" -n 3 bash "$0" --one || true

total=0
nfail=0
failed_lines=""
while read -r kind cache v; do
    [ -n "$kind" ] || continue
    resolve "$kind" "$cache" "$v"
    total=$((total + 1))
    if [ ! -f "$R_OUT" ] || [ ! -f "$R_MARK" ] || [ "$(cat "$R_MARK")" != "$(mark_want)" ]; then
        nfail=$((nfail + 1))
        failed_lines="${failed_lines}  - ${R_LABEL}${nl}"
    fi
done <<< "$TASKS"

if [ "$nfail" -gt 0 ]; then
    echo "NESTED K-SWEEP COMPLETED WITH ${nfail} FAILED RUN(S):"
    printf '%s' "$failed_lines"
    exit 1
fi
echo "NESTED K-SWEEP COMPLETED: all ${total} runs present."
