#!/usr/bin/env bash
#SBATCH --job-name=pcp_nested_ksweep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.out

export PY="${PY:-python}"
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
mkdir -p outputs/nested_ksweep logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

export JOBS="${SLURM_CPUS_PER_TASK:-4}"

RESULT_RE='^.. outputs/(nested_ksweep/|[^/]+_acp\.json$|analysis/(controlled_k_sweep_[A-Za-z0-9_]+|label_complexity_contrast_s3)\.json$)'
check_checkout () {
    local head_now dirty status
    head_now="$(git rev-parse --short HEAD)" || { echo "FAIL: git rev-parse failed"; return 1; }
    if [ "$STAMP" != "$head_now" ]; then
        echo "FAIL: the checkout is at ${head_now}, the job was submitted for ${STAMP}"
        return 1
    fi
    status="$(git status --porcelain --untracked-files=no)" || { echo "FAIL: git status failed"; return 1; }
    dirty="$(printf '%s' "$status" | grep -v -E "$RESULT_RE" || true)"
    if [ -n "$dirty" ]; then
        echo "FAIL: tracked files differ from ${STAMP}:"
        echo "$dirty"
        return 1
    fi
}
STAMP="${STAMP:-$(git rev-parse --short HEAD)}"
export STAMP
echo "code version: ${STAMP}"
check_checkout || exit 1
"$PY" --version

bash scripts/run_nested_ksweep.sh && check_checkout && echo "NESTED-KSWEEP JOB DONE"
