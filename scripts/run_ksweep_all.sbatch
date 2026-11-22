#!/usr/bin/env bash
#SBATCH --job-name=pcp_ksweep_all
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.out

export PY="${PY:-python}"
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
mkdir -p outputs/analysis logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

bash scripts/run_ksweep_all.sh
echo "K-SWEEP-ALL JOB DONE"
