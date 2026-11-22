#!/usr/bin/env bash
#SBATCH --job-name=pcp_grid
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.out

export PY="${PY:-python}"
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
mkdir -p outputs logs

if ! compgen -G "caches/*.npz" > /dev/null; then
  echo "No caches found under caches/"
  exit 1
fi

run_cell() {
  local cache="$1" m="$2"
  local name; name="$(basename "$cache" .npz)"
  if "$PY" -m pcpbench.cli run --dataset "$cache" --method "$m" \
       --alpha 0.10 --seeds 10 --out "outputs/${name}_${m}.json" \
       > "logs/${name}_${m}.log" 2>&1; then
    echo "OK:   ${name}/${m}"
  else
    echo "FAIL: ${name}/${m} (see logs/${name}_${m}.log)"
  fi
}
export -f run_cell
export PY

METHODS="${METHODS:-split mondrian weighted ppi acp oracle clustered}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

for c in caches/*.npz; do
  for m in $METHODS; do
    printf '%s %s\n' "$c" "$m"
  done
done | xargs -P 32 -n 2 bash -c 'run_cell "$0" "$1"'

"$PY" -m pcpbench.leaderboard --in outputs --out leaderboard.md
echo "=== leaderboard tail ==="
tail -60 leaderboard.md
echo "GRID DONE"
