# Quickstart

This walk-through runs `pcpbench` end to end on the built-in synthetic shift, then shows how to
point it at a real cache.

## 1. Install

From the root of the repository:

```bash
pip install -e .
```

Python 3.10 to 3.12, with numpy, scipy, and scikit-learn. The cache builders under
`scripts/` also need OpenCLIP, Hugging Face `datasets`, and Pillow: `pip install -e ".[caches]"`.

## 2. Run a method from the command line

```bash
pcpbench run --dataset synthetic --method mondrian --alpha 0.1 --seeds 5
```

Flags:

- `--dataset` accepts the literal `synthetic` (an in-memory generated shift) or a path to an
  `.npz` cache.
- `--method` is one of `split`, `mondrian`, `weighted`, `ppi`, `acp`, `oracle`, `clustered`.
- `--alpha` sets the miscoverage level (default 0.1, so the target coverage is 0.9).
- `--seeds` averages the row over that many seeds and reports the seed-to-seed standard
  deviation.
- `--m-labeled` sets how many labeled target points per class `ppi`, `acp`, and `oracle` draw
  from the pool, the half of the target test rows that is not scored. The default, 0, gives
  them the whole pool.
- `--score` sets the nonconformity score, `lac` (default) or `aps`, of every method but `acp`,
  which keeps its own APS score.
- `--out path.json` also writes the row to a file.

The rows of `leaderboard.md` use `--seeds 10` and every other flag at its default; the rows
of `leaderboard_aps.md` add `--score aps`.

The output is one JSON row. Its summary keys, rounded:

```json
{
  "dataset": "synthetic_shift",
  "method": "mondrian",
  "alpha": 0.1,
  "seeds": 5,
  "n_valid_seeds": 5,
  "n_failed_seeds": 0,
  "m_labeled": 0,
  "n_classes": 12,
  "marginal": 0.507,
  "marginal_std": 0.004,
  "worst_class": 0.002,
  "worst_class_std": 0.002,
  "frac_undercovered": 1.0,
  "frac_undercovered_std": 0.0,
  "mean_set_size": 0.507,
  "mean_set_size_std": 0.004,
  "n_eval": 3000.0,
  "n_pool": 3000.0,
  "frac_sig_undercovered": 1.0,
  "frac_sig_undercovered_std": 0.0,
  "score": "lac",
  "set_rule": "score_threshold"
}
```

The row also holds the values of every seed (`per_seed`), the coverage and the evaluation and
calibration counts of every class in every seed, and the number of classes the significance
test cannot judge in every seed.

The synthetic generator builds a deliberately hard shift so that a naive baseline collapses on
some classes. The numbers above are illustrative of the synthetic regime, not a real-data
result.

## 3. Run from Python

```python
from pcpbench.datasets.synthetic import make_synthetic_shift
from pcpbench.runner import run, METHODS

ds = make_synthetic_shift(K=8, n_per_class=500, shift=0.6, seed=0)

for method in METHODS:
    row = run(ds, method, alpha=0.1, seeds=5)
    print(method, row["worst_class"], row["frac_undercovered"], row["mean_set_size"])
```

## 4. The label-complexity curve and the pseudo-label ceiling

Two further diagnostics of the library:

```python
from pcpbench.metrics.label_complexity import label_complexity_curve
from pcpbench.metrics.oracle_ceiling import oracle_ceiling

# Worst-class coverage as the labeled target budget per class grows.
curve = label_complexity_curve(ds, alpha=0.1, m_grid=[10, 25, 50, 100, 200], seed=0)
for m, v in curve.items():
    print(m, v["worst_class"], v["mean_per_class"])

# Best-case efficiency gain per class from an unlabeled-target pseudo-label method.
xt, yt = ds.target_test()
gains = oracle_ceiling(xt, yt, alpha=0.1)
print(gains)  # values near 1 on collapsing classes: no unlabeled shortcut
```

## 5. Load a real cache

Load a dataset cache directly:

```python
from pcpbench.datasets.shift_dataset import ShiftDataset

ds = ShiftDataset.from_npz("caches/ntu60_xsub.npz")
sc_logits, sc_labels = ds.source_cal()   # split == 1
tt_logits, tt_labels = ds.target_test()  # split == 2
```

Cache schema, one `.npz` per dataset and shift configuration:

| key | dtype | shape | meaning |
|-----|-------|-------|---------|
| `logits` | float32 | `[N, K]` | model logits |
| `labels` | int64 | `[N]` | ground-truth class |
| `split` | int8 | `[N]` | 0 source train, 1 source calibration, 2 target test |
| `domain` | int32 | `[N]` | 0 source, 1 target |
| `meta` | json string | scalar | `name`, `shift_family`, `n_classes`, `source` |

The caches behind the paper ship as an archive of their own. Extracted at the root of the
repository, it fills `caches/` with one cache per condition of the board. `docs/DATASHEET.md`
describes them, and `leaderboard.md` is the board built from their row files under `outputs/`.

## 6. Add a method

A method is a class whose `fit` returns an object with `predict_set`, which maps logits to a
boolean `[N, K]` prediction-set mask. Registered in `pcpbench.runner.METHODS`, it runs like the
built-in methods:

```python
import numpy as np

from pcpbench.runner import METHODS, run


class FittedTopClass:
    def predict_set(self, logits):
        mask = np.zeros(logits.shape, dtype=bool)
        mask[np.arange(len(logits)), logits.argmax(axis=1)] = True
        return mask


class TopClass:
    def fit(self, cal_logits, cal_labels, *, alpha=0.1, seed=0):
        return FittedTopClass()


METHODS["topclass"] = TopClass
row = run(ds, "topclass", alpha=0.1, seeds=5)
print(row["worst_class"], row["mean_set_size"])
```

A new method receives the source calibration set, `alpha`, and `seed`, and its row records
`score` and `set_rule` as `null`. Add the entry to `METHODS` in `src/pcpbench/runner.py` to
run it as `pcpbench run --method topclass`.

## 7. Re-check the released results

From the root of the repository. These commands read the committed files and need no cache:

```bash
pip install -e ".[dev]"
python -m pytest -q
python scripts/rebuild_derived.py --check
python scripts/camera_ready_numbers.py --check
```

`rebuild_derived.py` rebuilds the board and every analysis computed from the row files, and
`camera_ready_numbers.py` writes every measured value and table body of the paper into
`paper/camera_ready/generated/`. With `--check`, each compares a fresh run with the committed
files. The nested class-subset trend of Appendix B.2 comes from the sweep files under
`outputs/nested_ksweep/`:

```bash
python scripts/class_count_trend.py --design nested \
    --inputs-glob 'outputs/nested_ksweep/nested_k_sweep_*.json.gz' \
    --no-shift-glob 'outputs/nested_ksweep/noshift_k_sweep_*.json.gz' \
    --expect-seeds 30 --expect-no-shift 10 \
    --out-json outputs/analysis/nested_trend.json --out-md paper/camera_ready/NESTED_TREND.md
```

The committed files were written with numpy 1.26. Under numpy 2.4 the JSON file differs from
the committed one in the last digits of some floats, and the report is byte-identical.

With the caches in place, `bash scripts/run_grid.sh` runs every method on every cache under
the protocol of the board, rewrites the row files under `outputs/`, and builds `leaderboard.md`
from every row file there. That board also holds the 60 rows of ten older CIFAR-10-C
conditions, which have no cache in the archive. The released board keeps the conditions of
the submitted one:

```bash
python -m pcpbench.leaderboard --in outputs --out leaderboard.md \
    --datasets-from outputs/analysis/leaderboard_submitted.md
```

`SCORE=aps bash scripts/run_grid.sh` does the same for `outputs/aps/` and `leaderboard_aps.md`.
On a Slurm cluster, `mkdir -p logs && sbatch scripts/run_grid.sbatch` runs the grid 32 cells
at a time. Weighted and ACP give back the released rows only at the thread count of the
released runs, one per cell (`docs/DATASHEET.md`, Thread count). The nested sweep job,
`scripts/run_nested_ksweep.sbatch`, records the git commit it runs and so needs a clone of the
repository.
