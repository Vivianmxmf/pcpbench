"""Registry of the numbers read from the committed artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Iterator

_REPO_ROOT = Path(__file__).resolve().parents[1]

VALIDITY_FLOOR = "outputs/analysis/validity_floor.json"
SWEEP_UNCERTAINTY = "outputs/analysis/sweep_uncertainty.json"
THRESHOLD_SENSITIVITY = "outputs/analysis/threshold_sensitivity.json"
LABEL_SWEEP = "outputs/analysis/label_complexity_contrast_s3.json"
SWEEP_SCRIPT = "scripts/sweep_uncertainty.py"
CLUSTERED = "src/pcpbench/calibrators/clustered.py"
RUNNER = "src/pcpbench/runner.py"

CLUSTERED_FLOOR_SETTINGS: tuple[str, ...] = ("cifar10c", "k_sweep", "cifar100c", "imagenet1k")
QUOTED_ORACLE_SETTINGS: tuple[str, ...] = ("k_sweep", "imagenet1k")
ANCHOR_METHODS: tuple[str, ...] = ("ppi", "oracle")


@dataclass(frozen=True)
class Entry:
    name: str
    value: str
    source: str
    key: str

    def line(self) -> str:
        return f"{self.name} | {self.value} | {self.source} | {self.key}"


def f3(x: float) -> str:
    return f"{x:.3f}"


def f2(x: float) -> str:
    return f"{x:.2f}"


def f1(x: float) -> str:
    return f"{x:.1f}"


def _num(x) -> str:
    if isinstance(x, bool):
        return str(x).lower()
    if isinstance(x, int):
        return str(x)
    return f3(x)


def _signed(name: str, x: float, fmt, source: str, key: str) -> Iterator[Entry]:
    yield Entry(name, fmt(x), source, key)
    yield Entry(f"{name}.magnitude", fmt(abs(x)), source, f"absolute value of {key}")


def _load(root: Path, rel: str) -> dict:
    return json.loads((root / rel).read_text(encoding="utf-8"))


def _load_script(root: Path, rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, root / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_FLOOR_F3 = ("floor_mean", "floor_sd", "floor_q05", "floor_q95", "seed_avg_floor",
             "fp_rate_at_0.90", "sig_power_at_0.80", "paper_oracle", "paper_ppi",
             "paper_frac_undercovered")
_FLOOR_INT = ("n_cal", "n_eval", "sig_cutoff", "paper_n_averaged", "seed_avg_seeds")


def _floor_prefix(row: dict) -> str:
    prefix = f"floor.{row['setting']}.K{row['K']}"
    if row["setting"] == "label_sweep":
        prefix += f".ncal{row['n_cal']}"
    return prefix


def floor_entries(root: Path) -> Iterator[Entry]:
    d = _load(root, VALIDITY_FLOOR)
    src = VALIDITY_FLOOR
    yield Entry("floor.level", f2(d["level"]), src, "level")
    yield Entry("floor.alpha", f2(d["alpha"]), src, "alpha")
    yield Entry("floor.reps", str(d["reps"]), src, "reps")
    zs = []
    zs_quoted = []
    for i, row in enumerate(d["rows"]):
        p = _floor_prefix(row)
        for k in _FLOOR_F3:
            v = row.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                yield Entry(f"{p}.{k}", f3(v), src, f"rows[{i}].{k}")
        for k in _FLOOR_INT:
            v = row.get(k)
            if isinstance(v, int) and not isinstance(v, bool):
                yield Entry(f"{p}.{k}", str(v), src, f"rows[{i}].{k}")
        for k in ("z", "z_seed_mean"):
            v = row.get(k)
            if v is not None:
                yield from _signed(f"{p}.{k}", v, f2, src, f"rows[{i}].{k}")
                if k == "z":
                    zs.append(abs(v))
                    if row["setting"] in QUOTED_ORACLE_SETTINGS:
                        zs_quoted.append(abs(v))
    yield Entry("floor.max_abs_z", f2(max(zs)), src,
                "max over rows of the absolute value of rows[].z")
    yield Entry("floor.max_abs_z.k_sweep_and_imagenet1k", f2(max(zs_quoted)), src,
                "max over rows[setting in k_sweep, imagenet1k] of |rows[].z|")
    pc = d["power_curve"]
    for k in ("first_above_half", "above_half_from", "first_size_at_max"):
        yield Entry(f"floor.power_curve.{k}", str(pc[k]), src, f"power_curve.{k}")
    for k in ("power_at_first", "max_power", "power_at_n_cal_50"):
        yield Entry(f"floor.power_curve.{k}", f3(pc[k]), src, f"power_curve.{k}")


def sweep_entries(root: Path) -> Iterator[Entry]:
    d = _load(root, SWEEP_UNCERTAINTY)
    src = SWEEP_UNCERTAINTY
    cfg = d["config"]
    yield Entry("sweep.config.B", str(cfg["B"]), src, "config.B")
    yield Entry("sweep.config.n_moderate_caches", str(len(cfg["moderate_caches"])), src,
                "len(config.moderate_caches)")
    yield Entry("sweep.config.n_floored_caches", str(len(cfg["floored_caches"])), src,
                "len(config.floored_caches)")
    ctl = d["controlled"]
    for i, r in enumerate(ctl["per_k"]):
        p = f"sweep.per_k.{r['method']}.K{r['k']}"
        key = f"controlled.per_k[{i}]"
        yield Entry(f"{p}.n_records", str(r["n_records"]), src, f"{key}.n_records")
        for k in ("mean", "se", "ci_lo", "ci_hi", "floor_mean", "floor_sd"):
            yield Entry(f"{p}.{k}", f3(r[k]), src, f"{key}.{k}")
        yield from _signed(f"{p}.excess", r["excess"], f3, src, f"{key}.excess")
    for i, r in enumerate(ctl["decay"]):
        p = f"sweep.decay.{r['method']}"
        key = f"controlled.decay[{i}]"
        for k in ("decay_mean", "decay_se", "decay_ci_lo", "decay_ci_hi"):
            yield Entry(f"{p}.{k}", f3(r[k]), src, f"{key}.{k}")
        for k in ("ratio_to_oracle", "ratio_ci_lo", "ratio_ci_hi"):
            yield Entry(f"{p}.{k}", f1(r[k]), src, f"{key}.{k}")
        yield Entry(f"{p}.displayed_endpoint_ratio", f1(r["paper_convention_ratio"]), src,
                    f"{key}.paper_convention_ratio")
    ratio_excl_one = all(r["ratio_ci_lo"] > 1.0 for r in ctl["decay"])
    yield Entry("sweep.decay.every_ratio_interval_excludes_one", str(ratio_excl_one).lower(),
                src, "all(controlled.decay[].ratio_ci_lo > 1)")
    for i, r in enumerate(ctl["vs_oracle"]):
        p = f"sweep.vs_oracle.{r['method']}.K{r['k']}"
        key = f"controlled.vs_oracle[{i}]"
        for k in ("diff_mean", "diff_ci_lo", "diff_ci_hi"):
            yield from _signed(f"{p}.{k}", r[k], f3, src, f"{key}.{k}")
        yield Entry(f"{p}.n_below_oracle", str(r["n_below_oracle"]), src,
                    f"{key}.n_below_oracle")
    k100 = [r for r in ctl["vs_oracle"] if r["k"] == 100]
    excl_zero = all(r["diff_ci_hi"] < 0 or r["diff_ci_lo"] > 0 for r in k100)
    yield Entry("sweep.vs_oracle.K100.every_interval_excludes_zero", str(excl_zero).lower(),
                src, "controlled.vs_oracle[k == 100]: diff_ci_hi < 0 or diff_ci_lo > 0")
    below = all(r["below_floor_mean"] for r in ctl["floor_check_k100"])
    yield Entry("sweep.floor_check_k100.every_non_oracle_interval_below_floor_mean",
                str(below).lower(), src, "all(controlled.floor_check_k100[].below_floor_mean)")
    n_excl = 0
    for i, r in enumerate(d["floored_caches"]):
        p = f"sweep.floored.{r['cache']}.K{r['k']}"
        key = f"floored_caches[{i}]"
        for k in ("mean", "se", "ci_lo", "ci_hi", "floor_mean", "floor_sd"):
            yield Entry(f"{p}.{k}", f3(r[k]), src, f"{key}.{k}")
        yield Entry(f"{p}.n_seeds", str(r["n_seeds"]), src, f"{key}.n_seeds")
        yield from _signed(f"{p}.mean_minus_floor", r["mean"] - r["floor_mean"], f3, src,
                           f"{key}.mean minus {key}.floor_mean")
        se_seed_mean = r["floor_sd"] / math.sqrt(r["n_seeds"])
        yield Entry(f"{p}.floor_se_of_seed_mean", f3(se_seed_mean), src,
                    f"{key}.floor_sd / sqrt({key}.n_seeds)")
        yield from _signed(f"{p}.gap_in_seed_mean_se", (r["mean"] - r["floor_mean"]) / se_seed_mean,
                           f1, src, f"({key}.mean minus {key}.floor_mean) / floor_se_of_seed_mean")
        contains = r["ci_lo"] <= r["floor_mean"] <= r["ci_hi"]
        n_excl += not contains
        yield Entry(f"{p}.interval_contains_floor_mean", str(contains).lower(), src,
                    f"{key}.ci_lo <= {key}.floor_mean <= {key}.ci_hi")
    yield Entry("sweep.floored.n_cells", str(len(d["floored_caches"])), src,
                "len(floored_caches)")
    yield Entry("sweep.floored.n_cells_interval_excludes_floor_mean", str(n_excl), src,
                "count of floored_caches[] whose interval excludes floor_mean")
    ls = d["label_sweep"]
    for i, r in enumerate(ls["per_m"]):
        if r["mean"] is None:
            continue
        p = f"sweep.label.{r['method']}.m{r['m']}"
        key = f"label_sweep.per_m[{i}]"
        for k in ("mean", "se", "ci_lo", "ci_hi"):
            yield Entry(f"{p}.{k}", f3(r[k]), src, f"{key}.{k}")
        if r["excess"] is not None:
            yield from _signed(f"{p}.excess", r["excess"], f3, src, f"{key}.excess")
    for i, r in enumerate(ls["ppi_vs_oracle"]):
        p = f"sweep.label.ppi_minus_oracle.m{r['m']}"
        key = f"label_sweep.ppi_vs_oracle[{i}]"
        for k in ("diff_mean", "diff_ci_lo", "diff_ci_hi"):
            yield from _signed(f"{p}.{k}", r[k], f3, src, f"{key}.{k}")
    for k in ("smallest_m_ppi_meets_floor", "smallest_m_oracle_meets_floor"):
        yield Entry(f"sweep.label.{k}", str(ls[k]), src, f"label_sweep.{k}")
    first: dict[str, int] = {}
    for r in ls["per_m"]:
        if r["mean"] is not None and r["method"] not in first:
            first[r["method"]] = r["m"]
    for method, m in first.items():
        yield Entry(f"sweep.label.first_defined_m.{method}", str(m), src,
                    f"smallest label_sweep.per_m[method == {method}].m with a defined mean")
    caps = {r["reused_from_m"] for r in ls["per_m"] if r["reused_from_m"] is not None}
    if len(caps) == 1:
        yield Entry("sweep.label.pool_cap_per_class", str(caps.pop()), src,
                    "the one value label_sweep.per_m[].reused_from_m takes where it is set")
    su = _load_script(root, SWEEP_SCRIPT, "_registry_sweep_uncertainty")
    level = su.CI_HI_PCT - su.CI_LO_PCT
    yield Entry("sweep.interval_level_percent", f"{level:.0f}", SWEEP_SCRIPT,
                "CI_HI_PCT minus CI_LO_PCT")


def threshold_entries(root: Path) -> Iterator[Entry]:
    d = _load(root, THRESHOLD_SENSITIVITY)
    src = THRESHOLD_SENSITIVITY
    rg, gg = d["regime_grid"], d["guard_grid"]
    yield Entry("thresholds.n_regime_combinations", str(len(rg)), src, "len(regime_grid)")
    yield Entry("thresholds.n_guard_combinations", str(len(gg)), src, "len(guard_grid)")
    yield Entry("thresholds.n_combinations", str(len(rg) + len(gg)), src,
                "len(regime_grid) + len(guard_grid)")
    for k in ("recovered", "regime_a_marginal", "regime_b_marginal"):
        for v in sorted({r[k] for r in rg}):
            yield Entry(f"thresholds.regime_grid.{k}.{f2(v)}", f2(v), src,
                        f"distinct regime_grid[].{k}")
    for k in ("vacuous_set_frac", "vacuous_overcover_frac", "overcover_marginal"):
        for v in sorted({r[k] for r in gg}):
            yield Entry(f"thresholds.guard_grid.{k}.{f2(v)}", f2(v), src,
                        f"distinct guard_grid[].{k}")
    for grid in ("regime_grid", "guard_grid"):
        for label, mm in d["summary"][grid].items():
            for end in ("min", "max"):
                yield Entry(f"thresholds.summary.{grid}.{label}.{end}", str(mm[end]), src,
                            f"summary.{grid}.{label}.{end}")
    for end in ("min", "max"):
        yield Entry(f"thresholds.summary.guard_grid_demoted_bound.{end}",
                    str(d["summary"]["guard_grid_demoted_bound"][end]), src,
                    f"summary.guard_grid_demoted_bound.{end}")
    dom = d["summary"]["regime_b_dominance"]
    yield Entry("thresholds.regime_b_always_largest", str(dom["always_largest"]).lower(), src,
                "summary.regime_b_dominance.always_largest")
    yield Entry("thresholds.regime_b_closest_margin", str(dom["closest_margin"]), src,
                "summary.regime_b_dominance.closest_margin")
    rb_all = [r["counts"]["regime_b"] for r in rg + gg]
    yield Entry("thresholds.regime_b.min_over_all", str(min(rb_all)), src,
                "min over regime_grid and guard_grid of counts.regime_b")
    yield Entry("thresholds.regime_b.max_over_all", str(max(rb_all)), src,
                "max over regime_grid and guard_grid of counts.regime_b")
    for k, v in d["default_counts"].items():
        yield Entry(f"thresholds.default_counts.{k}", str(v), src, f"default_counts.{k}")
    settled = [(i, r) for i, r in enumerate(gg) if r.get("divider_settled")]
    for i, r in settled:
        yield Entry("thresholds.default_guard_cell.demoted_min", str(r["demoted_min"]), src,
                    f"guard_grid[{i}].demoted_min (divider_settled)")
        yield Entry("thresholds.default_guard_cell.demoted_max", str(r["demoted_max"]), src,
                    f"guard_grid[{i}].demoted_max (divider_settled)")
    yield Entry("thresholds.max_ambiguous_rows", str(max(r["demoted_ambiguous_rows"] for r in gg)),
                src, "max guard_grid[].demoted_ambiguous_rows")


def label_sweep_entries(root: Path) -> Iterator[Entry]:
    d = _load(root, LABEL_SWEEP)
    for i, m in enumerate(d["m_values"]):
        yield Entry(f"label_sweep.m_values[{i}]", str(m), LABEL_SWEEP, f"m_values[{i}]")
    yield Entry("label_sweep.seeds", str(d["seeds"]), LABEL_SWEEP, "seeds")


def clustered_entries(root: Path) -> Iterator[Entry]:
    """The clustered rule at each (K, calibration points per class) of the floor settings, both
    read from the rows of validity_floor.json."""
    sys.path.insert(0, str(root / "src"))
    from pcpbench.calibrators.clustered import default_clustering_parameters

    rows = _load(root, VALIDITY_FLOOR)["rows"]
    for i, row in enumerate(rows):
        k, n = row["K"], row["n_cal"]
        if row["setting"] not in CLUSTERED_FLOOR_SETTINGS or not isinstance(n, int):
            continue
        ncl, nk = default_clustering_parameters(k, n)
        p = f"clustered.{row['setting']}.K{k}"
        key = f"default_clustering_parameters(rows[{i}].K, rows[{i}].n_cal)"
        yield Entry(f"{p}.n_clustering", str(ncl), CLUSTERED, f"{key}[0]")
        yield Entry(f"{p}.num_clusters", str(nk), CLUSTERED, f"{key}[1]")
        yield Entry(f"{p}.points_per_class", str(n), VALIDITY_FLOOR, f"rows[{i}].n_cal")


def significance_entries(root: Path) -> Iterator[Entry]:
    sys.path.insert(0, str(root / "src"))
    from pcpbench import runner

    yield Entry("significance.level", f2(runner.SIG_Q), RUNNER, "SIG_Q")


def anchor_entries(root: Path) -> Iterator[Entry]:
    """Synthetic anchors, recomputed from the committed code (deterministic)."""
    sys.path.insert(0, str(root / "src"))
    from pcpbench.datasets.synthetic import make_synthetic_shift
    from pcpbench.runner import run

    for m in ANCHOR_METHODS:
        row = run(make_synthetic_shift(), m, alpha=0.1, seeds=5)
        key = f"run(make_synthetic_shift(), '{m}', alpha=0.1, seeds=5)"
        for k in ("frac_undercovered", "frac_sig_undercovered"):
            yield Entry(f"anchor.{m}.{k}", f3(row[k]), RUNNER, f"{key}['{k}']")


BOARD = "leaderboard.md"
APS_BOARD = "leaderboard_aps.md"
BOARD_DIFF = "outputs/analysis/leaderboard_rerun_diff.json"
ANALYZE_SCRIPT = "scripts/analyze_leaderboard.py"
ROWS_DIR = "outputs"
NO_SHIFT_DIR = "outputs/noshift"
ANALYSIS_DIR = "outputs/analysis"
REVIEWED_VERSION = "outputs/analysis/reviewed_version.json"
Z_BOUND = 2
NO_SHIFT_QUOTED = ("split", "mondrian", "oracle", "clustered")
_DIFF_FIELD = {"marginal": "marginal", "worst_class": "worst_class",
               "frac_undercovered": "frac_undercovered", "mean_set_size": "mean_set_size",
               "set/K": "set_over_k"}


def analyze_module(root: Path):
    """``scripts/analyze_leaderboard.py``, the committed analysis code."""
    return _load_script(root, ANALYZE_SCRIPT, "_registry_analyze_leaderboard")


def rerun_board(root: Path, al) -> list:
    """Rows of the regenerated leaderboard."""
    return al.parse_leaderboard((root / BOARD).read_text(encoding="utf-8"))


def released_board(root: Path, al) -> list:
    """Rows of the released leaderboard, rebuilt from the regenerated board and the committed
    diff of the two: the rows only the rerun has are dropped, every cell the diff lists as
    changed takes its old printed value, and K is derived again from the restored cells."""
    diff = _load(root, BOARD_DIFF)
    only_new = {(r["dataset"], r["method"]) for r in diff["rows_only_in_new"]}
    changed = {(c["dataset"], c["method"]): c["cells"] for c in diff["changed_rows"]}
    demoted_before = {(d["dataset"], d["method"]): d["demoted_old"] for d in diff["demotion_changes"]}
    rows = []
    for row in rerun_board(root, al):
        key = (row.dataset, row.method)
        if key in only_new:
            continue
        cells = changed.get(key)
        if cells:
            row = row._replace(**{_DIFF_FIELD[c]: float(cell["old"]) for c, cell in cells.items()})
            row = row._replace(k=al._class_count(row.family, row.mean_set_size, row.set_over_k))
        if key in demoted_before:
            row = row._replace(demoted=demoted_before[key])
        rows.append(row)
    if len(rows) != diff["n_rows_old"]:
        raise ValueError(f"rebuilt {len(rows)} released rows, the diff counts {diff['n_rows_old']}")
    return rows


def _printed_cells(text: str) -> dict[tuple[str, str], dict[str, str]]:
    header: list[str] = []
    out: dict[tuple[str, str], dict[str, str]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not header:
            header = cells
            continue
        if all(set(c) <= set("-:") for c in cells) or not any(cells[1:]):
            continue
        out[(cells[0], cells[1])] = dict(zip(header, cells))
    return out


def _headline(rows: list, al) -> list:
    return [r for r in al.select_headline(rows) if not r.demoted]


def _regime_short(al) -> dict[str, str]:
    return {al.RECOVERED: "recovered", al.REGIME_A: "regime_a", al.REGIME_B: "regime_b",
            al.REGIME_OTHER: "other"}


def _paper_regime_counts(root: Path) -> list[int]:
    counts = _load(_REPO_ROOT, REVIEWED_VERSION)["regime_counts"]
    return [counts[name] for name in ("non_demoted", "recovered", "regime_a", "regime_b", "other")]


def headline_entries(root: Path) -> Iterator[Entry]:
    """The paper's 404 non-demoted headline rows and their regime labels reproduced from the
    released board, each joined with its row on the regenerated board."""
    sys.path.insert(0, str(root / "src"))
    from pcpbench import runner

    al = analyze_module(root)
    released_rows = released_board(root, al)
    released = _headline(released_rows, al)
    labels = {(r.dataset, r.method): al.label_regime(r.marginal, r.worst_class) for r in released}
    short = _regime_short(al)
    counts = {name: sum(lab == full for lab in labels.values()) for full, name in short.items()}
    reproduced = [len(released)] + [counts[k] for k in ("recovered", "regime_a", "regime_b", "other")]
    if reproduced != _paper_regime_counts(root):
        raise ValueError(f"the released headline rows give {reproduced}, which do not reproduce "
                         f"{REVIEWED_VERSION} {_paper_regime_counts(root)}")
    src = BOARD_DIFF
    key = "released board: leaderboard.md with the old cells of the diff, select_headline, not demoted"
    yield Entry("regime.released.n_rows", str(len(released)), src, key)
    for name in ("recovered", "regime_a", "regime_b", "other"):
        yield Entry(f"regime.released.{name}", str(counts[name]), src,
                    f"{key}, label_regime == {name}")

    on_board = {(r.dataset, r.method): r for r in al.select_headline(rerun_board(root, al))}
    before = {(r.dataset, r.method) for r in al.select_headline(released_rows)}
    if set(on_board) != before:
        raise ValueError("the headline of the regenerated board and the released headline hold other "
                         f"rows: {sorted(set(on_board) ^ before)[:6]}")
    rerun = {k: on_board[k] for k in labels}
    printed = _printed_cells((root / BOARD).read_text(encoding="utf-8"))
    level = Decimal(f"{runner.SIG_Q}")
    sig = {k: Decimal(printed[k]["frac_sig_undercovered"]) for k in labels}
    for (ds, m), value in sig.items():
        exact = _load(root, f"{ROWS_DIR}/{ds}_{m}.json")["frac_sig_undercovered"]
        if (value > level) != (exact > float(level) + 1e-9):
            raise ValueError(f"{ds} {m}: the printed and the unrounded means decide the rule apart")
    passes = {k: sig[k] > level for k in labels}

    h = "rerun.headline"
    rows = list(rerun.values())
    yield Entry(f"{h}.n_rows", str(len(rows)), BOARD,
                "the headline rows not demoted on the released board, each read on leaderboard.md "
                "by dataset and method")
    yield Entry(f"{h}.mean_frac_undercovered", f3(mean(r.frac_undercovered for r in rows)), BOARD,
                "mean of frac_undercovered over those rows")
    yield Entry(f"{h}.mean_frac_sig_undercovered", f3(mean(float(v) for v in sig.values())), BOARD,
                "mean of frac_sig_undercovered over those rows")
    yield Entry(f"{h}.n_frac_sig_above_level", str(sum(passes.values())), BOARD,
                "count of those rows with frac_sig_undercovered > runner.SIG_Q")

    rg = "regime.floor_rule"
    for full, name in short.items():
        group = [k for k, lab in labels.items() if lab == full]
        yield Entry(f"{rg}.{name}.n_pass", str(sum(passes[k] for k in group)), BOARD,
                    f"rows labelled {name} on the released board whose frac_sig_undercovered on "
                    "leaderboard.md > runner.SIG_Q")
    for method in ("oracle", "ppi"):
        group = [k for k, lab in labels.items() if lab == al.REGIME_A and k[1] == method
                 and al.parse_dataset_name(k[0]).family == "cifar100c"]
        p = f"{rg}.regime_a.{method}_cifar100c"
        yield Entry(f"{p}.n_rows", str(len(group)), src,
                    f"regime-A rows of {method} on cifar100c, released board")
        yield Entry(f"{p}.n_pass", str(sum(passes[k] for k in group)), BOARD,
                    "those rows whose frac_sig_undercovered on leaderboard.md > runner.SIG_Q")
    oracle = [k for k in labels if k[1] == "oracle"]
    yield Entry(f"{rg}.oracle.n_rows", str(len(oracle)), src,
                "oracle rows among the released non-demoted headline rows")
    yield Entry(f"{rg}.oracle.n_pass", str(sum(passes[k] for k in oracle)), BOARD,
                "those rows whose frac_sig_undercovered on leaderboard.md > runner.SIG_Q")
    yield Entry(f"{rg}.recovered.oracle.n_pass",
                str(sum(passes[k] for k in oracle if labels[k] == al.RECOVERED)), BOARD,
                "those flagged oracle rows labelled recovered on the released board")


def aps_entries(root: Path) -> Iterator[Entry]:
    """The APS twin (<method>_aps, same dataset) of each of the 404 non-demoted headline rows,
    its regime label by the paper's thresholds and its worst class against the LAC worst class of
    the regenerated board."""
    al = analyze_module(root)
    released = _headline(released_board(root, al), al)
    lac = {(r.dataset, r.method): r for r in rerun_board(root, al)}
    twins = {(r.dataset, r.method): r for r in
             al.parse_leaderboard((root / APS_BOARD).read_text(encoding="utf-8"))}
    pairs = [(r, twins[(r.dataset, f"{r.method}_aps")]) for r in released
             if (r.dataset, f"{r.method}_aps") in twins]
    without = sorted({r.method for r in released if (r.dataset, f"{r.method}_aps") not in twins})
    if without != ["acp"]:
        raise ValueError(f"headline methods without an APS twin: {without}")
    same = [al.label_regime(r.marginal, r.worst_class) == al.label_regime(t.marginal, t.worst_class)
            for r, t in pairs]
    diffs = [t.worst_class - lac[(r.dataset, r.method)].worst_class for r, t in pairs]
    src = APS_BOARD
    yield Entry("aps.n_rows", str(len(released)), BOARD_DIFF, "the released non-demoted headline rows")
    yield Entry("aps.n_rows_with_twin", str(len(pairs)), src, "those rows with a <method>_aps row")
    yield Entry("aps.n_rows_without_twin", str(len(released) - len(pairs)), src,
                "those rows without one (acp)")
    yield Entry("aps.n_same_regime", str(sum(same)), src,
                "twins whose label_regime equals the released row's")
    for name, value in (("median", median(diffs)), ("min", min(diffs)), ("max", max(diffs))):
        yield from _signed(f"aps.worst_minus_lac.{name}", value, f3, src,
                           f"{name} over the twins of worst_class minus the leaderboard.md worst_class")
    yield Entry("aps.n_twins_demoted", str(sum(t.demoted for _, t in pairs)), src,
                "twins below the divider of leaderboard_aps.md")
    for method in sorted({r.method for r, _ in pairs}):
        idx = [i for i, (r, _) in enumerate(pairs) if r.method == method]
        yield Entry(f"aps.{method}.n_rows_with_twin", str(len(idx)), src, f"twins of {method} rows")
        yield Entry(f"aps.{method}.n_same_regime", str(sum(same[i] for i in idx)), src,
                    f"twins of {method} rows with the same label_regime")
    for name, group in (("oracle_ppi", ("oracle", "ppi")), ("other_lac", ("split", "mondrian", "weighted"))):
        idx = [i for i, (r, _) in enumerate(pairs) if r.method in group]
        yield Entry(f"aps.{name}.n_rows_with_twin", str(len(idx)), src, f"twins of {', '.join(group)} rows")
        yield Entry(f"aps.{name}.n_same_regime", str(sum(same[i] for i in idx)), src,
                    f"twins of {', '.join(group)} rows with the same label_regime")
    b = [i for i, (r, _) in enumerate(pairs)
         if al.label_regime(r.marginal, r.worst_class) == al.REGIME_B]
    keep = [i for i in b if same[i]]
    yield Entry("aps.regime_b.n_rows_with_twin", str(len(b)), src,
                "released regime-B rows with a <method>_aps row")
    yield Entry("aps.regime_b.n_keep_regime_b", str(len(keep)), src,
                "those whose twin's label_regime is regime B")
    yield Entry("aps.regime_b.n_twins_demoted", str(sum(pairs[i][1].demoted for i in b)), src,
                "those whose twin sits below the divider of leaderboard_aps.md")
    yield Entry("aps.regime_b.n_keep_regime_b_demoted", str(sum(pairs[i][1].demoted for i in keep)),
                src, "twins that keep regime B and sit below the divider")


def clustered_rerun_entries(root: Path) -> Iterator[Entry]:
    """Per rerun dataset, whether the clustered rule skips clustering or forms clusters, and the
    clustered method on the controlled estimate."""
    sys.path.insert(0, str(root / "src"))
    from pcpbench.calibrators.clustered import default_clustering_parameters, get_quantile_threshold

    datasets = sorted(p.name[:-len("_clustered.json")]
                      for p in (root / ROWS_DIR).glob("*_clustered.json"))
    skipped = form = mixed = form_imagenet = equal_worst = 0
    for ds in datasets:
        c = _load(root, f"{ROWS_DIR}/{ds}_clustered.json")
        s = _load(root, f"{ROWS_DIR}/{ds}_split.json")
        n_thresh = get_quantile_threshold(c["alpha"])
        decisions = []
        for full, part in zip(s["n_cal_per_class_per_seed"], c["n_cal_per_class_per_seed"]):
            clustering = [a - b for a, b in zip(full, part)]
            if min(clustering) < 0:
                raise ValueError(f"{ds}: a calibration part larger than the source set")
            n_min = max(min(full), n_thresh)
            _, num_clusters = default_clustering_parameters(sum(x >= n_min for x in full), n_min)
            n_rare = sum(x < n_thresh for x in clustering)
            decisions.append(not (c["n_classes"] - n_rare > num_clusters and num_clusters > 1))
        if all(decisions):
            skipped += 1
            equal_worst += c["worst_class"] == s["worst_class"]
        elif not any(decisions):
            form += 1
            form_imagenet += ds.startswith("imagenet1k")
        else:
            mixed += 1
    src = f"{ROWS_DIR}/"
    key = "outputs/<dataset>_clustered.json and _split.json, n_cal_per_class_per_seed"
    yield Entry("clustered.rerun.n_datasets", str(len(datasets)), src, "count of *_clustered.json rows")
    yield Entry("clustered.rerun.n_skipped", str(skipped), src, f"{key}: the rule skips in every seed")
    yield Entry("clustered.rerun.n_clusters_form", str(form), src, f"{key}: clusters form in every seed")
    yield Entry("clustered.rerun.n_mixed", str(mixed), src, f"{key}: the seeds disagree")
    yield Entry("clustered.rerun.n_clusters_form_imagenet", str(form_imagenet), src,
                f"{key}: clusters form in every seed on an imagenet1k dataset")
    yield Entry("clustered.rerun.n_skipped_worst_equals_split", str(equal_worst), src,
                "skipped datasets whose clustered worst_class equals split's")

    su = _load_script(root, SWEEP_SCRIPT, "_registry_sweep_uncertainty")
    import numpy as np

    analysis = root / ANALYSIS_DIR
    idx = su.index_controlled_records(
        {c: su.load_controlled_records(analysis, c) for c in su.MODERATE_CACHES})
    stats = su.controlled_per_k(idx, su.MODERATE_CACHES, ("clustered",), su.K_VALUES,
                                su.N_SUBSET_SEEDS, np.random.default_rng(su.DEFAULT_SEED), su.DEFAULT_B)
    for s in stats:
        p = f"clustered.controlled.K{s.k}"
        key = f"sweep_uncertainty.controlled_per_k(clustered, K {s.k}) on the five moderate caches"
        for k in ("mean", "ci_lo", "ci_hi", "floor_mean"):
            yield Entry(f"{p}.{k}", f3(getattr(s, k)), SWEEP_SCRIPT, f"{key}.{k}")
        yield from _signed(f"{p}.excess", s.excess, f3, SWEEP_SCRIPT, f"{key}.excess")
        yield Entry(f"{p}.interval_below_floor_mean", str(s.ci_hi < s.floor_mean).lower(),
                    SWEEP_SCRIPT, f"{key}: ci_hi < floor_mean")


def no_shift_entries(root: Path) -> Iterator[Entry]:
    """The no-shift reference over every cache, read from the per-cache outputs."""
    files = sorted((root / NO_SHIFT_DIR).glob("*.json"))
    data = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in files}
    src = f"{NO_SHIFT_DIR}/"
    yield Entry("no_shift.n_caches", str(len(data)), src, "count of outputs/noshift/*.json")
    methods = sorted({e["method"] for d in data.values() for e in d["summary"]})
    for m in methods:
        zs = [e["z"] for d in data.values() for e in d["summary"]
              if e["method"] == m and e["floor_status"] == "ok"]
        p = f"no_shift.{m}"
        key = f"summary[method == {m}, floor_status == ok].z over the caches"
        yield Entry(f"{p}.n_floor_defined", str(len(zs)), src, f"count of {key}")
        yield Entry(f"{p}.n_within_two_sd", str(sum(abs(z) <= Z_BOUND for z in zs)), src,
                    f"count of |z| <= {Z_BOUND} in {key}")
        yield Entry(f"{p}.n_below_two_sd", str(sum(z < -Z_BOUND for z in zs)), src,
                    f"count of z < -{Z_BOUND} in {key}")
        yield from _signed(f"{p}.most_negative_z", min(zs), f2, src, f"min of {key}")
        yield from _signed(f"{p}.largest_z", max(zs), f2, src, f"max of {key}")
    imagenet = {k: d for k, d in data.items() if k.startswith("imagenet1k")}
    statuses = {d["floor"]["clustered"]["status"] for d in imagenet.values()}
    no_cal = {d["floor"]["clustered"]["max_no_cal_classes"] for d in imagenet.values()}
    most = {max(max(r["row"]["n_cal_per_class_per_seed"][0]) for r in d["per_seed"]
                if r["method"] == "clustered") for d in imagenet.values()}
    floors = {(d["floor"]["clustered"]["mean"], d["floor"]["clustered"]["sd"]) for d in imagenet.values()}
    if len(statuses) != 1 or len(no_cal) != 1 or len(most) != 1 or len(floors) != 1:
        raise ValueError("the ImageNet no-shift outputs disagree on the clustered floor")
    p = "no_shift.imagenet"
    yield Entry(f"{p}.n_caches", str(len(imagenet)), src, "outputs/noshift/imagenet1k*.json")
    yield Entry(f"{p}.clustered.status", statuses.pop(), src, "floor.clustered.status, equal on every one")
    yield Entry(f"{p}.clustered.max_no_cal_classes", str(no_cal.pop()), src,
                "floor.clustered.max_no_cal_classes, equal on every one")
    yield Entry(f"{p}.clustered.max_cal_points_per_class", str(most.pop()), src,
                "max over per_seed[method == clustered] of row.n_cal_per_class_per_seed[0]")
    fm, fsd = floors.pop()
    yield Entry(f"{p}.clustered.floor_mean", f3(fm), src, "floor.clustered.mean, equal on every one")
    yield Entry(f"{p}.clustered.floor_sd", f3(fsd), src, "floor.clustered.sd, equal on every one")


def ksweep_all_entries(root: Path) -> Iterator[Entry]:
    """Every cache with a controlled K-sweep file, at K 100, against the floor mean and the
    floor's standard error of a mean of the sweep's seeds, the floor sd at K 100 with 50 and 50
    points of validity_floor.json over the square root of the number of subset seeds."""
    row = next(r for r in _load(root, VALIDITY_FLOOR)["rows"]
               if r["setting"] == "k_sweep" and r["K"] == 100)
    files = sorted((root / ANALYSIS_DIR).glob("controlled_k_sweep_*.json"))
    seeds = {_load(root, f"{ANALYSIS_DIR}/{p.name}")["config"]["subset_seeds"] for p in files}
    if len(seeds) != 1:
        raise ValueError(f"the K-sweep files differ in subset_seeds: {sorted(seeds)}")
    n_seeds = seeds.pop()
    se = row["floor_sd"] / math.sqrt(n_seeds)
    lo, hi = row["floor_mean"] - Z_BOUND * se, row["floor_mean"] + Z_BOUND * se
    means: dict[str, dict[str, float]] = {}
    sets: dict[str, dict[str, float]] = {}
    for p in files:
        recs = [r for r in _load(root, f"{ANALYSIS_DIR}/{p.name}")["records"] if r["k"] == 100]
        for m in sorted({r["method"] for r in recs}):
            mine = [r for r in recs if r["method"] == m]
            if len(mine) != n_seeds:
                raise ValueError(f"{p.name}: {m} has {len(mine)} K 100 records")
            means.setdefault(m, {})[p.stem] = mean(r["worst_class"] for r in mine)
            sets.setdefault(m, {})[p.stem] = mean(r["set_over_k"] for r in mine)
    src = f"{ANALYSIS_DIR}/"
    yield Entry("ksweep_all.n_caches", str(len(files)), src, "count of controlled_k_sweep_*.json")
    for backbone, is_l in (("ViT-B", False), ("ViT-L", True)):
        yield Entry(f"ksweep_all.n_caches.{backbone}",
                    str(sum(("_vitL_" in p.stem) == is_l for p in files)), src,
                    f"count of those files {'with' if is_l else 'without'} vitL in the name")
    yield Entry("ksweep_all.n_seeds", str(n_seeds), src, "config.subset_seeds, equal in every file")
    p = "ksweep_all.K100"
    yield Entry(f"{p}.floor_mean", f3(row["floor_mean"]), VALIDITY_FLOOR,
                "rows[setting == k_sweep, K == 100].floor_mean")
    yield Entry(f"{p}.floor_sd", f3(row["floor_sd"]), VALIDITY_FLOOR,
                "rows[setting == k_sweep, K == 100].floor_sd")
    yield Entry(f"{p}.floor_se_of_seed_mean", f3(se), VALIDITY_FLOOR,
                "floor_sd / sqrt(config.subset_seeds)")
    for m, by_cache in sorted(means.items()):
        vals = list(by_cache.values())
        if any(min(abs(x - lo), abs(x - hi)) < 1e-9 for x in vals):
            raise ValueError(f"{m}: a five-seed mean sits on the edge of the floor band")
        key = f"records[k == 100, method == {m}], mean over seeds, per cache"
        yield Entry(f"{p}.{m}.n_below", str(sum(x < lo for x in vals)), src,
                    f"count of {key} below floor_mean - {Z_BOUND} floor_se_of_seed_mean")
        yield Entry(f"{p}.{m}.n_within", str(sum(lo <= x <= hi for x in vals)), src,
                    f"count of {key} within {Z_BOUND} floor_se_of_seed_mean of floor_mean")
        yield Entry(f"{p}.{m}.n_above", str(sum(x > hi for x in vals)), src,
                    f"count of {key} above floor_mean + {Z_BOUND} floor_se_of_seed_mean")
        yield Entry(f"{p}.{m}.min_mean", f3(min(vals)), src, f"min of {key}")
        yield Entry(f"{p}.{m}.max_mean", f3(max(vals)), src, f"max of {key}")
        above = [sets[m][c] for c, x in by_cache.items() if x > hi]
        if above:
            yield Entry(f"{p}.{m}.min_set_over_k_above", f3(min(above)), src,
                        f"min over the caches above the band of the mean set_over_k of {m}")
            yield Entry(f"{p}.{m}.max_set_over_k_above", f3(max(above)), src,
                        f"max over the caches above the band of the mean set_over_k of {m}")


def label_sweep_set_size_entries(root: Path) -> Iterator[Entry]:
    """The mean set size of every method at every budget of the label sweep, from the summary
    rows the rerun added, at one decimal."""
    for i, s in enumerate(_load(root, LABEL_SWEEP)["summary"]):
        if s["mean_set_size_mean"] is None:
            continue
        yield Entry(f"label_sweep.set_size.{s['method']}.m{s['m']}", f1(s["mean_set_size_mean"]),
                    LABEL_SWEEP, f"summary[{i}].mean_set_size_mean")


def ntu_row_entries(root: Path) -> Iterator[Entry]:
    """The NTU to N-UCLA row of every method on the regenerated leaderboard, and whether each
    class at zero coverage in every seed is flagged as significantly undercovered in every seed,
    by the runner's own test on the stored counts."""
    sys.path.insert(0, str(root / "src"))
    from pcpbench import runner
    from pcpbench.metrics.validity_floor import (SIG_TESTABLE, significance_class_status,
                                                 significance_cutoff)

    rows = {m: _load(root, f"{ROWS_DIR}/ntu2nucla_{m}.json") for m in sorted(runner.METHODS)}
    zero_class = flagged = 0
    for m, row in rows.items():
        cov, n_eval, n_cal = (row["per_class_coverage_per_seed"], row["n_eval_per_class_per_seed"],
                              row["n_cal_per_class_per_seed"])
        zeros = [c for c in range(row["n_classes"]) if all(seed[c] == 0 for seed in cov)]
        if not zeros:
            continue
        zero_class += 1
        every = True
        for s in range(row["seeds"]):
            status = significance_class_status(n_eval[s], n_cal[s], row["alpha"])
            for c in zeros:
                cut = significance_cutoff(n_eval[s][c], n_cal[s][c], row["alpha"], runner.SIG_Q)
                hits = round(cov[s][c] * n_eval[s][c])
                every &= bool(status[c] == SIG_TESTABLE and hits <= cut)
        flagged += every
    src = f"{ROWS_DIR}/"
    for m, row in rows.items():
        yield Entry(f"ntu2nucla.{m}.frac_sig_undercovered", f3(row["frac_sig_undercovered"]), src,
                    f"ntu2nucla_{m}.json frac_sig_undercovered")
    fracs = [row["frac_sig_undercovered"] for row in rows.values()]
    yield Entry("ntu2nucla.frac_sig_undercovered.min", f3(min(fracs)), src, "min over the methods")
    yield Entry("ntu2nucla.frac_sig_undercovered.max", f3(max(fracs)), src, "max over the methods")
    yield Entry("ntu2nucla.n_methods", str(len(rows)), RUNNER, "len(runner.METHODS)")
    yield Entry("ntu2nucla.n_methods_with_a_zero_class", str(zero_class), src,
                "methods with a class at coverage 0 in every seed")
    yield Entry("ntu2nucla.n_methods_flagging_the_zero_class_every_seed", str(flagged), src,
                "those methods whose every such class is testable and at most the cutoff in every seed")
    seeds = {row["seeds"] for row in rows.values()}
    if len(seeds) != 1:
        raise ValueError("the NTU to N-UCLA rows differ in seeds")
    yield Entry("ntu2nucla.seeds", str(seeds.pop()), src, "seeds, equal in every row")


def board_diff_entries(root: Path) -> Iterator[Entry]:
    """The cell-level diff of the regenerated leaderboard against the released one."""
    d = _load(root, BOARD_DIFF)
    src = BOARD_DIFF
    for k in ("n_rows_old", "n_rows_new", "n_shared_rows", "n_changed_rows", "n_changed_cells",
              "n_regime_changes"):
        yield Entry(f"board_diff.{k}", str(d[k]), src, k)
    yield Entry("board_diff.n_rows_only_in_new", str(len(d["rows_only_in_new"])), src,
                "len(rows_only_in_new)")
    yield Entry("board_diff.n_demotion_changes", str(len(d["demotion_changes"])), src,
                "len(demotion_changes)")
    yield Entry("board_diff.changed_methods", " ".join(sorted({c["method"] for c in d["changed_rows"]})),
                src, "distinct changed_rows[].method")
    for col, n in d["changed_cells_per_column"].items():
        yield Entry(f"board_diff.changed_cells.{col}", str(n), src, f"changed_cells_per_column.{col}")
        moves = [c["cells"][col]["abs_diff"] for c in d["changed_rows"] if col in c["cells"]]
        fmt = f2 if col == "set/K" else f3
        yield Entry(f"board_diff.max_move.{col}", fmt(max(moves)), src,
                    f"max of changed_rows[].cells.{col}.abs_diff")


def _table_values(rows: list, al) -> dict[str, dict[str, list[str]]]:
    _, summ = al.exhibit_k_scaling(al.select_headline(rows))
    delta = {s.method: [str(s.n_cells_informative), f"{s.mean_delta_worst_informative:+.3f}",
                        f"{s.mean_delta_worst_all:+.3f}"] for s in summ}
    vitl_rows = [r for r in rows if r.structured and r.backbone == "ViT-L"
                 and r.method in al.HEADLINE_METHODS]
    cells_l, _ = al.exhibit_k_scaling(vitl_rows)
    vitl = {m: [f"{mean(c.delta_worst for c in cells_l if c.method == m and c.informative):+.3f}",
                delta[m][1]] for m in delta}
    imagenet = {}
    counts = set()
    for m in delta:
        cells = [r for r in rows if r.dataset.startswith("imagenet1k") and r.method == m]
        counts.add(len(cells))
        imagenet[m] = [f3(mean(r.marginal for r in cells)), f3(mean(r.worst_class for r in cells)),
                       f3(mean(r.frac_undercovered for r in cells)),
                       f1(mean(r.mean_set_size for r in cells)), f3(mean(r.set_over_k for r in cells))]
    if len(counts) != 1:
        raise ValueError(f"the methods differ in their number of ImageNet-1k rows: {sorted(counts)}")
    return {"tab:delta": delta, "tab:vitl": vitl, "tab:imagenet": imagenet}


def build_registry(root: Path = _REPO_ROOT) -> list[Entry]:
    """Every entry, in the order of the groups below."""
    root = Path(root)
    groups: Iterable[Iterator[Entry]] = (
        floor_entries(root),
        sweep_entries(root),
        threshold_entries(root),
        label_sweep_entries(root),
        clustered_entries(root),
        significance_entries(root),
        anchor_entries(root),
        headline_entries(root),
        clustered_rerun_entries(root),
        aps_entries(root),
        no_shift_entries(root),
        ksweep_all_entries(root),
        label_sweep_set_size_entries(root),
        ntu_row_entries(root),
        board_diff_entries(root),
    )
    out: list[Entry] = []
    for g in groups:
        out.extend(g)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=_REPO_ROOT,
                    help="repository root holding the committed artifacts")
    args = ap.parse_args(argv)
    for e in build_registry(args.root):
        print(e.line())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
