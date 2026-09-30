"""Figures for Chapter 9 (Results and discussion).

Usage:
    uv run python -m ems.figures

Run after ems.experiments, ems.summary and ems.report. Reads results/comparison.csv,
results/training_episodes.csv (written by ems.report) and the SB3 Monitor logs in
results/models/, and simulates one deterministic trace for Figure 4. Writes four PNG
files to results/. Every figure regenerates from this one command (gate G7).

    fig9_cost.png          operating cost by controller and scenario        (Figure 2)
    fig9_fuel_split.png    no-load and marginal fuel by controller           (Figure 3)
    fig9_soc_traces.png    state of charge and genset operation, three days  (Figure 4)
    fig9_rl_training.png   RL training cost above cycle charging, same episodes (Figure 5)
    fig10_families.png     best of each method family, by scenario            (Figure 7)
    fig10_mechanism.png    cost saving against genset on-time saved            (Figure 8)

Colour follows the job it does. Controllers are grouped in three families (fixed
rules, optimisation-based, learned), one categorical slot each; the three slots pass
the colour-vision check in every pair. Text is never coloured; every figure also
has a table in the chapter with the same numbers.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ems.components.genset import is_on  # noqa: E402
from ems.experiments import EVALUATION_DAYS  # noqa: E402
from ems.report import TRAINING_EPISODES  # noqa: E402
from ems.run import CONTROLLERS  # noqa: E402
from ems.scenarios import expected_scenario_profiles, scenario_profiles  # noqa: E402
from ems.system import IntegratedSystem  # noqa: E402
from ems.vessel import REFERENCE  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
MODELS = RESULTS / "models"

SCENARIOS = ["standard", "high_load", "cloudy"]
SCENARIO_LABEL = {"standard": "Standard", "high_load": "High load", "cloudy": "Cloudy"}
ORDER = ["rule_based", "cycle_charging", "mpc", "mpc_perfect", "offline_optimum",
         "rl_t0", "rl_t1", "rl_t2"]
LABEL = {
    "rule_based": "Rule-based",
    "cycle_charging": "Cycle charging",
    "mpc": "MPC",
    "mpc_perfect": "MPC, perfect forecast",
    "offline_optimum": "Offline optimum",
    "rl_t0": "RL, seed 0",
    "rl_t1": "RL, seed 1",
    "rl_t2": "RL, seed 2",
}

# Categorical slots 1-3 of the reference palette (validated all-pairs, light mode).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
FAMILY = {"Fixed rule": BLUE, "Optimisation-based": ORANGE, "Learned (RL)": AQUA}
FAMILY_OF = {"rule_based": "Fixed rule", "cycle_charging": "Fixed rule",
             "mpc": "Optimisation-based", "mpc_perfect": "Optimisation-based",
             "offline_optimum": "Optimisation-based",
             "rl_t0": "Learned (RL)", "rl_t1": "Learned (RL)", "rl_t2": "Learned (RL)"}
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e6e5e1"
SURFACE = "#ffffff"

# Figure 4: a three-day window after the first week of the standard scenario, seed 0.
TRACE_SCENARIO, TRACE_SEED, TRACE_START_DAY, TRACE_DAYS = "standard", 0, 7, 3
TRACE_CONTROLLERS = ["rule_based", "cycle_charging", "offline_optimum"]
# Figure 5: rolling mean over this many episodes (row in the contract).
ROLLING_EPISODES = 50

# Layout, in inches: the text width of the thesis page is 16 cm.
WIDTH = 6.3

# Conclusion figures: the three method families compared as a synthesis, and the
# mechanism that the comparison exposes.
FAMILY_MEMBERS = {
    "Fixed rule": ["cycle_charging", "rule_based"],
    "Optimisation-based": ["mpc", "mpc_perfect", "offline_optimum"],
    "Learned (RL)": ["rl_t0", "rl_t1", "rl_t2"],
}
BASELINE = "rule_based"


# ---------------------------------------------------------------- style
def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.edgecolor": INK_3,
        "axes.labelcolor": INK_2,
        "axes.linewidth": 0.6,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "savefig.dpi": 300,
        "svg.hashsalt": "ems",
    })


def _save(fig: plt.Figure, name: str) -> None:
    out = RESULTS / name
    # Fixed metadata so two runs write byte-identical files (gate G7).
    fig.savefig(out, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    print(f"  wrote {out}")


def _clean(ax: plt.Axes, grid_axis: str) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- data
def load_runs() -> dict[tuple[str, str], list[dict[str, float]]]:
    with (RESULTS / "comparison.csv").open(encoding="utf-8") as fh:
        fh.readline()
        rows = list(csv.DictReader(fh))
    runs: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for r in rows:
        runs[(r["scenario"], r["controller"])].append(
            {k: float(v) for k, v in r.items() if k not in ("scenario", "controller")})
    return runs


def _family_legend(fig: plt.Figure) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY.values()]
    fig.legend(handles, list(FAMILY), loc="lower center", ncol=len(FAMILY),
               bbox_to_anchor=(0.5, -0.02), handlelength=1.0)


# ---------------------------------------------------------------- Figure 2
def fig_cost(runs) -> None:
    """Total operating cost, one panel per scenario, one bar per controller."""
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(WIDTH, 2.9), sharey=True)
    y = np.arange(len(ORDER))[::-1]
    top = max(r["total_eur"] for v in runs.values() for r in v)
    for ax, s in zip(axes, SCENARIOS):
        for yi, c in zip(y, ORDER):
            v = [r["total_eur"] for r in runs[(s, c)]]
            m = float(np.mean(v))
            ax.barh(yi, m, height=0.62, color=FAMILY[FAMILY_OF[c]], edgecolor=SURFACE, linewidth=1)
            ax.errorbar(m, yi, xerr=[[m - min(v)], [max(v) - m]], color=INK, lw=0.6, capsize=1.5)
            ax.text(m + top * 0.02, yi, f"{m:.0f}", va="center", fontsize=6.5, color=INK_2)
        ax.set_title(SCENARIO_LABEL[s], color=INK)
        ax.set_xlim(0, top * 1.22)
        ax.set_xlabel("Cost over 30 days (EUR)")
        _clean(ax, "x")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LABEL[c] for c in ORDER])
    _family_legend(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, "fig9_cost.png")


# ---------------------------------------------------------------- Figure 3
def fig_fuel_split(runs) -> None:
    """Fuel split into the no-load part (genset running at all) and the marginal part."""
    dt = REFERENCE.timestep_hours
    idle = REFERENCE.genset_intercept_l_per_h_kw * REFERENCE.genset_rated_kw
    slope = REFERENCE.genset_slope_l_per_kwh
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(WIDTH, 2.9), sharey=True)
    y = np.arange(len(ORDER))[::-1]
    top = max(r["fuel_l"] for v in runs.values() for r in v)
    for ax, s in zip(axes, SCENARIOS):
        for yi, c in zip(y, ORDER):
            noload = float(np.mean([idle * r["genset_on_steps"] * dt for r in runs[(s, c)]]))
            marginal = float(np.mean([slope * r["diesel_kwh"] for r in runs[(s, c)]]))
            ax.barh(yi, marginal, height=0.62, color=BLUE, edgecolor=SURFACE, linewidth=1)
            ax.barh(yi, noload, left=marginal, height=0.62, color=ORANGE, edgecolor=SURFACE, linewidth=1)
            ax.text(marginal + noload + top * 0.02, yi, f"{marginal + noload:.0f}",
                    va="center", fontsize=6.5, color=INK_2)
        ax.set_title(SCENARIO_LABEL[s], color=INK)
        ax.set_xlim(0, top * 1.22)
        ax.set_xlabel("Fuel over 30 days (L)")
        _clean(ax, "x")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LABEL[c] for c in ORDER])
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    fig.legend(handles, ["Marginal fuel (F1 × energy)", "No-load fuel (F0 × rating × hours on)"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02), handlelength=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, "fig9_fuel_split.png")


# ---------------------------------------------------------------- Figure 4
def fig_soc_traces() -> None:
    """SoC and genset operation over three days, for three controllers."""
    days = TRACE_START_DAY + TRACE_DAYS
    real = scenario_profiles(TRACE_SCENARIO, EVALUATION_DAYS, TRACE_SEED)
    expected = expected_scenario_profiles(TRACE_SCENARIO, EVALUATION_DAYS)
    per_day = round(24 / REFERENCE.timestep_hours)
    lo, hi = TRACE_START_DAY * per_day, days * per_day
    hours = np.arange(lo, hi) * REFERENCE.timestep_hours
    fig, axes = plt.subplots(len(TRACE_CONTROLLERS), 1, figsize=(WIDTH, 4.2), sharex=True)
    for ax, c in zip(axes, TRACE_CONTROLLERS):
        result = IntegratedSystem(*real, CONTROLLERS[c](real, expected)).run()
        on = is_on(result.diesel_kw[lo:hi])
        # Genset running: shaded spans, no second axis.
        edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            ax.axvspan(hours[a], hours[b - 1] + REFERENCE.timestep_hours, color=ORANGE,
                       alpha=0.25, lw=0)
        for bound in (REFERENCE.soc_min, REFERENCE.soc_max):
            ax.axhline(bound, color=INK_3, lw=0.6, ls="--")
        ax.plot(hours, result.soc[lo:hi], color=BLUE, lw=1.4)
        ax.set_ylim(0, 1)
        ax.set_ylabel("SoC")
        ax.set_title(LABEL[c], loc="left", color=INK)
        _clean(ax, "y")
    axes[-1].set_xlabel("Time from the start of the run (h)")
    axes[-1].set_xticks(np.arange(lo, hi + 1, per_day // 4) * REFERENCE.timestep_hours)
    handles = [plt.Line2D([], [], color=BLUE, lw=1.4),
               plt.Rectangle((0, 0), 1, 1, color=ORANGE, alpha=0.25),
               plt.Line2D([], [], color=INK_3, lw=0.6, ls="--")]
    fig.legend(handles, ["State of charge", "Genset running", "SoC window 0.10–0.90"],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01), handlelength=1.6)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, "fig9_soc_traces.png")


# ---------------------------------------------------------------- Figure 5
def fig_rl_training() -> None:
    """RL training cost above cycle charging, on the same training episodes."""
    from ems.rl.env import TRAIN_EPISODE_DAYS
    from ems.rl.train import N_ENVS

    steps_per_episode = TRAIN_EPISODE_DAYS * round(24 / REFERENCE.timestep_hours)
    with TRAINING_EPISODES.open(encoding="utf-8") as fh:
        fh.readline()
        rows = list(csv.DictReader(fh))
    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    colours = {"rl_t0": BLUE, "rl_t1": ORANGE, "rl_t2": AQUA}
    kernel = np.ones(ROLLING_EPISODES) / ROLLING_EPISODES
    gaps = {}
    for name, colour in colours.items():
        eps = sorted((r for r in rows if r["policy"] == name), key=lambda r: float(r["t"]))
        excess = np.array([float(r["rl"]) - float(r["cycle_charging"]) for r in eps])
        smooth = np.convolve(excess, kernel, mode="valid")
        x = (np.arange(len(smooth)) + ROLLING_EPISODES) * steps_per_episode
        ax.plot(x, smooth, color=colour, lw=1.4, label=f"RL, training seed {name.removeprefix('rl_t')}")
        for r in eps:
            gaps[(r["scenario"], r["profile_seed"])] = float(r["rule_based"]) - float(r["cycle_charging"])
    rule_gap = float(np.mean(list(gaps.values())))
    ax.axhline(0.0, color=INK_3, lw=0.8, ls="--")
    ax.axhline(rule_gap, color=INK_3, lw=0.8, ls="--")
    xmax = ax.get_xlim()[1]
    ax.text(xmax, 0.0, " Cycle charging", va="center", fontsize=6.5, color=INK_2)
    ax.text(xmax, rule_gap, " Rule-based (mean)", va="center", fontsize=6.5, color=INK_2)
    ax.set_xlabel(f"Training timesteps (sum over the {N_ENVS} environments)")
    ax.set_ylabel("Cost above cycle charging\n(EUR per 7-day episode)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    _clean(ax, "y")
    ax.legend(loc="upper right", bbox_to_anchor=(0.86, 1.0))
    fig.tight_layout()
    _save(fig, "fig9_rl_training.png")


# ------------------------------------------------- Figure 6: three families
def _saving_vs_baseline(runs, metric):
    """Mean saving, in percent, of every controller against the baseline.

    Returns {(scenario, controller): pct}. The baseline itself is 0 by definition.
    """
    out = {}
    for s in SCENARIOS:
        base = float(np.mean([r[metric] for r in runs[(s, BASELINE)]]))
        for c in ORDER:
            m = float(np.mean([r[metric] for r in runs[(s, c)]]))
            out[(s, c)] = 0.0 if base == 0 else 100.0 * (base - m) / base
    return out


def fig_families(runs) -> None:
    """Best member of each of the three method families, by scenario.

    This is the synthesis figure for the conclusions: it answers "which family of
    methods saves most?" in one view, which the per-controller figures of the
    results chapter do not.
    """
    save = _saving_vs_baseline(runs, "total_eur")
    fams = list(FAMILY_MEMBERS)
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    x = np.arange(len(SCENARIOS))
    w = 0.8 / len(fams)
    for i, fam in enumerate(fams):
        members = FAMILY_MEMBERS[fam]
        # The RL family has three policies and no single best; show the mean.
        vals = [float(np.mean([save[(s, c)] for c in members])) for s in SCENARIOS]
        errs_lo, errs_hi = [], []
        for j, s in enumerate(SCENARIOS):
            v = [save[(s, c)] for c in members]
            errs_lo.append(vals[j] - min(v))
            errs_hi.append(max(v) - vals[j])
        off = (i - len(fams) / 2 + 0.5) * w
        span = f"{min(vals):.0f}–{max(vals):.0f}%"
        ax.bar(x + off, vals, w, color=FAMILY[fam], edgecolor=SURFACE, linewidth=1,
               label=f"{fam} — mean of {len(members)} methods, range {span}",
               yerr=[errs_lo, errs_hi],
               error_kw={"elinewidth": 0.6, "capthick": 0.6})
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABEL[s] for s in SCENARIOS])
    ax.set_ylabel("Operating-cost saving vs rule-based (%)")
    ax.axhline(0, color=INK_3, lw=0.8)
    _clean(ax, "y")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1)
    fig.tight_layout()
    _save(fig, "fig10_families.png")


# ------------------------------------------- Figure 7: the shared mechanism
def fig_mechanism(runs) -> None:
    """Cost saving against genset on-time saved, for every controller and scenario.

    The single most useful result in the thesis: the saving is almost entirely
    explained by how much less time the genset runs, whatever the method. One
    point per (scenario, controller), coloured by family, with the fitted line.
    """
    save_cost = _saving_vs_baseline(runs, "total_eur")
    save_time = _saving_vs_baseline(runs, "genset_on_steps")
    fig, ax = plt.subplots(figsize=(WIDTH, 3.4))
    xs, ys = [], []
    for c in ORDER:
        if c == BASELINE:
            continue
        for s in SCENARIOS:
            x, y = save_time[(s, c)], save_cost[(s, c)]
            xs.append(x)
            ys.append(y)
            ax.plot(x, y, "o", ms=4.5, color=FAMILY[FAMILY_OF[c]], mec=SURFACE, mew=0.6)
    xs_a, ys_a = np.array(xs), np.array(ys)
    slope, intercept = np.polyfit(xs_a, ys_a, 1)
    pred = slope * xs_a + intercept
    ss_res = float(((ys_a - pred) ** 2).sum())
    ss_tot = float(((ys_a - ys_a.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    grid = np.linspace(0, xs_a.max(), 50)
    ax.plot(grid, slope * grid + intercept, color=INK_3, lw=1.0, ls="--", zorder=1)
    ax.annotate(f"slope {slope:.2f}\n$R^2$ = {r2:.3f}",
                xy=(0.97, 0.06), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=7, color=INK_2)
    ax.set_xlabel("Reduction in genset on-time vs rule-based (%)")
    ax.set_ylabel("Operating-cost saving vs rule-based (%)")
    _clean(ax, "both")
    _family_legend(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, "fig10_mechanism.png")


def main() -> int:
    _style()
    runs = load_runs()
    fig_cost(runs)
    fig_fuel_split(runs)
    fig_soc_traces()
    fig_rl_training()
    fig_families(runs)
    fig_mechanism(runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
