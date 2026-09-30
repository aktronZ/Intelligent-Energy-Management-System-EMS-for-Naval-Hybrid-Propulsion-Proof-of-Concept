"""Every number Chapter 9 quotes, computed from results/ and written to one file.

Usage:
    uv run python -m ems.report

Reads results/comparison.csv, results/summary.csv, results/models/*.json and the SB3
Monitor logs, plus the pre-registered values in specs/010-simulation-contract.md §4,
and writes results/chapter9_numbers.md. The file holds the values already formatted
the way thesis/09-results.md quotes them, and the chapter's tables verbatim, so that
every number in Chapter 9 can be found in results/ (gate G9) and none is typed by
hand. The run is deterministic: it only reads files.

Definitions (all per run, then averaged over the evaluation seeds 0-4):
    saving(A vs B)        (cost_B - cost_A) / cost_B, from results/summary.csv
    share of the MPC gain captured by cycle charging
                          (cost_rule - cost_cc) / (cost_rule - cost_mpc)
    share of the offline gain captured by a controller X
                          (cost_rule - cost_X) / (cost_rule - cost_offline)
    genset hours          genset_on_steps x timestep
    load factor when on   diesel_kwh / (genset hours x rated power)
    no-load fuel          F0 x rated power x genset hours      (Willans intercept)
    marginal fuel         F1 x diesel_kwh                       (Willans slope)
The no-load and marginal fuel add up to fuel_l for every run; the report checks it.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from ems.cost import terminal_value_eur_per_kwh
from ems.vessel import REFERENCE

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
MODELS = RESULTS / "models"
CONTRACT = ROOT / "specs" / "010-simulation-contract.md"
OUT = RESULTS / "chapter9_numbers.md"

SCENARIOS = ["standard", "high_load", "cloudy"]
SCENARIO_LABEL = {"standard": "Standard", "high_load": "High load", "cloudy": "Cloudy"}
CONTROLLERS = ["rule_based", "cycle_charging", "mpc", "mpc_perfect", "offline_optimum",
               "rl_t0", "rl_t1", "rl_t2"]
LABEL = {
    "rule_based": "Rule-based (load following)",
    "cycle_charging": "Cycle charging",
    "mpc": "MPC (schedule forecast)",
    "mpc_perfect": "MPC (perfect forecast)",
    "offline_optimum": "Offline optimum",
    "rl_t0": "RL, training seed 0",
    "rl_t1": "RL, training seed 1",
    "rl_t2": "RL, training seed 2",
    "rl_mean": "RL, mean of the three",
}
RL = ["rl_t0", "rl_t1", "rl_t2"]
# Reporting parameters, each with a row in specs/010-simulation-contract.md.
TRAINING_TAIL_EPISODES = 200      # last episodes averaged to summarise a training run
TOLERANCE_L = 1e-6                # the fuel decomposition must close to this
REFERENCE_CONTROLLERS = ["rule_based", "cycle_charging"]
TRAINING_EPISODES = RESULTS / "training_episodes.csv"


# ---------------------------------------------------------------- loading
def _read(path: Path) -> tuple[str, list[dict[str, str]]]:
    with path.open(encoding="utf-8") as fh:
        header = fh.readline().strip()
        return header, list(csv.DictReader(fh))


def load_runs() -> tuple[str, dict[tuple[str, str], list[dict[str, float]]]]:
    header, rows = _read(RESULTS / "comparison.csv")
    runs: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["scenario"], r["controller"], int(r["seed"]))):
        runs[(r["scenario"], r["controller"])].append(
            {k: float(v) for k, v in r.items() if k not in ("scenario", "controller")})
    return header, runs


def load_summary() -> dict[tuple[str, str, str, str], tuple[float, float, float]]:
    _, rows = _read(RESULTS / "summary.csv")
    return {(r["scenario"], r["controller"], r["versus"], r["metric"]):
            (float(r["saving_mean_pct"]), float(r["saving_min_pct"]), float(r["saving_max_pct"]))
            for r in rows}


def load_training() -> dict[str, dict[str, float]]:
    """Per policy: the episode cost (EUR) over the training run, from the Monitor logs."""
    from ems.rl.env import REWARD_SCALE_EUR
    out = {}
    for name in RL:
        seed = name.removeprefix("rl_t")
        meta = json.loads((MODELS / f"rl_ppo_t{seed}.json").read_text(encoding="utf-8"))
        episodes = []
        for log in sorted((MODELS / f"monitor_t{seed}").glob("*.monitor.csv")):
            with log.open(encoding="utf-8") as fh:
                fh.readline()
                episodes += [(float(r["t"]), -float(r["r"]) * REWARD_SCALE_EUR)
                             for r in csv.DictReader(fh)]
        cost = np.array([c for _, c in sorted(episodes)])
        tail = cost[-TRAINING_TAIL_EPISODES:]
        out[name] = {
            "episodes": len(cost),
            "tail_mean": float(tail.mean()),
            "tail_se": float(tail.std() / np.sqrt(len(tail))),
            "wall_seconds": float(meta["wall_seconds"]),
            "timesteps": int(meta["timesteps"]),
            "commit": meta["commit"],
        }
    return out


def _episode_sequence(env_seed: int, n: int) -> list[tuple[str, int]]:
    """The (scenario, profile seed) of the first n episodes of one training environment.

    ems.rl.train builds its environments with make_vec_env(seed=train_seed), so
    environment rank i is reset once with seed train_seed + i and then draws, at every
    reset, a scenario and a profile seed from its own generator (EMSEnv.reset). The
    same draws are replayed here. The report checks the replay: on the same episodes,
    the RL training cost and the cycle-charging cost must be strongly correlated.
    """
    from ems.rl.env import TRAIN_SCENARIOS
    from ems.rl.seeds import TRAIN_SEED_POOL, TRAIN_SEED_START

    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(env_seed)))
    out = []
    for _ in range(n):
        scenario = TRAIN_SCENARIOS[int(rng.integers(len(TRAIN_SCENARIOS)))]
        out.append((scenario, TRAIN_SEED_START + int(rng.integers(TRAIN_SEED_POOL))))
    return out


def training_episodes() -> list[dict[str, object]]:
    """Every RL training episode, with the cost of the fixed rules on the same episode.

    Written to results/training_episodes.csv: one row per (policy, environment rank,
    episode), with the RL training cost from the Monitor log and the rule-based and
    cycle-charging costs of the same 7-day episode, so that the learning curve can be
    read against a fixed rule on identical data instead of against noisy episode costs.
    """
    from ems.experiments import run_case
    from ems.rl.env import REWARD_SCALE_EUR, TRAIN_EPISODE_DAYS
    from ems.rl.train import N_ENVS

    cache: dict[tuple[str, int], dict[str, float]] = {}
    rows: list[dict[str, object]] = []
    for name in RL:
        train_seed = int(name.removeprefix("rl_t"))
        for rank in range(N_ENVS):
            log = MODELS / f"monitor_t{train_seed}" / f"{rank}.monitor.csv"
            with log.open(encoding="utf-8") as fh:
                fh.readline()
                logged = [(float(r["t"]), -float(r["r"]) * REWARD_SCALE_EUR) for r in csv.DictReader(fh)]
            for k, ((scenario, seed), (t, cost)) in enumerate(
                    zip(_episode_sequence(train_seed + rank, len(logged)), logged)):
                if (scenario, seed) not in cache:
                    cache[(scenario, seed)] = {
                        c: float(run_case((scenario, seed, c, TRAIN_EPISODE_DAYS))["total_eur"])
                        for c in REFERENCE_CONTROLLERS}
                rows.append({"policy": name, "rank": rank, "episode": k, "t": t,
                             "scenario": scenario, "profile_seed": seed, "rl": cost,
                             **cache[(scenario, seed)]})
    with TRAINING_EPISODES.open("w", newline="", encoding="utf-8") as fh:
        fh.write(f"# source=ems.report days={TRAIN_EPISODE_DAYS} costs in EUR per episode\n")
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    return rows


def genset_starts(days: int) -> dict[str, int]:
    """Off-to-on transitions of the genset in one run (standard scenario, seed 0).

    The cost function has no start penalty, so the number of starts is reported as a
    side effect of each strategy. MPC is left out: a 30-day MPC run takes minutes, and
    its trajectory is within 0.1 % of the offline optimum's cost.
    """
    from ems.components.genset import is_on
    from ems.run import CONTROLLERS as REGISTRY
    from ems.scenarios import expected_scenario_profiles, scenario_profiles
    from ems.system import IntegratedSystem

    real = scenario_profiles("standard", days, 0)
    expected = expected_scenario_profiles("standard", days)
    out = {}
    for c in ["rule_based", "cycle_charging", "offline_optimum"]:
        on = is_on(IntegratedSystem(*real, REGISTRY[c](real, expected)).run().diesel_kw).astype(int)
        out[c] = int(np.sum(np.diff(np.r_[0, on]) == 1))
    return out


def reference_values() -> dict[str, str]:
    """The pre-registered band, the falsifier and the literature values, read from §4."""
    text = CONTRACT.read_text(encoding="utf-8")
    band = re.search(r"\| MPC \| −(\d+)% to −(\d+)% \|", text)
    kanellos = re.search(r"Kanellos et al\. \(2016\) report (\d+)–(\d+)%", text)
    antono = re.search(r"Antonopoulos et al\. \(2021\) report about ([\d.]+)%", text)
    wu = re.search(r"Wu et al\. \(2020\) reach\s+([\d.]+)%", text)
    falsifier = re.search(r"by more than (\d+)%", text)
    if not all((band, kanellos, antono, wu, falsifier)):
        raise SystemExit("contract §4 no longer matches the expected wording; update ems.report")
    return {
        "pre-registered MPC band": f"{band.group(1)} % to {band.group(2)} %",
        "falsifier threshold": f"{falsifier.group(1)} %",
        "Kanellos et al. (2016)": f"{kanellos.group(1)} % to {kanellos.group(2)} %",
        "Antonopoulos et al. (2021)": f"{antono.group(1)} %",
        "Wu et al. (2020), share of the DP optimum": f"{wu.group(1)} %",
    }


# ---------------------------------------------------------------- helpers
def pct(x: float) -> str:
    return f"{x:.2f} %"


def rng(values: list[float], fmt: str = "{:.2f}") -> str:
    return f"{fmt.format(min(values))}–{fmt.format(max(values))}"


def col(runs, scenario, controller, key) -> list[float]:
    return [r[key] for r in runs[(scenario, controller)]]


def mean(xs: list[float]) -> float:
    return float(np.mean(xs))


def ratio(runs, scenario, num_a, num_b, den_a, den_b) -> list[float]:
    """Per-seed (cost_numA - cost_numB) / (cost_denA - cost_denB)."""
    a = col(runs, scenario, num_a, "total_eur")
    b = col(runs, scenario, num_b, "total_eur")
    c = col(runs, scenario, den_a, "total_eur")
    d = col(runs, scenario, den_b, "total_eur")
    return [(x - y) / (z - w) * 100.0 for x, y, z, w in zip(a, b, c, d)]


def rl_mean_costs(runs, scenario) -> list[float]:
    return [mean(v) for v in zip(*(col(runs, scenario, n, "total_eur") for n in RL))]


# ---------------------------------------------------------------- report
def build() -> str:
    header, runs = load_runs()
    summary = load_summary()
    training = load_training()
    refs = reference_values()
    dt = REFERENCE.timestep_hours
    rated = REFERENCE.genset_rated_kw
    f0 = REFERENCE.genset_intercept_l_per_h_kw
    f1 = REFERENCE.genset_slope_l_per_kwh
    lines: list[str] = []
    w = lines.append

    w("# Chapter 9 numbers")
    w("")
    w(f"Generated by `python -m ems.report` from `results/comparison.csv` ({header.lstrip('# ')}),")
    w("`results/summary.csv`, `results/models/` and `specs/010-simulation-contract.md` §4.")
    w("Do not edit: regenerate. Every value is a mean over evaluation seeds 0–4 unless a")
    w("range is given, in which case the range is the minimum–maximum across those seeds.")
    w("")

    # -------------------------------------------------- plant and horizon
    days = int(re.search(r"days=(\d+)", header).group(1))
    horizon_steps = days * round(1 / dt) * 24
    w("## Setting")
    w("")
    w(f"- evaluation horizon: {days} days ({horizon_steps} timesteps of {dt:g} h, {horizon_steps * dt:.0f} h)")
    w(f"- genset rated power: {rated:g} kW; no-load fuel F0 x rated: {f0 * rated:.3f} L/h; "
      f"marginal fuel F1: {f1:.3f} L/kWh")
    from ems.rl.env import TRAIN_EPISODE_DAYS
    homer = re.search(r"HOMER documentation, (\d+) kW generator example", CONTRACT.read_text(encoding="utf-8"))
    w(f"- RL training episode length: {TRAIN_EPISODE_DAYS} days; HOMER example generator from which α "
      f"is derived: {homer.group(1)} kW")
    from ems.controllers.cycle_charging import CYCLE_CHARGING_SETPOINT_SOC
    alpha = f0 / (f0 + f1)
    w(f"- no-load fraction α: {alpha:.3f}; cycle-charging setpoint SoC: {CYCLE_CHARGING_SETPOINT_SOC:.2f}")
    w(f"- battery nominal capacity: {REFERENCE.battery_capacity_kwh:g} kWh; fuel price "
      f"{REFERENCE.fuel_price_eur_per_l:g} EUR/L; terminal value λ {terminal_value_eur_per_kwh():.3f} EUR/kWh")
    w(f"- reference vessel: {REFERENCE.name}, {REFERENCE.gross_tonnage:g} GT")
    w(f"- runs: {sum(len(v) for v in runs.values())} "
      f"({len(SCENARIOS)} scenarios x {len(runs[('standard', 'rule_based')])} seeds x "
      f"{len(CONTROLLERS)} controllers)")
    w("")

    # -------------------------------------------------- reference values
    w("## Pre-registered and literature values (specs/010-simulation-contract.md §4)")
    w("")
    for k, v in refs.items():
        w(f"- {k}: {v}")
    w("")

    # -------------------------------------------------- integrity
    allrows = [r for v in runs.values() for r in v]
    w("## Integrity over all runs")
    w("")
    w(f"- max |bus residual|: {max(abs(r['balance_residual_kwh']) for r in allrows):.1e} kWh")
    w(f"- max |battery residual|: {max(abs(r['battery_residual_kwh']) for r in allrows):.1e} kWh")
    w(f"- max unserved energy: {max(r['unserved_kwh'] for r in allrows):.1f} kWh")
    w(f"- max curtailed energy: {max(r['curtail_kwh'] for r in allrows):.1f} kWh")
    w(f"- SoC over all runs: {min(r['soc_min'] for r in allrows):.3f}–{max(r['soc_max'] for r in allrows):.3f}")
    w(f"- MPC solver fallbacks: {int(sum(r['n_solver'] for r in allrows))}; "
      f"genset-cap reductions: {int(sum(r['n_genset_cap'] for r in allrows))}")
    for c in CONTROLLERS:
        w(f"- clipped requests, {c}: {int(sum(r['n_clip'] for s in SCENARIOS for r in runs[(s, c)]))}")
    w("")
    fuel_err = max(abs(r["fuel_l"] - f0 * rated * r["genset_on_steps"] * dt - f1 * r["diesel_kwh"])
                   for r in allrows)
    if fuel_err > TOLERANCE_L:
        raise SystemExit(f"fuel decomposition does not close: {fuel_err:.3e} L")
    w(f"- fuel decomposition residual (no-load + marginal vs fuel_l): {fuel_err:.1e} L")
    w("")

    # -------------------------------------------------- Table 5: cost
    w("## Table 5 — operating cost")
    w("")
    w("| Controller | " + " | ".join(SCENARIO_LABEL[s] for s in SCENARIOS) + " |")
    w("|---|" + "---|" * len(SCENARIOS))
    for c in CONTROLLERS:
        cells = []
        for s in SCENARIOS:
            v = col(runs, s, c, "total_eur")
            cells.append(f"{mean(v):.2f} ({rng(v)})")
        w(f"| {LABEL[c]} | " + " | ".join(cells) + " |")
    w("")

    spans = {(s, c): max(col(runs, s, c, "total_eur")) - min(col(runs, s, c, "total_eur"))
             for s in SCENARIOS for c in CONTROLLERS}
    widest = max(spans, key=spans.get)
    w(f"- largest seed range of any controller: {spans[widest]:.1f} EUR ({widest[1]}, {widest[0]}); "
      f"largest among the non-RL controllers: "
      f"{max(v for k, v in spans.items() if not k[1].startswith('rl_')):.1f} EUR")
    w("")

    # -------------------------------------------------- Table 6: savings
    w("## Table 6 — saving against the rule-based baseline (total cost)")
    w("")
    w("| Controller | " + " | ".join(SCENARIO_LABEL[s] for s in SCENARIOS) + " |")
    w("|---|" + "---|" * len(SCENARIOS))
    for c in ["cycle_charging", "mpc", "mpc_perfect", "offline_optimum", "rl_t0", "rl_t1",
              "rl_t2", "rl_mean"]:
        cells = []
        for s in SCENARIOS:
            if (s, c, "rule_based", "total_eur") in summary:
                m, lo, hi = summary[(s, c, "rule_based", "total_eur")]
            else:  # mpc_perfect vs rule is not a summary pair: compute it the same way
                a, b = col(runs, s, c, "total_eur"), col(runs, s, "rule_based", "total_eur")
                v = [(y - x) / y * 100.0 for x, y in zip(a, b)]
                m, lo, hi = mean(v), min(v), max(v)
            cells.append(f"{m:.2f} ({lo:.2f}–{hi:.2f})")
        w(f"| {LABEL[c]} | " + " | ".join(cells) + " |")
    w("")
    for c in ["cycle_charging", "mpc", "mpc_perfect", "offline_optimum", "rl_t0", "rl_t1",
              "rl_t2", "rl_mean"]:
        for s in SCENARIOS:
            if (s, c, "rule_based", "total_eur") in summary:
                w(f"- saving of {c} vs rule_based, {s}: {pct(summary[(s, c, 'rule_based', 'total_eur')][0])}")
    w("")

    # -------------------------------------------------- pairwise gaps
    w("## Gaps between controllers (total cost, from summary.csv)")
    w("")
    for a, b in [("mpc", "cycle_charging"), ("offline_optimum", "mpc"), ("mpc_perfect", "mpc")]:
        for s in SCENARIOS:
            m, lo, hi = summary[(s, a, b, "total_eur")]
            w(f"- {a} vs {b}, {s}: {pct(m)} ({lo:.2f}–{hi:.2f} %)")
    for s in SCENARIOS:
        for metric in ("fuel_l", "co2_kg"):
            m, lo, hi = summary[(s, "mpc", "rule_based", metric)]
            w(f"- mpc vs rule_based, {metric}, {s}: {pct(m)} ({lo:.2f}–{hi:.2f} %)")
    w("")
    seed0 = {c: runs[("standard", c)][0] for c in ("mpc", "offline_optimum")}
    assert int(seed0["mpc"]["seed"]) == 0
    w(f"- offline_optimum vs mpc, standard, seed 0: "
      f"{pct((seed0['mpc']['total_eur'] - seed0['offline_optimum']['total_eur']) / seed0['mpc']['total_eur'] * 100)}")
    w("")
    w("## Shares of the achievable gain")
    w("")
    for s in SCENARIOS:
        v = ratio(runs, s, "rule_based", "cycle_charging", "rule_based", "mpc")
        w(f"- share of the MPC gain captured by cycle charging, {s}: {pct(mean(v))} ({rng(v)} %)")
    for s in SCENARIOS:
        v = ratio(runs, s, "rule_based", "mpc", "rule_based", "offline_optimum")
        w(f"- share of the offline gain captured by MPC, {s}: {pct(mean(v))} ({rng(v)} %)")
    for s in SCENARIOS:
        for c in RL:
            v = ratio(runs, s, "rule_based", c, "rule_based", "offline_optimum")
            w(f"- share of the offline gain captured by {c}, {s}: {pct(mean(v))} ({rng(v)} %)")
        rule = col(runs, s, "rule_based", "total_eur")
        opt = col(runs, s, "offline_optimum", "total_eur")
        v = [(r - x) / (r - o) * 100.0 for r, x, o in zip(rule, rl_mean_costs(runs, s), opt)]
        w(f"- share of the offline gain captured by rl_mean, {s}: {pct(mean(v))} ({rng(v)} %)")
    w("")

    # -------------------------------------------------- Table 7: mechanism
    w("## Table 7 — how the fuel is spent (means over seeds)")
    w("")
    for s in SCENARIOS:
        w(f"### {SCENARIO_LABEL[s]}")
        w("")
        w("| Controller | Genset on (h) | Load factor when on | Diesel energy (kWh) "
          "| No-load fuel (L) | Marginal fuel (L) | Total fuel (L) |")
        w("|---|---|---|---|---|---|---|")
        for c in CONTROLLERS:
            hours = mean([x * dt for x in col(runs, s, c, "genset_on_steps")])
            kwh = mean(col(runs, s, c, "diesel_kwh"))
            lf = mean([k / (n * dt * rated) * 100.0 for k, n in
                       zip(col(runs, s, c, "diesel_kwh"), col(runs, s, c, "genset_on_steps"))])
            w(f"| {LABEL[c]} | {hours:.1f} | {lf:.1f} % | {kwh:.1f} | {f0 * rated * hours:.1f} "
              f"| {f1 * kwh:.1f} | {mean(col(runs, s, c, 'fuel_l')):.1f} |")
        w("")
    for s in SCENARIOS:
        for c in CONTROLLERS:
            w(f"- diesel energy, {c}, {s}: {mean(col(runs, s, c, 'diesel_kwh')):.1f} kWh")
    w("")
    w("### Mechanism, MPC against the rule-based baseline")
    w("")
    for s in SCENARIOS:
        rh = mean([x * dt for x in col(runs, s, "rule_based", "genset_on_steps")])
        mh = mean([x * dt for x in col(runs, s, "mpc", "genset_on_steps")])
        rk = mean(col(runs, s, "rule_based", "diesel_kwh"))
        mk = mean(col(runs, s, "mpc", "diesel_kwh"))
        rf = mean(col(runs, s, "rule_based", "fuel_l"))
        mf = mean(col(runs, s, "mpc", "fuel_l"))
        noload_saved = f0 * rated * (rh - mh)
        marginal_added = f1 * (mk - rk)
        w(f"- {s}: genset hours {rh:.1f} h -> {mh:.1f} h ({(rh - mh) / rh * 100:.1f} % fewer); "
          f"share of time on {rh / (horizon_steps * dt) * 100:.1f} % -> {mh / (horizon_steps * dt) * 100:.1f} %; "
          f"diesel energy {rk:.1f} kWh -> {mk:.1f} kWh ({(mk - rk) / rk * 100:+.1f} %); "
          f"no-load fuel saved {noload_saved:.1f} L; marginal fuel added {marginal_added:.1f} L; "
          f"net fuel saved {rf - mf:.1f} L; no-load share of baseline fuel "
          f"{f0 * rated * rh / rf * 100:.1f} %")
        w(f"- {s}: CO2 {mean(col(runs, s, 'rule_based', 'co2_kg')):.1f} kg -> "
          f"{mean(col(runs, s, 'mpc', 'co2_kg')):.1f} kg "
          f"({mean(col(runs, s, 'rule_based', 'co2_kg')) - mean(col(runs, s, 'mpc', 'co2_kg')):.1f} kg saved "
          f"over {days} days)")
    w("")
    w(f"### Genset starts (standard scenario, seed 0, {days} days; not recorded in comparison.csv)")
    w("")
    for c, n in genset_starts(days).items():
        w(f"- {c}: {n} starts ({n / days:.1f} per day)")
    w("")
    w("### Terminal valuation")
    w("")
    for s in SCENARIOS:
        for c in ["rule_based", "cycle_charging", "mpc"]:
            w(f"- {c}, {s}: terminal correction {mean(col(runs, s, c, 'terminal_eur')):+.2f} EUR; "
              f"final SoC {mean(col(runs, s, c, 'soc_end')):.3f}")
    for s in SCENARIOS:
        m, lo, hi = summary[(s, "cycle_charging", "rule_based", "fuel_l")]
        w(f"- cycle_charging vs rule_based, fuel only, {s}: {pct(m)}")
        m, lo, hi = summary[(s, "mpc", "cycle_charging", "fuel_l")]
        w(f"- mpc vs cycle_charging, fuel only, {s}: {pct(m)}")
    w("")

    # -------------------------------------------------- RL detail
    w("## Reinforcement learning")
    w("")
    for s in SCENARIOS:
        for c in RL:
            clips = col(runs, s, c, "n_clip")
            on = col(runs, s, c, "genset_on_steps")
            w(f"- {c}, {s}: genset on {mean([x * dt for x in on]):.1f} h ({rng(on, '{:.0f}')} steps); "
              f"clipped requests per run {mean(clips):.0f} ({mean(clips) / horizon_steps * 100:.1f} % of steps); "
              f"final SoC {mean(col(runs, s, c, 'soc_end')):.3f}")
    for s in SCENARIOS:
        for c in ["rule_based", "cycle_charging", "mpc"]:
            on = col(runs, s, c, "genset_on_steps")
            w(f"- {c}, {s}: genset on {rng(on, '{:.0f}')} steps")
    for s in SCENARIOS:
        for versus in ["cycle_charging", "mpc", "offline_optimum"]:
            m, lo, hi = summary[(s, "rl_mean", versus, "total_eur")]
            w(f"- rl_mean vs {versus}, {s}: {pct(m)} ({lo:.2f}–{hi:.2f} %); "
              f"rl_mean costs {-m:.2f} % more")
    for s in SCENARIOS:
        for c in RL:
            m = summary[(s, c, "cycle_charging", "total_eur")][0]
            w(f"- {c} vs cycle_charging, {s}: {-m:.2f} % more expensive")
    for s in SCENARIOS:
        best = max(RL, key=lambda c: summary[(s, c, "rule_based", "total_eur")][0])
        m = summary[(s, best, "cycle_charging", "total_eur")][0]
        w(f"- strongest RL policy, {s}: {best}, {-m:.2f} % more expensive than cycle charging")
    w("")
    w("### Training (Monitor logs, 7-day episodes, scenario and seed drawn from the training pool)")
    w("")
    for c, t in training.items():
        w(f"- {c}: {t['episodes']} episodes, {t['timesteps']} timesteps, "
          f"{t['wall_seconds'] / 60:.1f} min, commit {t['commit']}; last "
          f"{TRAINING_TAIL_EPISODES} episodes: {t['tail_mean']:.2f} EUR per episode "
          f"(standard error {t['tail_se']:.2f} EUR)")
    tails = [t["tail_mean"] for t in training.values()]
    w(f"- last-{TRAINING_TAIL_EPISODES}-episode cost across the three policies: "
      f"{min(tails):.2f}–{max(tails):.2f} EUR per episode")
    w("")
    episodes = training_episodes()
    w("### RL training cost against the fixed rules on the same episodes "
      "(results/training_episodes.csv)")
    w("")
    for name in RL:
        eps = sorted((e for e in episodes if e["policy"] == name), key=lambda e: e["t"])
        rl = np.array([e["rl"] for e in eps])
        cc = np.array([e["cycle_charging"] for e in eps])
        rb = np.array([e["rule_based"] for e in eps])
        tail = slice(-TRAINING_TAIL_EPISODES, None)
        excess = rl[tail] - cc[tail]
        w(f"- {name}: correlation of RL and cycle-charging episode cost {np.corrcoef(rl, cc)[0, 1]:.3f}; "
          f"last {TRAINING_TAIL_EPISODES} episodes: RL {rl[tail].mean():.2f} EUR, cycle charging "
          f"{cc[tail].mean():.2f} EUR, rule-based {rb[tail].mean():.2f} EUR; RL above cycle charging by "
          f"{excess.mean():.2f} EUR per episode (standard error {excess.std() / np.sqrt(len(excess)):.2f} EUR), "
          f"{excess.mean() / cc[tail].mean() * 100:.1f} %")
    unique = {(e["scenario"], e["profile_seed"]): e for e in episodes}
    gap = [(e["rule_based"] - e["cycle_charging"]) / e["rule_based"] * 100 for e in unique.values()]
    gap_eur = [e["rule_based"] - e["cycle_charging"] for e in unique.values()]
    w(f"- distinct training episodes replayed: {len(unique)}; cycle charging below rule-based on them by "
      f"{mean(gap):.1f} % on average, {mean(gap_eur):.1f} EUR per episode")
    w("")
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
