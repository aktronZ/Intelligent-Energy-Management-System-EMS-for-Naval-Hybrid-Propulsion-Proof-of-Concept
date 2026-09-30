"""Experiment matrix: every controller, every scenario, every seed.

Usage:
    uv run python -m ems.experiments                 # full matrix, 30 days, seeds 0-4
    uv run python -m ems.experiments --seeds 0 1 --days 7 --workers 2
    uv run python -m ems.experiments --no-rl         # S5 controllers only (no trained models)

Writes results/comparison.csv: one row per (scenario, seed, controller), with the
cost breakdown from ems.cost and the plant-intervention counts gate G6 needs. The
header records the commit, so every number traces to a run (gate G9). The run is
deterministic: two runs of the same commit write byte-identical files (gate G3).

Controllers:
    rule_based       pre-registered baseline (TASK-071), load-following
    cycle_charging   sensitivity baseline for the falsifier investigation
    offline_optimum  perfect-foresight ceiling (TASK-072)
    mpc              receding horizon with the schedule forecast (TASK-073, main case)
    mpc_perfect      receding horizon with a perfect forecast (diagnostic)
    rl_t0, rl_t1...  PPO policies, one per training seed (TASK-074); they need
                     results/models/rl_ppo_t{seed}.zip from `python -m ems.rl.train`
The evaluation seeds are ems.rl.seeds.EVALUATION_SEEDS, never used in training.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ems.cost import evaluate
from ems.rl.seeds import EVALUATION_SEEDS, rl_controller_names
from ems.run import CONTROLLERS, _git_commit
from ems.scenarios import expected_scenario_profiles, scenario_profiles
from ems.system import IntegratedSystem

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

S5_SCENARIOS = ["standard", "high_load", "cloudy"]
S5_CONTROLLERS = ["rule_based", "cycle_charging", "offline_optimum", "mpc", "mpc_perfect"]
EVALUATION_DAYS = 30
SEEDS = list(EVALUATION_SEEDS)

FIELDS = [
    "scenario", "seed", "controller", "total_eur", "fuel_eur", "terminal_eur", "fuel_l",
    "co2_kg", "diesel_kwh", "genset_on_steps", "curtail_kwh", "unserved_kwh", "soc_end",
    "soc_min", "soc_max", "balance_residual_kwh", "battery_residual_kwh",
    "n_clip", "n_genset_cap", "n_solver",
]


def run_case(case: tuple[str, int, str, int]) -> dict[str, object]:
    """Simulate one (scenario, seed, controller) and score it."""
    scenario, seed, name, days = case
    t0 = time.time()
    real = scenario_profiles(scenario, days, seed)
    exp = expected_scenario_profiles(scenario, days)
    controller = CONTROLLERS[name](real, exp)
    result = IntegratedSystem(*real, controller).run()
    cost = evaluate(result)
    row: dict[str, object] = {"scenario": scenario, "seed": seed, "controller": name}
    row.update(cost.as_dict())
    row.update(
        soc_min=float(result.soc.min()),
        soc_max=float(result.soc.max()),
        balance_residual_kwh=result.balance_residual_kwh,
        battery_residual_kwh=result.battery_residual_kwh,
        n_clip=result.n_clip,
        n_genset_cap=result.n_genset_cap,
        n_solver=getattr(controller, "n_solver", 0),
    )
    return row


def _fmt(value: object) -> str:
    return f"{value:.9f}" if isinstance(value, float) else str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S5 controller comparison")
    parser.add_argument("--days", type=int, default=EVALUATION_DAYS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--scenarios", nargs="+", default=S5_SCENARIOS)
    parser.add_argument("--controllers", nargs="+", default=None,
                        help="default: the S5 controllers plus every RL policy")
    parser.add_argument("--no-rl", action="store_true", help="leave out the RL policies")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", default="comparison.csv")
    args = parser.parse_args(argv)
    if args.controllers is None:
        args.controllers = S5_CONTROLLERS + ([] if args.no_rl else rl_controller_names())

    cases = [(s, seed, c, args.days) for s in args.scenarios for seed in args.seeds
             for c in args.controllers]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_case, cases))

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / args.out
    with out.open("w", newline="", encoding="utf-8") as fh:
        fh.write(f"# commit={_git_commit()} days={args.days} seeds={' '.join(map(str, args.seeds))}\n")
        writer = csv.DictWriter(fh, fieldnames=FIELDS[:-1], extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row[k]) for k in FIELDS[:-1]})
    print(f"wrote {out}")
    for row in rows:
        print(f"  {row['scenario']:<10} s{row['seed']} {row['controller']:<16} "
              f"{row['total_eur']:9.3f} EUR  on={row['genset_on_steps']:5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
