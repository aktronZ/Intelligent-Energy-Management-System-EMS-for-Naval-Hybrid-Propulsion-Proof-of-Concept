"""Command-line entry point for the simulator.

Runs the integrated system for a configurable number of days and profile, with a
selectable controller, and writes a trace to results/ that the acceptance gates
can check.

Usage:
    uv run python -m ems.run --profile standard --days 7
    uv run python -m ems.run --profile standard --days 30 --seed 42 --controller rule_based
    uv run python -m ems.run --profile scenarios
    uv run python -m ems.run --profile cloudy --days 30 --controller mpc

The development follows the spiral methodology of the abstract. Round 1 is the
minimum viable simulator with a trivial controller. Round 2 adds the rule-based
baseline. Round 3 adds the optimisation-based controllers. Each round keeps the
system fully operational.

The output CSV records its seed and git commit in the header so that every number
can be traced back to a specific run. Two runs with the same seed and controller
are byte-identical, which is gate G3.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from ems.controllers.cycle_charging import CycleChargingController
from ems.controllers.mpc import MPCController
from ems.controllers.offline_optimum import OfflineOptimumController
from ems.controllers.rule_based import RuleBasedController
from ems.rl.seeds import TRAIN_SEEDS
from ems.cost import evaluate
from ems.scenarios import SCENARIOS, expected_scenario_profiles, scenario_profiles
from ems.system import IntegratedSystem

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


class TrivialController:
    """Discharges the battery to serve load and charges from surplus.

    Not a sensible energy management strategy. It exists so that Round 1 has a
    controller to exercise end to end; the rule-based baseline of Round 2
    replaces it. The dispatch in IntegratedSystem is authoritative and clips
    whatever this requests to what is physically feasible.
    """

    def decide(self, soc, pv_kw, wind_kw, load_kw, timestep, dt_hours):  # noqa: ARG002
        net = load_kw - pv_kw - wind_kw
        battery = max(-5.0, min(5.0, -net))
        return battery, 0.0


# The controller registry. Each entry builds a controller from the realised and
# expected (noise-free) profiles of the run; controllers that need no foresight
# ignore them. Adding a controller here makes it available to the command line
# and to ems.experiments without changing the simulator.
CONTROLLERS = {
    "trivial": lambda real, exp: TrivialController(),
    "rule_based": lambda real, exp: RuleBasedController(),
    "cycle_charging": lambda real, exp: CycleChargingController(),
    "offline_optimum": lambda real, exp: OfflineOptimumController(*real),
    "mpc": lambda real, exp: MPCController.with_forecast("schedule", real, exp),
    "mpc_perfect": lambda real, exp: MPCController.with_forecast("perfect", real, exp),
}


def _rl(train_seed: int):
    def build(real, exp):  # noqa: ARG001 - the policy sees only the forecast, never `real`
        from ems.rl.controller import RLController

        return RLController.load(train_seed, exp)

    return build


# One RL controller per training seed (S6); each loads results/models/rl_ppo_t{seed}.zip.
for _seed in TRAIN_SEEDS:
    CONTROLLERS[f"rl_t{_seed}"] = _rl(_seed)


def _git_commit() -> str:
    """Short hash of the current commit, or a placeholder if unavailable."""
    candidates = ["git", r"C:\Program Files\Git\cmd\git.exe", r"C:\Program Files\Git\bin\git.exe"]
    for git in candidates:
        try:
            out = subprocess.run(
                [git, "rev-parse", "--short", "HEAD"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except Exception:
            continue
    return "no-git"


def run(
    profile: str,
    days: int,
    seed: int,
    controller_name: str = "trivial",
) -> Path:
    """Run the simulator and write the trace. Returns the output path.

    ``profile`` is a scenario name from ems.scenarios (``standard`` by default).
    """
    if controller_name not in CONTROLLERS:
        raise SystemExit(
            f"unknown controller: {controller_name}. "
            f"choose one of: {', '.join(CONTROLLERS)}"
        )
    if profile not in SCENARIOS:
        raise SystemExit(f"unknown profile: {profile}. choose one of: {', '.join(SCENARIOS)}")

    real = scenario_profiles(profile, days, seed)
    exp = expected_scenario_profiles(profile, days)
    controller = CONTROLLERS[controller_name](real, exp)
    result = IntegratedSystem(*real, controller).run()
    cost = evaluate(result)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"sim_{profile}_d{days}_s{seed}_{controller_name}.csv"

    dt = result.timestep_hours
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([f"# profile={profile}", f"# days={days}", f"# seed={seed}"])
        writer.writerow(
            [
                f"# controller={controller_name}",
                f"# commit={_git_commit()}",
                f"# balance_residual_kwh={result.balance_residual_kwh:.3e}",
                f"# battery_residual_kwh={result.battery_residual_kwh:.3e}",
            ]
        )
        writer.writerow(
            [
                f"# total_eur={cost.total_eur:.6f}",
                f"# fuel_l={cost.fuel_l:.6f}",
                f"# n_clip={result.n_clip}",
                f"# n_genset_cap={result.n_genset_cap}",
                f"# unserved_kwh={cost.unserved_kwh:.6f}",
            ]
        )
        writer.writerow(
            ["hour", "load_kw", "pv_kw", "wind_kw", "battery_kw", "diesel_kw", "curtail_kw", "soc"]
        )
        for t in range(len(result.soc)):
            writer.writerow(
                [
                    f"{t * dt:.2f}",
                    f"{result.load_kw[t]:.6f}",
                    f"{result.pv_kw[t]:.6f}",
                    f"{result.wind_kw[t]:.6f}",
                    f"{result.battery_kw[t]:.6f}",
                    f"{result.diesel_kw[t]:.6f}",
                    f"{result.curtail_kw[t]:.6f}",
                    f"{result.soc[t]:.6f}",
                ]
            )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Minimum viable EMS simulator")
    parser.add_argument("--profile", default="standard", help="scenario name, or 'scenarios'")
    parser.add_argument("--days", type=int, default=7, help="number of days to simulate")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--controller",
        default="trivial",
        choices=sorted(CONTROLLERS),
        help="controller to run",
    )
    args = parser.parse_args(argv)

    if args.profile == "scenarios":
        from ems.scenarios import run_scenarios

        out = run_scenarios(seed=args.seed)
        print(f"wrote {out}")
        return 0

    out = run(args.profile, args.days, args.seed, args.controller)

    print(f"wrote {out}")
    print(f"  profile={args.profile} days={args.days} seed={args.seed} controller={args.controller}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
