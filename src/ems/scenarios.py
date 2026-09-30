"""Behavioural and operational scenarios.

Runs the simulator under a range of operating conditions and checks that it
behaves correctly in each. This is the behavioural and operational test of
Chapter 6.3, and it is the stage that confirms the simulator is trustworthy
before any controller is judged against it.

The scenarios stress the parts of the model that could plausibly fail: a
high-load case that empties the battery, a cloudy case that removes the solar
surplus, and a long-horizon case that tests whether the state of charge drifts
out of its window over time.

Every scenario asserts the power balance, battery energy conservation, the
state-of-charge window and zero unserved energy, so a behavioural failure is a
hard error rather than a number to inspect.

The same scenario definitions feed the controller comparison (ems.experiments)
and, in their expected (noise-free) form, the MPC's schedule forecast.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ems.components.genset import is_on
from ems.controllers.rule_based import RuleBasedController
from ems.profiles import (
    STEPS_PER_DAY,
    expected_load_profile,
    expected_solar_profile,
    expected_wind_profile,
    load_profile,
    solar_profile,
    wind_profile,
)
from ems.system import IntegratedSystem

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

HIGH_LOAD_SCALE = 1.5     # longer working day with more transit
CLOUDY_SOLAR_SCALE = 0.2  # heavy cloud cover

SCENARIOS = ["standard", "high_load", "cloudy", "long_horizon"]
DAYS = {"standard": 7, "high_load": 7, "cloudy": 7, "long_horizon": 30}


def _apply(name: str, load: np.ndarray, pv: np.ndarray, wind: np.ndarray):
    if name in ("standard", "long_horizon"):
        pass
    elif name == "high_load":
        load = load * HIGH_LOAD_SCALE
    elif name == "cloudy":
        pv = pv * CLOUDY_SOLAR_SCALE
    else:
        raise SystemExit(f"unknown scenario: {name}")
    return load, pv, wind


def scenario_profiles(name: str, days: int, seed: int):
    """Realised load (kW), solar factor and wind factor for a named scenario."""
    steps = days * STEPS_PER_DAY
    return _apply(name, load_profile(steps, seed=seed), solar_profile(steps, seed=seed),
                  wind_profile(steps, seed=seed))


def expected_scenario_profiles(name: str, days: int):
    """Noise-free expectation of the same scenario. Takes no seed by design."""
    steps = days * STEPS_PER_DAY
    return _apply(name, expected_load_profile(steps), expected_solar_profile(steps),
                  expected_wind_profile(steps))


def _metrics(result) -> dict[str, float]:
    """Reduce a run to the numbers the gates and the thesis care about."""
    dt = result.timestep_hours
    return {
        "balance_residual_kwh": result.balance_residual_kwh,
        "battery_residual_kwh": result.battery_residual_kwh,
        "soc_min": float(result.soc.min()),
        "soc_max": float(result.soc.max()),
        "load_kwh": float(result.load_kw.sum() * dt),
        "pv_kwh": float(result.pv_kw.sum() * dt),
        "wind_kwh": float(result.wind_kw.sum() * dt),
        "diesel_kwh": float(result.diesel_kw.sum() * dt),
        "curtail_kwh": float(result.curtail_kw.sum() * dt),
        "unserved_kwh": result.unserved_kwh,
        "soc_mean": float(result.soc.mean()),
        "genset_on_steps": int(is_on(result.diesel_kw).sum()),
        "genset_mean_kw_when_on": float(result.diesel_kw[is_on(result.diesel_kw)].mean())
        if is_on(result.diesel_kw).any() else 0.0,
    }


def run_scenarios(seed: int = 0) -> Path:
    """Run every scenario and write a summary. Returns the output path."""
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"scenarios_s{seed}.csv"

    rows: list[dict[str, float | str]] = []
    failures: list[str] = []

    for name in SCENARIOS:
        days = DAYS[name]
        load, pv, wind = scenario_profiles(name, days, seed)
        result = IntegratedSystem(load, pv, wind, RuleBasedController()).run()
        m = _metrics(result)

        if not result.energy_balance_ok:
            failures.append(
                f"{name}: bus residual {m['balance_residual_kwh']:.3e}, "
                f"battery residual {m['battery_residual_kwh']:.3e}"
            )
        if m["soc_min"] < 0.10 - 1e-9:
            failures.append(f"{name}: SoC minimum {m['soc_min']:.4f} below 0.10")
        if m["soc_max"] > 0.90 + 1e-9:
            failures.append(f"{name}: SoC maximum {m['soc_max']:.4f} above 0.90")
        if m["unserved_kwh"] > 0.0:
            failures.append(f"{name}: unserved energy {m['unserved_kwh']:.3f} kWh")

        rows.append({"scenario": name, "days": days, **m})

    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {out}")
    for row in rows:
        print(
            f"  {row['scenario']:<14} days={row['days']:<3} "
            f"load={row['load_kwh']:.1f} pv={row['pv_kwh']:.1f} wind={row['wind_kwh']:.1f} "
            f"diesel={row['diesel_kwh']:.1f} soc=[{row['soc_min']:.2f},{row['soc_max']:.2f}]"
        )

    if failures:
        raise SystemExit("FAILED behavioural assertions:\n  " + "\n  ".join(failures))

    print("\nall scenarios satisfy the power balance and the state-of-charge window")
    return out


if __name__ == "__main__":
    run_scenarios()
    raise SystemExit(0)
