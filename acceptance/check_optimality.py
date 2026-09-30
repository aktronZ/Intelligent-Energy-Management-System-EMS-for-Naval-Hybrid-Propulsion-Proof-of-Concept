"""Optimality-order and Pareto-honesty gates (G5, G6), plus G1/G2 for every controller.

Risk controlled: a controller comparison that is physically invalid, reported
without its violation counts, or ordered in a way that exposes a bug.

Reads results/comparison.csv (written by `uv run python -m ems.experiments`).

    G1   every run closes the bus balance and the battery energy to < 1e-6 kWh
    G2   every run keeps 0.10 <= SoC <= 0.90, and serves all the load
    G5   for every scenario and seed: cost(offline) <= cost(MPC) <= cost(rule-based)
         * left: the offline optimum is exact up to its SoC grid (checked against an
           LP in the tests); it must also undercut every other controller in the
           comparison, including the sensitivity baseline. A failure means the DP is
           not finding the optimum;
         * right: an empirical result; a failure is reported as a finding under the
           pre-registered falsifier.
    G6   every MPC row carries its violation counts (n_clip, n_genset_cap,
         n_solver), and they are printed next to the cost.

    HELD-OUT  (S6, TASK-074) no evaluation seed is in the RL training pool.

The cycle-charging sensitivity baseline and the RL policies are printed for
review. RL is never gated on winning: "the number is reported even if RL loses
the baseline" (TASK-074). It is gated on G1/G2 like every controller, and the
offline optimum must undercut it (G5, left).

Exit status: 0 all gates pass; 1 any gate fails or the comparison is missing.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPARISON = ROOT / "results" / "comparison.csv"

TOL_KWH = 1e-6
TOL_EUR = 1e-9
G6_COUNTS = ("n_clip", "n_genset_cap", "n_solver")


def _read() -> list[dict[str, str]]:
    with COMPARISON.open(encoding="utf-8") as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    return list(csv.DictReader(lines))


def check() -> int:
    if not COMPARISON.exists():
        print("FAIL  results/comparison.csv missing: run `uv run python -m ems.experiments`")
        return 1
    rows = _read()
    failures: list[str] = []

    for r in rows:
        tag = f"{r['scenario']}/s{r['seed']}/{r['controller']}"
        if abs(float(r["balance_residual_kwh"])) >= TOL_KWH or abs(float(r["battery_residual_kwh"])) >= TOL_KWH:
            failures.append(f"G1 {tag}: residuals {r['balance_residual_kwh']}, {r['battery_residual_kwh']}")
        if float(r["soc_min"]) < 0.10 - 1e-9 or float(r["soc_max"]) > 0.90 + 1e-9:
            failures.append(f"G2 {tag}: SoC [{r['soc_min']}, {r['soc_max']}]")
        if float(r["unserved_kwh"]) > 0.0:
            failures.append(f"G2 {tag}: unserved {r['unserved_kwh']} kWh")

    by_case: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for r in rows:
        by_case[(r["scenario"], r["seed"])][r["controller"]] = r

    print(f"{'scenario':<10} {'seed':>4} {'rule_based':>11} {'cycle_chg':>10} {'mpc':>10} "
          f"{'offline':>10} {'MPC gain':>9}  G6 counts (clip, cap, solver)")
    for (scenario, seed), c in sorted(by_case.items()):
        need = {"rule_based", "mpc", "offline_optimum"}
        if not need <= c.keys():
            failures.append(f"G5 {scenario}/s{seed}: missing {sorted(need - c.keys())}")
            continue
        rb, mpc, off = (float(c[k]["total_eur"]) for k in ("rule_based", "mpc", "offline_optimum"))
        for other, row in c.items():
            if other != "offline_optimum" and off > float(row["total_eur"]) + TOL_EUR:
                failures.append(f"G5 {scenario}/s{seed}: offline {off:.6f} > {other} "
                                f"{float(row['total_eur']):.6f} (the optimum is not optimal)")
        if mpc > rb + TOL_EUR:
            failures.append(f"G5 {scenario}/s{seed}: MPC {mpc:.6f} > rule-based {rb:.6f} (finding)")
        missing = [k for k in G6_COUNTS if c["mpc"].get(k, "") == ""]
        if missing:
            failures.append(f"G6 {scenario}/s{seed}: MPC row lacks {missing}")
        cc = c.get("cycle_charging", {}).get("total_eur", "nan")
        counts = ", ".join(c["mpc"].get(k, "?") for k in G6_COUNTS)
        print(f"{scenario:<10} {seed:>4} {rb:11.3f} {float(cc):10.3f} {mpc:10.3f} {off:10.3f} "
              f"{100 * (rb - mpc) / rb:8.2f}%  ({counts})")

    # Held-out: the evaluation seeds must never be RL training seeds.
    from ems.rl.seeds import training_seeds

    pool = set(training_seeds())
    leaked = sorted({int(r["seed"]) for r in rows} & pool)
    if leaked:
        failures.append(f"HELD-OUT: evaluation seeds {leaked} are in the RL training pool")

    rl_names = sorted({r["controller"] for r in rows if r["controller"].startswith("rl_")})
    if not rl_names:
        print("\nREVIEW  no RL rows yet: S6 is not evaluated (train, then re-run experiments)")
    else:
        print(f"\nRL (held-out seeds; reported even where it loses)")
        print(f"{'scenario':<10} {'seed':>4} {'policy':<7} {'cost':>10} {'vs rule':>8} {'vs cycle':>9} "
              f"{'vs MPC':>8}  counts (clip, cap)")
        for (scenario, seed), c in sorted(by_case.items()):
            for name in rl_names:
                if name not in c:
                    failures.append(f"RL {scenario}/s{seed}: missing {name}")
                    continue
                x = float(c[name]["total_eur"])
                rel = lambda k: 100 * (float(c[k]["total_eur"]) - x) / float(c[k]["total_eur"])  # noqa: E731
                print(f"{scenario:<10} {seed:>4} {name:<7} {x:10.3f} {rel('rule_based'):7.2f}% "
                      f"{rel('cycle_charging'):8.2f}% {rel('mpc'):7.2f}%  "
                      f"({c[name]['n_clip']}, {c[name]['n_genset_cap']})")

    if failures:
        print("\nFAIL")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"\nOK  {len(rows)} runs: G1, G2, G5, G6 and held-out hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(check())
