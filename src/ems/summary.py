"""Summarise results/comparison.csv into results/summary.csv.

Usage: uv run python -m ems.summary

For every scenario, the relative saving of one controller against another,
averaged over seeds, with the minimum and maximum across seeds:

    saving(A vs B) = (cost_B - cost_A) / cost_B

Every number the Results chapter quotes about S5 must come from this file or
from comparison.csv (gate G9).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

PAIRS = [
    ("mpc", "rule_based"),              # the pre-registered comparison
    ("offline_optimum", "rule_based"),  # the ceiling against the pre-registered baseline
    ("cycle_charging", "rule_based"),   # what a better fixed rule captures
    ("mpc", "cycle_charging"),          # what forecasting and optimisation add to it
    ("offline_optimum", "mpc"),         # how far MPC is from the ceiling
    ("mpc_perfect", "mpc"),             # the value of a perfect forecast
]
# S6: every RL policy (one per training seed) against the others, plus their mean.
RL_VERSUS = ["rule_based", "cycle_charging", "mpc", "offline_optimum"]
METRICS = ["total_eur", "fuel_l", "co2_kg", "genset_on_steps"]


def main() -> int:
    src = RESULTS / "comparison.csv"
    with src.open(encoding="utf-8") as fh:
        header = fh.readline().strip()
        rows = list(csv.DictReader(fh))

    table: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(dict)
    for r in rows:
        table[(r["scenario"], r["seed"])][r["controller"]] = {m: float(r[m]) for m in METRICS}

    # The mean over RL training seeds is a controller of its own: "rl_mean".
    for case in table.values():
        rl = [v for k, v in case.items() if k.startswith("rl_t")]
        if rl:
            case["rl_mean"] = {m: sum(r[m] for r in rl) / len(rl) for m in METRICS}
    rl_names = sorted({k for c in table.values() for k in c if k.startswith("rl_")})
    pairs = PAIRS + [(a, b) for a in rl_names for b in RL_VERSUS]

    scenarios = sorted({s for s, _ in table})
    out_rows = []
    for scenario in scenarios:
        cases = [c for (s, _), c in sorted(table.items()) if s == scenario]
        for a, b in pairs:
            for metric in METRICS:
                vals = [(c[b][metric] - c[a][metric]) / c[b][metric] * 100.0 for c in cases
                        if a in c and b in c and c[b][metric] != 0.0]
                if not vals:
                    continue
                out_rows.append({
                    "scenario": scenario, "controller": a, "versus": b, "metric": metric,
                    "saving_mean_pct": f"{sum(vals) / len(vals):.2f}",
                    "saving_min_pct": f"{min(vals):.2f}",
                    "saving_max_pct": f"{max(vals):.2f}",
                    "seeds": len(vals),
                })

    out = RESULTS / "summary.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        fh.write(f"{header} source=comparison.csv\n")
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {out}")
    for r in out_rows:
        if r["metric"] == "total_eur":
            print(f"  {r['scenario']:<10} {r['controller']:<16} vs {r['versus']:<15} "
                  f"{r['saving_mean_pct']:>6}%  [{r['saving_min_pct']}, {r['saving_max_pct']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
