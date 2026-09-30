"""Tests for the offline optimum (TASK-072) and MPC (TASK-073)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from ems.controllers.mpc import MPCController
from ems.controllers.offline_optimum import OfflineOptimumController, net_demand
from ems.controllers.rule_based import RuleBasedController
from ems.cost import evaluate, terminal_value_eur_per_kwh
from ems.controllers.cycle_charging import CycleChargingController
from ems.optim import make_grid, solve
from ems.scenarios import expected_scenario_profiles, scenario_profiles
from ems.system import IntegratedSystem
from ems.vessel import REFERENCE as V


def _run(profiles, controller):
    r = IntegratedSystem(*profiles, controller).run()
    return r, evaluate(r)


def _lp_linear_fuel(demand: np.ndarray) -> float:
    """The same problem with no no-load fuel, as an LP (072 section 3-4, F1)."""
    n = len(demand)
    dt, e, eta = V.timestep_hours, V.battery_capacity_kwh, V.leg_efficiency
    price_kwh = V.fuel_price_eur_per_l * V.genset_slope_l_per_kwh
    lam = terminal_value_eur_per_kwh()
    # variables: ch, dis, dg, cu, soc  (5n)
    c = np.zeros(5 * n)
    c[2 * n:3 * n] = price_kwh * dt
    c[5 * n - 1] = -lam * e
    a = lil_matrix((2 * n, 5 * n))
    b = np.zeros(2 * n)
    for t in range(n):
        a[t, n + t], a[t, 2 * n + t], a[t, t], a[t, 3 * n + t] = 1, 1, -1, -1
        b[t] = demand[t]
        r = n + t
        a[r, 4 * n + t] = 1
        a[r, t] = -eta * dt / e
        a[r, n + t] = dt / (eta * e)
        if t:
            a[r, 4 * n + t - 1] = -1
        else:
            b[r] = V.soc_initial
    bounds = ([(0, V.battery_power_max_kw)] * 2 * n + [(0, V.genset_rated_kw)] * n
              + [(0, None)] * n + [(V.soc_min, V.soc_max)] * n)
    res = linprog(c, A_eq=a.tocsr(), b_eq=b, bounds=bounds, method="highs")
    assert res.status == 0
    return float(res.fun + lam * e * V.soc_initial)


class TestOfflineOptimum:
    def test_planned_value_equals_evaluated_cost(self) -> None:
        p = scenario_profiles("standard", 2, seed=0)
        c = OfflineOptimumController(*p)
        r, e = _run(p, c)
        assert abs(c.planned_value_eur - e.total_eur) < 1e-9
        assert np.abs(c.planned_soc - r.soc).max() < 1e-9
        assert r.n_clip == 0 and r.n_genset_cap == 0 and r.energy_balance_ok

    def test_dp_matches_lp_when_there_is_no_idle_fuel(self) -> None:
        for seed in (0, 4):
            p = scenario_profiles("standard", 2, seed=seed)
            demand = net_demand(*p)
            exact_cost, _, _, _ = solve(demand, make_grid(0.001), V.soc_initial, intercept=0.0)
            lp_value = _lp_linear_fuel(demand)
            assert exact_cost >= lp_value - 1e-9                  # the LP is the continuous optimum
            assert exact_cost <= lp_value + 0.001 * abs(lp_value)  # DP within 0.1 %

    def test_offline_beats_both_rule_baselines_in_every_scenario(self) -> None:
        for name in ("standard", "high_load", "cloudy"):
            p = scenario_profiles(name, 3, seed=2)
            _, rb = _run(p, RuleBasedController())
            _, cc = _run(p, CycleChargingController())
            _, opt = _run(p, OfflineOptimumController(*p))
            assert opt.total_eur <= rb.total_eur, name
            assert opt.total_eur <= cc.total_eur, name
            assert opt.unserved_kwh == 0.0, name

    def test_no_needless_curtailment(self) -> None:
        # Regression for the grid-only DP, which wasted battery energy as curtailment.
        p = scenario_profiles("standard", 3, seed=0)
        r, opt = _run(p, OfflineOptimumController(*p))
        _, rb = _run(p, RuleBasedController())
        assert opt.curtail_kwh <= rb.curtail_kwh + 0.5


class TestMPC:
    def test_perfect_forecast_full_horizon_reproduces_offline_optimum(self) -> None:
        p = scenario_profiles("cloudy", 1, seed=1)
        _, opt = _run(p, OfflineOptimumController(*p))
        steps = len(p[0])
        mpc = MPCController(net_demand(*p), horizon=steps, grid_step=0.001)
        _, got = _run(p, mpc)
        assert abs(got.total_eur - opt.total_eur) < 1e-9

    def test_offline_is_a_lower_bound_for_mpc(self) -> None:
        p = scenario_profiles("standard", 2, seed=3)
        e = expected_scenario_profiles("standard", 2)
        _, opt = _run(p, OfflineOptimumController(*p))
        mpc = MPCController.with_forecast("schedule", p, e)
        r, got = _run(p, mpc)
        assert opt.total_eur <= got.total_eur + 1e-9
        assert r.energy_balance_ok and mpc.n_solver == 0

    def test_schedule_forecast_is_seed_independent(self) -> None:
        e = expected_scenario_profiles("high_load", 1)
        a = MPCController.with_forecast("schedule", scenario_profiles("high_load", 1, 0), e)
        b = MPCController.with_forecast("schedule", scenario_profiles("high_load", 1, 9), e)
        assert np.array_equal(a.forecast, b.forecast)
