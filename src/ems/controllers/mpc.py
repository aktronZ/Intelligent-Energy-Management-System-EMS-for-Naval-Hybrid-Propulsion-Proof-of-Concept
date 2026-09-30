"""Model predictive control — the online EMS strategy.

Every 15 minutes: take the measured SoC, solve the ems.optim dynamic programme
over a 24 h receding horizon, apply only the first battery decision, repeat.
The current step uses the measured load and renewables (the simulator passes
them to ``decide``); later steps use a forecast:

  * "schedule" (main case): the noise-free expectation of the scenario's
    profiles (ems.scenarios.expected_scenario_profiles). It takes no seed, so it
    cannot leak the realised noise.
  * "perfect": the realised future. Diagnostic only; it isolates the effect of
    the finite horizon.

The horizon shrinks near the end of the evaluation period, so the MPC never plans
past it. The terminal valuation at the end of each horizon is the same lambda the
offline optimum and ems.cost use.

The decision at each step is evaluated from the exact measured SoC (ems.optim,
best_action). The value function comes from a 0.002 SoC grid, coarser than the
offline optimum's 0.001, to keep 2,880 solves per month affordable.

Gate G6 counts: n_solver (no feasible action, fallback to the rule-based
decision), plus n_clip and n_genset_cap from the simulator.
Formulation: specs/073-mpc-formulation.md.
"""

from __future__ import annotations

import numpy as np

from ems.controllers.offline_optimum import net_demand
from ems.controllers.rule_based import RuleBasedController
from ems.optim import backward, best_action, make_grid
from ems.profiles import STEPS_PER_DAY

MPC_GRID_STEP = 0.002
MPC_HORIZON_STEPS = STEPS_PER_DAY


class MPCController:
    """Receding-horizon DP controller.

    Parameters
    ----------
    forecast_demand_kw : array
        Forecast net demand (load minus renewables, kW) for every step of the run.
        For step t the controller uses the measured value, and the forecast for
        t+1 ... t+N-1.
    horizon : int
        Horizon length in steps.
    """

    def __init__(self, forecast_demand_kw, horizon: int = MPC_HORIZON_STEPS,
                 grid_step: float = MPC_GRID_STEP) -> None:
        self.forecast = np.asarray(forecast_demand_kw, dtype=float)
        self.horizon = int(horizon)
        self.grid = make_grid(grid_step)
        self.fallback = RuleBasedController()
        self.n_solver = 0

    @classmethod
    def with_forecast(cls, kind, realised, expected, horizon: int = MPC_HORIZON_STEPS):
        """Build from (load, pv factor, wind factor) tuples for the realised and expected profiles."""
        source = realised if kind == "perfect" else expected
        if kind not in ("perfect", "schedule"):
            raise ValueError(f"unknown forecast: {kind}")
        return cls(net_demand(*source), horizon=horizon)

    def decide(self, soc, pv_kw, wind_kw, load_kw, timestep, dt_hours):
        end = min(timestep + self.horizon, len(self.forecast))
        demand = self.forecast[timestep:end].copy()
        demand[0] = load_kw - pv_kw - wind_kw          # measured, not forecast

        try:
            values = backward(demand, self.grid)
            power, _, _ = best_action(soc, demand[0], values[1], self.grid)
        except RuntimeError:
            self.n_solver += 1
            return self.fallback.decide(soc, pv_kw, wind_kw, load_kw, timestep, dt_hours)
        return power, 0.0
