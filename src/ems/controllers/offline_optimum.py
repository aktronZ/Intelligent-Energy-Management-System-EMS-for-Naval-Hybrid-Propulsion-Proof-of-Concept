"""Offline optimum — the upper reference (ceiling) of the evaluation sandwich.

Perfect foresight of the whole evaluation horizon: the dynamic programme of
ems.optim is solved once over every step, on the 0.001 SoC grid, and the
resulting battery schedule is replayed through IntegratedSystem. The reported
cost is the simulator trace scored by ems.cost, never the DP value. The exact
cost of the planned schedule equals the evaluated cost, which the tests check.
The DP is exact up to its SoC grid and interpolation; the tests bound that error
against an independent LP.

Formulation and problem-class argument: specs/072-offline-optimum-formulation.md.
"""

from __future__ import annotations

import numpy as np

from ems.components.propulsion import propulsion_load
from ems.components.pv import pv_power
from ems.components.wind import wind_power
from ems.optim import make_grid, solve
from ems.vessel import REFERENCE

OFFLINE_GRID_STEP = 0.001


def net_demand(load_kw: np.ndarray, pv_factor: np.ndarray, wind_factor: np.ndarray) -> np.ndarray:
    """Load minus renewables, in kW, through the same component models as the plant."""
    return propulsion_load(load_kw) - pv_power(pv_factor) - wind_power(wind_factor)


class OfflineOptimumController:
    """Replays the perfect-foresight DP schedule."""

    def __init__(self, load_kw, pv_factor, wind_factor, grid_step: float = OFFLINE_GRID_STEP) -> None:
        self.grid = make_grid(grid_step)
        demand = net_demand(np.asarray(load_kw), np.asarray(pv_factor), np.asarray(wind_factor))
        self.planned_value_eur, self.schedule_kw, self.planned_soc, self.dp_estimate_eur = solve(
            demand, self.grid, REFERENCE.soc_initial
        )

    def decide(self, soc, pv_kw, wind_kw, load_kw, timestep, dt_hours):  # noqa: ARG002
        return float(self.schedule_kw[timestep]), 0.0
