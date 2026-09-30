"""Actions and observations of the RL agent — one definition for training and evaluation.

ACTIONS. The agent chooses a genset setpoint, not a raw battery power:

    action 0      FOLLOW: genset off; the battery serves the net demand (and stores
                  surplus). The genset runs only for what the battery cannot cover.
    action i > 0  RUN: genset at GENSET_LEVELS[i] x rated power; the battery takes the
                  difference (charging the excess, or covering a shortfall).

Every action maps to one battery request, so the agent still decides the battery
power, as TASK-074 requires. The genset parameterisation matters because the fuel
curve has a no-load term: with a continuous battery-power action the agent would
almost never land exactly on "genset off" and would pay the full no-load fuel for
a sliver of genset output, the same trap the grid-only DP fell into (072 §10).
Action 0 reproduces the pre-registered rule-based baseline exactly, and actions
{0, rated} are enough to express the cycle-charging rule; the tests check both.

OBSERVATION (float32, all roughly in [-1, 2]):

    0  state of charge, rescaled to [0, 1] over the 0.10-0.90 window
    1  measured net demand now (load - renewables), / genset rating
    2  genset running in the previous step (0 or 1)
    3  sin(2 pi * time of day)
    4  cos(2 pi * time of day)
    5  forecast mean net demand, next FORECAST_WINDOWS[0] steps, / rating
    6  forecast mean net demand, next FORECAST_WINDOWS[1] steps, / rating
    7  forecast mean net demand, next FORECAST_WINDOWS[2] steps, / rating

The forecast is the same seed-free schedule forecast the MPC uses (expected
profiles), so RL and MPC see the same information about the future.
"""

from __future__ import annotations

import math

import numpy as np

from ems.profiles import STEPS_PER_DAY
from ems.vessel import REFERENCE

GENSET_LEVELS = (0.0, 0.5, 0.75, 1.0)   # fractions of rated power; 0 means FOLLOW
FORECAST_WINDOWS = (4, 16, 48)          # 1 h, 4 h and 12 h at 15-minute steps
N_ACTIONS = len(GENSET_LEVELS)
N_FEATURES = 5 + len(FORECAST_WINDOWS)


def battery_request(action: int, net_demand_kw: float) -> float:
    """Battery request (+ discharge / - charge) for an action at the given net demand."""
    if action == 0:
        return float(net_demand_kw)                    # follow: battery covers it all
    setpoint = GENSET_LEVELS[action] * REFERENCE.genset_rated_kw
    return float(net_demand_kw - setpoint)             # battery takes the difference


def forecast_means(forecast_kw: np.ndarray) -> np.ndarray:
    """Forward rolling means of the forecast for every step and window: (steps, windows).

    Near the end of the run the window shrinks to what remains, as for the MPC horizon.
    """
    f = np.asarray(forecast_kw, dtype=float)
    n = len(f)
    csum = np.concatenate([[0.0], np.cumsum(f)])
    out = np.empty((n, len(FORECAST_WINDOWS)))
    start = np.arange(n)
    for j, w in enumerate(FORECAST_WINDOWS):
        # Window of the w steps after the current one: t+1 .. t+w.
        lo = np.minimum(start + 1, n)
        hi = np.minimum(start + 1 + w, n)
        count = np.maximum(hi - lo, 1)
        out[:, j] = np.where(hi > lo, (csum[hi] - csum[lo]) / count, f[start])
    return out


def observation(soc: float, net_demand_kw: float, genset_was_on: bool, timestep: int,
                forecast_row: np.ndarray) -> np.ndarray:
    """The observation vector for one step."""
    rated = REFERENCE.genset_rated_kw
    phase = 2.0 * math.pi * (timestep % STEPS_PER_DAY) / STEPS_PER_DAY
    head = [
        (soc - REFERENCE.soc_min) / REFERENCE.usable_soc_window,
        net_demand_kw / rated,
        1.0 if genset_was_on else 0.0,
        math.sin(phase),
        math.cos(phase),
    ]
    return np.asarray(head + list(np.asarray(forecast_row) / rated), dtype=np.float32)
