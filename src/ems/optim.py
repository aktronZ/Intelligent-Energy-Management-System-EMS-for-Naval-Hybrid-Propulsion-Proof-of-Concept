"""Dynamic programme over the state of charge — shared by the offline optimum and MPC.

Formulation: specs/072-offline-optimum-formulation.md (sections 3-5.2) and
specs/073-mpc-formulation.md. The genset fuel curve has a no-load term, which
makes the dispatch problem non-convex (a MILP with one binary per step). The
problem has a single continuous state, the SoC, so it is solved by backward
induction on a SoC grid instead:

    V_T(s) = -lambda * E * (s - s_0)                         terminal valuation
    V_t(s) = min over actions a of  g_t(s, a) + V_{t+1}(s')
    g_t    = price * dt * (F0 * P_rated * [x > 0] + F1 * x),   x = d_t - p_battery

d_t is the net demand (load minus renewables). Two kinds of action are
available from every state:

  * GRID actions: move to a grid point. The genset absorbs the difference, so the
    SoC can be steered exactly onto the grid without wasting energy.
  * FOLLOW action: genset off, the battery follows the net demand exactly
    (discharging the deficit, or charging the surplus and curtailing only what the
    battery cannot take). The next SoC is generally off the grid, and V_{t+1} is
    linearly interpolated there.

The FOLLOW action matters. With grid actions alone, a battery covering the load
on its own could only choose a power on the grid (about 0.08 kW apart at a 0.001
grid). Discharging a little too little starts the genset for a sliver and pays
the whole no-load fuel, so the DP discharged a little too much and curtailed the
excess. That wasted about 24 kWh a month and let a simple cycle-charging rule beat
the "optimum" (found 2026-09-27, see the 072 implementation record).

Constraints are enforced on every action: battery power within the 0.5 C limit,
SoC within [0.10, 0.90], genset output within [0, P_rated], a single signed
battery power per step (no simultaneous charge and discharge).

The forward pass starts from the exact SoC and, at every step, evaluates every
action from the exact current state with the exact stage cost and the
(interpolated) V_{t+1}. The trajectory it returns is therefore a real,
feasible schedule, and its cost is exact. Only the V estimate is approximate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ems.components.genset import GENSET_ON_THRESHOLD_KW
from ems.cost import terminal_value_eur_per_kwh
from ems.vessel import REFERENCE

# Slack when turning a power limit into a whole number of grid steps.
_GRID_SLACK = 1e-9


@dataclass(frozen=True)
class SocGrid:
    """A uniform SoC grid and the grid-to-grid transitions allowed on it."""

    step: float
    n_states: int
    socs: np.ndarray
    k: np.ndarray            # SoC change of each transition, in grid steps
    power_kw: np.ndarray     # battery power of each transition, + discharge / - charge

    def soc(self, i: int) -> float:
        return float(self.socs[i])

    def index(self, soc: float) -> int:
        return int(round((soc - REFERENCE.soc_min) / self.step))


def make_grid(step: float) -> SocGrid:
    """Build the grid and its transition set for a given SoC resolution."""
    span = REFERENCE.soc_max - REFERENCE.soc_min
    n_states = int(round(span / step)) + 1
    if abs((n_states - 1) * step - span) > _GRID_SLACK:
        raise ValueError(f"grid step {step} does not divide the SoC window")
    e, eta, dt = REFERENCE.battery_capacity_kwh, REFERENCE.leg_efficiency, REFERENCE.timestep_hours
    pmax = REFERENCE.battery_power_max_kw
    k_charge = math.floor(pmax * eta * dt / e / step + _GRID_SLACK)
    k_discharge = math.floor(pmax * dt / (eta * e) / step + _GRID_SLACK)
    k = np.arange(-k_discharge, k_charge + 1)
    socs = REFERENCE.soc_min + np.arange(n_states) * step
    return SocGrid(step=step, n_states=n_states, socs=socs, k=k, power_kw=_battery_power(k * step))


def _battery_power(dsoc: np.ndarray) -> np.ndarray:
    """Battery power (+ discharge) that changes the SoC by ``dsoc`` in one step."""
    e, eta, dt = REFERENCE.battery_capacity_kwh, REFERENCE.leg_efficiency, REFERENCE.timestep_hours
    dsoc = np.asarray(dsoc, dtype=float)
    return np.where(dsoc > 0.0, -dsoc * e / (eta * dt), -dsoc * e * eta / dt)


def fuel_cost(diesel_kw: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    """EUR of fuel for one step at the given genset output (the ems.cost fuel curve)."""
    diesel = np.asarray(diesel_kw, dtype=float)
    on = diesel > GENSET_ON_THRESHOLD_KW
    litres = (intercept * REFERENCE.genset_rated_kw * on + slope * diesel) * REFERENCE.timestep_hours
    return litres * REFERENCE.fuel_price_eur_per_l


def _coefficients(intercept, slope):
    f0 = REFERENCE.genset_intercept_l_per_h_kw if intercept is None else intercept
    f1 = REFERENCE.genset_slope_l_per_kwh if slope is None else slope
    return f0, f1


def _follow(socs: np.ndarray, d: float, f0: float, f1: float):
    """FOLLOW action from each SoC: genset off unless the battery cannot cover d.

    Returns (battery power, stage cost, next SoC), vectorised over ``socs``.
    """
    e, eta, dt = REFERENCE.battery_capacity_kwh, REFERENCE.leg_efficiency, REFERENCE.timestep_hours
    pmax = REFERENCE.battery_power_max_kw
    if d > 0.0:
        can = np.minimum(pmax, np.maximum(socs - REFERENCE.soc_min, 0.0) * e * eta / dt)
        p = np.minimum(d, can)
        cost = fuel_cost(d - p, f0, f1)
        nxt = socs - p * dt / (eta * e)
        return p, cost, nxt
    room = np.minimum(pmax, np.maximum(REFERENCE.soc_max - socs, 0.0) * e / (eta * dt))
    ch = np.minimum(-d, room)
    nxt = socs + eta * ch * dt / e
    return -ch, np.zeros_like(socs), nxt


def terminal_values(grid: SocGrid, soc_initial: float | None = None) -> np.ndarray:
    """V_T on the grid: credit for energy left in the battery, relative to the initial SoC."""
    s0 = REFERENCE.soc_initial if soc_initial is None else soc_initial
    return -terminal_value_eur_per_kwh() * REFERENCE.battery_capacity_kwh * (grid.socs - s0)


def backward(net_demand_kw, grid: SocGrid, intercept=None, slope=None, terminal=None) -> np.ndarray:
    """Value function on the grid for every step: array of shape (steps + 1, n_states)."""
    f0, f1 = _coefficients(intercept, slope)
    d = np.asarray(net_demand_kw, dtype=float)
    steps, n = len(d), grid.n_states
    cap = REFERENCE.genset_rated_kw

    nxt = np.arange(n)[:, None] + grid.k[None, :]
    blocked = np.where((nxt >= 0) & (nxt < n), 0.0, np.inf)
    nxt = np.clip(nxt, 0, n - 1)

    values = np.empty((steps + 1, n))
    values[steps] = terminal_values(grid) if terminal is None else terminal
    for t in range(steps - 1, -1, -1):
        v_next = values[t + 1]
        x = d[t] - grid.power_kw
        g = np.where(x > cap + GENSET_ON_THRESHOLD_KW, np.inf, fuel_cost(np.clip(x, 0.0, None), f0, f1))
        best_grid = (g[None, :] + v_next[nxt] + blocked).min(axis=1)
        _, cost_f, s_f = _follow(grid.socs, d[t], f0, f1)
        follow_ok = (d[t] - _follow(grid.socs, d[t], f0, f1)[0]) <= cap + GENSET_ON_THRESHOLD_KW
        best_follow = np.where(follow_ok, cost_f + np.interp(s_f, grid.socs, v_next), np.inf)
        values[t] = np.minimum(best_grid, best_follow)
    return values


def best_action(soc: float, d: float, v_next: np.ndarray, grid: SocGrid, intercept=None, slope=None):
    """Best action from an exact SoC: (battery power kW, stage cost EUR, value estimate)."""
    f0, f1 = _coefficients(intercept, slope)
    cap = REFERENCE.genset_rated_kw
    pmax = REFERENCE.battery_power_max_kw
    p = _battery_power(grid.socs - soc)
    x = d - p
    ok = (np.abs(p) <= pmax + GENSET_ON_THRESHOLD_KW) & (x <= cap + GENSET_ON_THRESHOLD_KW)
    g = fuel_cost(np.clip(x, 0.0, None), f0, f1)
    cand = np.where(ok, g + v_next, np.inf)
    j = int(np.argmin(cand))
    best = (float(p[j]), float(g[j]), float(cand[j]))

    pf, cf, sf = _follow(np.array([soc]), d, f0, f1)
    if d - pf[0] <= cap + GENSET_ON_THRESHOLD_KW:
        vf = float(cf[0] + np.interp(sf[0], grid.socs, v_next))
        if vf < best[2]:
            best = (float(pf[0]), float(cf[0]), vf)
    if not np.isfinite(best[2]):
        raise RuntimeError("no feasible action from this state")
    return best


def solve(net_demand_kw, grid: SocGrid, soc_start: float, intercept=None, slope=None):
    """Offline schedule: backward induction, then the exact forward pass.

    Returns (exact cost of the schedule in EUR including the terminal valuation,
    battery power schedule in kW, SoC after each step, DP value estimate).
    """
    d = np.asarray(net_demand_kw, dtype=float)
    values = backward(d, grid, intercept, slope)
    e, eta, dt = REFERENCE.battery_capacity_kwh, REFERENCE.leg_efficiency, REFERENCE.timestep_hours
    soc = soc_start
    powers = np.empty(len(d))
    socs = np.empty(len(d))
    total = 0.0
    for t in range(len(d)):
        p, g, _ = best_action(soc, d[t], values[t + 1], grid, intercept, slope)
        soc = soc + (eta * max(-p, 0.0) * dt - max(p, 0.0) * dt / eta) / e
        soc = min(max(soc, REFERENCE.soc_min), REFERENCE.soc_max)
        powers[t], socs[t] = p, soc
        total += g
    total += -terminal_value_eur_per_kwh() * e * (soc - REFERENCE.soc_initial)
    return total, powers, socs, float(np.interp(soc_start, grid.socs, values[0]))
