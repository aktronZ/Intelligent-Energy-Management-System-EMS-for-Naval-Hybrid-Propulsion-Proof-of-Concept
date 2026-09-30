"""Diesel generator model — a documented increment to the Ma'arif et al. plant.

Fuel follows the linear (Willans-line) fuel curve used by HOMER:

    F [L/h] = F0 * P_rated * on + F1 * P,     on = P > GENSET_ON_THRESHOLD_KW

F0 is the no-load fuel per rated kW and F1 the marginal fuel per kWh. Both derive
from the assumed 30 % full-load efficiency and the no-load fraction of HOMER's
worked example, so full-load consumption matches the efficiency exactly. The
no-load term is what penalises running the genset at low load, and it is the
mechanism an energy management system can act on.

Sources: specs/010-simulation-contract.md, "Diesel generator" (decision 2026-09-27).
"""

from __future__ import annotations

import numpy as np

from ems.vessel import REFERENCE

# Diesel output above this counts as the genset running.
GENSET_ON_THRESHOLD_KW = 1e-9


def is_on(power_kw: np.ndarray) -> np.ndarray:
    """True where the genset is running."""
    return np.asarray(power_kw, dtype=float) > GENSET_ON_THRESHOLD_KW


def fuel_litres(power_kw: np.ndarray, dt_hours: float | None = None) -> np.ndarray:
    """Fuel burnt in each timestep, in litres.

    Limiting cases
    --------------
    * zero output burns exactly zero (the genset is off);
    * rated output burns exactly SFC_full * P_rated * dt.
    """
    dt = REFERENCE.timestep_hours if dt_hours is None else dt_hours
    p = np.asarray(power_kw, dtype=float)
    no_load = REFERENCE.genset_intercept_l_per_h_kw * REFERENCE.genset_rated_kw * is_on(p)
    return (no_load + REFERENCE.genset_slope_l_per_kwh * p) * dt


def co2_kg(litres: np.ndarray) -> np.ndarray:
    """CO2 emitted by burning the given fuel, in kg."""
    return np.asarray(litres, dtype=float) * REFERENCE.diesel_density_kg_per_l * REFERENCE.diesel_co2_kg_per_kg
