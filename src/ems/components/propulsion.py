"""Propulsion load model.

Maps the duty-cycle load profile to a propulsion power demand. The profile is
reconstructed from published measurements: Ma'arif et al. (2026) report an average
power draw of 2-3 kW at 2.5-3.5 knots during fishing operations, rising above
3.5 kW at speeds over 4 knots.

Source: the 15 kW motor rating is an assumption recorded in
specs/010-simulation-contract.md.
"""

from __future__ import annotations

import numpy as np

from ems.vessel import REFERENCE


def propulsion_load(load_profile_kw: np.ndarray, motor_kw: float | None = None) -> np.ndarray:
    """Propulsion power demand in kW at each timestep.

    Parameters
    ----------
    load_profile_kw : array
        Base load demand in kW from the duty cycle.
    motor_kw : float, optional
        Rated motor power in kW. Defaults to the reference vessel.

    Returns
    -------
    array of propulsion power in kW, clipped to the rated motor power.

    Limiting cases
    --------------
    * a zero profile gives exactly zero propulsion power;
    * a profile at the rated power is not exceeded.
    """
    rated = REFERENCE.motor_rated_kw if motor_kw is None else motor_kw
    power = np.asarray(load_profile_kw, dtype=float)
    return np.clip(power, 0.0, rated)
