"""Wind generation model.

Converts a wind power factor and the rated turbine power into a generation power.
The factor already accounts for the turbine power curve, so the model is a
scaled production term.

Source: rated power 1.5 kW and capacity factor 0.22, both assumptions recorded in
specs/010-simulation-contract.md, following Nomak and Çiçek (2025) for the turbine
model structure.
"""

from __future__ import annotations

import numpy as np

from ems.vessel import REFERENCE


def wind_power(wind_factor: np.ndarray, rated_kw: float | None = None) -> np.ndarray:
    """Wind generation power in kW.

    Parameters
    ----------
    wind_factor : array of 0 to 1
        Production factor accounting for the turbine power curve and wake.
    rated_kw : float, optional
        Rated turbine power in kW. Defaults to the reference vessel.

    Returns
    -------
    array of generation power in kW, clipped to the rated power.

    Limiting cases
    --------------
    * a factor of 0 gives exactly zero generation;
    * a factor of 1 gives exactly the rated power.
    """
    rated = REFERENCE.wind_rated_kw if rated_kw is None else rated_kw
    power = np.asarray(wind_factor, dtype=float) * rated
    return np.clip(power, 0.0, rated)
