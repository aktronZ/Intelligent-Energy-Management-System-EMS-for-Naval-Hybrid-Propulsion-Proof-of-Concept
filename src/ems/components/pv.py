"""Photovoltaic generation model.

Converts a plane-of-array irradiance factor and the installed capacity into a
generation power. Assumes maximum power point tracking, which is standard on
marine photovoltaic installations.

Source: capacity 2.18 kWp and capacity factor 0.17 from Ma'arif et al. (2026).
"""

from __future__ import annotations

import numpy as np

from ems.vessel import REFERENCE


def pv_power(irradiance_factor: np.ndarray, capacity_kwp: float | None = None) -> np.ndarray:
    """Photovoltaic generation power in kW.

    Parameters
    ----------
    irradiance_factor : array of 0 to 1
        Plane-of-array irradiance as a fraction of the rated irradiance.
    capacity_kwp : float, optional
        Installed capacity in kWp. Defaults to the reference vessel.

    Returns
    -------
    array of generation power in kW, clipped to the installed capacity.

    Limiting cases
    --------------
    * zero irradiance gives exactly zero generation;
    * an irradiance factor of 1 gives exactly the installed capacity.
    """
    capacity = REFERENCE.pv_capacity_kwp if capacity_kwp is None else capacity_kwp
    power = np.asarray(irradiance_factor, dtype=float) * capacity
    return np.clip(power, 0.0, capacity)
