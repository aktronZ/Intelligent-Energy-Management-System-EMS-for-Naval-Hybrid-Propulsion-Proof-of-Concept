"""Battery storage model.

Tracks the state of charge (a fraction of NOMINAL capacity) under charge and
discharge power. The round-trip efficiency is split evenly between the two legs,
eta_c = eta_d = sqrt(eta_rt), and the state of charge is confined to the
0.10-0.90 window, which leaves 15.36 kWh usable out of 19.2 kWh nominal.

Energy is conserved exactly: the power limits below are the powers that land the
state of charge on the window bound, so soc_update never has to clip. A request
that would leave the window is a defect in the caller and raises.

Sources: nominal capacity 19.2 kWh from Ma'arif et al. (2026); round-trip
efficiency 92 %, the SoC window and the 0.5 C power limit are assumptions recorded
in specs/010-simulation-contract.md (corrected 2026-09-27, TASK-065 D2-D5).
"""

from __future__ import annotations

from ems.vessel import REFERENCE

# Floating-point slack when comparing a state of charge with its bounds.
SOC_TOLERANCE = 1e-9


def soc_update(
    soc: float,
    charge_kw: float,
    discharge_kw: float,
    dt_hours: float | None = None,
    capacity_kwh: float | None = None,
) -> float:
    """Advance the state of charge by one timestep.

    Parameters
    ----------
    soc : float
        State of charge at the start of the timestep, fraction of nominal.
    charge_kw, discharge_kw : float
        Charging and discharging power in kW, non-negative, not both positive.
    dt_hours : float, optional
        Timestep duration in hours. Defaults to the reference vessel.
    capacity_kwh : float, optional
        Nominal capacity in kWh. Defaults to the reference vessel.

    Raises
    ------
    ValueError
        If the powers are negative, both positive, or would leave the SoC window.
    """
    dt = REFERENCE.timestep_hours if dt_hours is None else dt_hours
    nominal = REFERENCE.battery_capacity_kwh if capacity_kwh is None else capacity_kwh
    eta = REFERENCE.leg_efficiency

    if charge_kw < 0.0 or discharge_kw < 0.0:
        raise ValueError("charge_kw and discharge_kw must be non-negative")
    if charge_kw > 0.0 and discharge_kw > 0.0:
        raise ValueError("battery cannot charge and discharge in the same timestep")

    new_soc = soc + (eta * charge_kw * dt - discharge_kw * dt / eta) / nominal
    if new_soc < REFERENCE.soc_min - SOC_TOLERANCE or new_soc > REFERENCE.soc_max + SOC_TOLERANCE:
        raise ValueError(
            f"state of charge {new_soc:.12f} would leave the window "
            f"[{REFERENCE.soc_min}, {REFERENCE.soc_max}]"
        )
    # Only floating-point noise remains; pinning it to the bound creates no energy.
    return min(max(new_soc, REFERENCE.soc_min), REFERENCE.soc_max)


def max_charge_kw(soc: float) -> float:
    """Largest charging power that keeps the state of charge within the window."""
    room = max(0.0, REFERENCE.soc_max - soc)
    to_bound = room * REFERENCE.battery_capacity_kwh / (REFERENCE.leg_efficiency * REFERENCE.timestep_hours)
    return min(REFERENCE.battery_power_max_kw, to_bound)


def max_discharge_kw(soc: float) -> float:
    """Largest discharging power that keeps the state of charge within the window."""
    available = max(0.0, soc - REFERENCE.soc_min)
    to_bound = available * REFERENCE.battery_capacity_kwh * REFERENCE.leg_efficiency / REFERENCE.timestep_hours
    return min(REFERENCE.battery_power_max_kw, to_bound)
