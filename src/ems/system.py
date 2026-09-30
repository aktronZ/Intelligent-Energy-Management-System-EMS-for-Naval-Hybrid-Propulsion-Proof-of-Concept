"""Integrated system model.

Wires the component models together and enforces the instantaneous power balance
at every timestep. This is the structure defined in Chapter 4.3, and gate G1
requires it to close to within 1e-6 kWh over every simulation day.

The dispatch is authoritative: a controller only requests a battery power. The
dispatch clips the request to what the state of charge allows, runs the diesel
generator for whatever the renewables and the battery do not cover, reduces a
charge request that would push the genset above its rating, and curtails any
renewable surplus that can be neither used nor stored. Every such intervention is
counted, because gate G6 reports a controller's cost together with how often the
plant had to overrule it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ems.components.battery import max_charge_kw, max_discharge_kw, soc_update
from ems.components.propulsion import propulsion_load
from ems.components.pv import pv_power
from ems.components.wind import wind_power
from ems.vessel import REFERENCE

TOLERANCE_KWH = 1e-6
# A request differing from the applied power by more than this counts as clipped.
REQUEST_TOLERANCE_KW = 1e-6


@dataclass(frozen=True)
class StepOutcome:
    """The plant's response to one battery request."""

    battery_kw: float        # applied, + discharge / - charge
    diesel_kw: float
    curtail_kw: float
    unserved_kw: float
    soc: float               # state of charge after the step
    clipped: bool            # the applied power differs from the request
    genset_capped: bool      # a charge request was reduced to respect the genset rating


def dispatch_step(soc: float, pv_kw: float, wind_kw: float, load_kw: float,
                  requested_kw: float) -> StepOutcome:
    """Apply one battery request to the plant. The single source of plant physics.

    Used by IntegratedSystem for every controller and by the RL environment, so
    the agent trains on exactly the plant it is evaluated on.
    """
    cap = REFERENCE.genset_rated_kw
    requested = float(requested_kw)

    # 1. The battery cannot go beyond what the state of charge allows.
    battery = min(max(requested, -max_charge_kw(soc)), max_discharge_kw(soc))

    # 2. The genset cannot go beyond its rating: give up charge first.
    net = load_kw - pv_kw - wind_kw - battery          # what the genset must supply
    capped = False
    if net > cap and battery < 0.0:
        battery = min(0.0, battery + (net - cap))
        net = load_kw - pv_kw - wind_kw - battery
        capped = True

    diesel = min(max(net, 0.0), cap)
    unserved = max(net - cap, 0.0)                     # must be zero in every scenario
    curtail = max(-net, 0.0)
    charge = -battery if battery < 0.0 else 0.0
    discharge = battery if battery > 0.0 else 0.0
    return StepOutcome(
        battery_kw=battery,
        diesel_kw=diesel,
        curtail_kw=curtail,
        unserved_kw=unserved,
        soc=soc_update(soc, charge, discharge, REFERENCE.timestep_hours),
        clipped=abs(battery - requested) > REQUEST_TOLERANCE_KW,
        genset_capped=capped,
    )


@dataclass
class SystemResult:
    """The outcome of one simulation run."""

    timestep_hours: float
    initial_soc: float
    soc: np.ndarray
    pv_kw: np.ndarray
    wind_kw: np.ndarray
    load_kw: np.ndarray
    battery_kw: np.ndarray
    requested_battery_kw: np.ndarray
    diesel_kw: np.ndarray
    curtail_kw: np.ndarray
    unserved_kw: np.ndarray
    n_clip: int = 0
    n_genset_cap: int = 0
    balance_residual_kwh: float = 0.0
    battery_residual_kwh: float = 0.0

    @property
    def energy_balance_ok(self) -> bool:
        """True when the bus balance and the battery energy both close within tolerance."""
        return (
            abs(self.balance_residual_kwh) < TOLERANCE_KWH
            and abs(self.battery_residual_kwh) < TOLERANCE_KWH
        )

    @property
    def unserved_kwh(self) -> float:
        return float(self.unserved_kw.sum() * self.timestep_hours)


@dataclass
class IntegratedSystem:
    """The hybrid electric vessel as a single dispatchable system.

    The controller is any object with a ``decide`` method returning the requested
    battery power in kW (positive for discharge, negative for charge) and a diesel
    power, which the dispatch ignores: diesel always follows from the balance.
    """

    load_profile_kw: np.ndarray
    pv_factor: np.ndarray
    wind_factor: np.ndarray
    controller: object
    initial_soc: float = REFERENCE.soc_initial

    def __post_init__(self) -> None:
        n = len(self.load_profile_kw)
        if not (len(self.pv_factor) == len(self.wind_factor) == n):
            raise ValueError("load, pv and wind profiles must have equal length")

    def run(self) -> SystemResult:
        """Simulate every timestep and return the trace."""
        n = len(self.load_profile_kw)
        dt = REFERENCE.timestep_hours
        eta = REFERENCE.leg_efficiency

        pv_all = pv_power(self.pv_factor)
        wind_all = wind_power(self.wind_factor)
        load_all = propulsion_load(self.load_profile_kw)

        soc = self.initial_soc
        soc_trace = np.empty(n)
        battery_trace = np.empty(n)
        request_trace = np.empty(n)
        diesel_trace = np.empty(n)
        curtail_trace = np.empty(n)
        unserved_trace = np.zeros(n)
        n_clip = 0
        n_cap = 0

        for t in range(n):
            pv, wind, load = float(pv_all[t]), float(wind_all[t]), float(load_all[t])
            requested, _ = self.controller.decide(
                soc=soc, pv_kw=pv, wind_kw=wind, load_kw=load, timestep=t, dt_hours=dt,
            )
            out = dispatch_step(soc, pv, wind, load, requested)
            soc = out.soc
            n_clip += out.clipped
            n_cap += out.genset_capped
            battery_kw, diesel, curtail, unserved = (
                out.battery_kw, out.diesel_kw, out.curtail_kw, out.unserved_kw,
            )

            soc_trace[t] = soc
            battery_trace[t] = battery_kw
            request_trace[t] = float(requested)
            diesel_trace[t] = diesel
            curtail_trace[t] = curtail
            unserved_trace[t] = unserved

        charge_trace = np.clip(-battery_trace, 0.0, None)
        discharge_trace = np.clip(battery_trace, 0.0, None)

        # Independent checks. Bus: generation + unserved = load + charge + curtailment.
        bus = float(
            np.sum(pv_all + wind_all + discharge_trace + diesel_trace + unserved_trace
                   - load_all - charge_trace - curtail_trace) * dt
        )
        # Battery: stored energy change equals what the efficiencies let through.
        stored = (soc_trace[-1] - self.initial_soc) * REFERENCE.battery_capacity_kwh if n else 0.0
        through = float(np.sum(eta * charge_trace - discharge_trace / eta) * dt)

        return SystemResult(
            timestep_hours=dt,
            initial_soc=self.initial_soc,
            soc=soc_trace,
            pv_kw=np.asarray(pv_all, dtype=float),
            wind_kw=np.asarray(wind_all, dtype=float),
            load_kw=np.asarray(load_all, dtype=float),
            battery_kw=battery_trace,
            requested_battery_kw=request_trace,
            diesel_kw=diesel_trace,
            curtail_kw=curtail_trace,
            unserved_kw=unserved_trace,
            n_clip=n_clip,
            n_genset_cap=n_cap,
            balance_residual_kwh=bus,
            battery_residual_kwh=float(stored - through),
        )
