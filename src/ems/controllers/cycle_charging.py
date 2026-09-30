"""Cycle-charging rule — a SENSITIVITY baseline, not the pre-registered one.

The pre-registered baseline (TASK-071, ems.controllers.rule_based) is a
load-following rule: the genset supplies exactly the deficit the battery cannot,
so on this plant it runs for long periods at low load and pays the no-load fuel
each time. The pre-registered falsifier says an MPC gain above 10 % points to a
defect "in the model or the baseline" and must be investigated before reporting.
This controller is part of that investigation.

Cycle charging is the other standard industrial dispatch rule for diesel-battery
hybrids: once the genset has to start, it runs at rated power and the excess
charges the battery, until the state of charge reaches a setpoint; then the
battery serves the load again. It is still a fixed rule with no forecast, so the
gap between it and MPC is the part of the MPC gain that a better rule cannot
capture.

The setpoint is an assumption recorded in specs/010-simulation-contract.md.
"""

from __future__ import annotations

from ems.components.battery import max_charge_kw, max_discharge_kw
from ems.vessel import REFERENCE

CYCLE_CHARGING_SETPOINT_SOC = 0.80


class CycleChargingController:
    """Genset at rated power until the battery reaches the setpoint."""

    def __init__(self, setpoint_soc: float = CYCLE_CHARGING_SETPOINT_SOC) -> None:
        self.setpoint = setpoint_soc
        self.genset_running = False

    def decide(self, soc, pv_kw, wind_kw, load_kw, timestep, dt_hours):  # noqa: ARG002
        net = load_kw - pv_kw - wind_kw
        rated = REFERENCE.genset_rated_kw

        if net <= 0.0:
            # Renewable surplus: the genset stops and the battery takes what it can.
            self.genset_running = False
            return -min(-net, max_charge_kw(soc)), 0.0

        if self.genset_running and soc >= self.setpoint:
            self.genset_running = False
        if not self.genset_running and max_discharge_kw(soc) >= net:
            return net, 0.0                           # the battery covers the deficit alone

        # The genset must run: at rated power, the excess charging the battery.
        self.genset_running = True
        return -min(rated - net, max_charge_kw(soc)), 0.0
