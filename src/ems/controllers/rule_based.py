"""Rule-based baseline controller.

The lower reference of the evaluation sandwich. It is a deterministic threshold
policy on the battery state of charge and the net load, of the kind that dominates
industrial applications because it is simple, robust and computationally light.

Policy:
  * On a power deficit, discharge the battery while the state of charge is above
    the lower bound; otherwise burn diesel.
  * On a renewable surplus, charge the battery while the state of charge is below
    the upper bound; otherwise curtail.

The controller requests; the dispatch in IntegratedSystem clips the request to
what the current state of charge allows and absorbs any remainder in the diesel
generator or in curtailment. It therefore cannot violate the power balance or the
state-of-charge window, which is what gate G4 checks.
"""

from __future__ import annotations

from ems.components.battery import max_charge_kw, max_discharge_kw
from ems.vessel import REFERENCE


class RuleBasedController:
    """A deterministic threshold controller.

    Parameters
    ----------
    soc_high : float
        State of charge above which the battery is considered full enough to
        discharge. Defaults to the vessel's upper bound.
    soc_low : float
        State of charge below which the battery is considered too empty to
        discharge. Defaults to the vessel's lower bound.
    """

    def __init__(self, soc_high: float | None = None, soc_low: float | None = None) -> None:
        self.soc_high = REFERENCE.soc_max if soc_high is None else soc_high
        self.soc_low = REFERENCE.soc_min if soc_low is None else soc_low

    def decide(self, soc, pv_kw, wind_kw, load_kw, timestep, dt_hours):  # noqa: ARG002
        """Return the requested battery and diesel power in kW."""
        net = load_kw - pv_kw - wind_kw

        if net > 0.0:
            # Deficit: discharge if there is charge to spare, else diesel.
            if soc > self.soc_low:
                battery = min(net, max_discharge_kw(soc))
            else:
                battery = 0.0
            diesel = net - max(battery, 0.0)
        else:
            # Surplus: charge if there is room, else curtail.
            if soc < self.soc_high:
                battery = -min(-net, max_charge_kw(soc))
            else:
                battery = 0.0
            diesel = 0.0

        return float(battery), float(diesel)
