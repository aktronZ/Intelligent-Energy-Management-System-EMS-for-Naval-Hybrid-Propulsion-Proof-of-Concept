"""Operating cost: the one function every controller is scored by.

Decision 2026-09-27 (specs/010-simulation-contract.md, "Operating cost"):

    cost = fuel cost at the constant price, through the Willans fuel curve
           - lambda * E_nominal * (SoC_end - SoC_0)          terminal valuation
    lambda = fuel price * F1 * eta_d

The terminal term credits energy left in the battery with the diesel it would
displace at the margin, and charges a controller that ends emptier than it
started the same amount. It makes the comparison independent of where each
controller happens to leave the battery (072 formulation note, section 4).

A controller is never scored by its own solver objective, only by this function
applied to the simulator's trace. CO2 is reported alongside, never added.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ems.components.genset import co2_kg, fuel_litres, is_on
from ems.system import SystemResult
from ems.vessel import REFERENCE


def terminal_value_eur_per_kwh() -> float:
    """lambda: value of one kWh stored in the battery, EUR per kWh."""
    return REFERENCE.fuel_price_eur_per_l * REFERENCE.genset_slope_l_per_kwh * REFERENCE.leg_efficiency


@dataclass(frozen=True)
class CostBreakdown:
    total_eur: float
    fuel_eur: float
    terminal_eur: float
    fuel_l: float
    co2_kg: float
    diesel_kwh: float
    genset_on_steps: int
    curtail_kwh: float
    unserved_kwh: float
    soc_end: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def evaluate(result: SystemResult) -> CostBreakdown:
    """Score a simulation trace."""
    dt = result.timestep_hours
    litres = fuel_litres(result.diesel_kw, dt)
    fuel_l = float(litres.sum())
    fuel_eur = fuel_l * REFERENCE.fuel_price_eur_per_l
    soc_end = float(result.soc[-1])
    terminal = -terminal_value_eur_per_kwh() * REFERENCE.battery_capacity_kwh * (soc_end - result.initial_soc)
    return CostBreakdown(
        total_eur=fuel_eur + terminal,
        fuel_eur=fuel_eur,
        terminal_eur=terminal,
        fuel_l=fuel_l,
        co2_kg=float(co2_kg(litres).sum()),
        diesel_kwh=float(result.diesel_kw.sum() * dt),
        genset_on_steps=int(is_on(result.diesel_kw).sum()),
        curtail_kwh=float(result.curtail_kw.sum() * dt),
        unserved_kwh=result.unserved_kwh,
        soc_end=soc_end,
    )
