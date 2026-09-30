"""Reference vessel definition and hybrid electrical system sizing.

The quantitative plant is the 5 GT solar-electric fishing boat of Ma'arif et al.
(2026), peer-reviewed in Results in Engineering. EcoSlim is the motivating precedent
and is never simulated: its renewable share is about 4% of its 220 kW propulsion,
which leaves an EMS almost nothing to optimise, and its published battery data is
internally contradictory. See specs/010-simulation-contract.md section 1.

The diesel generator is a documented increment (decision 2026-09-27): Ma'arif's boat
has none. Its fuel curve is the linear Willans-line form used by HOMER.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Vessel:
    """The reference vessel and its hybrid electrical system.

    Every value here has a provenance row in
    specs/010-simulation-contract.md. Do not add a field without adding a row.
    """

    name: str = "Ma'arif 5 GT solar-electric fishing boat"
    gross_tonnage: float = 5.0
    pv_capacity_kwp: float = 2.18
    wind_rated_kw: float = 1.5
    motor_rated_kw: float = 15.0
    battery_capacity_kwh: float = 19.2
    battery_c_rate_per_h: float = 0.5
    round_trip_efficiency: float = 0.92
    soc_min: float = 0.10
    soc_max: float = 0.90
    soc_initial: float = 0.50
    timestep_minutes: int = 15
    pv_capacity_factor: float = 0.17
    wind_capacity_factor: float = 0.22

    # Diesel generator (documented increment).
    genset_rated_kw: float = 6.0
    genset_full_load_efficiency: float = 0.30
    homer_example_intercept_l_per_h_kw: float = 0.033
    homer_example_slope_l_per_h_kw: float = 0.273

    # Fuel.
    diesel_lhv_mj_per_kg: float = 42.7
    diesel_density_kg_per_l: float = 0.85
    diesel_co2_kg_per_kg: float = 3.206
    fuel_price_eur_per_l: float = 0.85
    mj_per_kwh: float = 3.6

    # ------------------------------------------------------------------ battery
    @property
    def usable_soc_window(self) -> float:
        """Fraction of nominal capacity the controller may use."""
        return self.soc_max - self.soc_min

    @property
    def battery_usable_kwh(self) -> float:
        """Usable energy: the SoC window applied once to the nominal capacity."""
        return self.battery_capacity_kwh * self.usable_soc_window

    @property
    def leg_efficiency(self) -> float:
        """Charge and discharge efficiency, each the square root of the round trip."""
        return math.sqrt(self.round_trip_efficiency)

    @property
    def battery_power_max_kw(self) -> float:
        """Charge and discharge power limit from the C-rate."""
        return self.battery_c_rate_per_h * self.battery_capacity_kwh

    # ------------------------------------------------------------------ genset
    @property
    def genset_sfc_full_l_per_kwh(self) -> float:
        """Specific fuel consumption at full load, from the full-load efficiency."""
        return self.mj_per_kwh / (
            self.genset_full_load_efficiency * self.diesel_lhv_mj_per_kg * self.diesel_density_kg_per_l
        )

    @property
    def genset_no_load_fraction(self) -> float:
        """No-load fuel as a fraction of full-load fuel, from the HOMER worked example."""
        f0 = self.homer_example_intercept_l_per_h_kw
        return f0 / (f0 + self.homer_example_slope_l_per_h_kw)

    @property
    def genset_intercept_l_per_h_kw(self) -> float:
        """F0: no-load fuel per rated kW."""
        return self.genset_no_load_fraction * self.genset_sfc_full_l_per_kwh

    @property
    def genset_slope_l_per_kwh(self) -> float:
        """F1: marginal fuel per kWh of output."""
        return (1.0 - self.genset_no_load_fraction) * self.genset_sfc_full_l_per_kwh

    # ------------------------------------------------------------------ time
    @property
    def timestep_hours(self) -> float:
        return self.timestep_minutes / 60.0


# The single instance used everywhere. Constructing a second one is a defect.
REFERENCE = Vessel()
