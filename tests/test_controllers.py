"""Tests for the rule-based baseline controller.

Gate G4 requires the baseline to never violate the power balance or the
state-of-charge window over the full evaluation horizon.
"""

from __future__ import annotations

from ems.components.battery import max_charge_kw, max_discharge_kw
from ems.controllers.rule_based import RuleBasedController
from ems.profiles import load_profile, solar_profile, wind_profile
from ems.system import IntegratedSystem


class TestRuleBasedPolicy:
    def controller(self) -> RuleBasedController:
        return RuleBasedController()

    def test_discharges_on_deficit_when_charged(self) -> None:
        c = self.controller()
        battery, diesel = c.decide(soc=0.8, pv_kw=0.0, wind_kw=0.0, load_kw=3.0,
                                   timestep=0, dt_hours=0.25)
        assert battery > 0.0
        assert diesel >= 0.0

    def test_burns_diesel_on_deficit_when_empty(self) -> None:
        c = self.controller()
        battery, diesel = c.decide(soc=0.1, pv_kw=0.0, wind_kw=0.0, load_kw=3.0,
                                   timestep=0, dt_hours=0.25)
        assert battery == 0.0
        assert diesel > 0.0

    def test_charges_on_surplus_when_not_full(self) -> None:
        c = self.controller()
        battery, diesel = c.decide(soc=0.5, pv_kw=3.0, wind_kw=0.0, load_kw=0.0,
                                   timestep=0, dt_hours=0.25)
        assert battery < 0.0
        assert diesel == 0.0

    def test_curtails_on_surplus_when_full(self) -> None:
        c = self.controller()
        battery, diesel = c.decide(soc=0.9, pv_kw=3.0, wind_kw=0.0, load_kw=0.0,
                                   timestep=0, dt_hours=0.25)
        assert battery == 0.0
        assert diesel == 0.0

    def test_diesel_covers_what_battery_cannot(self) -> None:
        c = self.controller()
        # Empty battery, large deficit: diesel must cover the whole load.
        battery, diesel = c.decide(soc=0.1, pv_kw=0.0, wind_kw=0.0, load_kw=4.0,
                                   timestep=0, dt_hours=0.25)
        assert battery == 0.0
        assert abs(diesel - 4.0) < 1e-9


class TestBaselineFeasibility:
    """Gate G4: the baseline never violates the balance or the SoC window."""

    def test_baseline_is_feasible_over_full_horizon(self) -> None:
        system = IntegratedSystem(
            load_profile(), solar_profile(), wind_profile(), RuleBasedController()
        )
        result = system.run()
        assert result.energy_balance_ok
        assert result.soc.min() >= 0.10 - 1e-9
        assert result.soc.max() <= 0.90 + 1e-9

    def test_baseline_is_deterministic(self) -> None:
        a = IntegratedSystem(
            load_profile(), solar_profile(), wind_profile(), RuleBasedController()
        ).run()
        b = IntegratedSystem(
            load_profile(), solar_profile(), wind_profile(), RuleBasedController()
        ).run()
        import numpy as np

        assert np.array_equal(a.soc, b.soc)
