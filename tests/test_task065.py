"""Regression tests for the simulator defects fixed in TASK-065 (2026-09-27)."""

from __future__ import annotations

import numpy as np

from ems.components.battery import max_discharge_kw, soc_update
from ems.components.genset import co2_kg, fuel_litres
from ems.controllers.rule_based import RuleBasedController
from ems.cost import evaluate
from ems.profiles import (
    STEPS_PER_DAY,
    expected_load_profile,
    expected_solar_profile,
    expected_wind_profile,
    load_profile,
    solar_profile,
)
from ems.scenarios import expected_scenario_profiles, scenario_profiles
from ems.system import IntegratedSystem
from ems.vessel import REFERENCE as V

DT = V.timestep_hours


class TestD1DailyProfiles:
    def test_solar_is_zero_every_night(self) -> None:
        s = solar_profile(7 * STEPS_PER_DAY, seed=3)
        hour = (np.arange(len(s)) % STEPS_PER_DAY) * DT
        assert np.all(s[(hour < 6.0) | (hour >= 18.0)] == 0.0)

    def test_every_day_has_the_same_load_energy_within_noise(self) -> None:
        one_day = load_profile(STEPS_PER_DAY, seed=0).sum() * DT
        week = load_profile(7 * STEPS_PER_DAY, seed=0).reshape(7, STEPS_PER_DAY).sum(axis=1) * DT
        assert np.all(np.abs(week - one_day) <= 0.10 * one_day)

    def test_one_day_run_is_unchanged_by_the_fix(self) -> None:
        # The first day of a long run equals a one-day run with the same seed only in
        # its deterministic part; the transit steps must coincide exactly.
        a = load_profile(STEPS_PER_DAY, seed=0)
        b = load_profile(30 * STEPS_PER_DAY, seed=0)[:STEPS_PER_DAY]
        assert np.array_equal(a == 3.5, b == 3.5)


class TestD2EnergyConservation:
    def test_rule_based_conserves_battery_energy_over_30_days(self) -> None:
        for name in ("standard", "high_load", "cloudy"):
            load, pv, wind = scenario_profiles(name, 30, seed=1)
            r = IntegratedSystem(load, pv, wind, RuleBasedController()).run()
            assert abs(r.battery_residual_kwh) < 1e-6, name
            assert r.energy_balance_ok, name


class TestD3D4Battery:
    def test_round_trip_is_92_percent(self) -> None:
        # Charge for one step, then discharge back to the starting SoC.
        p = 4.0
        s1 = soc_update(0.5, charge_kw=p, discharge_kw=0.0)
        stored_kwh = (s1 - 0.5) * V.battery_capacity_kwh
        delivered_kwh = stored_kwh * V.leg_efficiency
        assert abs(delivered_kwh / (p * DT) - 0.92) < 1e-12

    def test_usable_window_delivers_15_36_kwh_of_stored_energy(self) -> None:
        soc, stored = 0.90, 0.0
        while max_discharge_kw(soc) > 0.0:
            p = max_discharge_kw(soc)
            new = soc_update(soc, 0.0, p)
            stored += (soc - new) * V.battery_capacity_kwh
            soc = new
        assert abs(stored - 15.36) < 1e-9

    def test_power_limit_is_half_c(self) -> None:
        assert abs(V.battery_power_max_kw - 9.6) < 1e-12


class TestD6Genset:
    def test_off_burns_nothing(self) -> None:
        assert fuel_litres(np.array([0.0]))[0] == 0.0

    def test_full_load_matches_30_percent_efficiency(self) -> None:
        litres = fuel_litres(np.array([V.genset_rated_kw]), dt_hours=1.0)[0]
        fuel_mj = litres * V.diesel_density_kg_per_l * V.diesel_lhv_mj_per_kg
        assert abs(V.genset_rated_kw * 3.6 / fuel_mj - 0.30) < 1e-12

    def test_no_load_fraction_follows_homer_example(self) -> None:
        assert abs(V.genset_no_load_fraction - 0.033 / 0.306) < 1e-12

    def test_part_load_is_less_efficient(self) -> None:
        low = fuel_litres(np.array([1.0]), 1.0)[0] / 1.0
        high = fuel_litres(np.array([6.0]), 1.0)[0] / 6.0
        assert low > high

    def test_co2_uses_density_and_factor(self) -> None:
        assert abs(co2_kg(np.array([1.0]))[0] - 0.85 * 3.206) < 1e-12

    def test_dispatch_caps_genset_by_giving_up_charge(self) -> None:
        class ChargeFlat:
            def decide(self, soc, pv_kw, wind_kw, load_kw, timestep, dt_hours):  # noqa: ARG002
                return -V.battery_power_max_kw, 0.0

        load, pv, wind = scenario_profiles("high_load", 2, seed=0)
        r = IntegratedSystem(load, pv, wind, ChargeFlat(), initial_soc=0.10).run()
        assert r.diesel_kw.max() <= V.genset_rated_kw + 1e-12
        assert r.n_genset_cap > 0
        assert r.unserved_kwh == 0.0
        assert r.energy_balance_ok


class TestForecastAndCost:
    def test_expected_profiles_are_the_noise_means(self) -> None:
        load = expected_load_profile(STEPS_PER_DAY)
        assert np.all((load == 0.30) | (load == 3.5) | (load == 2.5))
        assert solar_profile(STEPS_PER_DAY, seed=0).sum() != expected_solar_profile().sum()
        assert expected_wind_profile().min() >= 0.0

    def test_expected_scenario_takes_no_seed_and_is_stable(self) -> None:
        a = expected_scenario_profiles("cloudy", 2)
        b = expected_scenario_profiles("cloudy", 2)
        for x, y in zip(a, b):
            assert np.array_equal(x, y)

    def test_terminal_valuation_is_signed(self) -> None:
        load, pv, wind = scenario_profiles("standard", 1, seed=0)
        r = IntegratedSystem(load, pv, wind, RuleBasedController()).run()
        c = evaluate(r)
        assert (c.terminal_eur > 0.0) == (c.soc_end < V.soc_initial)
        assert abs(c.total_eur - (c.fuel_eur + c.terminal_eur)) < 1e-12
