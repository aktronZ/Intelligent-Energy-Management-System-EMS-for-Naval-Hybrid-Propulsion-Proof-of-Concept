"""Unit tests for the component models.

Each model must behave correctly in its limiting cases. Run: uv run pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest

from ems.components.battery import max_discharge_kw, max_charge_kw, soc_update
from ems.components.pv import pv_power
from ems.components.propulsion import propulsion_load
from ems.components.wind import wind_power
from ems.profiles import STEPS_PER_DAY, load_profile, solar_profile, wind_profile


class TestPV:
    def test_zero_irradiance_gives_zero(self) -> None:
        assert np.allclose(pv_power(np.zeros(STEPS_PER_DAY)), 0.0)

    def test_full_irradiance_gives_capacity(self) -> None:
        out = pv_power(np.ones(STEPS_PER_DAY))
        assert np.allclose(out, 2.18)

    def test_over_capacity_is_clipped(self) -> None:
        out = pv_power(np.full(STEPS_PER_DAY, 5.0))
        assert np.all(out <= 2.18)


class TestWind:
    def test_zero_factor_gives_zero(self) -> None:
        assert np.allclose(wind_power(np.zeros(STEPS_PER_DAY)), 0.0)

    def test_full_factor_gives_rated(self) -> None:
        out = wind_power(np.ones(STEPS_PER_DAY))
        assert np.allclose(out, 1.5)


class TestPropulsion:
    def test_zero_profile_gives_zero(self) -> None:
        assert np.allclose(propulsion_load(np.zeros(STEPS_PER_DAY)), 0.0)

    def test_excess_is_clipped_to_motor(self) -> None:
        out = propulsion_load(np.full(STEPS_PER_DAY, 999.0))
        assert np.all(out <= 15.0)


class TestBattery:
    def test_charge_increases_soc(self) -> None:
        assert soc_update(0.5, charge_kw=1.0, discharge_kw=0.0) > 0.5

    def test_discharge_decreases_soc(self) -> None:
        assert soc_update(0.5, charge_kw=0.0, discharge_kw=1.0) < 0.5

    def test_simultaneous_charge_and_discharge_raises(self) -> None:
        with pytest.raises(ValueError):
            soc_update(0.5, charge_kw=1.0, discharge_kw=1.0)

    def test_overcharge_raises_instead_of_clipping(self) -> None:
        # Silently clipping here created energy (TASK-065 D2); it must now raise.
        with pytest.raises(ValueError):
            soc_update(0.89, charge_kw=50.0, discharge_kw=0.0)

    def test_overdischarge_raises_instead_of_clipping(self) -> None:
        with pytest.raises(ValueError):
            soc_update(0.11, charge_kw=0.0, discharge_kw=50.0)

    def test_max_charge_lands_exactly_on_upper_bound(self) -> None:
        out = soc_update(0.89, charge_kw=max_charge_kw(0.89), discharge_kw=0.0)
        assert abs(out - 0.90) < 1e-12

    def test_max_discharge_lands_exactly_on_lower_bound(self) -> None:
        out = soc_update(0.11, charge_kw=0.0, discharge_kw=max_discharge_kw(0.11))
        assert abs(out - 0.10) < 1e-12

    def test_max_charge_decreases_near_full(self) -> None:
        assert max_charge_kw(0.89) < max_charge_kw(0.50)

    def test_max_discharge_decreases_near_empty(self) -> None:
        assert max_discharge_kw(0.11) < max_discharge_kw(0.50)


class TestProfiles:
    def test_load_profile_shape(self) -> None:
        out = load_profile()
        assert out.shape == (STEPS_PER_DAY,)
        assert np.all(out >= 0.0)

    def test_solar_profile_bounds(self) -> None:
        out = solar_profile()
        assert np.all(out >= 0.0)
        assert np.all(out <= 1.0)

    def test_solar_profile_is_zero_at_night(self) -> None:
        out = solar_profile()
        assert out[0] == 0.0

    def test_determinism(self) -> None:
        assert np.array_equal(load_profile(seed=42), load_profile(seed=42))

    def test_profiles_are_seeded_and_reproducible(self) -> None:
        a = solar_profile(seed=7)
        b = solar_profile(seed=7)
        assert np.array_equal(a, b)
