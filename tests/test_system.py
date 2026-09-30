"""Tests for the integrated system model.

Gate G1 requires the energy balance to close to within 1e-6 kWh over every
simulation day. Gate G2 requires the state of charge to stay within the usable
window. Gate G3 requires determinism for a given seed.
"""

from __future__ import annotations

import numpy as np

from ems.profiles import load_profile, solar_profile, wind_profile
from ems.system import TOLERANCE_KWH, IntegratedSystem


class TrivialController:
    """Discharges the battery to serve load and charges from surplus.

    Not a sensible energy management strategy — it exists to exercise the
    integrated model and to verify that the power balance closes.
    """

    def decide(self, soc, pv_kw, wind_kw, load_kw, timestep, dt_hours):  # noqa: ARG002
        net = load_kw - pv_kw - wind_kw
        battery = max(-5.0, min(5.0, -net))
        return battery, 0.0


def _run() -> object:
    return IntegratedSystem(
        load_profile(), solar_profile(), wind_profile(), TrivialController()
    ).run()


def test_energy_balance_closes() -> None:
    result = _run()
    assert result.energy_balance_ok, (
        f"energy balance residual {result.balance_residual_kwh} exceeds {TOLERANCE_KWH}"
    )


def test_soc_stays_within_window() -> None:
    result = _run()
    assert result.soc.min() >= 0.10 - 1e-9
    assert result.soc.max() <= 0.90 + 1e-9


def test_determinism() -> None:
    a = _run()
    b = _run()
    assert np.array_equal(a.soc, b.soc)
    assert np.array_equal(a.battery_kw, b.battery_kw)


def test_profiles_have_equal_length() -> None:
    assert len(load_profile()) == len(solar_profile()) == len(wind_profile())
