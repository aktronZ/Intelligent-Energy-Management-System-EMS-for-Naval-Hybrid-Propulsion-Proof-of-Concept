"""Load and generation profiles at 15-minute resolution.

The load profile is reconstructed from published measurements rather than invented.
Ma'arif et al. (2026) report, for the 5 GT solar-electric fishing boat, an average
power draw of 2-3 kW at 2.5-3.5 knots during fishing operations, rising above
3.5 kW at speeds over 4 knots. The daily profile below reproduces those published
bounds across a representative fishing day. The hourly shape is a reconstruction;
the average and peak are the published values.

The solar profile is reconstructed from a clear-sky Mediterranean irradiance model
scaled by the published capacity factor. The wind profile is reconstructed from the
published capacity factor.

Every profile repeats daily: the time of day is the step index modulo the steps
per day (fixed 2026-09-27, TASK-065 D1; before that a multi-day run stretched one
day over the whole run).

Each profile has two forms that share one code path:
  * the realised profile, with seeded noise (ems.random);
  * the expected profile, with every noise draw replaced by its mean. This is the
    MPC's schedule forecast. It takes no seed, so it cannot leak the realisation.
"""

from __future__ import annotations

import numpy as np

from ems.random import rng

TIMESTEP_MINUTES = 15
TIMESTEP_HOURS = TIMESTEP_MINUTES / 60.0
STEPS_PER_DAY = int(24 / TIMESTEP_HOURS)

# Mean of a U(0, 1) draw: what the expected profiles use in place of noise.
UNIFORM_MEAN = 0.5


def day_fraction(steps: int) -> np.ndarray:
    """Time of day, as a fraction of the day, at each step of a run."""
    return (np.arange(steps) % STEPS_PER_DAY) / STEPS_PER_DAY


def _check(steps: int) -> None:
    if steps < 1:
        raise ValueError("steps must be positive")


def _load(steps: int, draws: np.ndarray) -> np.ndarray:
    profile = np.full(steps, 0.30, dtype=float)  # anchored auxiliary load
    day = day_fraction(steps)
    transit = ((day >= 0.21) & (day < 0.25)) | ((day >= 0.63) & (day < 0.67))
    fishing = (day >= 0.25) & (day < 0.63)
    profile[transit] = 3.5                       # published peak, above 4 knots
    profile[fishing] = 2.0 + 1.0 * draws[fishing]
    return profile


def _solar(steps: int, draws: np.ndarray) -> np.ndarray:
    day = day_fraction(steps)
    bell = np.exp(-0.5 * ((day - 0.5) / 0.16) ** 2)
    daylight = (day >= 0.25) & (day < 0.75)      # night is exactly zero
    bell = bell * daylight
    mean = bell.mean()
    if mean <= 0.0:
        return np.zeros(steps, dtype=float)
    profile = np.clip(bell * (0.17 / mean), 0.0, 1.0)
    noise = 1.0 + 0.05 * draws - 0.025          # light cloud variability, mean 1
    return np.clip(profile * noise, 0.0, 1.0)


def _wind(steps: int, draws: np.ndarray) -> np.ndarray:
    day = day_fraction(steps)
    base = 0.5 + 0.3 * np.sin(2.0 * np.pi * (day - 0.25))
    mean = base.mean()
    profile = base * (0.22 / mean if mean > 0.0 else 0.0)
    noise = 1.0 + 0.10 * draws - 0.05            # mean 1
    return np.clip(profile * noise, 0.0, 1.0)


def load_profile(steps: int = STEPS_PER_DAY, seed: int = 0) -> np.ndarray:
    """Propulsion and auxiliary load in kW at each timestep (realised)."""
    _check(steps)
    return _load(steps, rng(seed).random(steps))


def solar_profile(steps: int = STEPS_PER_DAY, seed: int = 0) -> np.ndarray:
    """Plane-of-array irradiance factor, 0 to 1, at each timestep (realised)."""
    _check(steps)
    return _solar(steps, rng(seed).random(steps))


def wind_profile(steps: int = STEPS_PER_DAY, seed: int = 0) -> np.ndarray:
    """Wind power factor, 0 to 1, at each timestep (realised)."""
    _check(steps)
    return _wind(steps, rng(seed).random(steps))


def expected_load_profile(steps: int = STEPS_PER_DAY) -> np.ndarray:
    """Load with the fishing-speed draw replaced by its mean (2.5 kW while fishing)."""
    _check(steps)
    return _load(steps, np.full(steps, UNIFORM_MEAN))


def expected_solar_profile(steps: int = STEPS_PER_DAY) -> np.ndarray:
    """Solar factor with the cloud noise replaced by its mean (factor 1)."""
    _check(steps)
    return _solar(steps, np.full(steps, UNIFORM_MEAN))


def expected_wind_profile(steps: int = STEPS_PER_DAY) -> np.ndarray:
    """Wind factor with the noise replaced by its mean (factor 1)."""
    _check(steps)
    return _wind(steps, np.full(steps, UNIFORM_MEAN))
