"""Seeded randomness.

Rule: no randomness anywhere else in the package. Every stochastic run takes an
explicit seed, so two runs with the same seed are byte-identical (gate G3).
"""

from __future__ import annotations

import numpy as np


def rng(seed: int) -> np.random.Generator:
    """Return a generator for the given seed.

    A fresh generator per call, never a module-level global, so that concurrent
    runs cannot contaminate each other.
    """
    return np.random.default_rng(seed)
