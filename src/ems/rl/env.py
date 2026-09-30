"""Gymnasium environment: the hybrid vessel's plant, one 15-minute step at a time.

Physics come from ems.system.dispatch_step (the same function every controller is
evaluated on). Reward is minus the fuel cost of the step, with the terminal
valuation of ems.cost added on the last step, all divided by REWARD_SCALE_EUR:

    sum of rewards * REWARD_SCALE_EUR  ==  -ems.cost.evaluate(trace).total_eur

so maximising return is exactly minimising the cost the thesis reports (a test
checks the identity).

Episodes. Training episodes draw a scenario and a profile seed at random from the
TRAINING seed pool, which is disjoint from the evaluation seeds (0-4) by
construction: the reported number is held-out performance (TASK-074). An
explicit (scenario, seed, days) can be passed through ``options`` for tests.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ems.components.genset import fuel_litres, is_on
from ems.components.propulsion import propulsion_load
from ems.components.pv import pv_power
from ems.components.wind import wind_power
from ems.controllers.offline_optimum import net_demand
from ems.cost import terminal_value_eur_per_kwh
from ems.rl.features import N_ACTIONS, N_FEATURES, battery_request, forecast_means, observation
from ems.rl.seeds import EVALUATION_SEEDS, TRAIN_SEED_POOL, TRAIN_SEED_START, training_seeds  # noqa: F401
from ems.scenarios import expected_scenario_profiles, scenario_profiles
from ems.system import dispatch_step
from ems.vessel import REFERENCE

REWARD_SCALE_EUR = 0.1
TRAIN_EPISODE_DAYS = 7
TRAIN_SCENARIOS = ("standard", "high_load", "cloudy")
# Observation bounds: comfortably outside every value the features can take.
OBS_BOUND = 10.0


class EMSEnv(gym.Env):
    """One episode = one scenario, one profile seed, a whole number of days."""

    metadata = {"render_modes": []}

    def __init__(self, days: int = TRAIN_EPISODE_DAYS, scenarios=TRAIN_SCENARIOS) -> None:
        super().__init__()
        self.days = days
        self.scenarios = tuple(scenarios)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(-OBS_BOUND, OBS_BOUND, shape=(N_FEATURES,), dtype=np.float32)
        self._t = 0

    # ------------------------------------------------------------------ episode
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        scenario = options.get("scenario") or self.scenarios[int(self.np_random.integers(len(self.scenarios)))]
        profile_seed = options.get("profile_seed")
        if profile_seed is None:
            profile_seed = TRAIN_SEED_START + int(self.np_random.integers(TRAIN_SEED_POOL))
        days = int(options.get("days", self.days))

        load, pv_f, wind_f = scenario_profiles(scenario, days, profile_seed)
        self.pv = pv_power(pv_f)
        self.wind = wind_power(wind_f)
        self.load = propulsion_load(load)
        self.forecast = forecast_means(net_demand(*expected_scenario_profiles(scenario, days)))
        self.n = len(self.load)
        self._t = 0
        self.soc = REFERENCE.soc_initial
        self.genset_on = False
        self.scenario, self.profile_seed = scenario, profile_seed
        return self._obs(), {"scenario": scenario, "profile_seed": profile_seed}

    def _obs(self) -> np.ndarray:
        t = self._t
        net = self.load[t] - self.pv[t] - self.wind[t]
        return observation(self.soc, net, self.genset_on, t, self.forecast[t])

    # ------------------------------------------------------------------ step
    def step(self, action):
        t = self._t
        pv, wind, load = float(self.pv[t]), float(self.wind[t]), float(self.load[t])
        out = dispatch_step(self.soc, pv, wind, load, battery_request(int(action), load - pv - wind))
        cost = float(fuel_litres(np.array([out.diesel_kw]))[0]) * REFERENCE.fuel_price_eur_per_l

        self.soc = out.soc
        self.genset_on = bool(is_on(np.array([out.diesel_kw]))[0])
        self._t += 1
        terminated = self._t >= self.n
        if terminated:
            cost -= terminal_value_eur_per_kwh() * REFERENCE.battery_capacity_kwh * (
                self.soc - REFERENCE.soc_initial
            )
        info = {"cost_eur": cost, "diesel_kw": out.diesel_kw, "unserved_kw": out.unserved_kw}
        obs = self._obs() if not terminated else np.zeros(self.observation_space.shape, np.float32)
        return obs, -cost / REWARD_SCALE_EUR, terminated, False, info
