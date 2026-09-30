"""Tests for the RL environment and controller (S6, TASK-074).

They need gymnasium but not Stable-Baselines3 or a trained model: policies here are
hand-written stand-ins with the same ``predict`` interface.
"""

from __future__ import annotations

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")

from ems.controllers.cycle_charging import CycleChargingController  # noqa: E402
from ems.controllers.rule_based import RuleBasedController  # noqa: E402
from ems.cost import evaluate  # noqa: E402
from ems.rl.controller import RLController  # noqa: E402
from ems.rl.env import (  # noqa: E402
    EVALUATION_SEEDS,
    REWARD_SCALE_EUR,
    EMSEnv,
    training_seeds,
)
from ems.rl.features import N_ACTIONS, battery_request  # noqa: E402
from ems.scenarios import expected_scenario_profiles, scenario_profiles  # noqa: E402
from ems.system import IntegratedSystem  # noqa: E402

OPTIONS = {"scenario": "standard", "profile_seed": 3, "days": 2}


def _rollout(policy, options=OPTIONS):
    """Run one env episode; return (sum of rewards * scale, actions, observations)."""
    env = EMSEnv()
    obs, _ = env.reset(seed=0, options=options)
    total, actions, observations, done = 0.0, [], [], False
    while not done:
        assert env.observation_space.contains(obs)
        observations.append(obs)
        action = policy(env, obs)
        actions.append(action)
        obs, reward, done, truncated, _ = env.step(action)
        assert not truncated
        total += reward * REWARD_SCALE_EUR
    return total, np.array(actions), np.array(observations)


class _ActionReplay:
    """Controller replaying an action sequence, to score it through the simulator."""

    def __init__(self, actions):
        self.actions = actions

    def decide(self, soc, pv_kw, wind_kw, load_kw, timestep, dt_hours):  # noqa: ARG002
        return battery_request(int(self.actions[timestep]), load_kw - pv_kw - wind_kw), 0.0


def _profiles(options=OPTIONS):
    return scenario_profiles(options["scenario"], options["days"], options["profile_seed"])


class TestEnvironment:
    def test_episode_length_and_spaces(self) -> None:
        rng = np.random.default_rng(0)
        _, actions, obs = _rollout(lambda env, o: int(rng.integers(N_ACTIONS)))
        assert len(actions) == 2 * 96
        assert obs.dtype == np.float32

    def test_return_equals_minus_the_thesis_cost(self) -> None:
        rng = np.random.default_rng(1)
        ret, actions, _ = _rollout(lambda env, o: int(rng.integers(N_ACTIONS)))
        r = IntegratedSystem(*_profiles(), _ActionReplay(actions)).run()
        assert abs(ret + evaluate(r).total_eur) < 1e-9
        assert r.energy_balance_ok and r.unserved_kwh == 0.0

    def test_follow_only_is_the_rule_based_baseline(self) -> None:
        ret, _, _ = _rollout(lambda env, o: 0)
        rb = evaluate(IntegratedSystem(*_profiles(), RuleBasedController()).run())
        assert abs(ret + rb.total_eur) < 1e-9

    def test_cycle_charging_is_expressible(self) -> None:
        cc = CycleChargingController()

        def policy(env, obs):
            t = env._t
            net = env.load[t] - env.pv[t] - env.wind[t]
            request, _ = cc.decide(env.soc, env.pv[t], env.wind[t], env.load[t], t, 0.25)
            return 0 if (net <= 0.0 or request == net) else N_ACTIONS - 1

        ret, _, _ = _rollout(policy)
        ref = evaluate(IntegratedSystem(*_profiles(), CycleChargingController()).run())
        assert abs(ret + ref.total_eur) < 1e-9

    def test_training_draws_only_training_seeds(self) -> None:
        env = EMSEnv(days=1)
        pool = set(training_seeds())
        for k in range(50):
            _, info = env.reset(seed=k)
            assert info["profile_seed"] in pool
        assert not pool & set(EVALUATION_SEEDS)


class _RecordingPolicy:
    """A stand-in for an SB3 model: fixed rule, and records what it was shown."""

    def __init__(self, rule):
        self.rule, self.seen = rule, []

    def predict(self, obs, deterministic=True):  # noqa: ARG002
        self.seen.append(np.array(obs))
        return np.array(self.rule(obs)), None


class TestController:
    def test_controller_sees_what_the_env_shows(self) -> None:
        # A policy that depends on the genset feature exercises the feedback path.
        rule = lambda o: N_ACTIONS - 1 if (o[0] < 0.3 or (o[2] > 0.5 and o[0] < 0.8)) else 0  # noqa: E731
        ret, _, env_obs = _rollout(lambda env, o: rule(o))
        policy = _RecordingPolicy(rule)
        ctrl = RLController(policy, expected_scenario_profiles(OPTIONS["scenario"], OPTIONS["days"]))
        cost = evaluate(IntegratedSystem(*_profiles(), ctrl).run())
        assert np.array_equal(np.array(policy.seen), env_obs)
        assert abs(ret + cost.total_eur) < 1e-9

    def test_missing_model_says_how_to_train(self) -> None:
        with pytest.raises(SystemExit):
            RLController.load(987, expected_scenario_profiles("standard", 1))
