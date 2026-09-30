"""A trained RL policy as an ems controller.

It builds the observation from exactly the features the environment trained on,
asks the policy for an action (deterministically), and turns the action into a
battery request. The simulator and ems.cost then score it like every other
controller.

The "genset ran last step" feature is obtained by applying the same pure plant
function (ems.system.dispatch_step) to the request the controller just made:
it is what the simulator is about to do, so no feedback channel is needed.

The policy is anything with ``predict(obs, deterministic=True)``; a
Stable-Baselines3 model is loaded from results/models by ``load``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ems.components.genset import is_on
from ems.controllers.offline_optimum import net_demand
from ems.rl.features import battery_request, forecast_means, observation
from ems.system import dispatch_step

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "results" / "models"


def model_path(train_seed: int) -> Path:
    return MODELS / f"rl_ppo_t{train_seed}.zip"


class RLController:
    """Runs a trained policy on the plant."""

    def __init__(self, policy, expected_profiles) -> None:
        self.policy = policy
        self.forecast = forecast_means(net_demand(*expected_profiles))
        self.genset_on = False
        self.actions = np.zeros(len(self.forecast), dtype=np.int64)

    @classmethod
    def load(cls, train_seed: int, expected_profiles):
        path = model_path(train_seed)
        if not path.exists():
            raise SystemExit(
                f"no trained model at {path}. Train it first:\n"
                f"  uv run python -m ems.rl.train --train-seeds {train_seed}"
            )
        import torch  # imported here so the rest of ems never needs torch
        from stable_baselines3 import PPO

        torch.set_num_threads(1)  # experiments run one process per core
        return cls(PPO.load(str(path), device="cpu"), expected_profiles)

    def decide(self, soc, pv_kw, wind_kw, load_kw, timestep, dt_hours):  # noqa: ARG002
        net = load_kw - pv_kw - wind_kw
        obs = observation(soc, net, self.genset_on, timestep, self.forecast[timestep])
        action, _ = self.policy.predict(obs, deterministic=True)
        action = int(np.asarray(action).item())
        self.actions[timestep] = action
        request = battery_request(action, net)
        outcome = dispatch_step(soc, pv_kw, wind_kw, load_kw, request)
        self.genset_on = bool(is_on(np.array([outcome.diesel_kw]))[0])
        return request, 0.0
