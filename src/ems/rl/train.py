"""Train the RL controller with PPO (Stable-Baselines3).

Usage:
    uv run python -m ems.rl.train                       # training seeds 0 1 2, 1,000,000 steps each
    uv run python -m ems.rl.train --train-seeds 0 --timesteps 200000   # quick run

For each training seed it writes, under results/models/:
    rl_ppo_t{seed}.zip     the trained policy
    rl_ppo_t{seed}.json    what produced it: commit, seeds, hyperparameters, wall time
    monitor_t{seed}/       per-episode training returns (SB3 Monitor CSVs)

Training profiles come only from the training seed pool (ems.rl.env); the thesis
reports the policy's cost on the evaluation seeds 0-4, which it never saw
(TASK-074). Algorithm choice and hyperparameters: specs/074-rl-design.md.

Reproducibility: evaluation of a saved model is deterministic (gate G3). Training
itself is seeded, but PyTorch does not guarantee bit-identical training across
machines and library versions, which is why the model files are kept and cited.
"""

from __future__ import annotations

import argparse
import json
import time

from ems.rl.controller import MODELS, model_path
from ems.rl.env import TRAIN_EPISODE_DAYS, TRAIN_SCENARIOS, EMSEnv
from ems.rl.seeds import (
    EVALUATION_SEEDS,
    TRAIN_SEED_POOL,
    TRAIN_SEED_START,
    TRAIN_SEEDS,
    training_seeds,
)
from ems.run import _git_commit

TOTAL_TIMESTEPS = 1000000
N_ENVS = 4
PPO_HYPERPARAMETERS = {
    "gamma": 0.995,          # effective horizon about 200 steps, about two days
    "n_steps": 2048,         # per environment, per update
    "batch_size": 256,
    "learning_rate": 0.0003, # SB3 default
    "ent_coef": 0.01,        # keeps exploring the four discrete actions
}


def train(train_seed: int, timesteps: int) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.env_util import make_vec_env

    assert not set(EVALUATION_SEEDS) & set(training_seeds()), "evaluation seeds leak into training"
    check_env(EMSEnv(), warn=True)

    MODELS.mkdir(parents=True, exist_ok=True)
    vec_env = make_vec_env(EMSEnv, n_envs=N_ENVS, seed=train_seed,
                           monitor_dir=str(MODELS / f"monitor_t{train_seed}"))
    model = PPO("MlpPolicy", vec_env, seed=train_seed, device="cpu", verbose=1, **PPO_HYPERPARAMETERS)

    t0 = time.time()
    model.learn(total_timesteps=timesteps)
    wall = time.time() - t0
    model.save(str(model_path(train_seed)))

    meta = {
        "commit": _git_commit(),
        "algorithm": "PPO (stable-baselines3), MlpPolicy",
        "train_seed": train_seed,
        "timesteps": timesteps,
        "n_envs": N_ENVS,
        "hyperparameters": PPO_HYPERPARAMETERS,
        "episode_days": TRAIN_EPISODE_DAYS,
        "scenarios": list(TRAIN_SCENARIOS),
        "profile_seed_pool": [TRAIN_SEED_START, TRAIN_SEED_START + TRAIN_SEED_POOL - 1],
        "evaluation_seeds_held_out": list(EVALUATION_SEEDS),
        "wall_seconds": round(wall, 1),
    }
    model_path(train_seed).with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {model_path(train_seed)}  ({wall / 60:.1f} min)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the RL controller (PPO)")
    parser.add_argument("--train-seeds", type=int, nargs="+", default=list(TRAIN_SEEDS))
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    args = parser.parse_args(argv)
    for seed in args.train_seeds:
        train(seed, args.timesteps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
