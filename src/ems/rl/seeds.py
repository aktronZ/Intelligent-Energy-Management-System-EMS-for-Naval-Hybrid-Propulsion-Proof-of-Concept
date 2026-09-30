"""Profile-seed pools: training and evaluation are disjoint by construction.

No dependency on gymnasium or torch, so the experiment runner and the gates can
import it. TASK-074: the reported number is the held-out performance.
"""

from __future__ import annotations

EVALUATION_SEEDS = (0, 1, 2, 3, 4)
TRAIN_SEED_START = 1000          # training profiles use seeds 1000 ... 1000 + pool - 1
TRAIN_SEED_POOL = 10000
TRAIN_SEEDS = (0, 1, 2)          # PPO initialisation seeds: one trained policy per seed


def training_seeds() -> range:
    return range(TRAIN_SEED_START, TRAIN_SEED_START + TRAIN_SEED_POOL)


def rl_controller_names(train_seeds=TRAIN_SEEDS) -> list[str]:
    return [f"rl_t{s}" for s in train_seeds]
