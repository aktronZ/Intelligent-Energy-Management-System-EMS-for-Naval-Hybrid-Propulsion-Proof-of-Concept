"""Reinforcement-learning controller (S6, TASK-074).

    features.py    actions and observations, shared by training and evaluation
    env.py         Gymnasium environment around the plant (ems.system.dispatch_step)
    controller.py  a trained policy as an ems controller, scored like every other
    train.py       PPO training with Stable-Baselines3

Design and rationale: specs/074-rl-design.md.
"""
