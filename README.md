# EMS for a small hybrid vessel: simulation framework

Code of the Final Degree Project *Intelligent Energy Management System (EMS) for Naval Hybrid
Propulsion: Proof of Concept* (Akram Chellou Bakkali, Facultat de Nàutica de Barcelona, UPC, 2026).

The package `ems` simulates a 5 GT solar-electric fishing boat, extended with a 1.5 kW wind
turbine and a 6 kW diesel generator, in 15-minute steps. It compares five energy management
strategies on the same plant and with the same cost function:

| Strategy | Family | Role |
|---|---|---|
| Rule-based (load following) | Rule-based | Lower reference (baseline) |
| Cycle charging | Rule-based | Better fixed rule |
| Model predictive control (MPC) | Optimisation, online | Proposed EMS |
| Offline optimum (dynamic programming) | Optimisation, offline | Upper reference |
| Three PPO policies | Reinforcement learning | AI-based EMS |

## How the code is organised

| Part | What it contains |
|---|---|
| `src/ems/vessel.py`, `src/ems/components/` | Parameters of the vessel and models of the solar panels, wind turbine, battery, motor and generator |
| `src/ems/profiles.py`, `src/ems/scenarios.py` | Daily load, sun and wind profiles, and the test scenarios |
| `src/ems/system.py` | The plant: applies the physical limits at every step and checks the energy balance |
| `src/ems/cost.py` | The single cost function that converts the trace of any run into its fuel cost |
| `src/ems/controllers/`, `src/ems/optim.py` | Rule-based, cycle charging, offline optimum (dynamic programming) and MPC |
| `src/ems/rl/` | Reinforcement-learning environment, training and controller |
| `src/ems/experiments.py` | Runs the 120 simulations (3 scenarios × 5 seeds × 8 controllers) and writes the comparison file |
| `tests/`, `acceptance/` | Unit tests and the automatic acceptance checks |
| `specs/010-simulation-contract.md` | Source or assumption of every parameter used by the simulator |

## How to reproduce the results

Requirements: Python 3.13 and the [uv](https://docs.astral.sh/uv/) package manager.

```bash
uv sync                               # install the exact library versions
uv run python -m ems.rl.train         # train the three RL policies (about 3.6 min each)
uv run python -m ems.experiments      # run the 120 simulations
uv run python -m ems.report           # compute every number of Chapter 9
uv run python -m ems.figures          # draw the figures of Chapter 9
uv run python -m acceptance.run       # run all automatic checks
```

A single run can be launched from the command line, for example:

```bash
uv run python -m ems.run --profile standard --days 7 --seed 0 --controller rule_based
```

Every run is deterministic: the same seed and the same version of the code give byte-identical
result files. The outputs are written to `results/`.

## Main result

In the standard scenario, MPC reduces the fuel cost and CO₂ of the load-following rule by 28 %
(18–28 % across scenarios). A cycle-charging rule already obtains 94–97 % of that saving; the
learned policies improve on the load-following rule but not on cycle charging. See the thesis
for the full discussion.
