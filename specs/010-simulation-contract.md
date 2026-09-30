# 010 — Simulation contract

This is the binding contract between the reference vessel, the simulator and the experiment
matrix. Everything in it is either sourced, derived, measured, or explicitly assumed.

## 1. Reference vessel

Two vessels serve different roles. This split is deliberate and must not be blurred.

### 1.1 Motivating precedent — EcoSlim (narrative only, never simulated)

Real vessel built by Drassanes Dalmau, Arenys de Mar, launched 31 March 2011, with technical
support from the UPC. It is the local proof that a multi-source shipboard EMS exists.

| Parameter | Value | Unit | Source | Source type |
|---|---|---|---|---|
| Length overall | 24 | m | New Atlas (2011) | published |
| Maximum beam | 10.5 | m | New Atlas (2011) | published |
| Passenger capacity | 150 | persons | New Atlas (2011) | published |
| Propulsion power | 2 × 110 | kW | NauticalExpo catalogue | published |
| Solar panels | 40 | count, monocrystalline, deck-mounted | New Atlas (2011) | published |
| Wind turbines | 2 | count | New Atlas (2011) | published |
| Renewable generation, combined peak | 9.5 | kW | New Atlas (2011) | published |
| Thermal generator | diesel-electric | — | New Atlas (2011) | published |
| Auxiliary fuel cell | 2 | kW, hydrogen | New Atlas (2011) | published |
| Continuous range | 4 | h at 6–7 kn | New Atlas (2011) | published |
| Top speed | 12 | kn | New Atlas (2011) | published |

**Battery capacity: UNRESOLVED — do not quote either figure.**

Two published sources conflict and are arithmetically irreconcilable:

- New Atlas (2011): "a bank of 90 lead-acid batteries"
- NauticalExpo catalogue: "1,130 kWh (388 V)"

388 V ÷ 90 cells = 4.31 V per cell, which is not a real cell voltage. One of the two figures is
wrong or refers to a different vessel. Both are therefore **excluded from the simulation**.

**Consequence:** EcoSlim is used only as the motivating precedent in Chapters 3 and 4.1. It is
never the simulated plant. Its renewable share is 9.5 kW against 220 kW of propulsion — about
4% — which is exactly why it makes a poor simulation plant and a good illustration of why
renewables alone do not drive the EMS problem.

### 1.2 Quantitative plant — Ma'arif et al. (2026)

5 GT solar-electric fishing boat, field-measurement-based, peer-reviewed in *Results in
Engineering*. Already cited in Chapter 2.3. This is the plant the simulator models.

| Parameter | Value | Unit | Source | Source type |
|---|---|---|---|---|
| Vessel type | solar-electric fishing boat | — | Ma'arif et al. (2026) | published |
| Gross tonnage | 5 | GT | Ma'arif et al. (2026) | published |
| Photovoltaic capacity | 2.18 | kWp | Ma'arif et al. (2026) | published |
| Battery capacity | 19.2 | kWh | Ma'arif et al. (2026) | published |

**Wind generation is absent from Ma'arif's plant and is added as a documented increment.** The
wind component model follows Nomak and Çiçek (2025), also already cited in Chapter 2.3, which
covers solar, wind and hydro generation on a 12.5 m sailboat.

## 2. Model parameterisation

Derived values follow from the sources above plus stated assumptions. Each row names its
derivation or its rationale.

| Parameter | Value | Unit | Basis | Source type |
|---|---|---|---|---|
| Battery usable capacity | 15.36 | kWh | 19.2 kWh × 0.80 usable window | derived |
| Usable SoC window | 80 | % of nominal | assumption — conservative window for cycle life | assumption |
| Wind turbine rated power | 1.5 | kW | assumption — small marine unit for a 5 GT craft | assumption |
| Propulsion motor rated power | 15 | kW | assumption — adequate for a 5 GT vessel, within generation capability | assumption |
| State of charge, minimum | 0.10 | fraction of nominal | assumption — lower bound of the usable window | assumption |
| State of charge, maximum | 0.90 | fraction of nominal | assumption — upper bound of the usable window | assumption |
| Round-trip efficiency | 92 | % | assumption — lithium-ion, conservative | assumption |
| PV capacity factor, annual | 17 | % | assumption — Mediterranean, fixed tilt | assumption |
| Wind capacity factor, annual | 22 | % | assumption — marine, small turbine | assumption |
| Timestep | 15 | min | assumption — EMS decision horizon granularity | assumption |
| Demand profile resolution | 15 | min | matches timestep | derived |
| Marine diesel LHV | 42.7 | MJ/kg | standard value | published |
| Marine diesel density | 0.85 | kg/L | standard value | published |
| Marine diesel CO₂ factor | 3.206 | kg CO₂/kg fuel | IPCC default for marine gas oil | published |
| Fuel price, constant case | 0.85 | EUR/L | assumption — 2026 Mediterranean bunker | assumption |
| Fuel price, TOU peak | 1.25 | EUR/L | assumption — peak window 18:00–23:00 | assumption |
| Fuel price, TOU off-peak | 0.62 | EUR/L | assumption | assumption |
| Demand charge | 6.50 | EUR/kW/month | assumption — exceeds fuel price to create arbitrage | assumption |
| EU ETS allowance price | 75 | EUR/t CO₂ | assumption — 2026 forward market | assumption |
| Evaluation horizon | 30 | days | assumption — long enough to average weather | assumption |

### Derived fractions used in code

The capacity and efficiency factors above are recorded as percentages because that is
how they are reported in the source literature. The code uses fractions. This table
records the conversion so that the provenance gate can trace it.

| Parameter | Value | Unit | Basis | Source type |
|---|---|---|---|---|
| Round-trip efficiency, fraction | 0.92 | — | 92 % ÷ 100 | derived |
| Per-leg efficiency, charge and discharge | 0.959 | — | √0.92, so that η_c·η_d equals the 92 % round trip (TASK-065 D3a, 2026-09-27); computed in code with `math.sqrt` | derived |
| PV capacity factor, fraction | 0.17 | — | 17 % ÷ 100 | derived |
| Wind capacity factor, fraction | 0.22 | — | 22 % ÷ 100 | derived |
| Timestep, hours | 0.25 | h | 15 min ÷ 60 min/h | derived |
| Minutes per hour | 60 | min/h | unit definition | published |
| Hours per day | 24 | h/day | unit definition | published |
| Days per month | 30 | day/month | convention, matches evaluation horizon | assumption |

### Profile reconstruction parameters

The load, solar and wind profiles are reconstructed from published bounds rather
than taken as raw time series. The shape parameters below define that
reconstruction. The power levels are published; the temporal shape is assumed.

| Parameter | Value | Unit | Basis | Source type |
|---|---|---|---|---|
| Auxiliary load, anchored | 0.30 | kW | assumption — hotel load at anchor | assumption |
| Load peak power | 3.5 | kW | Ma'arif et al. (2026), draw above 4 knots | published |
| Load mean power, fishing | 2.0 | kW | Ma'arif et al. (2026), 2.5-3.5 knots | published |
| Load spread, fishing | 1.0 | kW | Ma'arif et al. (2026): 2-3 kW while fishing, so U(0,1) × 1.0 kW above 2.0 kW | published |
| Transit window, morning start | 0.21 | fraction of day | assumption — 05:02 departure (first 15-min step 05:15) | assumption |
| Transit window, morning end | 0.25 | fraction of day | assumption — 06:00 arrival | assumption |
| Fishing window, start | 0.25 | fraction of day | assumption — primary working hours | assumption |
| Fishing window, end | 0.63 | fraction of day | assumption | assumption |
| Return transit, start | 0.63 | fraction of day | assumption | assumption |
| Return transit, end | 0.67 | fraction of day | assumption | assumption |
| Solar noon | 0.5 | fraction of day | convention | published |
| Solar bell width | 0.16 | fraction of day | assumption — clear-sky approximation | assumption |
| Daylight window, start | 0.25 | fraction of day | assumption — 06:00 | assumption |
| Daylight window, end | 0.75 | fraction of day | assumption — 18:00 | assumption |
| Solar noise amplitude | 0.05 | fraction | assumption — light cloud variability | assumption |
| Solar noise offset | 0.025 | fraction | assumption | assumption |
| Wind base level | 0.5 | fraction | assumption | assumption |
| Wind diurnal amplitude | 0.3 | fraction | assumption | assumption |
| Wind noise amplitude | 0.05 | fraction | assumption | assumption |
| Wind noise span | 0.10 | fraction | derived — twice the amplitude, `1 + 0.10·U − 0.05` | derived |
| Cloudy scenario solar scale | 0.2 | fraction | assumption — heavy cloud cover | assumption |
| High-load scenario load scale | 1.5 | factor | assumption — longer working day with more transit | assumption |
| Initial state of charge | 0.50 | fraction of nominal | assumption — mid-window start | assumption |

### Battery power (decision 2026-09-27, TASK-065 D5)

| Parameter | Value | Unit | Basis | Source type |
|---|---|---|---|---|
| Battery C-rate, charge and discharge | 0.5 | 1/h | assumption — conservative continuous rate for cycle life | assumption |
| Battery power limit | 9.6 | kW | 0.5 /h × 19.2 kWh | derived |

### Diesel generator — documented increment (decision 2026-09-27, A1/A2)

Ma'arif et al. (2026) describe a solar-electric boat without a genset. The genset is added as
a documented increment, as wind is in Chapter 4.1.3. Its fuel use follows the linear
(Willans-line) fuel curve used by HOMER, `F [L/h] = F0·P_rated + F1·P_gen`, where `F0` is the
no-load fuel per rated kW and `F1` the marginal fuel (HOMER Energy, *Generator fuel curve
intercept coefficient* and *Generator fuel curve slope*, HOMER Pro/Grid documentation).

| Parameter | Value | Unit | Basis | Source type |
|---|---|---|---|---|
| Genset rated power | 6 | kW | assumption — covers the 5.25 kW high-load peak with margin | assumption |
| Genset full-load electrical efficiency | 30 | % | assumption — typical small marine diesel genset | assumption |
| Genset full-load efficiency, fraction | 0.30 | — | 30 % ÷ 100 | derived |
| Megajoules per kilowatt-hour | 3.6 | MJ/kWh | unit definition | published |
| Full-load specific fuel consumption | 0.331 | L/kWh | 3.6 ÷ (0.30 × 42.7 × 0.85); computed in code | derived |
| HOMER worked example, intercept coefficient | 0.033 | L/h/kW rated | HOMER documentation, 50 kW generator example | published |
| HOMER worked example, slope | 0.273 | L/h/kW | HOMER documentation, 50 kW generator example | published |
| No-load fuel fraction α | 0.108 | — | 0.033 ÷ (0.033 + 0.273); computed in code. Taken from a 50 kW unit: small gensets usually idle less efficiently, so α is likely **understated** and the EMS gain with it | derived |
| Genset on threshold | 1e-9 | kW | numerical tolerance: diesel above this counts as running | assumption |

`F0 = α · SFC_full` and `F1 = (1 − α) · SFC_full`, so full-load consumption equals the 30 %
efficiency exactly.

### Operating cost used by every controller (decision 2026-09-27, A2/A3)

- **Objective:** fuel cost only, at the constant price 0.85 EUR/L through the fuel curve above.
  CO₂ is reported in kg alongside the cost, never added to it.
- **Rows kept but not used in the objective:** the TOU fuel prices, the demand charge and the
  EU ETS price. A time-varying price on fuel already in the tank has no physical basis. EU
  ETS maritime and FuelEU Maritime apply from 5,000 GT, and this vessel is 5 GT. The ETS
  price may be used only as a shadow-price sensitivity.
- **Terminal valuation (A3):** every controller's cost is corrected by
  `−λ · E_nominal · (SoC_end − SoC_0)`, with `λ = π_fuel · F1 · η_d`: the diesel a stored kWh
  displaces at the margin. See `specs/072-offline-optimum-formulation.md` §4.

### Numerical parameters of the optimisers

| Parameter | Value | Unit | Basis | Source type |
|---|---|---|---|---|
| Offline optimum SoC grid | 0.001 | fraction of nominal | assumption — resolution of the dynamic programme | assumption |
| MPC SoC grid | 0.002 | fraction of nominal | assumption — twice the offline grid, so every MPC trajectory is feasible for the offline DP | assumption |
| MPC horizon | 96 | steps | 24 h ÷ 0.25 h; the duty cycle is daily | derived |
| Forecast mean of U(0,1) noise | 0.5 | — | expectation of the uniform distribution | derived |
| Request tolerance for the violation count | 1e-6 | kW | numerical tolerance | assumption |

### Reinforcement-learning controller (S6, 2026-09-27)

Design: `specs/074-rl-design.md`. The training hyperparameters are author assumptions
unless marked as the library default. None were tuned on the evaluation seeds.

| Parameter | Value | Unit | Basis | Source type |
|---|---|---|---|---|
| Genset setpoint levels (actions 1–3) | 0.5, 0.75, 1.0 | fraction of rated | assumption — the offline optimum runs the genset between 5 and 6 kW only; 0.5 and 0.75 leave room for lower settings | assumption |
| Genset setpoint, half | 0.5 | fraction of rated | as above | assumption |
| Genset setpoint, three-quarters | 0.75 | fraction of rated | as above | assumption |
| Genset setpoint, full | 1.0 | fraction of rated | as above | assumption |
| Forecast window, short | 4 | steps | 1 h | assumption |
| Forecast window, medium | 16 | steps | 4 h | assumption |
| Forecast window, long | 48 | steps | 12 h | assumption |
| Reward scale | 0.1 | EUR | brings the per-step reward to order 1 | assumption |
| Observation bound | 10 | — | outside every feature's range | assumption |
| Training episode length | 7 | days | assumption — one week of duty cycles per episode | assumption |
| Training profile seed pool, start | 1000 | — | disjoint from evaluation seeds 0–4 | assumption |
| Training profile seed pool, size | 10000 | — | as above | assumption |
| PPO training steps per policy | 1000000 | steps | assumption | assumption |
| Parallel training environments | 4 | — | assumption | assumption |
| Discount factor γ | 0.995 | — | effective horizon about 200 steps (about 2 days) | assumption |
| PPO rollout length | 2048 | steps per environment | Stable-Baselines3 default | published |
| PPO minibatch size | 256 | — | assumption | assumption |
| PPO learning rate | 0.0003 | — | Stable-Baselines3 default | published |
| PPO entropy coefficient | 0.01 | — | assumption — keeps exploring the four actions | assumption |

### Sensitivity baseline (falsifier investigation, 2026-09-27)

| Parameter | Value | Unit | Basis | Source type |
|---|---|---|---|---|
| Cycle-charging setpoint SoC | 0.80 | fraction of nominal | assumption — the genset charges the battery to near the upper bound, leaving headroom for solar surplus | assumption |

The cycle-charging controller is **not** the pre-registered baseline. It is reported
alongside it, so that the MPC gain can be split into what a better fixed rule would capture
and what only forecasting and optimisation capture.

### Reporting parameters for Chapter 9 (2026-09-27)

Choices of how results are summarised, not of how they are produced. None changes a
number in `results/comparison.csv`. Used by `src/ems/report.py` and `src/ems/figures.py`.

| Parameter | Value | Unit | Basis | Source type |
|---|---|---|---|---|
| Training-log summary window | 200 | episodes | assumption — last ~13 % of the 1496 episodes per policy; long enough for a standard error below 1 EUR | assumption |
| Training-curve rolling mean | 50 | episodes | assumption — smooths episode-to-episode scenario changes in Figure 5 | assumption |
| Trace window, start | 7 | days | assumption — after the first week, so the initial SoC of 0.50 no longer shapes the trace | assumption |
| Trace window, length | 3 | days | assumption — three duty cycles are enough to show the pattern | assumption |
| Fuel decomposition tolerance | 1e-6 | L | numerical tolerance, as for the request tolerance | assumption |

## 3. Energy balance

Power balance at every timestep, with sign convention:

```
P_pv(t) + P_wind(t) + P_batt_discharge(t) + P_diesel(t)
    = P_prop(t) + P_aux(t) + P_batt_charge(t) + P_curtail(t)
```

- Generation and load are positive.
- `P_batt` is a signed variable; the battery cannot charge and discharge in the same step.
- `P_curtail(t) >= 0` is spill from non-dispatchable renewables.
- Storage state update: `SoC(t+1) = SoC(t) + (η_c·P_ch·dt − P_dis·dt/η_d) / E_nominal`,
  with `E_nominal = 19.2 kWh` and `η_c = η_d = √0.92`. SoC is a fraction of nominal
  capacity throughout, so the 0.10–0.90 window gives the 15.36 kWh usable of Chapter 4.
  (Corrected 2026-09-27, TASK-065 D3/D4: the earlier formula divided by `E_usable` and
  applied 0.92 on each leg, which gave 12.3 kWh usable and an 84.6 % round trip.)
- Genset: `0 ≤ P_diesel(t) ≤ 6 kW`. The dispatch reduces a battery-charge request that would
  exceed this, and counts it.

## 4. Pre-registered expectation

**This section is committed before any simulation code is written.** The commit that adds it
predates the first line of `src/ems`. The git history is the evidence that the prediction was
not reverse-engineered from the result.

### Prediction

| Strategy | Expected cost vs. rule-based baseline |
|---|---|
| Rule-based baseline | reference, 0% |
| MPC | −1% to −5% |
| RL | between the baseline and the DP optimum |
| DP / MILP offline optimum | best achievable, the ceiling |

These ranges follow the values already cited in Chapter 2.3: Kanellos et al. (2016) report 1–6%,
Antonopoulos et al. (2021) report about 3.5% for MPC over rule-based, and Wu et al. (2020) reach
96.9% of the DP optimum with RL.

### Expected decomposition

The achievable EMS gain is bounded multiplicatively:

```
gain  ≈  renewable share  ×  dispatch flexibility  ×  forecast quality
```

Each factor is measured by the simulator. This is why a small result is expected and is not a
failure: renewable penetration on the reference plant is low, so the schedulable energy is a
small fraction of total energy.

### Falsifier

> **If MPC improves operating cost by more than 10%, or fails to beat the rule-based
> baseline, the defect is in the model or the baseline — not in the result. Investigate before
> reporting.**

An MPC that "wins" by an implausible margin is evidence of a broken baseline, an unphysical
battery, or a leak in the energy balance. A result that loses is a legitimate finding.

### Finding recorded 2026-09-27 — not an amendment

Recorded before any MPC or RL result existed. The prediction and falsifier above are
**unchanged**. While formulating the optimisers (`specs/072-offline-optimum-formulation.md`
§6), two structural facts were established:

1. With a constant fuel price and a genset whose fuel is proportional to output, the
   rule-based baseline is provably optimal. No controller can beat it.
2. With the Willans genset curve chosen above (decision A2), the baseline runs the genset at
   low load for many hours and pays the no-load fuel each time. The offline ceiling found in
   the scratch probe (illustrative parameters) was 21–45 %. MPC could therefore exceed the 10 %
   falsifier threshold for a structural reason (a load-following baseline on an
   idle-penalised genset), not because of a model or baseline defect.

Whether section 4 should be amended in light of (2) is the author's decision (A4). Until then,
any MPC result above 10 % is reported as found, together with the investigation that the
falsifier requires.

### RL gate

RL is evaluated on **held-out seeds** — profiles the policy never trained on. The number is
reported even if RL loses the baseline. Losing is a valid outcome; hiding it is not.

## 5. Acceptance criteria

Each row is a gate. All must pass before the next stage opens.

| Gate | Criterion |
|---|---|
| G1 Energy balance | `|ΣP_gen − ΣP_load| < 1e-6` kWh over every simulation day |
| G2 Storage bounds | `0.10 <= SoC <= 0.90` for every timestep, every run |
| G3 Determinism | two runs with the same seed produce byte-identical output |
| G4 Baseline feasibility | rule-based baseline never violates G1 or G2 |
| G5 Optimality order | `cost(DP) <= cost(MPC) <= cost(rule-based)` |
| G6 Pareto honesty | MPC cost is reported with its constraint-violation count, not alone |
| G7 Reproducibility | every figure regenerates from one command with no manual step |
| G8 Provenance | no numeric literal in code lacks a row in section 2 |
| G9 No over-claim | every number in Results traces to a file in `results/` with a seed and commit |

## 6. Out of scope

- Network protection, short-circuit analysis, harmonics, EMI
- Hull hydrodynamics beyond a resistance-versus-speed table
- Navigation, collision avoidance, autopilot
- Any hardware construction, wiring, or measurement
- Certification and class approval
