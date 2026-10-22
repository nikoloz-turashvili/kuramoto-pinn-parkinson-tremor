# Physics-Informed Neural Modeling of Kuramoto Oscillators

From surrogate modeling to hybrid and advanced hybrid models for Parkinsonian tremor dynamics.

**Author:** Nikoloz Turashvili, Kutaisi International University (KIU)
**Course:** Advanced Topics in Numerical Mathematics (Projects #1, #2, #3)

## Overview

This repository contains the code for a three-stage project built around the Kuramoto model of coupled phase oscillators:

| Part | Topic | File |
|------|-------|------|
| I | Neural surrogate of the Kuramoto flow map `Φ_T : θ(0) → θ(T)` | `3.py` |
| II | Hybrid model: Kuramoto physics + residual neural network (ALAMEDA PD tremor data) | `integrate_model.py` |
| III | Advanced hybrid model: physics + residual network + state-dependent coupling correction `ΔK(θ)` | `integrate_model.py` |

## Part I: Surrogate model (`3.py`)

- Kuramoto system with N = 10 oscillators, K = 2.0, σ_ω = 0.8, T = 5.0.
- Training data generated with `scipy.integrate.solve_ivp` (Dormand–Prince RK45).
- Phases are embedded as (cos θ, sin θ) to avoid 2π wrapping.
- Physics-informed loss: MSE + order-parameter consistency + phase-variance preservation.
- Early stopping, plots of training history, phase distributions and order parameter r(t).

Run:

```bash
python surrogate_expB.py
```

Reported result (test set): mean angular error 0.299 rad (17.16°), order-parameter MAE 0.0080.

## Parts II and III: Hybrid and advanced hybrid models (`integrate_model.py`)

Pipeline:

1. Load the ALAMEDA PD tremor dataset and reduce it with PCA to N oscillators.
2. Extract phases with the Hilbert transform and unwrap them.
3. Estimate natural frequencies ω_i (median phase velocity) and a nominal coupling K.
4. Train `ResidualNN` on the residual `θ̇_data − f_Kuramoto(θ)`  (hybrid model).
5. Freeze the residual network and train `CouplingNN` to learn `ΔK(θ)` (advanced model).
6. Evaluate test MSE of physics-only, hybrid and advanced models, and compare rollouts and the order parameter r(t).

Model:

```
dθ_i/dt = ω_i + (K + ΔK(θ)) / N · Σ_j sin(θ_j − θ_i) + r_i(θ)
```

Run:

```bash
python integrate_model.py
```

### Data

The dataset is **not included** in this repository. Download the ALAMEDA PD tremor dataset separately and place it in the project folder as:

```
ALAMEDA_PD_tremor_dataset.csv
```

(or change `csv_path` in `main()`).

### Notes and limitations

- The time step is fixed to `dt = 0.01` by assumption. The dataset consists of feature rows (4151 samples, 99 features), not a raw continuous signal, so "time" is a sample-index scale and not measured seconds.
- Reported improvements of the coupling network over the hybrid model are small (0.37% for N = 5, 0.13% for N = 25) and were obtained with a single seed.

## Requirements

```bash
pip install -r requirements.txt
```

## Paper

The accompanying report is in `paper/Kuramoto_Surrogate_Hybrid_AdvancedHybrid_Paper.pdf` (if included).

## References

1. Acebrón et al., *The Kuramoto model: A simple paradigm for synchronization phenomena*, Rev. Mod. Phys. 77 (2005).
2. Strogatz, *From Kuramoto to Crawford*, Physica D 143 (2000).
3. Raissi, Perdikaris, Karniadakis, *Physics-informed neural networks*, J. Comput. Phys. 378 (2019).
