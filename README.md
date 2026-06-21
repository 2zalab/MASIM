# MASIM — Multi-Agent System for Intelligent Manufacturing

> **A Multi-Agent Intelligent Framework for Real-Time Manufacturing Process Optimization:
> Integrating XGBoost, Deep Reinforcement Learning, and Contract Net Protocol in Smart Factories**
>
> *Submitted to: Journal of Intelligent Manufacturing (Q1 — Springer)*
> *Author: Touza Isaac — Université de Maroua, Cameroun*

---

## Overview

MASIM is a four-layer hierarchical multi-agent framework designed for real-time
optimization of flexible job-shop manufacturing environments. It integrates three
complementary AI paradigms:

| Component | Role | Algorithm |
|-----------|------|-----------|
| Predictive Maintenance | Detect imminent machine failures from sensor streams | XGBoost + rolling-window features |
| Resource Allocation | Learn optimal job-to-machine assignment policies | Double Deep Q-Network (DDQN) |
| Distributed Coordination | Conflict-free resource commitment without central arbitration | Contract Net Protocol (CNP) |

The framework is evaluated on a simulated flexible job-shop (6 CNC machines, 4 disruption
scenarios) and validated with two real industrial datasets.

---

## Repository Structure

```
MASIM/
├── dataset/
│   ├── predictive_maintenance_dataset.csv   # 124,494 sensor readings, 1,169 devices
│   └── uci-secom.csv                        # 1,567 wafer measurements, 591 features
│
├── implementation/
│   ├── environment.py           # Flexible job-shop simulator (Weibull failure model)
│   ├── agents.py                # MAS agents (Machine, Maintenance, Resource, Production, Global)
│   ├── ml_models.py             # XGBoost predictor + LSTM anomaly detector
│   ├── rl_agent.py              # Double DQN agent + replay buffer
│   ├── baselines.py             # Centralized Control, EDD, Genetic Algorithm
│   └── real_data_experiments.py # Full experiment pipeline on real datasets
```

---

## Architecture

MASIM is organized as a **four-layer hierarchy**:

```
Layer 4 — Decision        [ Global Optimization Agent (DDQN) ]
                                        ↑
Layer 3 — Coordination    [ Production Agent ] [ Maintenance Agent ] [ Resource Agent ]
                                        ↑
Layer 2 — Local Agents    [ M1 (XGB) ] [ M2 (XGB) ] [ M3 (XGB) ] [ M4 (XGB) ] [ M5 (XGB) ]
                                        ↑
Layer 1 — Physical        [ CNC#1 ]    [ CNC#2 ]    [ CNC#3 ]    [ CNC#4 ]    [ CNC#5 ]
```

- **Machine Agents** embed an XGBoost failure predictor per CNC unit
- **Resource Agent** runs the Contract Net Protocol (CFP → BID → ASSIGN)
- **Production Agent** applies ATCS (Apparent Tardiness Cost with Setups) sequencing
- **Global Agent** executes the DDQN policy and monitors KPIs

---

## Datasets

### 1. Predictive Maintenance Dataset
- **Source**: Kaggle / Microsoft Azure ML
- **Size**: 124,494 daily sensor readings · 1,169 industrial devices
- **Features**: 9 raw sensor metrics (metric1–metric9) + date + device
- **Label**: binary failure (0/1)
- **Failure rate**: 0.085% (106 failure events) — extreme class imbalance
- **Challenge**: temporal structure, near-zero positive rate

### 2. UCI SECOM Semiconductor Dataset
- **Source**: UCI Machine Learning Repository
- **Size**: 1,567 wafer production runs · 591 process control parameters
- **Label**: Pass / Fail (6.6% defect rate)
- **Challenge**: high dimensionality, up to 40% missing values per feature

---

## Installation

```bash
# Clone the repository
git clone https://github.com/2zalab/Malaria---Paper.git
cd Malaria---Paper/MAS-Manufacturing-Paper

# Install dependencies
pip install xgboost imbalanced-learn scikit-learn matplotlib seaborn pandas numpy
```

**Tested with:**
```
Python        >= 3.9
xgboost       >= 1.7
imbalanced-learn >= 0.11
scikit-learn  >= 1.3
pandas        >= 2.0
numpy         >= 1.24
matplotlib    >= 3.7
seaborn       >= 0.12
```

---

## Usage

### Run the full experiment pipeline (real datasets)

```bash
cd implementation/
python real_data_experiments.py
```

This script:
1. Loads and preprocesses both real datasets
2. Engineers 39 temporal features for the PM dataset (lag-7 rolling mean/std/diff per sensor)
3. Applies SMOTE oversampling on the training split only (no leakage)
4. Trains XGBoost with PR-curve optimal threshold calibration
5. Runs 5-fold cross-validation on SECOM
6. Generates 8 publication-quality figures → `paper/figures/`
7. Saves all metrics → `paper/figures/paper_metrics_real.json`

**Expected runtime**: ~5–10 minutes on a standard laptop (CPU only).

---

## Key Results

### XGBoost Failure Prediction (Real Data)

| Metric | PM Dataset (test) | UCI SECOM (5-fold CV) |
|--------|------------------|----------------------|
| Accuracy | 0.9969 | 0.8503 |
| Recall | 0.2000 | 0.4762 |
| ROC-AUC | **0.7345** | **0.6787** |
| Avg. Precision | 0.0188 | 0.1628 |
| CV-F1 | — | 0.2791 |

> **Note on F1**: The low F1 on the PM dataset (0.094) is structurally
> inevitable at a 0.085% failure rate — ROC-AUC is the correct metric here.
> Both results are consistent with published benchmarks on these datasets.

### Scheduling KPIs (Simulation, average over 4 scenarios)

| Method | Jobs Completed | Energy (kWh) | Mean Tardiness (min) |
|--------|---------------|-------------|----------------------|
| Centralized Control | 33.25 | 255.74 | 131.59 |
| Rule-Based EDD | 28.25 | 234.65 | 162.44 |
| Genetic Algorithm | 34.25 | 260.62 | 63.70 |
| **MASIM (proposed)** | **36.75 (+10.5%)** | **219.67 (−14.1%)** | 126.66 |

MASIM achieves the highest throughput and lowest energy across all scenarios.

---

## Feature Engineering

For the PM dataset, 9 raw sensor metrics are enriched with:

```python
# Per-device rolling window features (lag-7)
for col in sensor_cols:
    df[f"{col}_roll7_mean"] = df.groupby("device")[col].transform(
        lambda x: x.rolling(7, min_periods=1).mean())
    df[f"{col}_roll7_std"]  = df.groupby("device")[col].transform(
        lambda x: x.rolling(7, min_periods=1).std().fillna(0))
    df[f"{col}_diff1"]      = df.groupby("device")[col].transform(
        lambda x: x.diff().fillna(0))

# Cross-sensor interactions
df["m1xm2"] = df["metric1"] * df["metric2"]
df["m5xm6"] = df["metric5"] * df["metric6"]
df["metric_sum"] = df[sensor_cols].sum(axis=1)
# Total: 39 features
```

Optimal classification threshold is selected via **PR-curve F1 maximisation**
instead of the default 0.5, critical for extreme imbalance settings.

---

## Author

**Touza Isaac**
- Department of Mathematics and Computer Science, University of Maroua, Cameroon
- Laboratoire de Recherche en Informatique, Université de Maroua
- ✉ isaac_touza@outlook.fr

---

## License

This repository is made available for academic and research purposes.
© 2025 Touza Isaac — Université de Maroua, Cameroun.
```
