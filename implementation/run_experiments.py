"""
Main experiment runner.
Trains XGBoost + DQN, runs 4 scenarios x 4 methods,
collects KPIs, saves results to JSON/CSV, generates figures.
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from environment import ManufacturingEnvironment
from ml_models import train_predictor, FEATURE_COLS, generate_sensor_dataset
from rl_agent import train_dqn, _build_state_vec
from agents import GlobalOptimizationAgent, MessageBus
from baselines import RuleBasedController, GeneticAlgorithmScheduler, CentralizedController

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)

SCENARIOS = ["normal", "machine_failure", "demand_surge", "resource_shortage"]
N_JOBS = 150
SIM_DUR = 1440.0   # 24 h in minutes

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────
# Step 1 – Train ML models
# ─────────────────────────────────────────────

print("=" * 60)
print("STEP 1 – Training XGBoost failure predictor")
print("=" * 60)
predictor, feat_names, df_sensor, lstm = train_predictor(n_samples=8000)
xgb_metrics = predictor.metrics

# Feature importance table
fi_df = pd.DataFrame({
    "feature": feat_names,
    "importance": predictor.feature_importances_
}).sort_values("importance", ascending=False)
fi_df.to_csv(os.path.join(RESULTS_DIR, "xgb_feature_importance.csv"), index=False)
print(fi_df.to_string(index=False))


# ─────────────────────────────────────────────
# Step 2 – Train DQN
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2 – Training DQN Resource Allocation Agent")
print("=" * 60)
dqn_agent = train_dqn(
    lambda: ManufacturingEnvironment(n_jobs_total=N_JOBS, sim_duration=SIM_DUR),
    n_episodes=25,
)
dqn_rewards = dqn_agent.rewards_history


# ─────────────────────────────────────────────
# Step 3 – Run all scenarios x methods
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3 – Running 4 scenarios × 4 methods")
print("=" * 60)

methods = {
    "Centralized Control": CentralizedController(),
    "Rule-Based (EDD)": RuleBasedController(),
    "Genetic Algorithm": GeneticAlgorithmScheduler(pop_size=30, generations=40),
    "Proposed MAS+XGB+DQN": None,   # handled separately
}

all_results = {}   # method -> scenario -> final_kpi

for method_name, controller in methods.items():
    all_results[method_name] = {}
    for scenario in SCENARIOS:
        env = ManufacturingEnvironment(n_jobs_total=N_JOBS, sim_duration=SIM_DUR)

        if method_name == "Proposed MAS+XGB+DQN":
            bus = MessageBus()
            agent = GlobalOptimizationAgent(env, bus, rl_policy=dqn_agent, predictor=predictor)
            kpis = agent.run(scenario=scenario)
        else:
            kpis = controller.run(env, scenario=scenario)

        final = kpis[-1] if kpis else {}
        all_results[method_name][scenario] = final
        print(f"  [{method_name:<28}] [{scenario:<20}] "
              f"Jobs={final.get('jobs_completed',0):3d} "
              f"Util={final.get('avg_utilization',0)*100:.1f}% "
              f"Tard={final.get('avg_tardiness',0):.1f}min "
              f"Energy={final.get('total_energy_kwh',0):.1f}kWh "
              f"Fail={final.get('machine_failures',0)}")


# ─────────────────────────────────────────────
# Step 4 – Statistical analysis & table building
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4 – Aggregating results")
print("=" * 60)

records = []
for mname, scenarios in all_results.items():
    for sc, kpi in scenarios.items():
        records.append({
            "method": mname,
            "scenario": sc,
            "jobs_completed": kpi.get("jobs_completed", 0),
            "avg_utilization_pct": round(kpi.get("avg_utilization", 0) * 100, 2),
            "avg_tardiness_min": round(kpi.get("avg_tardiness", 0), 2),
            "total_energy_kwh": round(kpi.get("total_energy_kwh", 0), 2),
            "machine_failures": kpi.get("machine_failures", 0),
            "makespan_min": SIM_DUR,   # normalized to window
        })

df_results = pd.DataFrame(records)
df_results.to_csv(os.path.join(RESULTS_DIR, "experiment_results.csv"), index=False)
print(df_results.to_string(index=False))


# Summary table (averaged across scenarios)
summary = df_results.groupby("method").agg({
    "jobs_completed":      "mean",
    "avg_utilization_pct": "mean",
    "avg_tardiness_min":   "mean",
    "total_energy_kwh":    "mean",
    "machine_failures":    "mean",
}).round(2)

# Compute improvement vs Centralized Control baseline
baseline = summary.loc["Centralized Control"]
summary["util_gain_pct"] = ((summary["avg_utilization_pct"] - baseline["avg_utilization_pct"])
                             / baseline["avg_utilization_pct"] * 100).round(1)
summary["tard_reduction_pct"] = ((baseline["avg_tardiness_min"] - summary["avg_tardiness_min"])
                                  / max(baseline["avg_tardiness_min"], 1) * 100).round(1)
summary["energy_saving_pct"] = ((baseline["total_energy_kwh"] - summary["total_energy_kwh"])
                                  / max(baseline["total_energy_kwh"], 1) * 100).round(1)

print("\n=== Summary (avg across 4 scenarios) ===")
print(summary.to_string())
summary.to_csv(os.path.join(RESULTS_DIR, "summary_table.csv"))


# ─────────────────────────────────────────────
# Step 5 – Save all metrics to JSON for paper
# ─────────────────────────────────────────────

paper_metrics = {
    "xgboost": {
        "accuracy":  round(xgb_metrics["accuracy"],  4),
        "precision": round(xgb_metrics["precision"], 4),
        "recall":    round(xgb_metrics["recall"],    4),
        "f1":        round(xgb_metrics["f1"],        4),
        "roc_auc":   round(xgb_metrics["roc_auc"],   4),
        "cv_f1_5fold": round(xgb_metrics["cv_f1"],   4),
        "confusion_matrix": xgb_metrics["confusion"],
        "n_training_samples": 8000,
    },
    "dqn": {
        "n_episodes": 25,
        "final_epsilon": round(dqn_agent.epsilon, 4),
        "avg_reward_last5": round(float(np.mean(dqn_rewards[-5:])), 2),
        "best_reward": round(float(np.max(dqn_rewards)), 2),
    },
    "scenarios": {
        sc: {
            mname: all_results[mname][sc]
            for mname in all_results
        }
        for sc in SCENARIOS
    },
    "summary": summary.reset_index().to_dict(orient="records"),
}

with open(os.path.join(RESULTS_DIR, "paper_metrics.json"), "w") as f:
    json.dump(paper_metrics, f, indent=2, default=float)

print(f"\nAll results saved to {RESULTS_DIR}/")
print("Done.")
