# your_repo/experiments/run_benchmarks.py
"""
Unified Benchmark Runner: Compares BaselineRunner vs. OptimizedAgent.

This script executes the training and prediction phases for both agents, 
tracks performance metrics (time, accuracy), and outputs a formatted table.
This script replaces the old evaluation/benchmark.py and uses the new modular structure.
"""
import time
import numpy as np
import torch
import gc
import json
import os
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from tabulate import tabulate

# 导入依赖：Agent Runner
from .baseline import BaselineRunner
from ..models.ensemble import OptimizedAgent 
# 导入依赖：Monitoring
from ..utils.monitoring import get_max_memory_usage_mb

# --- 配置 ---
N_TRAIN = 60000
N_TEST = 10000
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

def load_data():
    """Loads and splits the MNIST data."""
    print("Loading MNIST data (N=70000)...")
    try:
        mn = fetch_openml('mnist_784', version=1, parser='auto')
        X = mn.data.values.astype(np.float32).reshape(-1, 784)
        y = mn.target.astype(np.int64).values
    except Exception as e:
        print(f"Failed to load MNIST: {e}")
        return None, None, None, None

    X_train, X_test = X[:N_TRAIN], X[N_TRAIN:N_TRAIN + N_TEST]
    y_train, y_test = y[:N_TRAIN], y[N_TRAIN:N_TRAIN + N_TEST]
    print(f"Data loaded: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    return X_train, y_train, X_test, y_test

def track_agent_performance(agent_class, X_train, y_train, X_test, y_test, name):
    """Initializes, trains, predicts, and records metrics for a given agent."""
    print(f"\n--- Starting Evaluation for {name} ---")
    
    agent = agent_class()
    metrics = {'Agent': name}

    # 1. Training Phase
    t_train_start = time.time()
    try:
        agent.train(X_train, y_train)
        metrics['Train Time (s)'] = agent.metrics.get('train_time_s', round(time.time() - t_train_start, 2))
    except RuntimeError as e:
        print(f"[{name}] Training failed: {e}")
        metrics.update({'Accuracy': 'N/A', 'Train Time (s)': round(time.time() - t_train_start, 2), 
                        'Predict Time (s)': 'N/A', 'Models': 0})
        return metrics

    # 2. Prediction Phase
    # Note: BaselineRunner.predict returns (preds, time), OptimizedAgent.predict returns preds
    if name.startswith("Baseline"):
        preds, predict_time = agent.predict(X_test)
        metrics['Predict Time (s)'] = round(predict_time, 2)
    else: # Optimized Agent logic
        t_predict_start = time.time()
        preds = agent.predict(X_test)
        predict_time = time.time() - t_predict_start
        metrics['Predict Time (s)'] = round(predict_time, 2)


    # 3. Accuracy and Model Count
    metrics['Accuracy'] = round(accuracy_score(y_test, preds), 4)
    metrics['Models'] = len(getattr(agent, 'ensemble', [1])) # Count ensemble models or default to 1

    # 4. Resource Usage (Approximate Peak Memory)
    # Note: get_max_memory_usage_mb() tracks peak usage *since process start*.
    # We run it after training/prediction to capture overall peak.
    metrics['Peak Memory (MB)'] = round(get_max_memory_usage_mb(), 2) 
    
    # Cleanup
    del agent, preds
    gc.collect()

    return metrics

def run_all_benchmarks():
    """Main function to run all experiments."""
    X_train, y_train, X_test, y_test = load_data()
    if X_train is None:
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    results = []

    # 1. Evaluate Baseline Agent
    baseline_result = track_agent_performance(BaselineRunner, X_train, y_train, X_test, y_test, 
                                              "Baseline Agent (Shallow MLP)")
    results.append(baseline_result)
    
    # 2. Evaluate Optimized Agent
    optimized_result = track_agent_performance(OptimizedAgent, X_train, y_train, X_test, y_test, 
                                                "Optimized Agent (PCA+Deep+Ensemble)")
    results.append(optimized_result)

    # --- 结果记录与展示 ---
    
    # Save raw results for report.ipynb reproducibility
    with open(os.path.join(RESULTS_DIR, 'final_benchmarks.json'), 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {RESULTS_DIR}/final_benchmarks.json")


    print("\n\n#####################################################")
    print("##             ML-Arena Agent Benchmark            ##")
    print("#####################################################")
    
    # 格式化输出表格 (用于复制到报告中)
    headers = ["Agent", "Accuracy", "Train Time (s)", "Predict Time (s)", "Models", "Peak Memory (MB)"]
    table = [[r.get(h, 'N/A') for h in headers] for r in results]
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
    
    print("#####################################################")

if __name__ == "__main__":
    # To run: python -m experiments.run_benchmarks
    run_all_benchmarks()
