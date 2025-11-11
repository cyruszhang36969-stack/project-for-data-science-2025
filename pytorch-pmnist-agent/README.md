# 🏆 ML-Arena PyTorch Agent: Permuted MNIST (High-Performance Ensemble)

This repository contains the advanced PyTorch Agent developed for the ML-Arena 1-Minute Permuted MNIST competition, focusing on maximizing accuracy (targeting >0.99) under strict 60-second time and 4GB memory constraints.

The solution employs **Ensemble Learning**, **Principal Component Analysis (PCA)** for aggressive dimensionality reduction, and a **Time-Adaptive Training Strategy** to prevent competition timeout.

---

## 🚀 1. Installation

### Prerequisites

* Python 3.8+
* Git

### Setup Steps

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install dependencies** (`requirements.txt`):
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install the package in editable mode** (required to run `experiments`):
    ```bash
    pip install -e .
    ```

---

## 🛠️ 2. Usage and Benchmarking

The core benchmarking logic is located in the `experiments` module.

### Run Full Benchmark Comparison

Execute the unified benchmark script to compare the **Optimized Agent** against the **Baseline Runner**, tracking time, accuracy, and peak memory usage.

```bash
# Must be run from the repository root (your_repo/)
python -m experiments.run_benchmarks
