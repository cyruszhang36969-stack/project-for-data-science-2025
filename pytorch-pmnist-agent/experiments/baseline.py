# your_repo/experiments/baseline.py
"""
Baseline Agent Experiment — Simple PyTorch MLP.
This script runs the baseline model, which lacks PCA, ensemble, and adaptive training,
for performance comparison against the OptimizedAgent.
It now integrates monitoring utilities for detailed time tracking.
"""
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import gc

# 导入依赖：从 models.mlp 导入 MLP 结构
from ..models.mlp import MLP
# 导入依赖：从 utils.optimization 导入预处理函数
from ..utils.optimization import standardize_torch 
# 导入依赖：从 utils.monitoring 导入计时函数
from ..utils.monitoring import time_it, get_max_memory_usage_mb

# --------- Baseline 超参 ---------
BASE_DEVICE = 'cpu'
BASE_BATCH_SIZE = 64
BASE_EPOCHS = 3
BASE_LR = 0.01
BASE_HIDDEN_DIMS = [256, 128] 
BASE_INPUT_DIM = 784 
BASE_SEED = 100

class BaselineRunner:
    """
    一个简单的 PyTorch MLP 基线 Agent，用于运行和记录实验。
    已集成时间追踪。
    """
    def __init__(self, device=BASE_DEVICE):
        self.device = torch.device(device)
        self.model = None
        self.mean = None
        self.std = None
        self.is_trained = False
        self.metrics = {}

    @time_it
    def _run_train_loop(self, X_std, y):
        """内部函数：执行核心训练循环，由 time_it 装饰器计时。"""
        input_dim = X_std.shape[1]
        self.model = MLP(input_dim=input_dim, hidden_dims=BASE_HIDDEN_DIMS).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=BASE_LR)
        
        self.model.train()
        n_samples = X_std.shape[0]
        
        for epoch in range(BASE_EPOCHS):
            perm = torch.randperm(n_samples, device=self.device)
            X_epoch = X_std[perm]
            y_epoch = y[perm]
            
            for start in range(0, n_samples, BASE_BATCH_SIZE):
                end = min(start + BASE_BATCH_SIZE, n_samples)
                xb = X_epoch[start:end]
                yb = y_epoch[start:end]

                optimizer.zero_grad()
                out = self.model(xb)
                loss = F.cross_entropy(out, yb)
                loss.backward()
                optimizer.step()
        
        # 返回模型和参数到 CPU
        self.model.cpu()
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


    def train(self, X_train, y_train):
        torch.manual_seed(BASE_SEED)
        np.random.seed(BASE_SEED)
        
        # 1. Data Preparation
        if X_train.ndim == 3:
            N = X_train.shape[0]
            X_train = X_train.reshape(N, -1)
            
        X = torch.from_numpy(X_train.astype(np.float32) / 255.0).to(self.device)
        y = torch.from_numpy(y_train.squeeze().astype(np.int64)).to(self.device)

        # Standardize (must store mean/std for prediction)
        X_std, self.mean, self.std = standardize_torch(X)
        
        # 2. Run Training with Time Tracking
        # _run_train_loop 返回 (None, total_time)
        _, total_time = self._run_train_loop(X_std, y)
        
        self.is_trained = True
        self.metrics['train_time_s'] = round(total_time, 2)
        # Note: Max memory tracking is usually done outside the function call 
        # or integrated into the system-level benchmark script.
        # self.metrics['peak_memory_mb'] = get_max_memory_usage_mb() 
        
        print(f"Baseline training time: {total_time:.2f}s")
        gc.collect()

    @time_it
    def predict(self, X_test):
        if not self.is_trained:
            raise RuntimeError("Agent must be trained before predict()")

        if X_test.ndim == 3:
            X_test = X_test.reshape(X_test.shape[0], -1)
            
        Xt = torch.from_numpy(X_test.astype(np.float32) / 255.0).to('cpu')
        
        # Standardize using stored mean/std
        Xs = (Xt - self.mean) / self.std

        self.model.eval()
        batch = 2048
        preds_probs = []
        with torch.no_grad():
            for start in range(0, Xs.shape[0], batch):
                end = min(start + batch, Xs.shape[0])
                xb = Xs[start:end]
                out = self.model(xb)
                probs = torch.softmax(out, dim=1).cpu().numpy()
                preds_probs.append(probs)
                
        probs_arr = np.vstack(preds_probs)
        preds = probs_arr.argmax(axis=1).astype(np.int64)
        
        # Note: predict will return (preds, total_time) because of @time_it
        return preds

# --- 实验执行入口 (用于本地测试) ---
if __name__ == '__main__':
    from sklearn.datasets import fetch_openml
    from sklearn.metrics import accuracy_score
    
    print("Running Baseline Experiment...")
    
    mn = fetch_openml('mnist_784', version=1, parser='auto')
    X = mn.data.values.astype(np.float32).reshape(-1, 784)
    y = mn.target.astype(np.int64).values
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    
    runner = BaselineRunner()
    runner.train(X_train, y_train)
    
    preds, predict_time = runner.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    runner.metrics['accuracy'] = round(acc, 4)
    runner.metrics['predict_time_s'] = round(predict_time, 2)
    
    print("\n--- Baseline Results ---")
    print(f"Accuracy: {runner.metrics['accuracy']}")
    print(f"Train Time: {runner.metrics['train_time_s']}s")
    print(f"Predict Time: {runner.metrics['predict_time_s']}s")
