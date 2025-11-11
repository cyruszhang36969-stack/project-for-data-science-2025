# pytorch_pmnist_agent/agent.py
"""
强化版 PyTorch Agent — 面向 ML-Arena 1-Minute Permuted MNIST
目标：在 4GB 内存预算、2 核 CPU 的条件下尽量提升准确率（0.984 附近），并保证不超时（<60s）。
实现要点：
- 完全用 PyTorch（不依赖 sklearn）
- PCA 使用 torch.pca_lowrank（优先）或 SVD 回退
- 深度 MLP (512,384,256)
- n_models = 4（ensemble）
- 动态训练时间控制（train_time_limit 秒）
注意：本文件已重构，辅助函数 (PCA/Standardize/MLP) 已移至 .utils 模块。
"""
import time
import sys
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 从本地包导入辅助函数和MLP类
from .utils import MLP, compute_pca_torch, standardize_torch

# --------- 可调超参（可根据环境微调） ----------
PCA_TARGET = 400          # 目标 PCA 主成分数（激进版）
N_MODELS = 4              # ensemble 模型数量
TRAIN_SAMPLE_MAX = 60000  # 使用全部训练样本
TRAIN_TIME_LIMIT = 50.0   # 训练上限（秒），保留 ~10s 给预测与环境开销
DEVICE = 'cpu'            # 本竞赛以 CPU 为主；若 GPU 可用，可改为 'cuda'
BATCH_BASE = 512          # 基础 batch size（会基于剩余时间自适应）
HIDDEN_DIMS = [512, 384, 256]
LR = 0.006
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 8            # 最大 epoch（会自适应）
SEED_BASE = 42

# --------- Agent 实现 ---------
class Agent:
    def __init__(self, device=DEVICE):
        # 每个子模型会保存： model, mean, std, X_mean (for PCA), components
        self.ensemble = []
        self.device = torch.device(device)
        self.is_trained = False
        self.train_time_limit = TRAIN_TIME_LIMIT
        self.pca_target = PCA_TARGET
        self.n_models = N_MODELS
        self.hidden_dims = HIDDEN_DIMS # 确保 MLP 知道网络结构

    def train(self, X_train, y_train):
        """
        X_train: np.ndarray (N,28,28) or (N,784)
        y_train: np.ndarray (N,) or (N,1)
        """
        t0 = time.time()
        torch.manual_seed(SEED_BASE)
        np.random.seed(SEED_BASE)

        # as torch tensors (float32)
        if X_train.ndim == 3:
            N = X_train.shape[0]
            X_train = X_train.reshape(N, -1)
        # / 255.0 is included here
        X = torch.from_numpy(X_train.astype(np.float32) / 255.0).to(self.device)  # (N,784)
        y = torch.from_numpy(y_train.squeeze().astype(np.int64)).to(self.device)

        n_samples = min(TRAIN_SAMPLE_MAX, X.shape[0])
        X = X[:n_samples]
        y = y[:n_samples]

        self.ensemble = []
        # try to estimate memory headroom: free Python gc
        gc.collect()

        for i in range(self.n_models):
            model_start = time.time()
            seed = SEED_BASE + i * 101
            torch.manual_seed(seed)
            # 使用随机子集/排列，增加模型的差异性 (Permuted indexes)
            idx = torch.randperm(X.size(0), device=self.device)
            Xs = X[idx]
            ys = y[idx]

            # determine remaining time and adapt PCA / batch / epoch
            elapsed = time.time() - t0
            remaining = max(0.0, self.train_time_limit - elapsed)
            # leave ~6-8s for predict overhead
            target_train_time = max(5.0, remaining - 6.0)

            # heuristic: adapt hyperparameters based on remaining time
            if target_train_time > 25:
                pca_dim = self.pca_target
                batch_size = BATCH_BASE
                epochs = MAX_EPOCHS
            elif target_train_time > 12:
                pca_dim = max(200, int(self.pca_target * 0.6))
                batch_size = max(256, int(BATCH_BASE * 0.8))
                epochs = max(2, int(MAX_EPOCHS * 0.7))
            else:
                pca_dim = max(100, int(self.pca_target * 0.4))
                batch_size = max(128, int(BATCH_BASE * 0.5))
                epochs = 1

            # compute standardization & PCA on Xs (training subset)
            Xs_std, mean, std = standardize_torch(Xs)
            
            # --- PCA Calculation with Fallback ---
            try:
                Xs_pca, X_mean, components = compute_pca_torch(Xs_std, n_components=pca_dim)
            except RuntimeError as e:
                # fallback: reduce pca_dim until success
                success = False
                pd = pca_dim
                while pd >= 80 and not success:
                    try:
                        Xs_pca, X_mean, components = compute_pca_torch(Xs_std, n_components=pd)
                        success = True
                    except Exception:
                        pd = int(pd * 0.8)
                if not success:
                    raise RuntimeError("PCA computation failed even after reducing dimensions.")
                pca_dim = pd
            # --- End PCA Calculation ---

            # free some memory: del Xs_std
            del Xs_std
            gc.collect()

            input_dim = Xs_pca.shape[1]
            # Use the imported MLP class
            model = MLP(input_dim=input_dim, hidden_dims=self.hidden_dims).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            # scheduler: cosine warm restart-like simple decay per epoch
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

            # train loop (mini-batch)
            model.train()
            n_samples_pca = Xs_pca.shape[0]
            
            current_epoch = 0
            for epoch in range(epochs):
                perm = torch.randperm(n_samples_pca, device=self.device)
                X_epoch = Xs_pca[perm]
                y_epoch = ys[perm]
                for start in range(0, n_samples_pca, batch_size):
                    end = min(start + batch_size, n_samples_pca)
                    xb = X_epoch[start:end]
                    yb = y_epoch[start:end]

                    optimizer.zero_grad()
                    out = model(xb)
                    loss = nn.functional.cross_entropy(out, yb)
                    loss.backward()
                    optimizer.step()

                    # lightweight time check per batch
                    if (time.time() - t0) + 6.0 > self.train_time_limit:
                        break

                scheduler.step()
                current_epoch = epoch + 1
                # per-epoch time check
                if (time.time() - t0) + 6.0 > self.train_time_limit:
                    break

            # save model + preprocessing params
            self.ensemble.append({
                'model': model.cpu(),         # move to CPU to reduce resource tie-up
                'mean': mean.cpu(),
                'std': std.cpu(),
                'X_mean': X_mean.cpu(),
                'components': components.cpu(),
                'pca_dim': input_dim
            })

            # free large tensors
            del Xs_pca, X_epoch, y_epoch
            gc.collect()

            model_time = time.time() - model_start
            print(f"Model {i+1}/{self.n_models} trained in {model_time:.2f}s (pca={pca_dim}, batch={batch_size}, epochs={current_epoch})")
            # check total elapsed
            if (time.time() - t0) + 6.0 > self.train_time_limit:
                print("Approaching train time limit, stop building more models")
                break

        self.is_trained = True
        total_time = time.time() - t0
        print(f"Total training time: {total_time:.2f}s, models built: {len(self.ensemble)}")

        # free training tensors
        del X, y
        gc.collect()

    def predict(self, X_test):
        """
        X_test: np.ndarray (N,28,28) or (N,784)
        返回 np.ndarray (N,)
        """
        if not self.is_trained:
            raise RuntimeError("Agent must be trained before predict()")

        t0 = time.time()
        if X_test.ndim == 3:
            N = X_test.shape[0]
            X_test = X_test.reshape(N, -1)
        # Convert and normalize to CPU tensor
        Xt = torch.from_numpy(X_test.astype(np.float32) / 255.0).to('cpu') 

        probs_sum = None
        for idx, item in enumerate(self.ensemble):
            model = item['model']
            mean = item['mean']
            std = item['std']
            X_mean = item['X_mean']
            components = item['components']
            pca_dim = item['pca_dim']

            # standardize using training mean/std
            Xs = (Xt - mean) / std
            # PCA transform: projected = (Xs - X_mean) @ components
            Xs_centered = Xs - X_mean  # shapes broadcast
            # Use only the trained components dimensions
            Xp = Xs_centered.matmul(components[:, :pca_dim]) 

            # batch predict to save memory
            model.eval()
            batch = 2048
            preds_probs = []
            with torch.no_grad():
                for start in range(0, Xp.shape[0], batch):
                    end = min(start + batch, Xp.shape[0])
                    xb = Xp[start:end]
                    out = model(xb)
                    probs = torch.softmax(out, dim=1).cpu().numpy()
                    preds_probs.append(probs)
            probs_arr = np.vstack(preds_probs)
            if probs_sum is None:
                probs_sum = probs_arr
            else:
                probs_sum += probs_arr

        probs_mean = probs_sum / len(self.ensemble)
        preds = probs_mean.argmax(axis=1).astype(np.int64)

        t1 = time.time()
        print(f"Prediction time: {t1 - t0:.2f}s")
        return preds

# ===== 本地快速自检 (保持不变) =====
if __name__ == "__main__":
    # 快速加载 MNIST 做本地验证（仅用于本地调试）
    from sklearn.datasets import fetch_openml
    from sklearn.metrics import accuracy_score
    # NOTE: In a real package environment, running agent.py directly with __main__ 
    # might require special handling of relative imports (.utils). 
    # For local tests outside the package structure, you might temporarily need to 
    # mock or comment out the 'from .utils import ...' line.
    
    print("Loading MNIST (may take a moment)...")
    mn = fetch_openml('mnist_784', version=1)
    X = mn.data.values.reshape(-1, 28, 28)
    y = mn.target.astype(int).values

    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    # NOTE: The following lines will FAIL if run outside the package structure 
    # without utils.py in the same directory due to the relative import 'from .utils'.
    # For a full package test, use the 'evaluation/benchmark.py' script.
    
    from utils import MLP, compute_pca_torch, standardize_torch # <-- temporary fix for standalone test
    
    print("Testing with temporary local import assumption...")
    agent = Agent(device='cpu')
    print("Training (this may use more memory)...")
    agent.train(X_train, y_train)

    print("Predicting ...")
    preds = agent.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Local accuracy: {acc:.4f}")
    print("Note: Run `evaluation/benchmark.py` to test the full package.")
    print(f"Local accuracy: {acc:.4f}")

