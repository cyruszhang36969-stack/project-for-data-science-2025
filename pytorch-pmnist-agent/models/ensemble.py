# your_repo/models/ensemble.py
"""
Optimized Agent Core Logic — PyTorch Ensemble with PCA and Adaptive Training
This module contains the high-performance Agent designed for ML-Arena Permuted MNIST.
It implements ensemble, PCA, and time-adaptive hyperparameter tuning.
"""
import time
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 导入依赖：从 utils.optimization 导入预处理函数
from ..utils.optimization import compute_pca_torch, standardize_torch
# 导入依赖：从 models.mlp 导入 MLP 模型结构
from .mlp import MLP 

# --------- 可调超参（原 agent.py 的参数） ----------
PCA_TARGET = 400          # 目标 PCA 主成分数
N_MODELS = 4              # ensemble 模型数量
TRAIN_SAMPLE_MAX = 60000  # 使用全部训练样本
TRAIN_TIME_LIMIT = 50.0   # 训练上限（秒）
DEVICE = 'cpu'            # 竞赛以 CPU 为主
BATCH_BASE = 512          # 基础 batch size
HIDDEN_DIMS = [512, 384, 256]
LR = 0.006
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 8            # 最大 epoch
SEED_BASE = 42

# --------- Optimized Agent 实现 ---------
class OptimizedAgent:
    """
    Optimized Agent Class, incorporating PCA, Deep MLP, and Ensemble Learning
    with an adaptive training time mechanism.
    """
    def __init__(self, device=DEVICE):
        # 每个子模型会保存： model, mean, std, X_mean (for PCA), components
        self.ensemble = []
        self.device = torch.device(device)
        self.is_trained = False
        self.train_time_limit = TRAIN_TIME_LIMIT
        self.pca_target = PCA_TARGET
        self.n_models = N_MODELS
        self.hidden_dims = HIDDEN_DIMS

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
            
        X = torch.from_numpy(X_train.astype(np.float32) / 255.0).to(self.device)
        y = torch.from_numpy(y_train.squeeze().astype(np.int64)).to(self.device)

        n_samples = min(TRAIN_SAMPLE_MAX, X.shape[0])
        X = X[:n_samples]
        y = y[:n_samples]

        self.ensemble = []
        gc.collect()

        for i in range(self.n_models):
            model_start = time.time()
            seed = SEED_BASE + i * 101
            torch.manual_seed(seed)
            idx = torch.randperm(X.size(0), device=self.device)
            Xs = X[idx]
            ys = y[idx]

            # --- Adaptive Training Logic ---
            elapsed = time.time() - t0
            remaining = max(0.0, self.train_time_limit - elapsed)
            target_train_time = max(5.0, remaining - 6.0) # Reserve time for prediction

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
            # --- End Adaptive Training Logic ---

            # compute standardization & PCA
            Xs_std, mean, std = standardize_torch(Xs)
            
            # --- PCA Calculation with Fallback ---
            try:
                Xs_pca, X_mean, components = compute_pca_torch(Xs_std, n_components=pca_dim)
            except RuntimeError:
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

            del Xs_std
            gc.collect()

            input_dim = Xs_pca.shape[1]
            # Instantiate MLP model
            model = MLP(input_dim=input_dim, hidden_dims=self.hidden_dims).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
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

                    if (time.time() - t0) + 6.0 > self.train_time_limit:
                        break

                scheduler.step()
                current_epoch = epoch + 1
                if (time.time() - t0) + 6.0 > self.train_time_limit:
                    break

            # save model + preprocessing params
            self.ensemble.append({
                'model': model.cpu(),
                'mean': mean.cpu(),
                'std': std.cpu(),
                'X_mean': X_mean.cpu(),
                'components': components.cpu(),
                'pca_dim': input_dim
            })

            del Xs_pca, X_epoch, y_epoch
            gc.collect()

            model_time = time.time() - model_start
            print(f"Model {i+1}/{self.n_models} trained in {model_time:.2f}s (pca={pca_dim}, batch={batch_size}, epochs={current_epoch})")
            if (time.time() - t0) + 6.0 > self.train_time_limit:
                print("Approaching train time limit, stopping ensemble build.")
                break

        self.is_trained = True
        total_time = time.time() - t0
        print(f"Total training time: {total_time:.2f}s, models built: {len(self.ensemble)}")

    def predict(self, X_test):
        """
        X_test: np.ndarray (N,28,28) or (N,784)
        Returns np.ndarray (N,)
        """
        if not self.is_trained:
            raise RuntimeError("Agent must be trained before predict()")

        t0 = time.time()
        if X_test.ndim == 3:
            N = X_test.shape[0]
            X_test = X_test.reshape(N, -1)
        Xt = torch.from_numpy(X_test.astype(np.float32) / 255.0).to('cpu') 

        probs_sum = None
        for item in self.ensemble:
            model = item['model']
            mean = item['mean']
            std = item['std']
            X_mean = item['X_mean']
            components = item['components']
            pca_dim = item['pca_dim']

            # standardize
            Xs = (Xt - mean) / std
            # PCA transform
            Xs_centered = Xs - X_mean
            Xp = Xs_centered.matmul(components[:, :pca_dim]) 

            # batch predict
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
