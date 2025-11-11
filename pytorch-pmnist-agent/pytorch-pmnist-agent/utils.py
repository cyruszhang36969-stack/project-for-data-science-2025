# pytorch_pmnist_agent/utils.py
import torch
import torch.nn as nn

# --------- 工具函数：PCA（优先用 torch.pca_lowrank） ---------
def compute_pca_torch(X, n_components):
    """
    输入：X: torch.Tensor (n_samples, n_features) float32
    返回：X_pca (n_samples, n_components), X_mean (1, n_features), components (n_features, n_components)
    尝试使用 torch.pca_lowrank（更节省内存/速度）；回退到 torch.linalg.svd 若必要。
    """
    # center
    X_mean = X.mean(dim=0, keepdim=True)
    X_centered = X - X_mean
    # try pca_lowrank
    try:
        # torch.pca_lowrank 返回 Q, S, V (V 的列是主成分)
        # Note: If n_components is large, Q, S, V can be large.
        Q, S, V = torch.pca_lowrank(X_centered, q=n_components, center=False)
        components = V[:, :n_components]  # (n_features, n_components)
        X_pca = X_centered.matmul(components)
        return X_pca, X_mean, components
    except Exception:
        # fallback to SVD (可能更慢/占内存)
        try:
            # Note: SVD can be extremely memory intensive for large matrices (60000x784)
            U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
            components = Vt[:n_components].T  # (n_features, n_components)
            X_pca = X_centered.matmul(components)
            return X_pca, X_mean, components
        except Exception as e:
            raise RuntimeError("PCA failed: " + str(e))

# --------- 标准化 ---------
def standardize_torch(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(dim=0, keepdim=True)
    if std is None:
        std = X.std(dim=0, keepdim=True)
        std = std.clamp(min=1e-6)
    Xs = (X - mean) / std
    return Xs, mean, std

# --------- 简单 MLP ---------
# Move the MLP class here or keep it in agent.py/baseline_agent.py, 
# for now we keep it here for shared use.
def create_mlp(input_dim, hidden_dims, output_dim=10):
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU(inplace=True))
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super().__init__()
        self.net = create_mlp(input_dim, hidden_dims, output_dim)
    def forward(self, x):
        return self.net(x)