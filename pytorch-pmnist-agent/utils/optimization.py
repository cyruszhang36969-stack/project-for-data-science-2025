# your_repo/utils/optimization.py
import torch
import numpy as np

# --------- PCA（优先用 torch.pca_lowrank） ---------
def compute_pca_torch(X: torch.Tensor, n_components: int):
    """
    输入：X: torch.Tensor (n_samples, n_features) float32
    返回：X_pca (n_samples, n_components), X_mean (1, n_features), components (n_features, n_components)
    尝试使用 torch.pca_lowrank；回退到 torch.linalg.svd 若必要。
    """
    # center
    X_mean = X.mean(dim=0, keepdim=True)
    X_centered = X - X_mean
    
    # try pca_lowrank
    try:
        # torch.pca_lowrank returns Q, S, V (V columns are principal components)
        Q, S, V = torch.pca_lowrank(X_centered, q=n_components, center=False)
        components = V[:, :n_components]  # (n_features, n_components)
        X_pca = X_centered.matmul(components)
        return X_pca, X_mean, components
    except Exception:
        # fallback to SVD 
        try:
            U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
            components = Vt[:n_components].T  # (n_features, n_components)
            X_pca = X_centered.matmul(components)
            return X_pca, X_mean, components
        except Exception as e:
            # Re-raise with informative error
            raise RuntimeError(f"PCA failed (dim={n_components}) even after SVD fallback: " + str(e))

# --------- 标准化 ---------
def standardize_torch(X: torch.Tensor, mean=None, std=None):
    """
    Standardizes the input tensor X. If mean/std are not provided, computes them.
    """
    if mean is None:
        mean = X.mean(dim=0, keepdim=True)
    if std is None:
        std = X.std(dim=0, keepdim=True)
        # Prevent division by zero/near-zero variance features
        std = std.clamp(min=1e-6) 
        
    Xs = (X - mean) / std
    return Xs, mean, std
