# your_repo/models/mlp.py
import torch
import torch.nn as nn

# --------- 简单 MLP ---------
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int = 10):
        """
        Deep Multilayer Perceptron architecture for MNIST classification.
        """
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
