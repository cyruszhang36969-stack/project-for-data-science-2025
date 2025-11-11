# your_repo/agent.py
"""
ML-Arena Submission File for Permuted MNIST.
This file serves as the mandatory entry point for the competition.
It wraps the core OptimizedAgent logic defined in the models/ensemble module.
"""
import sys
import os

# 确保项目根目录在 sys.path 中，以便可以找到 models 模块
# WARNING: This path manipulation might not work in the ML-Arena container. 
# The safest method is to copy the *full* content of models/ensemble.py here, 
# and replace relative imports (e.g., '..utils.optimization') with absolute imports 
# or local definitions for the contest.

# For GitHub Package structure:
try:
    from models.ensemble import OptimizedAgent
except ImportError:
    # Fallback/Debug note if running standalone
    print("Agent could not import OptimizedAgent from 'models.ensemble'.")
    raise

# ML-Arena requires a class named 'Agent'
class Agent(OptimizedAgent):
    """
    The official Agent class submitted to ML-Arena. 
    It inherits all methods (train, predict) from the robust OptimizedAgent.
    """
    # No changes needed; inheritance ensures the methods are exposed correctly.
    pass

# Note on submission: For the actual submission, if imports fail, 
# replace the content above with the entire body of models/ensemble.py, 
# ensuring all helper functions (MLP, PCA functions) are also defined within this file.
