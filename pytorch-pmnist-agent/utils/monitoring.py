# your_repo/utils/monitoring.py
import time
import psutil
import os
from functools import wraps
import resource # For Unix-like systems, useful for maximum memory tracking

# --- 1. 计时装饰器 ---
def time_it(func):
    """
    Decorator to measure the execution time of a function.
    Returns the result of the function and the execution time in seconds.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        t1 = time.time()
        execution_time = t1 - t0
        return result, execution_time
    return wrapper

# --- 2. 内存/资源追踪函数 ---

def get_current_memory_usage_mb():
    """
    Returns the current process's memory usage in megabytes (MB).
    Requires 'psutil' package.
    """
    try:
        # Get current process PID
        pid = os.getpid()
        # Use psutil to get memory info
        process = psutil.Process(pid)
        # return Resident Set Size (RSS) in MB
        return process.memory_info().rss / (1024 * 1024)
    except NameError:
        # Fallback if psutil is not available
        return -1 
    except Exception:
        return -1

def get_max_memory_usage_mb():
    """
    Returns the maximum memory usage (Peak RSS) of the current process 
    since the beginning of the process, using the 'resource' module (Unix only).
    """
    try:
        # resource.RUSAGE_SELF returns current process statistics
        # ru_maxrss is the maximum resident set size used (often in KB or bytes, depends on OS)
        max_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        
        # On Linux/macOS, ru_maxrss is usually in KB. Convert to MB.
        # This conversion might need adjustment based on the specific execution environment (e.g., ML-Arena kernel OS)
        # We assume KB -> MB conversion here for standard Unix environments.
        return max_rss_kb / 1024 
    except ImportError:
        # resource module not available (e.g., Windows)
        return -1
    except Exception:
        # Other error (e.g., non-Linux/Unix system)
        return -1

# --- 3. 完整的性能摘要函数 (整合到实验脚本中更方便) ---
# Example of how these functions could be used in experiments/baseline.py or models/ensemble.py
# (This part is for illustration, not the final file content)

"""
# Example usage in an experiment script:

from .utils.monitoring import time_it, get_current_memory_usage_mb

@time_it
def run_training_phase(agent, X_train, y_train):
    agent.train(X_train, y_train)
    # Return peak memory recorded during this phase if possible
    
# In the main script:
initial_mem = get_current_memory_usage_mb()
(_, train_time) = run_training_phase(agent, X_train, y_train)
final_mem = get_current_memory_usage_mb()

print(f"Train time: {train_time:.2f}s")
print(f"Memory (Start/End): {initial_mem:.2f}MB / {final_mem:.2f}MB")
print(f"Max Memory (since process start): {get_max_memory_usage_mb():.2f}MB")
"""
