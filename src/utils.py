"""
Utility functions for CLTV prediction pipeline.
"""

import torch
import random
import numpy as np
import os


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get available device (CUDA if available)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def create_experiment_dir(base_dir: str = 'experiments') -> str:
    """Create timestamped experiment directory."""
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(base_dir, f'exp_{timestamp}')
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Experiment directory: {exp_dir}")
    return exp_dir
