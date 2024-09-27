"""
This module provides a function to determine the appropriate device for PyTorch based on the available backend.
"""

import torch

def get_device():
    """
    Get the device to use for torch.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.backends.cuda.is_built():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
