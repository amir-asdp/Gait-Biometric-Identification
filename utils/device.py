"""
Device management and reproducibility utilities.
"""

import os
import random

import numpy as np
import torch


def get_device(device_type: str = 'cuda', gpu_ids: list = [0]) -> torch.device:
    """
    Get computation device based on configuration and availability.
    
    Supports CUDA (NVIDIA GPUs), MPS (Apple Silicon), and CPU.
    
    Parameters
    ----------
    device_type : str
        Desired device type: 'cuda', 'mps', or 'cpu'.
    gpu_ids : list
        List of GPU IDs to use (for CUDA).
    
    Returns
    -------
    torch.device
        PyTorch device.
    """
    if device_type == 'cuda':
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_ids[0]}')
            print(f"Using CUDA device: {torch.cuda.get_device_name(gpu_ids[0])}")
            print(f"CUDA version: {torch.version.cuda}")
            
            # Set default GPU
            torch.cuda.set_device(gpu_ids[0])
            
            # Print memory info
            total_memory = torch.cuda.get_device_properties(gpu_ids[0]).total_memory / 1e9
            print(f"Total GPU memory: {total_memory:.2f} GB")
        else:
            print("CUDA not available, falling back to CPU")
            device = torch.device('cpu')
    
    elif device_type == 'mps':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using MPS (Apple Silicon) device")
        else:
            print("MPS not available, falling back to CPU")
            device = torch.device('cpu')
    
    elif device_type == 'cpu':
        device = torch.device('cpu')
        print("Using CPU device")
    
    else:
        raise ValueError(f"Unknown device type: {device_type}")
    
    return device


def setup_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Sets seeds for:
    - Python random module
    - NumPy
    - PyTorch (CPU and CUDA)
    
    Parameters
    ----------
    seed : int
        Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # For deterministic behavior (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")


def print_system_info():
    """Print system and environment information."""
    print("\n" + "=" * 50)
    print("System Information")
    print("=" * 50)
    print(f"Python version: {os.sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"Number of CPU cores: {os.cpu_count()}")
    print("=" * 50 + "\n")


def get_model_size(model: torch.nn.Module) -> dict:
    """
    Calculate model size in terms of parameters and memory.
    
    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model.
    
    Returns
    -------
    dict
        Dictionary with size information:
        - 'total_params': Total number of parameters
        - 'trainable_params': Number of trainable parameters
        - 'size_mb': Approximate model size in MB
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate size (assuming float32 = 4 bytes)
    size_mb = total_params * 4 / (1024 ** 2)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'size_mb': size_mb,
    }


def print_model_info(model: torch.nn.Module):
    """
    Print model information.
    
    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model.
    """
    size_info = get_model_size(model)
    
    print("\n" + "=" * 50)
    print("Model Information")
    print("=" * 50)
    print(f"Total parameters: {size_info['total_params']:,}")
    print(f"Trainable parameters: {size_info['trainable_params']:,}")
    print(f"Model size: {size_info['size_mb']:.2f} MB")
    print("=" * 50 + "\n")

