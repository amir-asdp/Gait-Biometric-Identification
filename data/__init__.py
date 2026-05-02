"""
Data module for gait biometric identification.
Handles dataset loading, preprocessing, and batch sampling.
"""

from .dataset import CASIABDataset, get_dataloader
from .transforms import GaitTransform
from .sampler import TripletSampler

__all__ = [
    'CASIABDataset',
    'get_dataloader',
    'GaitTransform',
    'TripletSampler',
]

