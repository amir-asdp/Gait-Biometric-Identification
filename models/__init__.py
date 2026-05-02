"""
Models module for gait biometric identification.

Includes:
- GRL (Gradient Reversal Layer)
- Backbone networks (GaitSet)
- Complete gait recognition models
"""

from .grl import GradientReversalLayer, ViewDiscriminator
from .backbone import GaitSet
from .gait_model import GaitRecognitionModel, build_model
from .losses import TripletLoss, CenterLoss, CombinedLoss

__all__ = [
    'GradientReversalLayer',
    'ViewDiscriminator',
    'GaitSet',
    'GaitRecognitionModel',
    'build_model',
    'TripletLoss',
    'CenterLoss',
    'CombinedLoss',
]

