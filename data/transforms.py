"""
Data augmentation and transformation for gait silhouettes.
"""

import random
from typing import Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class GaitTransform:
    """
    Transform for gait silhouette sequences.
    
    Applies various augmentations including:
    - Resizing
    - Horizontal flipping
    - Random rotation
    - Random erasing
    
    Parameters
    ----------
    resolution : Tuple[int, int]
        Target resolution (height, width).
    horizontal_flip : bool
        Whether to apply random horizontal flip.
    flip_prob : float
        Probability of horizontal flip.
    random_rotation : float
        Maximum rotation angle in degrees.
    random_erasing : bool
        Whether to apply random erasing.
    erasing_prob : float
        Probability of random erasing.
    """
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (64, 44),
        horizontal_flip: bool = False,
        flip_prob: float = 0.5,
        random_rotation: float = 0.0,
        random_erasing: bool = False,
        erasing_prob: float = 0.5,
    ):
        """Initialize transform."""
        self.resolution = resolution
        self.horizontal_flip = horizontal_flip
        self.flip_prob = flip_prob
        self.random_rotation = random_rotation
        self.random_erasing = random_erasing
        self.erasing_prob = erasing_prob
    
    def __call__(self, sils: torch.Tensor) -> torch.Tensor:
        """
        Apply transform to silhouette sequence.
        
        Parameters
        ----------
        sils : torch.Tensor
            Input silhouettes of shape [T, H, W].
        
        Returns
        -------
        torch.Tensor
            Transformed silhouettes of shape [T, H', W'].
        """
        T, H, W = sils.shape
        
        # Resize if needed
        if (H, W) != self.resolution:
            # Add channel dimension: [T, H, W] -> [T, 1, H, W]
            sils = sils.unsqueeze(1)
            # Resize: [T, 1, H, W] -> [T, 1, H', W']
            sils = F.interpolate(
                sils,
                size=self.resolution,
                mode='bilinear',
                align_corners=False,
            )
            # Remove channel dimension: [T, 1, H', W'] -> [T, H', W']
            sils = sils.squeeze(1)
        
        # Random horizontal flip
        if self.horizontal_flip and random.random() < self.flip_prob:
            sils = torch.flip(sils, dims=[2])  # Flip width dimension
        
        # Random rotation
        if self.random_rotation > 0:
            angle = random.uniform(-self.random_rotation, self.random_rotation)
            # Add channel dimension for rotation
            sils_with_channel = sils.unsqueeze(1)  # [T, 1, H, W]
            # Rotate each frame
            rotated_frames = []
            for i in range(T):
                frame = TF.rotate(sils_with_channel[i], angle, fill=0.0)
                rotated_frames.append(frame)
            sils = torch.stack(rotated_frames, dim=0).squeeze(1)  # [T, H, W]
        
        # Random erasing
        if self.random_erasing and random.random() < self.erasing_prob:
            sils = self._random_erasing(sils)
        
        return sils
    
    def _random_erasing(
        self,
        sils: torch.Tensor,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: float = 0.0,
    ) -> torch.Tensor:
        """
        Apply random erasing augmentation.
        
        Parameters
        ----------
        sils : torch.Tensor
            Input silhouettes of shape [T, H, W].
        scale : Tuple[float, float]
            Range of proportion of erased area.
        ratio : Tuple[float, float]
            Range of aspect ratio of erased area.
        value : float
            Value to fill erased area.
        
        Returns
        -------
        torch.Tensor
            Silhouettes with random erasing applied.
        """
        T, H, W = sils.shape
        area = H * W
        
        # Sample erasing parameters
        target_area = random.uniform(scale[0], scale[1]) * area
        aspect_ratio = random.uniform(ratio[0], ratio[1])
        
        h = int(round((target_area * aspect_ratio) ** 0.5))
        w = int(round((target_area / aspect_ratio) ** 0.5))
        
        if h < H and w < W:
            top = random.randint(0, H - h)
            left = random.randint(0, W - w)
            
            # Erase the region
            sils[:, top:top + h, left:left + w] = value
        
        return sils


class GaitNormalize:
    """
    Normalize gait silhouettes.
    
    Parameters
    ----------
    mean : float
        Mean for normalization.
    std : float
        Standard deviation for normalization.
    """
    
    def __init__(self, mean: float = 0.5, std: float = 0.5):
        """Initialize normalization."""
        self.mean = mean
        self.std = std
    
    def __call__(self, sils: torch.Tensor) -> torch.Tensor:
        """
        Normalize silhouettes.
        
        Parameters
        ----------
        sils : torch.Tensor
            Input silhouettes.
        
        Returns
        -------
        torch.Tensor
            Normalized silhouettes.
        """
        return (sils - self.mean) / self.std

