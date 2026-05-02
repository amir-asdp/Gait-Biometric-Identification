"""
GaitSet: A state-of-the-art backbone for gait recognition.

GaitSet learns gait representations from silhouette sequences using a set-based
approach that aggregates frame-level features into a fixed-length representation
regardless of sequence length.

References
----------
.. [1] Chao, H., He, Y., Zhang, J., & Feng, J. (2019). GaitSet: Regarding gait as a set 
       for cross-view gait recognition. In Proceedings of the AAAI Conference on 
       Artificial Intelligence (Vol. 33, pp. 8126-8133).
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SetBlock(nn.Module):
    """
    Basic convolutional block for set-based feature extraction.
    
    Architecture: Conv2d → BatchNorm → LeakyReLU
    
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple
        Kernel size for convolution.
    stride : int or tuple
        Stride for convolution.
    padding : int or tuple
        Padding for convolution.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        """Initialize SetBlock."""
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        
        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class GLConv(nn.Module):
    """
    Global-Local Convolution for capturing multi-scale features.
    
    Combines features from different receptive fields:
    - Local: 3x3 convolution
    - Global: 3x3 conv → max pooling → 3x3 conv → interpolation
    
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    halving : int
        Pooling factor for global branch.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        halving: int = 4,
    ):
        """Initialize GLConv."""
        super().__init__()
        
        self.halving = halving
        
        # Local branch
        self.local_conv = SetBlock(in_channels, out_channels)
        
        # Global branch
        self.global_conv1 = SetBlock(in_channels, out_channels)
        self.global_conv2 = SetBlock(out_channels, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [N, C, H, W].
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape [N, C', H, W].
        """
        # Local branch
        local_feat = self.local_conv(x)
        
        # Global branch
        # Downsample
        _, _, h, w = x.size()
        global_feat = F.max_pool2d(x, kernel_size=self.halving)
        
        # Process
        global_feat = self.global_conv1(global_feat)
        global_feat = self.global_conv2(global_feat)
        
        # Upsample back to original size
        global_feat = F.interpolate(global_feat, size=(h, w), mode='bilinear', align_corners=False)
        
        # Combine
        out = local_feat + global_feat
        
        return out


class GeMHPP(nn.Module):
    """
    Generalized Mean Horizontal Pyramid Pooling (GeMHPP).
    
    This pooling strategy:
    1. Divides features horizontally into multiple strips
    2. Applies Generalized Mean Pooling (GeM) to each strip
    3. Concatenates strip features
    
    GeM pooling: f_gem = (1/|X| * Σ x^p)^(1/p)
    where p is a learnable parameter (p=1: average, p=∞: max)
    
    Parameters
    ----------
    bins : List[int]
        Number of horizontal strips for each scale.
    in_channels : int
        Number of input channels.
    """
    
    def __init__(self, bins: List[int] = [16, 8, 4, 2, 1], in_channels: int = 128):
        """Initialize GeMHPP."""
        super().__init__()
        
        self.bins = bins
        self.in_channels = in_channels
        
        # Learnable p parameter for GeM pooling (initialized to 3.0)
        self.p = nn.Parameter(torch.ones(1) * 3.0)
    
    def gem_pooling(self, x: torch.Tensor, p: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
        """
        Generalized Mean Pooling.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        p : float
            Power parameter.
        eps : float
            Small constant for numerical stability.
        
        Returns
        -------
        torch.Tensor
            Pooled tensor.
        """
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [N, T, C, H, W] or [N, C, H, W].
        
        Returns
        -------
        torch.Tensor
            Pooled features of shape [N, T, C * sum(bins)] or [N, C * sum(bins)].
        """
        # Check if input has temporal dimension
        if x.dim() == 5:  # [N, T, C, H, W]
            n, t, c, h, w = x.size()
            x = x.view(n * t, c, h, w)
            has_temporal = True
        else:  # [N, C, H, W]
            n, c, h, w = x.size()
            has_temporal = False
        
        features = []
        
        for bin_num in self.bins:
            # Calculate strip height
            strip_h = h // bin_num
            
            for i in range(bin_num):
                # Extract strip
                start = i * strip_h
                end = (i + 1) * strip_h if i < bin_num - 1 else h
                strip = x[:, :, start:end, :]
                
                # Apply GeM pooling
                pooled = self.gem_pooling(strip, p=self.p.item())
                pooled = pooled.view(pooled.size(0), -1)  # [N*T or N, C]
                
                features.append(pooled)
        
        # Concatenate all features
        features = torch.cat(features, dim=1)  # [N*T or N, C * sum(bins)]
        
        # Reshape if temporal dimension exists
        if has_temporal:
            features = features.view(n, t, -1)  # [N, T, C * sum(bins)]
        
        return features


class TemporalPooling(nn.Module):
    """
    Temporal pooling over frame sequences.
    
    Aggregates frame-level features into a single representation using
    statistical pooling (max + mean).
    
    Parameters
    ----------
    pooling_type : str
        Type of pooling: 'max', 'mean', or 'statistics' (max + mean).
    """
    
    def __init__(self, pooling_type: str = 'statistics'):
        """Initialize temporal pooling."""
        super().__init__()
        self.pooling_type = pooling_type
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [N, T, C].
        
        Returns
        -------
        torch.Tensor
            Pooled tensor of shape [N, C] or [N, 2*C] for statistics pooling.
        """
        if self.pooling_type == 'max':
            return torch.max(x, dim=1)[0]
        
        elif self.pooling_type == 'mean':
            return torch.mean(x, dim=1)
        
        elif self.pooling_type == 'statistics':
            # Combine max and mean pooling
            max_pool = torch.max(x, dim=1)[0]
            mean_pool = torch.mean(x, dim=1)
            return torch.cat([max_pool, mean_pool], dim=1)
        
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")


class GaitSet(nn.Module):
    """
    GaitSet: Set-based gait recognition network.
    
    Architecture:
    -------------
    1. Frame-level feature extraction (CNN backbone)
    2. Set-level feature aggregation (Horizontal Pyramid Pooling)
    3. Temporal pooling (max + mean)
    4. FC layers for final embedding
    
    The key innovation is treating gait sequences as sets, making the model
    invariant to frame permutations and robust to sequence length variations.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for silhouettes).
    hidden_dim : int
        Dimension of hidden features.
    feature_channels : int
        Channels in feature maps before pooling.
    embedding_dim : int
        Dimension of final embedding.
    bins : List[int]
        Pyramid pooling bins.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 256,
        feature_channels: int = 128,
        embedding_dim: int = 256,
        bins: List[int] = [16, 8, 4, 2, 1],
    ):
        """Initialize GaitSet."""
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Frame-level feature extractor
        self.conv1 = SetBlock(in_channels, 32, kernel_size=5, padding=2)
        self.conv2 = SetBlock(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = SetBlock(32, 64, kernel_size=3, padding=1)
        self.conv4 = SetBlock(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global-Local convolutions
        self.gl_conv1 = GLConv(64, 128, halving=4)
        self.gl_conv2 = GLConv(128, feature_channels, halving=4)
        
        # Horizontal Pyramid Pooling
        self.hpp = GeMHPP(bins=bins, in_channels=feature_channels)
        
        # Temporal pooling
        self.temporal_pool = TemporalPooling(pooling_type='statistics')
        
        # Calculate HPP output dimension
        hpp_output_dim = feature_channels * sum(bins) * 2  # *2 for max+mean pooling
        
        # FC layers
        self.fc = nn.Sequential(
            nn.Linear(hpp_output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, embedding_dim),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input silhouettes of shape [N, T, H, W] where:
            - N: batch size
            - T: number of frames
            - H, W: height and width
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - embeddings: Final embeddings of shape [N, embedding_dim]
            - frame_features: Frame-level features before temporal pooling [N, T, feature_dim]
        """
        n, t, h, w = x.size()
        
        # Add channel dimension and merge batch and temporal dimensions
        x = x.unsqueeze(2)  # [N, T, 1, H, W]
        x = x.view(n * t, self.in_channels, h, w)  # [N*T, 1, H, W]
        
        # Frame-level feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        
        x = self.gl_conv1(x)
        x = self.gl_conv2(x)  # [N*T, C, H', W']
        
        # Restore temporal dimension for pooling
        _, c, h_new, w_new = x.size()
        x = x.view(n, t, c, h_new, w_new)  # [N, T, C, H', W']
        
        # Horizontal Pyramid Pooling
        frame_features = self.hpp(x)  # [N, T, feature_dim]
        
        # Temporal pooling
        set_features = self.temporal_pool(frame_features)  # [N, feature_dim]
        
        # FC layers for final embedding
        embeddings = self.fc(set_features)  # [N, embedding_dim]
        
        return embeddings, frame_features
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features for inference (returns only embeddings).
        
        Parameters
        ----------
        x : torch.Tensor
            Input silhouettes of shape [N, T, H, W].
        
        Returns
        -------
        torch.Tensor
            Embeddings of shape [N, embedding_dim].
        """
        embeddings, _ = self.forward(x)
        return embeddings

