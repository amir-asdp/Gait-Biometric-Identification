"""
Complete Gait Recognition Model.

Combines:
- GaitSet backbone for feature extraction
- Identity classification head
- Optional GRL with view discriminator for domain adaptation
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .backbone import GaitSet
from .grl import DomainAdaptationModule


class GaitRecognitionModel(nn.Module):
    """
    Complete gait recognition model with optional GRL.
    
    Architecture:
    -------------
    Input Silhouettes → GaitSet Backbone → Feature Embeddings
                                         ↓
                              Identity Classifier
                                         ↓
                          (Optional) GRL → View Discriminator
    
    The model can operate in two modes:
    1. Without GRL: Standard gait recognition
    2. With GRL: View-invariant gait recognition using domain adaptation
    
    Parameters
    ----------
    num_classes : int
        Number of identity classes.
    in_channels : int
        Number of input channels (1 for silhouettes).
    hidden_dim : int
        Hidden dimension for backbone.
    feature_channels : int
        Feature channels before pooling.
    embedding_dim : int
        Dimension of feature embeddings.
    bins : list
        Pyramid pooling bins.
    use_grl : bool
        Whether to use Gradient Reversal Layer.
    grl_config : Optional[dict]
        Configuration for GRL module.
    """
    
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        hidden_dim: int = 256,
        feature_channels: int = 128,
        embedding_dim: int = 256,
        bins: list = [16, 8, 4, 2, 1],
        use_grl: bool = True,
        grl_config: Optional[dict] = None,
    ):
        """Initialize gait recognition model."""
        super().__init__()
        
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.use_grl = use_grl
        
        # Backbone network
        self.backbone = GaitSet(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            feature_channels=feature_channels,
            embedding_dim=embedding_dim,
            bins=bins,
        )
        
        # Identity classification head
        self.identity_classifier = nn.Linear(embedding_dim, num_classes)
        
        # Optional GRL module
        if use_grl:
            if grl_config is None:
                grl_config = {
                    'num_views': 11,
                    'lambda_grl': 1.0,
                    'schedule': 'constant',
                    'num_of_warmup_epochs': 20,
                    'hidden_dims': [256, 128, 64],
                    'dropout': 0.3,
                }
            
            self.grl_module = DomainAdaptationModule(
                feature_dim=embedding_dim,
                num_views=grl_config['num_views'],
                lambda_grl=grl_config['lambda_grl'],
                schedule=grl_config['schedule'],
                num_of_warmup_epochs=grl_config['num_of_warmup_epochs'],
                hidden_dims=grl_config['hidden_dims'],
                dropout=grl_config['dropout'],
            )
        else:
            self.grl_module = None
    
    def forward(
        self,
        silhouettes: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Parameters
        ----------
        silhouettes : torch.Tensor
            Input silhouettes of shape [batch_size, num_frames, height, width].
        return_features : bool
            Whether to return intermediate features.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - 'embeddings': Feature embeddings [batch_size, embedding_dim]
            - 'identity_logits': Identity classification logits [batch_size, num_classes]
            - 'view_logits': View classification logits [batch_size, num_views] (if GRL enabled)
            - 'frame_features': Frame-level features (if return_features=True)
        """
        # Extract features using backbone
        embeddings, frame_features = self.backbone(silhouettes)
        
        # Identity classification
        identity_logits = self.identity_classifier(embeddings)
        
        outputs = {
            'embeddings': embeddings,
            'identity_logits': identity_logits,
        }
        
        # Apply GRL if enabled
        if self.use_grl and self.grl_module is not None:
            view_logits = self.grl_module(embeddings)
            outputs['view_logits'] = view_logits
        
        # Return frame features if requested
        if return_features:
            outputs['frame_features'] = frame_features
        
        return outputs
    
    def extract_features(self, silhouettes: torch.Tensor) -> torch.Tensor:
        """
        Extract features for inference/evaluation.
        
        Parameters
        ----------
        silhouettes : torch.Tensor
            Input silhouettes of shape [batch_size, num_frames, height, width].
        
        Returns
        -------
        torch.Tensor
            Feature embeddings of shape [batch_size, embedding_dim].
        """
        with torch.no_grad():
            embeddings = self.backbone.extract_features(silhouettes)
        return embeddings
    
    def update_grl_lambda(self, epoch: int, max_epochs: int):
        """
        Update GRL lambda based on training schedule.
        
        Parameters
        ----------
        epoch : int
            Current epoch.
        max_epochs : int
            Total number of epochs.
        """
        if self.use_grl and self.grl_module is not None:
            self.grl_module.update_lambda(epoch, max_epochs)
    
    def set_grl_lambda(self, lambda_grl: float):
        """
        Set GRL lambda value.
        
        Parameters
        ----------
        lambda_grl : float
            New lambda value.
        """
        if self.use_grl and self.grl_module is not None:
            self.grl_module.set_lambda(lambda_grl)
    
    def get_num_parameters(self) -> Dict[str, int]:
        """
        Get number of parameters in the model.
        
        Returns
        -------
        Dict[str, int]
            Dictionary with parameter counts for each component.
        """
        def count_parameters(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        param_counts = {
            'backbone': count_parameters(self.backbone),
            'classifier': count_parameters(self.identity_classifier),
            'total': count_parameters(self),
        }
        
        if self.use_grl and self.grl_module is not None:
            param_counts['grl'] = count_parameters(self.grl_module)
        
        return param_counts


def build_model(config: dict, num_classes: int) -> GaitRecognitionModel:
    """
    Build gait recognition model from configuration.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary.
    num_classes : int
        Number of identity classes.
    
    Returns
    -------
    GaitRecognitionModel
        Configured model.
    """
    model_cfg = config['model']
    backbone_cfg = model_cfg['backbone']
    grl_cfg = model_cfg['grl']
    
    # Build GRL configuration
    grl_config = None
    if grl_cfg['enabled']:
        grl_config = {
            'num_views': grl_cfg['discriminator']['num_views'],
            'lambda_grl': grl_cfg['lambda_grl'],
            'schedule': grl_cfg['schedule'],
            'num_of_warmup_epochs': grl_cfg['num_of_warmup_epochs'],
            'hidden_dims': grl_cfg['discriminator']['hidden_dims'],
            'dropout': grl_cfg['discriminator']['dropout'],
        }
    
    # Create model
    model = GaitRecognitionModel(
        num_classes=num_classes,
        in_channels=backbone_cfg['in_channels'],
        hidden_dim=backbone_cfg['hidden_dim'],
        feature_channels=backbone_cfg['feature_channels'],
        embedding_dim=backbone_cfg['embedding_dim'],
        bins=backbone_cfg['set_pooling']['bins'],
        use_grl=grl_cfg['enabled'],
        grl_config=grl_config,
    )
    
    return model

