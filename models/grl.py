"""
Gradient Reversal Layer (GRL) for Domain Adaptation.

The GRL is a key component for learning view-invariant gait representations.
It works by reversing gradients during backpropagation, forcing the feature
extractor to learn representations that are invariant to view angles while
maintaining discriminative power for identity recognition.

References
----------
.. [1] Ganin, Y., & Lempitsky, V. (2015). Unsupervised domain adaptation by
       backpropagation. In International conference on machine learning (pp. 1180-1189).
.. [2] Hu, M., Wang, Y., Zhang, Z., Little, J. J., & Huang, D. (2018). 
       View-invariant discriminative projection for multi-view gait-based human 
       identification. IEEE transactions on information forensics and security.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Gradient Reversal Function.
    
    This function acts as an identity transform in the forward pass but
    multiplies the gradient by -lambda in the backward pass.
    
    Forward:  y = x
    Backward: dx = -lambda * dy
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_grl: float) -> torch.Tensor:
        """
        Forward pass (identity).
        
        Parameters
        ----------
        ctx : context
            Context object for saving information for backward pass.
        x : torch.Tensor
            Input tensor.
        lambda_grl : float
            Gradient reversal strength.
        
        Returns
        -------
        torch.Tensor
            Output tensor (same as input).
        """
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """
        Backward pass (gradient reversal).
        
        Parameters
        ----------
        ctx : context
            Context object.
        grad_output : torch.Tensor
            Gradient from subsequent layer.
        
        Returns
        -------
        tuple
            (reversed gradient, None)
        """
        lambda_grl = ctx.lambda_grl
        grad_input = grad_output.neg() * lambda_grl
        return grad_input, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer (GRL).
    
    The GRL is used to train a domain-invariant feature extractor. During
    forward propagation, GRL acts as an identity transform. During backpropagation,
    GRL multiplies the gradient by -λ before passing it to the preceding layer.
    
    Mathematical Formulation:
    -------------------------
    The objective with GRL is to minimize:
    
        L = L_y(G_f(x), y) - λ * L_d(G_d(G_f(x)), d)
    
    where:
    - G_f: Feature extractor
    - G_d: Domain discriminator  
    - L_y: Task loss (identity classification)
    - L_d: Domain loss (view classification)
    - λ: Gradient reversal strength
    
    By maximizing the domain confusion (minimizing domain classification accuracy),
    the feature extractor learns view-invariant representations.
    
    Parameters
    ----------
    lambda_grl : float
        Gradient reversal strength. Higher values increase view-invariance but
        may reduce identity discrimination.
    schedule : str
        How to adjust lambda during training:
        - 'constant': Fixed lambda
        - 'progressive': Gradually increase from 0 to max
        - 'adaptive': Adjust based on domain discriminator accuracy
    """
    
    def __init__(
            self,
            lambda_grl: float = 1.0,
            schedule: str = 'constant',
            num_of_warmup_epochs: int = 20,
    ):
        """Initialize GRL."""
        super().__init__()
        self.lambda_grl = lambda_grl
        self.schedule = schedule
        self.current_lambda = 0.0 if schedule == 'progressive' else lambda_grl
        self.num_of_warmup_epochs = num_of_warmup_epochs
        self.epoch = 0
    
    def set_lambda(self, lambda_grl: float):
        """
        Set the gradient reversal strength.
        
        Parameters
        ----------
        lambda_grl : float
            New lambda value.
        """
        self.current_lambda = lambda_grl
    
    def update_lambda(self, epoch: int, max_epochs: int):
        """
        Update lambda based on training schedule.
        
        For progressive schedule, lambda increases from 0 to lambda_grl:
            λ_p = (2 / (1 + exp(-10 * p)) - 1) * λ
        where p = epoch / max_epochs
        
        Parameters
        ----------
        epoch : int
            Current epoch.
        max_epochs : int
            Total number of epochs.
        """
        self.epoch = epoch
        
        if self.schedule == 'progressive':
            if epoch < self.num_of_warmup_epochs:
                self.current_lambda = 0.0
            else:
                p = (epoch - self.num_of_warmup_epochs) / (max_epochs - self.num_of_warmup_epochs)
                # Gradually increase lambda using sigmoid-like schedule
                self.current_lambda = (2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p))) - 1.0) * self.lambda_grl
        elif self.schedule == 'constant':
            if epoch < self.num_of_warmup_epochs:
                self.current_lambda = 0.0
            else:
                self.current_lambda = self.lambda_grl
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features.
        
        Returns
        -------
        torch.Tensor
            Output (same as input in forward pass).
        """
        return GradientReversalFunction.apply(x, self.current_lambda)
    
    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return f'lambda_grl={self.lambda_grl}, schedule={self.schedule}, current_lambda={self.current_lambda:.4f}'


class ViewDiscriminator(nn.Module):
    """
    View Angle Discriminator for domain adaptation.
    
    This network tries to predict the view angle from features. By connecting
    it through the GRL, we force the feature extractor to learn view-invariant
    representations (features that the discriminator cannot classify by view).
    
    Architecture:
    -------------
    Features → FC → ReLU → Dropout → FC → ReLU → Dropout → FC → Softmax
    
    The discriminator should be moderately complex - too simple and it won't
    provide useful gradients, too complex and it may overpower the feature extractor.
    
    Parameters
    ----------
    input_dim : int
        Dimension of input features.
    hidden_dims : List[int]
        Dimensions of hidden layers.
    num_views : int
        Number of view angles to classify.
    dropout : float
        Dropout probability.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: list = [256, 128, 64],
        num_views: int = 11,
        dropout: float = 0.3,
    ):
        """Initialize view discriminator."""
        super().__init__()
        
        self.input_dim = input_dim
        self.num_views = num_views
        
        # Build discriminator layers
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, num_views))
        
        self.discriminator = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features of shape [batch_size, input_dim].
        
        Returns
        -------
        torch.Tensor
            View predictions of shape [batch_size, num_views] (logits).
        """
        return self.discriminator(x)


class DomainAdaptationModule(nn.Module):
    """
    Complete Domain Adaptation Module combining GRL and View Discriminator.
    
    This module encapsulates the full domain adaptation pipeline:
    1. Apply GRL to input features
    2. Classify view angle using discriminator
    3. Compute domain adaptation loss
    
    Parameters
    ----------
    feature_dim : int
        Dimension of input features.
    num_views : int
        Number of view angles.
    lambda_grl : float
        Gradient reversal strength.
    schedule : str
        Lambda scheduling strategy.
    hidden_dims : list
        Hidden dimensions for discriminator.
    dropout : float
        Dropout probability.
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        num_views: int = 11,
        lambda_grl: float = 1.0,
        schedule: str = 'constant',
        num_of_warmup_epochs: int = 20,
        hidden_dims: list = [256, 128, 64],
        dropout: float = 0.3,
    ):
        """Initialize domain adaptation module."""
        super().__init__()
        
        self.grl = GradientReversalLayer(
            lambda_grl=lambda_grl,
            schedule=schedule,
            num_of_warmup_epochs=num_of_warmup_epochs,
        )
        self.discriminator = ViewDiscriminator(
            input_dim=feature_dim,
            hidden_dims=hidden_dims,
            num_views=num_views,
            dropout=dropout,
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        features : torch.Tensor
            Input features of shape [batch_size, feature_dim].
        
        Returns
        -------
        torch.Tensor
            View predictions (logits) of shape [batch_size, num_views].
        """
        # Apply gradient reversal
        reversed_features = self.grl(features)
        
        # Classify view angle
        view_logits = self.discriminator(reversed_features)
        
        return view_logits
    
    def update_lambda(self, epoch: int, max_epochs: int):
        """Update GRL lambda based on training schedule."""
        self.grl.update_lambda(epoch, max_epochs)
    
    def set_lambda(self, lambda_grl: float):
        """Set GRL lambda."""
        self.grl.set_lambda(lambda_grl)

