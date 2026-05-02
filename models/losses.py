"""
Loss functions for gait recognition.

Includes:
- Triplet Loss for metric learning
- Center Loss for intra-class compactness
- Combined losses for multi-task learning
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Batch Hard Triplet Loss.
    
    For each anchor in a batch:
    1. Select the hardest positive (furthest same-identity sample)
    2. Select the hardest negative (closest different-identity sample)
    3. Compute triplet loss: L = max(d(a,p) - d(a,n) + margin, 0)
    
    This encourages the network to learn embeddings where same-identity samples
    are closer than different-identity samples by at least a margin.
    
    Parameters
    ----------
    margin : float
        Margin for triplet loss.
    mining : str
        Mining strategy: 'batch_hard' or 'batch_all'.
    distance : str
        Distance metric: 'euclidean' or 'cosine'.
    """
    
    def __init__(
        self,
        margin: float = 0.2,
        mining: str = 'batch_hard',
        distance: str = 'euclidean',
    ):
        """Initialize triplet loss."""
        super().__init__()
        self.margin = margin
        self.mining = mining
        self.distance = distance
    
    def _pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distances between embeddings.
        
        Parameters
        ----------
        embeddings : torch.Tensor
            Embeddings of shape [batch_size, embedding_dim].
        
        Returns
        -------
        torch.Tensor
            Distance matrix of shape [batch_size, batch_size].
        """
        if self.distance == 'euclidean':
            # Euclidean distance
            dot_product = torch.matmul(embeddings, embeddings.t())
            square_norm = torch.diag(dot_product)
            distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
            distances = torch.clamp(distances, min=0.0)
            distances = torch.sqrt(distances + 1e-16)  # Add epsilon for numerical stability
        
        elif self.distance == 'cosine':
            # Cosine distance (1 - cosine similarity)
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
            cosine_similarity = torch.matmul(normalized_embeddings, normalized_embeddings.t())
            distances = 1.0 - cosine_similarity
        
        else:
            raise ValueError(f"Unknown distance metric: {self.distance}")
        
        return distances
    
    def _get_triplet_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Get valid triplet mask.
        
        A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] (positive pair)
        - labels[i] != labels[k] (negative pair)
        
        Parameters
        ----------
        labels : torch.Tensor
            Labels of shape [batch_size].
        
        Returns
        -------
        torch.Tensor
            Boolean mask of shape [batch_size, batch_size, batch_size].
        """
        # Check that i, j, k are distinct
        indices_equal = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
        indices_not_equal = ~indices_equal
        i_not_equal_j = indices_not_equal.unsqueeze(2)
        i_not_equal_k = indices_not_equal.unsqueeze(1)
        j_not_equal_k = indices_not_equal.unsqueeze(0)
        
        distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k
        
        # Check if labels[i] == labels[j] and labels[i] != labels[k]
        label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        i_equal_j = label_equal.unsqueeze(2)
        i_equal_k = label_equal.unsqueeze(1)
        
        valid_labels = i_equal_j & (~i_equal_k)
        
        return distinct_indices & valid_labels
    
    def batch_hard_triplet_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batch hard triplet loss.
        
        Parameters
        ----------
        embeddings : torch.Tensor
            Embeddings of shape [batch_size, embedding_dim].
        labels : torch.Tensor
            Labels of shape [batch_size].
        
        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        # Compute pairwise distances
        pairwise_dist = self._pairwise_distances(embeddings)
        
        # For each anchor, get the hardest positive
        mask_anchor_positive = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask_anchor_positive = mask_anchor_positive - torch.eye(labels.size(0), device=labels.device)
        
        # Hardest positive: maximum distance among positives
        anchor_positive_dist = pairwise_dist * mask_anchor_positive
        hardest_positive_dist = torch.max(anchor_positive_dist, dim=1)[0]
        
        # For each anchor, get the hardest negative
        mask_anchor_negative = (labels.unsqueeze(0) != labels.unsqueeze(1)).float()
        
        # Hardest negative: minimum distance among negatives
        # Add large value to positives so they don't count as minimum
        max_anchor_negative_dist = torch.max(pairwise_dist, dim=1)[0]
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist.unsqueeze(1) * (1.0 - mask_anchor_negative)
        hardest_negative_dist = torch.min(anchor_negative_dist, dim=1)[0]
        
        # Combine to get triplet loss
        triplet_loss = torch.clamp(hardest_positive_dist - hardest_negative_dist + self.margin, min=0.0)
        triplet_loss = torch.mean(triplet_loss)
        
        return triplet_loss
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Parameters
        ----------
        embeddings : torch.Tensor
            Embeddings of shape [batch_size, embedding_dim].
        labels : torch.Tensor
            Labels of shape [batch_size].
        
        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        if self.mining == 'batch_hard':
            return self.batch_hard_triplet_loss(embeddings, labels)
        else:
            raise NotImplementedError(f"Mining strategy {self.mining} not implemented")


class CenterLoss(nn.Module):
    """
    Center Loss for deep face recognition.
    
    Center loss learns a center (mean) for each class and penalizes the distance
    between samples and their corresponding class centers. This encourages
    intra-class compactness.
    
    L_center = (1/2) * Σ ||x_i - c_{y_i}||^2
    
    where c_{y_i} is the center of class y_i.
    
    References
    ----------
    .. [1] Wen, Y., Zhang, K., Li, Z., & Qiao, Y. (2016). A discriminative feature 
           learning approach for deep face recognition. In European conference on 
           computer vision (pp. 499-515).
    
    Parameters
    ----------
    num_classes : int
        Number of classes.
    feature_dim : int
        Dimension of features.
    """
    
    def __init__(self, num_classes: int, feature_dim: int):
        """Initialize center loss."""
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # Initialize centers
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute center loss.
        
        Parameters
        ----------
        embeddings : torch.Tensor
            Embeddings of shape [batch_size, feature_dim].
        labels : torch.Tensor
            Labels of shape [batch_size].
        
        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        batch_size = embeddings.size(0)
        
        # Get centers for the batch
        centers_batch = self.centers[labels]  # [batch_size, feature_dim]
        
        # Compute center loss
        loss = torch.sum((embeddings - centers_batch) ** 2) / (2.0 * batch_size)
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss for gait recognition.
    
    Combines multiple losses:
    - Cross-entropy loss for identity classification
    - Triplet loss for metric learning
    - Center loss for intra-class compactness
    - View classification loss (for GRL)
    
    Parameters
    ----------
    num_classes : int
        Number of identity classes.
    embedding_dim : int
        Dimension of embeddings.
    triplet_margin : float
        Margin for triplet loss.
    loss_weights : dict
        Weights for each loss component.
    """
    
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        triplet_margin: float = 0.2,
        loss_weights: dict = None,
    ):
        """Initialize combined loss."""
        super().__init__()
        
        # Default loss weights
        if loss_weights is None:
            loss_weights = {
                'identity': 1.0,
                'triplet': 1.0,
                'center': 0.0005,
                'view': 0.5,
            }
        
        self.loss_weights = loss_weights
        
        # Identity classification loss
        self.identity_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Triplet loss
        if loss_weights['triplet'] > 0:
            self.triplet_criterion = TripletLoss(margin=triplet_margin, mining='batch_hard')
        else:
            self.triplet_criterion = None
        
        # Center loss
        if loss_weights['center'] > 0:
            self.center_criterion = CenterLoss(num_classes=num_classes, feature_dim=embedding_dim)
        else:
            self.center_criterion = None
        
        # View classification loss
        self.view_criterion = nn.CrossEntropyLoss()
    
    def forward(
        self,
        embeddings: torch.Tensor,
        identity_logits: torch.Tensor,
        identity_labels: torch.Tensor,
        view_logits: Optional[torch.Tensor] = None,
        view_labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute combined loss.
        
        Parameters
        ----------
        embeddings : torch.Tensor
            Feature embeddings of shape [batch_size, embedding_dim].
        identity_logits : torch.Tensor
            Identity classification logits of shape [batch_size, num_classes].
        identity_labels : torch.Tensor
            Identity labels of shape [batch_size].
        view_logits : Optional[torch.Tensor]
            View classification logits of shape [batch_size, num_views].
        view_labels : Optional[torch.Tensor]
            View labels of shape [batch_size].
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'total': Total loss
            - 'identity': Identity classification loss
            - 'triplet': Triplet loss (if enabled)
            - 'center': Center loss (if enabled)
            - 'view': View classification loss (if enabled)
        """
        losses = {}
        
        # Identity classification loss
        identity_loss = self.identity_criterion(identity_logits, identity_labels)
        losses['identity'] = identity_loss
        
        # Triplet loss
        if self.triplet_criterion is not None and self.loss_weights['triplet'] > 0:
            triplet_loss = self.triplet_criterion(embeddings, identity_labels)
            losses['triplet'] = triplet_loss
        else:
            losses['triplet'] = torch.tensor(0.0, device=embeddings.device)
        
        # Center loss
        if self.center_criterion is not None and self.loss_weights['center'] > 0:
            center_loss = self.center_criterion(embeddings, identity_labels)
            losses['center'] = center_loss
        else:
            losses['center'] = torch.tensor(0.0, device=embeddings.device)
        
        # View classification loss (for GRL)
        if view_logits is not None and view_labels is not None:
            view_loss = self.view_criterion(view_logits, view_labels)
            losses['view'] = view_loss
        else:
            losses['view'] = torch.tensor(0.0, device=embeddings.device)
        
        # Compute total loss
        total_loss = (
            self.loss_weights['identity'] * losses['identity'] +
            self.loss_weights['triplet'] * losses['triplet'] +
            self.loss_weights['center'] * losses['center'] +
            self.loss_weights['view'] * losses['view']
        )
        
        losses['total'] = total_loss
        
        return losses

