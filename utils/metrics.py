"""
Evaluation metrics for gait recognition.

Implements:
- Rank-k accuracy
- Mean Average Precision (mAP)
- Cumulative Match Characteristic (CMC) curve
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score


def compute_distance_matrix(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    metric: str = 'euclidean',
) -> np.ndarray:
    """
    Compute distance matrix between query and gallery features.
    
    Parameters
    ----------
    query_features : torch.Tensor
        Query features of shape [num_query, feature_dim].
    gallery_features : torch.Tensor
        Gallery features of shape [num_gallery, feature_dim].
    metric : str
        Distance metric: 'euclidean' or 'cosine'.
    
    Returns
    -------
    np.ndarray
        Distance matrix of shape [num_query, num_gallery].
    """
    if metric == 'euclidean':
        # Euclidean distance
        m, n = query_features.size(0), gallery_features.size(0)
        distmat = (
            torch.pow(query_features, 2).sum(dim=1, keepdim=True).expand(m, n)
            + torch.pow(gallery_features, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            - 2 * torch.matmul(query_features, gallery_features.t())
        )
        distmat = torch.sqrt(torch.clamp(distmat, min=1e-12))
    
    elif metric == 'cosine':
        # Cosine distance (1 - cosine similarity)
        query_norm = F.normalize(query_features, p=2, dim=1)
        gallery_norm = F.normalize(gallery_features, p=2, dim=1)
        distmat = 1 - torch.matmul(query_norm, gallery_norm.t())
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return distmat.cpu().numpy()


def evaluate_rank(
    distmat: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    query_cams: np.ndarray = None,
    gallery_cams: np.ndarray = None,
    max_rank: int = 50,
) -> Dict[str, float]:
    """
    Evaluate ranking metrics.
    
    Parameters
    ----------
    distmat : np.ndarray
        Distance matrix of shape [num_query, num_gallery].
    query_labels : np.ndarray
        Query identity labels of shape [num_query].
    gallery_labels : np.ndarray
        Gallery identity labels of shape [num_gallery].
    query_cams : np.ndarray, optional
        Query camera IDs (for cross-camera evaluation).
    gallery_cams : np.ndarray, optional
        Gallery camera IDs.
    max_rank : int
        Maximum rank for CMC computation.
    
    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'rank1': Rank-1 accuracy
        - 'rank5': Rank-5 accuracy
        - 'rank10': Rank-10 accuracy
        - 'mAP': Mean Average Precision
        - 'cmc': CMC curve (array)
    """
    num_query, num_gallery = distmat.shape
    
    if num_query == 0 or num_gallery == 0:
        raise ValueError("Query or gallery set is empty")
    
    # Sort indices by distance (ascending)
    indices = np.argsort(distmat, axis=1)
    
    # Compute CMC and mAP
    all_cmc = []
    all_AP = []
    num_valid_queries = 0
    
    for i in range(num_query):
        # Get query info
        query_label = query_labels[i]
        query_cam = query_cams[i] if query_cams is not None else None
        
        # Remove gallery samples from the same camera (if applicable)
        if query_cams is not None and gallery_cams is not None:
            # Remove same identity and same camera
            remove = (gallery_labels == query_label) & (gallery_cams == query_cam)
        else:
            remove = np.zeros(num_gallery, dtype=bool)
        
        # Get matches (same identity, different camera or no camera constraint)
        matches = (gallery_labels == query_label) & (~remove)
        
        if not np.any(matches):
            # No valid matches for this query
            continue
        
        num_valid_queries += 1
        
        # Remove invalid gallery samples
        valid_indices = np.where(~remove)[0]
        matches = matches[valid_indices]
        
        # Compute CMC for this query
        order = indices[i]
        order = order[~remove[order]]  # Remove invalid samples from ranking
        
        # Find positions of correct matches
        match_positions = np.where(matches[order])[0]
        
        # CMC: at rank k, is there at least one correct match?
        cmc = np.zeros(max_rank)
        if len(match_positions) > 0:
            first_match_pos = match_positions[0]
            if first_match_pos < max_rank:
                cmc[first_match_pos:] = 1.0
        
        all_cmc.append(cmc)
        
        # Compute Average Precision (AP)
        num_matches = np.sum(matches)
        matches_cumsum = np.cumsum(matches[order])
        precision = matches_cumsum / (np.arange(len(matches_cumsum)) + 1)
        
        # Only consider positions where there is a match
        AP = np.sum(precision[matches[order]]) / num_matches
        all_AP.append(AP)
    
    if num_valid_queries == 0:
        raise ValueError("No valid queries found")
    
    # Compute average CMC
    all_cmc = np.array(all_cmc).astype(np.float32)
    cmc = np.mean(all_cmc, axis=0)
    
    # Compute mAP
    mAP = np.mean(all_AP)
    
    results = {
        'rank1': cmc[0],
        'rank5': cmc[4] if max_rank >= 5 else cmc[-1],
        'rank10': cmc[9] if max_rank >= 10 else cmc[-1],
        'mAP': mAP,
        'cmc': cmc,
    }
    
    return results


def compute_cmc(
    distmat: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    max_rank: int = 50,
) -> np.ndarray:
    """
    Compute Cumulative Match Characteristic (CMC) curve.
    
    CMC(k) represents the probability that the correct match appears in the 
    top-k retrieved samples.
    
    Parameters
    ----------
    distmat : np.ndarray
        Distance matrix of shape [num_query, num_gallery].
    query_labels : np.ndarray
        Query labels.
    gallery_labels : np.ndarray
        Gallery labels.
    max_rank : int
        Maximum rank to compute.
    
    Returns
    -------
    np.ndarray
        CMC curve of shape [max_rank].
    """
    num_query = distmat.shape[0]
    
    # Sort by distance
    indices = np.argsort(distmat, axis=1)
    
    # Compute CMC
    cmc = np.zeros(max_rank)
    
    for i in range(num_query):
        query_label = query_labels[i]
        order = indices[i]
        
        # Find positions where gallery matches query
        matches = (gallery_labels[order] == query_label)
        
        if np.any(matches):
            # Position of first correct match
            first_match_pos = np.where(matches)[0][0]
            
            if first_match_pos < max_rank:
                cmc[first_match_pos:] += 1
    
    cmc = cmc / num_query
    
    return cmc


def compute_map(
    distmat: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
) -> float:
    """
    Compute Mean Average Precision (mAP).
    
    Average Precision (AP) for a query is the average of precision values at
    positions where relevant documents are retrieved.
    
    mAP is the mean of AP across all queries.
    
    Parameters
    ----------
    distmat : np.ndarray
        Distance matrix.
    query_labels : np.ndarray
        Query labels.
    gallery_labels : np.ndarray
        Gallery labels.
    
    Returns
    -------
    float
        Mean Average Precision.
    """
    num_query = distmat.shape[0]
    
    # Sort by distance (ascending)
    indices = np.argsort(distmat, axis=1)
    
    aps = []
    
    for i in range(num_query):
        query_label = query_labels[i]
        order = indices[i]
        
        # Binary relevance: 1 if same identity, 0 otherwise
        matches = (gallery_labels[order] == query_label).astype(int)
        
        if np.sum(matches) == 0:
            continue
        
        # Compute Average Precision
        num_matches = np.sum(matches)
        matches_cumsum = np.cumsum(matches)
        precision = matches_cumsum / (np.arange(len(matches)) + 1)
        
        # Average precision = sum of precisions at relevant positions / num relevant
        ap = np.sum(precision * matches) / num_matches
        aps.append(ap)
    
    if len(aps) == 0:
        return 0.0
    
    return np.mean(aps)


def evaluate_gait(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    metric: str = 'euclidean',
) -> Dict[str, float]:
    """
    Complete evaluation pipeline for gait recognition.
    
    Parameters
    ----------
    query_features : torch.Tensor
        Query features.
    gallery_features : torch.Tensor
        Gallery features.
    query_labels : np.ndarray
        Query labels.
    gallery_labels : np.ndarray
        Gallery labels.
    metric : str
        Distance metric.
    
    Returns
    -------
    Dict[str, float]
        Evaluation results.
    """
    # Compute distance matrix
    distmat = compute_distance_matrix(query_features, gallery_features, metric=metric)
    
    # Evaluate
    results = evaluate_rank(
        distmat=distmat,
        query_labels=query_labels,
        gallery_labels=gallery_labels,
        max_rank=50,
    )
    
    return results


class AverageMeter:
    """
    Computes and stores the average and current value.
    
    Useful for tracking metrics during training.
    """
    
    def __init__(self):
        """Initialize meter."""
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update statistics.
        
        Parameters
        ----------
        val : float
            New value.
        n : int
            Number of samples.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
    
    def __repr__(self) -> str:
        """String representation."""
        return f"AverageMeter(val={self.val:.4f}, avg={self.avg:.4f})"

