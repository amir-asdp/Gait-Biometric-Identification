"""
Visualization utilities for gait recognition results.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE


def plot_cmc_curve(
    cmc: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Cumulative Match Characteristic (CMC) Curve",
):
    """
    Plot CMC curve.
    
    Parameters
    ----------
    cmc : np.ndarray
        CMC values of shape [max_rank].
    save_path : Optional[str]
        Path to save the plot.
    title : str
        Plot title.
    """
    plt.figure(figsize=(10, 6))
    
    ranks = np.arange(1, len(cmc) + 1)
    plt.plot(ranks, cmc * 100, linewidth=2)
    
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Recognition Rate (%)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim([1, len(cmc)])
    plt.ylim([(np.min(cmc) * 100) - 5, 105])
    
    # Add rank-1, rank-5, rank-10 annotations
    for rank in [1, 5, 10]:
        if rank <= len(cmc):
            plt.axvline(x=rank, color='r', linestyle='--', alpha=0.5)
            plt.text(rank, cmc[rank-1] * 100 + 2, 
                    f'Rank-{rank}: {cmc[rank-1]*100:.2f}%',
                    fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"CMC curve saved to {save_path}")
    
    plt.close()


def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "t-SNE Visualization",
    n_components: int = 2,
    perplexity: float = 30.0,
    max_classes: int = 20,
):
    """
    Plot t-SNE visualization of features.
    
    Parameters
    ----------
    features : np.ndarray
        Feature vectors of shape [num_samples, feature_dim].
    labels : np.ndarray
        Class labels of shape [num_samples].
    save_path : Optional[str]
        Path to save the plot.
    title : str
        Plot title.
    n_components : int
        Number of dimensions for t-SNE.
    perplexity : float
        Perplexity parameter for t-SNE.
    max_classes : int
        Maximum number of classes to plot (for clarity).
    """
    print("Computing t-SNE embedding...")
    
    # Limit number of classes for clarity
    unique_labels = np.unique(labels)
    if len(unique_labels) > max_classes:
        print(f"Too many classes ({len(unique_labels)}), randomly selecting {max_classes}")
        selected_labels = np.random.choice(unique_labels, max_classes, replace=False)
        mask = np.isin(labels, selected_labels)
        features = features[mask]
        labels = labels[mask]
    
    # Apply t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    embeddings = tsne.fit_transform(features)
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        plt.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=[color],
            label=f'ID {label}',
            alpha=0.6,
            s=50,
        )
    
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE plot saved to {save_path}")
    
    plt.close()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
):
    """
    Plot confusion matrix.
    
    Parameters
    ----------
    confusion_matrix : np.ndarray
        Confusion matrix of shape [num_classes, num_classes].
    save_path : Optional[str]
        Path to save the plot.
    title : str
        Plot title.
    normalize : bool
        Whether to normalize the confusion matrix.
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(
        confusion_matrix,
        annot=False,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        cbar=True,
        square=True,
    )
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_training_curves(
    train_losses: list,
    val_metrics: dict = None,
    save_path: Optional[str] = None,
):
    """
    Plot training curves.
    
    Parameters
    ----------
    train_losses : list
        List of training losses per epoch.
    val_metrics : dict
        Dictionary of validation metrics per epoch.
    save_path : Optional[str]
        Path to save the plot.
    """
    fig, axes = plt.subplots(1, 2 if val_metrics else 1, figsize=(15, 5))
    
    if not val_metrics:
        axes = [axes]
    
    # Plot training loss
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot validation metrics
    if val_metrics:
        for metric_name, values in val_metrics.items():
            axes[1].plot(epochs[:len(values)], values, linewidth=2, label=metric_name)
        
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Validation Metrics', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.close()


def plot_distance_distribution(
    positive_distances: np.ndarray,
    negative_distances: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Distance Distribution",
):
    """
    Plot distribution of positive and negative pair distances.
    
    Parameters
    ----------
    positive_distances : np.ndarray
        Distances between positive pairs (same identity).
    negative_distances : np.ndarray
        Distances between negative pairs (different identity).
    save_path : Optional[str]
        Path to save the plot.
    title : str
        Plot title.
    """
    plt.figure(figsize=(10, 6))
    
    plt.hist(positive_distances, bins=50, alpha=0.6, label='Positive Pairs', color='green')
    plt.hist(negative_distances, bins=50, alpha=0.6, label='Negative Pairs', color='red')
    
    plt.xlabel('Distance', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distance distribution saved to {save_path}")
    
    plt.close()

