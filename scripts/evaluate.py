"""
Evaluation script for gait biometric identification.

Performs comprehensive evaluation including:
- Rank-k accuracy computation
- CMC curve generation
- t-SNE visualization
- Cross-view evaluation
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import get_dataloader
from models import build_model
from utils import (
    get_device,
    setup_seed,
    evaluate_gait,
    plot_cmc_curve,
    plot_tsne,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Gait Recognition Model')
    parser.add_argument(
        '--config',
        type=str,
        default='/gait_biometric_identification/configs/config.yaml',
        help='Path to configuration file',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for results',
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualizations',
    )
    return parser.parse_args()


@torch.no_grad()
def extract_all_features(model, dataloader, device):
    """Extract features for all samples."""
    model.eval()
    
    all_features = []
    all_labels = []
    all_views = []
    
    for batch in dataloader:
        silhouettes = batch['silhouettes'].to(device)
        subject_ids = batch['subject_ids']
        view_angles = batch['view_angles']
        
        features = model.extract_features(silhouettes)
        
        all_features.append(features.cpu())
        all_labels.append(subject_ids)
        all_views.append(view_angles)
    
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_views = torch.cat(all_views, dim=0).numpy()
    
    return all_features, all_labels, all_views


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    setup_seed(config['experiment']['seed'])
    device = get_device(config['device']['type'], config['device']['gpu_ids'])

    description_tag = config['evaluation']['description_tag']
    eval_result_folder = f"/{description_tag}" if description_tag is not None else ""
    
    # Create output directory
    output_dir = Path(f"{args.output_dir}/evaluation results" + eval_result_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build model
    num_train_classes = (
        config['dataset']['train']['subjects'][1] -
        config['dataset']['train']['subjects'][0] + 1
    )
    
    model = build_model(config, num_classes=num_train_classes)
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(f=args.checkpoint, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Build dataloaders
    gallery_loader = get_dataloader(config, mode='gallery')
    probe_loader = get_dataloader(config, mode='probe')
    
    # Extract features
    print("Extracting gallery features...")
    gallery_features, gallery_labels, gallery_views = extract_all_features(
        model, gallery_loader, device
    )
    
    print("Extracting probe features...")
    probe_features, probe_labels, probe_views = extract_all_features(
        model, probe_loader, device
    )
    
    # Evaluate
    print("\nEvaluating...")
    results = evaluate_gait(
        query_features=probe_features,
        gallery_features=gallery_features,
        query_labels=probe_labels,
        gallery_labels=gallery_labels,
        metric=config['evaluation']['metric'],
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Rank-1 Accuracy:  {results['rank1']*100:.2f}%")
    print(f"Rank-5 Accuracy:  {results['rank5']*100:.2f}%")
    print(f"Rank-10 Accuracy: {results['rank10']*100:.2f}%")
    print(f"mAP:              {results['mAP']*100:.2f}%")
    print("=" * 50)
    
    # Save results
    results_file = f"{output_dir}/results.txt"
    with open(results_file, 'w') as f:
        f.write("Gait Recognition Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Rank-1:  {results['rank1']*100:.2f}%\n")
        f.write(f"Rank-5:  {results['rank5']*100:.2f}%\n")
        f.write(f"Rank-10: {results['rank10']*100:.2f}%\n")
        f.write(f"mAP:     {results['mAP']*100:.2f}%\n")
    
    print(f"\nResults saved to {results_file}")
    
    # Visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        
        # CMC curve
        plot_cmc_curve(
            results['cmc'],
            save_path=f"{output_dir}/cmc_curve.png",
        )
        
        # t-SNE visualization
        # Combine gallery and probe for visualization
        all_features = torch.cat([gallery_features, probe_features], dim=0)
        all_labels = np.concatenate([gallery_labels, probe_labels], axis=0)
        
        plot_tsne(
            features=all_features.numpy(),
            labels=all_labels,
            save_path=f"{output_dir}/tsne_visualization.png",
            max_classes=len(np.unique(all_labels)),
        )
        
        print(f"Visualizations saved to {output_dir}")


if __name__ == '__main__':
    main()

