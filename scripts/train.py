"""
Training script for gait biometric identification.

This script implements the complete training pipeline including:
- Model initialization
- Data loading with gallery/probe split
- Training with GRL (optional)
- Validation and evaluation
- Checkpointing and logging
"""

import argparse
import os
import sys
import time
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import get_dataloader, TripletSampler, CASIABDataset
from models import GaitRecognitionModel, build_model, CombinedLoss
from utils import (
    get_device,
    setup_seed,
    print_system_info,
    print_model_info,
    AverageMeter,
    evaluate_gait,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Gait Recognition Model')
    parser.add_argument(
        '--config',
        type=str,
        default='/gait_biometric_identification/configs/config.yaml',
        help='Path to configuration file',
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from',
    )
    parser.add_argument(
        '--eval_only',
        action='store_true',
        help='Only run evaluation',
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to config file.
    
    Returns
    -------
    dict
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    save_path: str,
    best_metric: float = 0.0,
):
    """
    Save model checkpoint.
    
    Parameters
    ----------
    model : nn.Module
        Model to save.
    optimizer : optim.Optimizer
        Optimizer state.
    scheduler
        Learning rate scheduler.
    epoch : int
        Current epoch.
    save_path : str
        Path to save checkpoint.
    best_metric : float
        Best validation metric.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_metric': best_metric,
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: optim.Optimizer = None,
    scheduler = None,
) -> dict:
    """
    Load model checkpoint.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint.
    model : nn.Module
        Model to load weights into.
    optimizer : optim.Optimizer, optional
        Optimizer to load state.
    scheduler : optional
        Scheduler to load state.
    
    Returns
    -------
    dict
        Checkpoint information.
    """
    checkpoint = torch.load(f=checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"  - Epoch: {checkpoint['epoch'] + 1}")
    print(f"  - Best metric (Rank-1): {checkpoint.get('best_metric', 0.0) * 100:.2f}")
    
    return checkpoint


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: CombinedLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: dict,
    writer: SummaryWriter = None,
) -> dict:
    """
    Train for one epoch.
    
    Parameters
    ----------
    model : nn.Module
        Model to train.
    dataloader
        Training data loader.
    criterion : CombinedLoss
        Loss function.
    optimizer : optim.Optimizer
        Optimizer.
    device : torch.device
        Computation device.
    epoch : int
        Current epoch.
    config : dict
        Configuration.
    writer : SummaryWriter, optional
        TensorBoard writer.
    
    Returns
    -------
    dict
        Training statistics.
    """
    model.train()
    
    # Meters for tracking
    loss_meter = AverageMeter()
    identity_loss_meter = AverageMeter()
    triplet_loss_meter = AverageMeter()
    center_loss_meter = AverageMeter()
    view_loss_meter = AverageMeter()
    
    num_iters = len(dataloader)
    
    for i, batch in enumerate(dataloader):
        start_time = time.time()

        # Move data to device
        silhouettes = batch['silhouettes'].to(device)
        subject_ids = batch['subject_ids'].to(device)
        view_angles = batch['view_angles'].to(device)
        
        # Forward pass
        outputs = model(silhouettes)
        
        embeddings = outputs['embeddings']
        identity_logits = outputs['identity_logits']
        view_logits = outputs.get('view_logits', None)
        
        # Compute loss
        losses = criterion(
            embeddings=embeddings,
            identity_logits=identity_logits,
            identity_labels=subject_ids,
            view_logits=view_logits,
            view_labels=view_angles // 18,  # Convert angle to class (0-10)
        )
        
        total_loss = losses['total']
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if config['training']['grad_clip']['enabled']:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['grad_clip']['max_norm'],
            )
        
        optimizer.step()
        
        # Update meters
        batch_size = silhouettes.size(0)
        loss_meter.update(total_loss.item(), batch_size)
        identity_loss_meter.update(losses['identity'].item(), batch_size)
        triplet_loss_meter.update(losses['triplet'].item(), batch_size)
        center_loss_meter.update(losses['center'].item(), batch_size)
        view_loss_meter.update(losses['view'].item(), batch_size)
        
        # Log
        if (i + 1) % config['logging']['print_freq'] == 0:
            elapsed = time.time() - start_time
            sys.stdout.write(
                "  "
                f"Batch: [{i + 1}/{num_iters}] "
                f"Loss: {loss_meter.avg:.4f} "
                f"ID: {identity_loss_meter.avg:.4f}, "
                f"Tri: {triplet_loss_meter.avg:.4f}, "
                f"Cen: {center_loss_meter.avg:.4f}, "
                f"View: {view_loss_meter.avg:.4f}) "
                f"Time: {elapsed:.2f}s\n"
            )
            
            # TensorBoard logging
            if writer:
                global_step = epoch * num_iters + i
                writer.add_scalar('train/loss', loss_meter.avg, global_step)
                writer.add_scalar('train/identity_loss', identity_loss_meter.avg, global_step)
                writer.add_scalar('train/triplet_loss', triplet_loss_meter.avg, global_step)
                writer.add_scalar('train/center_loss', center_loss_meter.avg, global_step)
                writer.add_scalar('train/view_loss', view_loss_meter.avg, global_step)
    
    stats = {
        'loss': loss_meter.avg,
        'identity_loss': identity_loss_meter.avg,
        'triplet_loss': triplet_loss_meter.avg,
        'center_loss': center_loss_meter.avg,
        'view_loss': view_loss_meter.avg,
    }
    
    return stats


@torch.no_grad()
def evaluate(
    model: nn.Module,
    gallery_loader,
    probe_loader,
    device: torch.device,
    config: dict,
) -> dict:
    """
    Evaluate model on gallery-probe split.
    
    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    gallery_loader
        Gallery data loader.
    probe_loader
        Probe data loader.
    device : torch.device
        Computation device.
    config : dict
        Configuration.
    
    Returns
    -------
    dict
        Evaluation results.
    """
    model.eval()
    
    print("Extracting gallery features...")
    gallery_features = []
    gallery_labels = []
    
    for batch in gallery_loader:
        silhouettes = batch['silhouettes'].to(device)
        subject_ids = batch['subject_ids']
        
        features = model.extract_features(silhouettes)
        gallery_features.append(features.cpu())
        gallery_labels.append(subject_ids)
    
    gallery_features = torch.cat(gallery_features, dim=0)
    gallery_labels = torch.cat(gallery_labels, dim=0).numpy()
    
    print("Extracting probe features...")
    probe_features = []
    probe_labels = []
    
    for batch in probe_loader:
        silhouettes = batch['silhouettes'].to(device)
        subject_ids = batch['subject_ids']
        
        features = model.extract_features(silhouettes)
        probe_features.append(features.cpu())
        probe_labels.append(subject_ids)
    
    probe_features = torch.cat(probe_features, dim=0)
    probe_labels = torch.cat(probe_labels, dim=0).numpy()
    
    print(f"Gallery: {len(gallery_labels)} samples, Probe: {len(probe_labels)} samples")
    
    # Evaluate
    results = evaluate_gait(
        query_features=probe_features,
        gallery_features=gallery_features,
        query_labels=probe_labels,
        gallery_labels=gallery_labels,
        metric=config['evaluation']['metric'],
    )
    
    return results


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup
    print_system_info()
    setup_seed(config['experiment']['seed'])
    device = get_device(
        config['device']['type'],
        config['device']['gpu_ids'],
    )
    
    # Create output directory
    experiment_output_folder_name = f"{config['experiment']['description_label']}-v{config['experiment']['version']}"
    output_dir = Path(f"{config['experiment']['output_dir']}/{experiment_output_folder_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(args.config).absolute(), Path(f"{output_dir}/{experiment_output_folder_name}-config.yaml").absolute())
    
    # Create TensorBoard writer
    writer = None
    if config['logging']['use_tensorboard']:
        tensorboard_dir = f"{output_dir}/tensorboard"
        writer = SummaryWriter(tensorboard_dir)
    
    # Build model
    num_train_classes = (
        config['dataset']['train']['subjects'][1] -
        config['dataset']['train']['subjects'][0] + 1
    )
    
    model = build_model(config, num_classes=num_train_classes)
    model = model.to(device)
    
    print_model_info(model)
    
    # Build loss
    loss_weights = {
        'identity': config['loss']['identity_loss']['weight'],
        'triplet': config['loss']['triplet_loss']['weight'] if config['loss']['triplet_loss']['enabled'] else 0,
        'center': config['loss']['center_loss']['weight'] if config['loss']['center_loss']['enabled'] else 0,
        'view': config['loss']['view_loss']['weight'],
    }
    
    criterion = CombinedLoss(
        num_classes=num_train_classes,
        embedding_dim=config['model']['backbone']['embedding_dim'],
        triplet_margin=config['loss']['triplet_loss'].get('margin', 0.2),
        loss_weights=loss_weights,
    )
    criterion = criterion.to(device)
    
    # Build optimizer
    if config['training']['optimizer']['type'] == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['optimizer']['lr'],
            weight_decay=config['training']['optimizer']['weight_decay'],
            betas=config['training']['optimizer']['betas'],
            amsgrad=config['training']['optimizer']['amsgrad'],
        )
    elif config['training']['optimizer']['type'] == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['optimizer']['lr'],
            weight_decay=config['training']['optimizer']['weight_decay'],
            betas=config['training']['optimizer']['betas'],
            amsgrad=config['training']['optimizer']['amsgrad'],
        )
    else:
        raise NotImplementedError(f"Optimizer {config['training']['optimizer']['type']} not implemented")
    
    # Build scheduler
    scheduler = None
    if config['training']['scheduler']['type'] == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config['training']['scheduler']['milestones'],
            gamma=config['training']['scheduler']['gamma'],
        )
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_rank1 = 0.0
    
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = checkpoint['epoch'] + 1
        best_rank1 = checkpoint.get('best_metric', 0.0)
    elif config['training']['resume']:
        checkpoint = load_checkpoint(config['training']['resume'], model, optimizer, scheduler)
        start_epoch = checkpoint['epoch'] + 1
        best_rank1 = checkpoint.get('best_metric', 0.0)

    # Build dataloaders
    print(f"\n{"="*50}")
    print("Preparing datasets...\n")

    # Create training sampler
    dataset_cfg = config['dataset']
    train_dataset = CASIABDataset(
        data_root=dataset_cfg['data_root'],
        subjects=list(range(
            dataset_cfg['train']['subjects'][0],
            dataset_cfg['train']['subjects'][1] + 1,
        )),
        conditions=dataset_cfg['train']['conditions'],
        views='all',
        frame_num=dataset_cfg['input']['frame_num'],
        sample_type=dataset_cfg['input']['sample_type'],
        cache=dataset_cfg['cache_enabled'],
        dataset_tag="train_data",
    )
    
    train_sampler = TripletSampler(
        dataset=train_dataset,
        batch_size=config['training']['batch_size'],
        person_num=config['training']['person_num'],
        sample_num=config['training']['sample_num'],
    )
    
    train_loader = get_dataloader(config, mode='train', sampler=train_sampler)
    gallery_loader = get_dataloader(config, mode='gallery')
    probe_loader = get_dataloader(config, mode='probe')
    print(f"{"="*50}\n")
    
    # Training loop
    if not args.eval_only:
        print("\n" + "=" * 50)
        print("Starting Training")
        print("=" * 50)
        
        for epoch in range(start_epoch, config['training']['num_epochs']):
            start_time = time.time()

            print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
            
            # Update GRL lambda
            if config['model']['grl']['enabled']:
                model.update_grl_lambda(epoch, config['training']['num_epochs'])
            
            # Train
            train_stats = train_one_epoch(
                model, train_loader, criterion, optimizer,
                device, epoch, config, writer,
            )
            
            # Step scheduler
            if scheduler:
                scheduler.step()
                if writer:
                    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
            
            # Evaluate
            if (epoch + 1) % config['experiment']['save_checkpoint_freq'] == 0:
                print("\nEvaluating...")
                eval_results = evaluate(
                    model, gallery_loader, probe_loader, device, config,
                )
                
                print(f"\nEvaluation Results:")
                print(f"  Rank-1: {eval_results['rank1']*100:.2f}%")
                print(f"  Rank-5: {eval_results['rank5']*100:.2f}%")
                print(f"  Rank-10: {eval_results['rank10']*100:.2f}%")
                print(f"  mAP: {eval_results['mAP']*100:.2f}%")
                
                # TensorBoard logging
                if writer:
                    writer.add_scalar('eval/rank1', eval_results['rank1'] * 100, epoch)
                    writer.add_scalar('eval/rank5', eval_results['rank5'] * 100, epoch)
                    writer.add_scalar('eval/rank10', eval_results['rank10'] * 100, epoch)
                    writer.add_scalar('eval/mAP', eval_results['mAP'] * 100, epoch)
                
                # Save checkpoint
                is_best = eval_results['rank1'] > best_rank1
                if is_best:
                    best_rank1 = eval_results['rank1']
                    save_checkpoint(
                        model, optimizer, scheduler, epoch,
                        f'{output_dir}/best_model.pth',
                        best_rank1,
                    )
                
                # Save regular checkpoint
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    f'{output_dir}/checkpoint_epoch_{epoch+1}.pth',
                    best_rank1,
                )

            total_epoch_time = time.time() - start_time
            print(f"-------- Total Epoch Time: {total_epoch_time:.2f}s --------\n")
        
        print("\n" + "=" * 50)
        print("Training Complete")
        print(f"Best Rank-1: {best_rank1*100:.2f}%")
        print("=" * 50)
    
    # Final evaluation
    print("\nFinal Evaluation...")
    eval_results = evaluate(
        model, gallery_loader, probe_loader, device, config,
    )
    
    print(f"\nFinal Results:")
    print(f"  Rank-1: {eval_results['rank1']*100:.2f}%")
    print(f"  Rank-5: {eval_results['rank5']*100:.2f}%")
    print(f"  Rank-10: {eval_results['rank10']*100:.2f}%")
    print(f"  mAP: {eval_results['mAP']*100:.2f}%")
    
    if writer:
        writer.close()


if __name__ == '__main__':
    main()

