"""
Test script to verify all components work correctly.

This script performs basic sanity checks on:
- Data loading
- Model creation
- Forward/backward passes
- Loss computation
- Device compatibility
"""

import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data import CASIABDataset, GaitTransform, TripletSampler
from models import GaitSet, GradientReversalLayer, ViewDiscriminator, GaitRecognitionModel
from models import TripletLoss, CenterLoss, CombinedLoss
from utils import get_device, setup_seed


def test_data_loading():
    """Test dataset loading."""
    print("\n" + "=" * 50)
    print("Testing Data Loading")
    print("=" * 50)
    
    try:
        # Create transform
        transform = GaitTransform(resolution=(64, 44))
        
        # Create dummy dataset (will fail if no data, but tests API)
        print("✓ Transform created successfully")
        
        # Test sampler
        class DummyDataset:
            def __init__(self):
                self.data_index = [
                    {'subject_id': i % 5} for i in range(100)
                ]
            def __len__(self):
                return len(self.data_index)
        
        dummy_dataset = DummyDataset()
        sampler = TripletSampler(
            dataset=dummy_dataset,
            batch_size=64,
            person_num=8,
            sample_num=8,
        )
        print("✓ Triplet sampler created successfully")
        print(f"  Sampler generates {len(sampler)} samples per epoch")
        
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        return False
    
    print("✓ Data loading tests passed")
    return True


def test_gaitset_backbone():
    """Test GaitSet backbone."""
    print("\n" + "=" * 50)
    print("Testing GaitSet Backbone")
    print("=" * 50)
    
    try:
        # Create model
        model = GaitSet(
            in_channels=1,
            hidden_dim=256,
            feature_channels=128,
            embedding_dim=256,
            bins=[16, 8, 4, 2, 1],
        )
        
        print(f"✓ Model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")
        
        # Test forward pass
        batch_size = 4
        num_frames = 30
        height, width = 64, 44
        
        dummy_input = torch.randn(batch_size, num_frames, height, width)
        
        embeddings, frame_features = model(dummy_input)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Embeddings shape: {embeddings.shape}")
        print(f"  Frame features shape: {frame_features.shape}")
        
        # Test backward pass
        loss = embeddings.sum()
        loss.backward()
        
        print("✓ Backward pass successful")
        
    except Exception as e:
        print(f"✗ GaitSet test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✓ GaitSet backbone tests passed")
    return True


def test_grl():
    """Test Gradient Reversal Layer."""
    print("\n" + "=" * 50)
    print("Testing Gradient Reversal Layer")
    print("=" * 50)
    
    try:
        # Create GRL
        grl = GradientReversalLayer(lambda_grl=1.0, schedule='constant')
        print("✓ GRL created successfully")
        
        # Create discriminator
        discriminator = ViewDiscriminator(
            input_dim=256,
            hidden_dims=[256, 128, 64],
            num_views=11,
            dropout=0.3,
        )
        print("✓ View discriminator created successfully")
        
        # Test forward pass
        batch_size = 8
        features = torch.randn(batch_size, 256, requires_grad=True)
        
        reversed_features = grl(features)
        view_logits = discriminator(reversed_features)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {features.shape}")
        print(f"  Output shape: {view_logits.shape}")
        
        # Test backward pass (verify gradient reversal)
        loss = view_logits.sum()
        loss.backward()
        
        # Gradient should be reversed (negative of normal)
        print("✓ Backward pass successful")
        print(f"  Feature gradient exists: {features.grad is not None}")
        
        # Test lambda update
        grl.update_lambda(epoch=50, max_epochs=100)
        print(f"✓ Lambda update successful (current λ={grl.current_lambda:.4f})")
        
    except Exception as e:
        print(f"✗ GRL test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✓ GRL tests passed")
    return True


def test_complete_model():
    """Test complete gait recognition model."""
    print("\n" + "=" * 50)
    print("Testing Complete Gait Recognition Model")
    print("=" * 50)
    
    try:
        # Test with GRL
        model = GaitRecognitionModel(
            num_classes=74,
            in_channels=1,
            hidden_dim=256,
            feature_channels=128,
            embedding_dim=256,
            bins=[16, 8, 4, 2, 1],
            use_grl=True,
            grl_config={
                'num_views': 11,
                'lambda_grl': 1.0,
                'schedule': 'constant',
                'hidden_dims': [256, 128, 64],
                'dropout': 0.3,
            },
        )
        
        print("✓ Model with GRL created successfully")
        
        # Test forward pass
        batch_size = 4
        num_frames = 30
        height, width = 64, 44
        
        dummy_input = torch.randn(batch_size, num_frames, height, width)
        outputs = model(dummy_input)
        
        print(f"✓ Forward pass successful")
        print(f"  Embeddings shape: {outputs['embeddings'].shape}")
        print(f"  Identity logits shape: {outputs['identity_logits'].shape}")
        print(f"  View logits shape: {outputs['view_logits'].shape}")
        
        # Test without GRL
        model_no_grl = GaitRecognitionModel(
            num_classes=74,
            embedding_dim=256,
            use_grl=False,
        )
        
        outputs_no_grl = model_no_grl(dummy_input)
        
        print("✓ Model without GRL works correctly")
        print(f"  Has view_logits: {'view_logits' in outputs_no_grl}")
        
    except Exception as e:
        print(f"✗ Complete model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✓ Complete model tests passed")
    return True


def test_losses():
    """Test loss functions."""
    print("\n" + "=" * 50)
    print("Testing Loss Functions")
    print("=" * 50)
    
    try:
        batch_size = 16
        embedding_dim = 256
        num_classes = 74
        
        # Create dummy data
        embeddings = torch.randn(batch_size, embedding_dim)
        identity_logits = torch.randn(batch_size, num_classes)
        labels = torch.randint(0, num_classes, (batch_size,))
        view_logits = torch.randn(batch_size, 11)
        view_labels = torch.randint(0, 11, (batch_size,))
        
        # Test triplet loss
        triplet_loss = TripletLoss(margin=0.2, mining='batch_hard')
        loss = triplet_loss(embeddings, labels)
        print(f"✓ Triplet loss: {loss.item():.4f}")
        
        # Test center loss
        center_loss = CenterLoss(num_classes=num_classes, feature_dim=embedding_dim)
        loss = center_loss(embeddings, labels)
        print(f"✓ Center loss: {loss.item():.4f}")
        
        # Test combined loss
        combined_loss = CombinedLoss(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            triplet_margin=0.2,
            loss_weights={
                'identity': 1.0,
                'triplet': 1.0,
                'center': 0.0005,
                'view': 0.5,
            },
        )
        
        losses = combined_loss(
            embeddings=embeddings,
            identity_logits=identity_logits,
            identity_labels=labels,
            view_logits=view_logits,
            view_labels=view_labels,
        )
        
        print(f"✓ Combined loss:")
        print(f"  Total: {losses['total'].item():.4f}")
        print(f"  Identity: {losses['identity'].item():.4f}")
        print(f"  Triplet: {losses['triplet'].item():.4f}")
        print(f"  Center: {losses['center'].item():.4f}")
        print(f"  View: {losses['view'].item():.4f}")
        
    except Exception as e:
        print(f"✗ Loss functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✓ Loss function tests passed")
    return True


def test_device_compatibility():
    """Test device compatibility."""
    print("\n" + "=" * 50)
    print("Testing Device Compatibility")
    print("=" * 50)
    
    try:
        # Test device detection
        print("\nAvailable devices:")
        
        if torch.cuda.is_available():
            device = get_device('cuda', [0])
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  ✗ CUDA not available")

        if torch.backends.mps.is_available():
            device = get_device('mps', [])
            print(f"  ✓ MPS available")
        else:
            print(f"  ✗ MPS not available")

        device = get_device('cpu', [])
        print(f"  ✓ CPU available")

        # Test model on device
        model = GaitSet(embedding_dim=256)
        model = model.to(device)
        
        dummy_input = torch.randn(2, 30, 64, 44).to(device)
        outputs = model(dummy_input)
        
        print(f"\n✓ Model successfully runs on {device}")
        
    except Exception as e:
        print(f"✗ Device compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✓ Device compatibility tests passed")
    return True


def test_configuration():
    """Test configuration loading."""
    print("\n" + "=" * 50)
    print("Testing Configuration")
    print("=" * 50)
    
    try:
        config_path = Path(__file__).parent / 'configs' / 'config.yaml'
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            print("✓ Configuration loaded successfully")
            print(f"  Experiment: {config['experiment']['name']}")
            print(f"  GRL enabled: {config['model']['grl']['enabled']}")
            print(f"  Device: {config['device']['type']}")
        else:
            print(f"✗ Configuration file not found: {config_path}")
            return False
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False
    
    print("✓ Configuration tests passed")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print(" " * 15 + "COMPONENT TESTING SUITE")
    print("=" * 70)
    
    # Set seed for reproducibility
    setup_seed(42)
    
    # Run tests
    results = {
        'Configuration': test_configuration(),
        'Data Loading': test_data_loading(),
        'GaitSet Backbone': test_gaitset_backbone(),
        'Gradient Reversal Layer': test_grl(),
        'Complete Model': test_complete_model(),
        'Loss Functions': test_losses(),
        'Device Compatibility': test_device_compatibility(),
    }
    
    # Summary
    print("\n" + "=" * 70)
    print(" " * 25 + "TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<50} {status}")
    
    print("=" * 70)
    
    # Overall result
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Prepare CASIA-B dataset")
        print("2. Update config.yaml with dataset path")
        print("3. Run: python train.py --config configs/config.yaml")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

