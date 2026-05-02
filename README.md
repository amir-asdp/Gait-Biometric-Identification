# Gait Biometric Identification System

**University Undergraduate Thesis**: A state-of-the-art gait recognition system with Gradient Reversal Layer (GRL) for view-invariant learning.

## Features

- ✨ **Architecture**: Set-based deep learning for gait recognition
- 🔄 **Gradient Reversal Layer**: Domain adaptation for view-invariant features
- 🚀 **Multi-Device Support**: CUDA, MPS (Apple Silicon), and CPU
- 📊 **Comprehensive Evaluation**: Rank-k accuracy, mAP, CMC curves
- ⚙️ **Flexible Configuration**: YAML-based configuration system
- 📈 **TensorBoard Integration**: Real-time training monitoring

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Dataset Preparation

Download CASIA-B dataset and organize it as:
```
CASIA-B/casiab-128-end2end/
├── 001/
│   ├── nm-01/
│   │   ├── 000/
│   │   │   └── 000-sils.pkl
│   │   └── ...
│   └── ...
└── ...
```

Update the dataset path in `configs/config.yaml`:
```yaml
dataset:
  data_root: "/path/to/CASIA-B/casiab-128-end2end"
```

### Training

```bash
# Train with GRL enabled
python train.py --config configs/config.yaml

# Or use the shell script
cd scripts
./train.sh
```

### Evaluation

```bash
# Evaluate trained model
python scripts/evaluate.py \
    --config configs/config.yaml \
    --checkpoint output/best_model.pth \
    --output_dir evaluation_results \
    --visualize
```

### Monitoring

```bash
# Launch TensorBoard
tensorboard --logdir output/tensorboard
```

## Project Structure

```
gait_biometric_identification/
├── configs/              # Configuration files
├── data/                 # Dataset loaders and transforms
├── models/               # Model architectures and losses
├── utils/                # Utilities (metrics, visualization, device)
├── scripts/              # Training and evaluation scripts
├── train.py              # Main training script
├── requirements.txt      # Dependencies
├── wiki.md              # Comprehensive documentation
└── README.md            # This file
```

## Configuration

### Enable/Disable GRL

In `configs/config.yaml`:

```yaml
model:
  grl:
    enabled: true  # Set to false to disable GRL
    lambda_grl: 1.0
    schedule: "constant"  # or "progressive"
```

### Device Selection

```yaml
device:
  type: "cuda"  # Options: "cuda", "mps", "cpu"
  gpu_ids: [0]
```

### Training Parameters

```yaml
training:
  batch_size: 8
  person_num: 8  # P persons per batch
  sample_num: 16  # K samples per person
  num_epochs: 200
  
  optimizer:
    lr: 0.0001
```

## Results

Best performance on CASIA-B:

| Setting | Rank-1 | Rank-5 | mAP    |
|---------|--------|--------|--------|
| With GRL | 98.95% | 99.83% | 82.16% |
| Without GRL | 98.78% | 99.48% | 79.99% |

*Note: Results vary based on training settings and random seed*

## Key Components

### 1. Backbone
- Set-based feature extraction
- Horizontal Pyramid Pooling
- Temporal aggregation (max + mean)

### 2. Gradient Reversal Layer (GRL)
- Domain adaptation for view angles
- Adversarial training
- Configurable lambda scheduling

### 3. Loss Functions
- Identity classification loss
- Triplet loss (batch hard mining)
- Center loss (intra-class compactness)
- View classification loss (for GRL)

### 4. Evaluation Metrics
- Rank-k accuracy (k=1, 5, 10)
- Mean Average Precision (mAP)
- Cumulative Match Characteristic (CMC)

## Documentation

See [wiki.md](wiki.md) for comprehensive documentation including:
- Detailed architecture explanation
- Mathematical formulations
- Training pipeline details
- Troubleshooting guide
- Advanced topics

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA >= 11.0 (for GPU training)
- See [requirements.txt](requirements.txt) for complete list

## License

This project is for academic research purposes.

## Acknowledgments

- CASIA-B dataset providers
- PyTorch team

## Contact

For questions or issues, please refer to the [wiki.md](wiki.md) troubleshooting section.

