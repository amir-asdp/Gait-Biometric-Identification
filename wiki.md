# Gait Biometric Identification System Complete Documentation

## 1. Introduction

This project implements a **state-of-the-art gait biometric identification system** with an integrated **Gradient Reversal Layer (GRL)** for view-invariant learning.

### Key Features

- **Architecture**: Set-based deep learning model for gait recognition
- **Gradient Reversal Layer (GRL)**: Domain adaptation for view-invariant features
- **Multi-Device Support**: CUDA (NVIDIA), MPS (Apple Silicon), and CPU
- **Comprehensive Evaluation**: Rank-k accuracy, mAP, and CMC curves
- **Gallery-Probe Split**: Standard evaluation protocol for CASIA-B dataset
- **Flexible Configuration**: YAML-based configuration system

### What is Gait Recognition?

Gait recognition identifies individuals based on their walking patterns. Unlike face or fingerprint recognition, gait can be captured from a distance without subject cooperation, making it valuable for surveillance and security applications.

### Why GRL?

Traditional gait recognition models struggle when the view angle (camera position) during testing differs from training. The **Gradient Reversal Layer** solves this by forcing the model to learn features that are:
- **Discriminative** for identity classification
- **Invariant** to view angles

---

## 2. System Architecture

### High-Level Overview

```
Input Silhouettes
      ↓
[Data Loading & Preprocessing]
      ↓
[Backbone Network]
      ↓
[Feature Embeddings] ──→ [Identity Classifier]
      ↓                         ↓
[GRL (Optional)]          [Identity Loss]
      ↓
[View Discriminator]
      ↓
[View Loss]
```

### Component Interaction

1. **Data Pipeline**:
   - Loads silhouette sequences from CASIA-B dataset
   - Applies augmentation (flip, rotation, erasing)
   - Organizes batches for triplet loss training

2. **Feature Extraction**:
   - CNN backbone extracts frame-level features
   - Horizontal Pyramid Pooling aggregates spatial information
   - Temporal pooling creates sequence-level representation

3. **Multi-Task Learning**:
   - **Identity Classification**: Recognize who is walking
   - **View Invariance (via GRL)**: Learn features independent of camera angle
   - **Metric Learning**: Ensure same-person samples are similar

4. **Evaluation**:
   - Gallery-probe matching
   - Rank-k accuracy and mAP computation
   - Cross-view and cross-condition evaluation

---

## 3. Network Architecture

### 3.1 Backbone

**Purpose**: Extract discriminative gait features from silhouette sequences.

**Architecture**:

```
Input: [Batch, Frames, Height, Width]
      ↓
┌─────────────────────────────────┐
│  Frame-Level Feature Extraction │
│  ┌───────────────────────────┐  │
│  │ Conv2D + BN + LeakyReLU   │  │  32 channels
│  │ Conv2D + BN + LeakyReLU   │  │  32 channels
│  │ MaxPool2D (stride=2)      │  │
│  ├───────────────────────────┤  │
│  │ Conv2D + BN + LeakyReLU   │  │  64 channels
│  │ Conv2D + BN + LeakyReLU   │  │  64 channels
│  │ MaxPool2D (stride=2)      │  │
│  ├───────────────────────────┤  │
│  │ Global-Local Conv (128)   │  │  Multi-scale features
│  │ Global-Local Conv (128)   │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
      ↓
┌─────────────────────────────────┐
│  Horizontal Pyramid Pooling     │
│  ┌───────────────────────────┐  │
│  │ Bin 16: 16 strips         │  │
│  │ Bin 8:  8 strips          │  │
│  │ Bin 4:  4 strips          │  │
│  │ Bin 2:  2 strips          │  │
│  │ Bin 1:  1 strip           │  │
│  │ GeM Pooling per strip     │  │
│  │ Concatenate all strips    │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
      ↓
┌─────────────────────────────────┐
│  Temporal Pooling               │
│  Max Pooling + Mean Pooling     │
└─────────────────────────────────┘
      ↓
┌─────────────────────────────────┐
│  Fully Connected Layers         │
│  FC(hidden_dim) + BN + ReLU     │
│  Dropout(0.5)                   │
│  FC(embedding_dim)              │
└─────────────────────────────────┘
      ↓
Output: [Batch, Embedding_Dim]
```

**Key Components**:

1. **Global-Local Convolution (GLConv)**:
   ```
   Local Branch:  3x3 Conv
   Global Branch: 3x3 Conv → MaxPool → 3x3 Conv → Upsample
   Combine: Local + Global
   ```
   - Captures both fine-grained and coarse features
   - Improves robustness to scale variations

2. **Generalized Mean Horizontal Pyramid Pooling (GeMHPP)**:
   ```python
   For each pyramid level (16, 8, 4, 2, 1 strips):
       For each horizontal strip:
           Apply GeM pooling: (∑ x^p)^(1/p)
       Concatenate strip features
   ```
   - Multi-scale spatial aggregation
   - Learnable pooling parameter p
   - Captures both global structure and local details

3. **Temporal Pooling**:
   ```python
   Max Pooling: max(features, dim=time)
   Mean Pooling: mean(features, dim=time)
   Concatenate: [max_features, mean_features]
   ```
   - Set-based: Order-invariant
   - Handles variable sequence lengths
   - Combines complementary statistics

### 3.2 Gradient Reversal Layer (GRL)

**Purpose**: Learn view-invariant features through adversarial training.

**Mathematical Formulation**:

The GRL implements the following optimization:

$$
\min_{\theta_f, \theta_y} \max_{\theta_d} \mathcal{L}(\theta_f, \theta_y, \theta_d)
$$

where:
- $\theta_f$: Feature extractor parameters
- $\theta_y$: Identity classifier parameters
- $\theta_d$: Domain (view) discriminator parameters

$$
\mathcal{L} = \mathcal{L}_{\text{identity}}(y, \hat{y}) - \lambda \mathcal{L}_{\text{view}}(v, \hat{v})
$$

**Implementation**:

```python
# Forward pass: Identity transform
y = GRL(x)  # y = x

# Backward pass: Gradient reversal
dx = -λ * dy
```

**How It Works**:

1. **Forward Pass**:
   ```
   Features → [GRL] → View Discriminator → View Prediction
   ```
   - GRL acts as identity (features pass through unchanged)

2. **Backward Pass**:
   ```
   View Loss Gradient → [GRL] → Reversed Gradient → Feature Extractor
   ```
   - GRL multiplies gradient by -λ
   - Feature extractor receives reversed gradients
   - This forces it to learn view-invariant features

3. **Training Dynamics**:
   - **View Discriminator**: Tries to classify view angle from features
   - **Feature Extractor**: Tries to confuse the discriminator
   - **Result**: Features that work well for all views

**Lambda Scheduling**:

```python
# Constant: λ = λ_max
lambda = 1.0

# Progressive: Gradually increase from 0 to λ_max
p = epoch / max_epochs
lambda = (2 / (1 + exp(-10 * p)) - 1) * lambda_max
```

### 3.3 View Discriminator

**Architecture**:

```
Input: [Batch, Embedding_Dim]
      ↓
FC(256) → BN → ReLU → Dropout(0.3)
      ↓
FC(128) → BN → ReLU → Dropout(0.3)
      ↓
FC(64) → BN → ReLU → Dropout(0.3)
      ↓
FC(num_views=11)
      ↓
Output: [Batch, 11] (view angle logits)
```

**Purpose**: Classify view angle from features (to be confused by GRL).

---

## 4. Training Pipeline

### 4.1 Overall Training Flow

```python
for epoch in range(num_epochs):
    # 1. Update GRL lambda
    model.update_grl_lambda(epoch, num_epochs)
    
    # 2. Train one epoch
    for batch in train_loader:
        # Load data
        silhouettes, subject_ids, view_angles = batch
        
        # Forward pass
        embeddings, identity_logits = model(silhouettes)
        
        # Apply GRL (if enabled)
        view_logits = grl_module(embeddings)
        
        # Compute losses
        identity_loss = CrossEntropy(identity_logits, subject_ids)
        triplet_loss = TripletLoss(embeddings, subject_ids)
        view_loss = CrossEntropy(view_logits, view_angles)
        center_loss = CenterLoss(embeddings, subject_ids)
        
        total_loss = (w1 * identity_loss + 
                     w2 * triplet_loss + 
                     w3 * center_loss + 
                     w4 * view_loss)
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm(model.parameters(), max_norm=5.0)
        optimizer.step()
    
    # 3. Learning rate scheduling
    scheduler.step()
    
    # 4. Evaluation
    if epoch % eval_freq == 0:
        evaluate(model, gallery_loader, probe_loader)
    
    # 5. Save checkpoint
    save_checkpoint(model, optimizer, epoch)
```

### 4.2 Loss Functions

#### 4.2.1 Identity Classification Loss

```python
L_identity = CrossEntropyLoss(identity_logits, subject_ids)
```

- **Purpose**: Direct supervision for identity recognition
- **Label Smoothing**: Prevents overconfidence
  ```python
  CrossEntropyLoss(label_smoothing=0.1)
  ```

#### 4.2.2 Triplet Loss (Batch Hard Mining)

```python
For each anchor in batch:
    hardest_positive = max(distance(anchor, positive))
    hardest_negative = min(distance(anchor, negative))
    
    loss = max(hardest_positive - hardest_negative + margin, 0)

L_triplet = mean(loss)
```

- **Purpose**: Metric learning - ensure same-person samples are closer
- **Batch Hard Mining**: Select most difficult triplets
- **Margin**: Minimum separation between positive and negative pairs

**Distance Computation**:

```python
# Euclidean distance
distance = sqrt(sum((a - b)^2))

# Cosine distance
distance = 1 - (a · b) / (||a|| * ||b||)
```

#### 4.2.3 Center Loss

```python
L_center = (1/2) * sum(||embeddings - centers[labels]||^2)

# Center update
centers[label] = centers[label] - alpha * (centers[label] - mean(embeddings))
```

- **Purpose**: Intra-class compactness
- **Effect**: Pulls samples toward their class center
- **Benefits**: Reduces intra-class variance

#### 4.2.4 View Classification Loss (GRL)

```python
L_view = CrossEntropyLoss(view_logits, view_labels)
```

- **Purpose**: Adversarial training for view invariance
- **Note**: Gradient is reversed by GRL before reaching feature extractor

### 4.3 Batch Sampling Strategy

**Triplet Sampler**:

```python
Batch Structure:
- P persons per batch (e.g., 8)
- K samples per person (e.g., 16)
- Total batch size = P × K = 128

Sampling Process:
1. Randomly select P persons
2. For each person:
   - Sample K sequences (different views/conditions)
3. Construct batch of P×K samples
```

**Why This Strategy?**:
- Ensures each batch contains multiple samples from same identity
- Enables effective triplet mining within batch
- Improves metric learning efficiency

---

## 5. Evaluation Pipeline

### 5.1 Gallery-Probe Protocol

**CASIA-B Standard Protocol**:

```
Training Set:
- Subjects: 001-098 (98 subjects)
- Conditions: nm-01 to nm-06 (normal walking)
- Views: All 11 views (0°-180°)

Test Set (Gallery):
- Subjects: 099-124 (26 subjects)
- Conditions: nm-05, nm-06
- Views: All 11 views

Test Set (Probe):
- Subjects: 075-124 (same 50 subjects)
- Conditions:
  * Normal: nm-01, nm-02, nm-03, nm-04
  * Bag: bg-01, bg-02
  * Clothing: cl-01, cl-02
- Views: All 11 views
```

**Evaluation Process**:

```python
# 1. Extract features
gallery_features = extract_features(gallery_loader)
probe_features = extract_features(probe_loader)

# 2. Compute distance matrix
distance_matrix = compute_distances(probe_features, gallery_features)
# Shape: [num_probe, num_gallery]

# 3. For each probe sample
for i, probe in enumerate(probes):
    # Sort gallery by distance (ascending)
    ranking = argsort(distance_matrix[i])
    
    # Check if correct match appears in top-k
    correct_label = probe_labels[i]
    correct_positions = where(gallery_labels[ranking] == correct_label)
    
    # Update metrics
    if first_correct_position < k:
        rank_k_correct += 1
```

### 5.2 Evaluation Metrics

#### 5.2.1 Rank-k Accuracy

**Definition**: Percentage of queries where the correct match appears in the top-k retrieved samples.

```
Rank-k = (Number of queries with correct match in top-k) / (Total queries) × 100%
```

**Common Values**:
- **Rank-1**: Most important - is the top match correct?
- **Rank-5**: Is the correct match in top 5?
- **Rank-10**: Is the correct match in top 10?

#### 5.2.2 Mean Average Precision (mAP)

**Average Precision (AP)** for a single query:

```
AP = (sum of (Precision at k × relevance(k))) / (number of relevant items)

where:
- Precision at k = (correct matches in top k) / k
- relevance(k) = 1 if item k is relevant, 0 otherwise
```

**Mean Average Precision**:

```
mAP = mean(AP across all queries)
```

**Interpretation**: mAP considers the ranking quality, not just whether correct match is in top-k.

#### 5.2.3 Cumulative Match Characteristic (CMC) Curve

**Definition**: Plot of Rank-k accuracy vs. k.

```python
for k in range(1, max_rank + 1):
    CMC[k] = Rank-k Accuracy
```

**Visualization**:
```
Recognition Rate (%)
100%|                    _______________
    |                __/
    |             __/
 50%|         __/
    |     __/
  0%|___/________________________
     1    5    10    20    50   Rank
```

### 5.3 Cross-View Evaluation

**Purpose**: Evaluate robustness to view angle changes.

```python
For each gallery view angle (e.g., 0°, 18°, ...):
    For each probe view angle:
        Compute metrics
        Store results in matrix

# Results matrix
results[gallery_view][probe_view] = rank1_accuracy
```

**Expected Behavior**:
- **Without GRL**: Performance drops significantly when views differ
- **With GRL**: More consistent performance across view pairs

---

## 6. File Structure

```
gait_biometric_identification/
│
├── configs/
│   └── config.yaml                    # Main configuration file
│
├── data/
│   ├── __init__.py                    # Data module exports
│   ├── dataset.py                     # CASIA-B dataset loader
│   ├── transforms.py                  # Data augmentation
│   └── sampler.py                     # Triplet batch sampler
│
├── models/
│   ├── __init__.py                    # Model module exports
│   ├── backbone.py                    # Backbone architecture
│   ├── grl.py                         # Gradient Reversal Layer
│   ├── gait_model.py                  # Complete gait recognition model
│   └── losses.py                      # Loss functions
│
├── utils/
│   ├── __init__.py                    # Utils module exports
│   ├── device.py                      # Device management & seeding
│   ├── metrics.py                     # Evaluation metrics
│   └── visualization.py               # Plotting functions
│
├── scripts/
│   ├── train.sh                       # Training shell script
│   └── evaluate.py                    # Evaluation script
│
├── train.py                           # Main training script
├── requirements.txt                   # Python dependencies
├── wiki.md                            # This documentation
└── README.md                          # Quick start guide
```

### 6.1 Key File Descriptions

#### **configs/config.yaml**
- Central configuration for all hyperparameters
- Dataset paths and split settings
- Model architecture parameters
- GRL enable/disable toggle
- Training and evaluation settings
- Device configuration (CUDA/MPS/CPU)

#### **data/dataset.py**
- `CASIABDataset`: PyTorch Dataset for CASIA-B
- Handles pickle file loading
- Implements frame sampling strategies
- Manages gallery/probe splits
- Supports data caching

#### **data/transforms.py**
- `GaitTransform`: Augmentation pipeline
- Resizing to target resolution
- Horizontal flipping
- Random rotation and erasing

#### **data/sampler.py**
- `TripletSampler`: Batch sampler for triplet loss
- Ensures P persons × K samples per batch
- Enables efficient triplet mining

#### **models/backbone.py**
- `SetBlock`: Basic conv block
- `GLConv`: Global-Local convolution
- `GeMHPP`: Horizontal pyramid pooling
- `TemporalPooling`: Set-based aggregation

#### **models/grl.py**
- `GradientReversalFunction`: Custom autograd function
- `GradientReversalLayer`: GRL module
- `ViewDiscriminator`: View angle classifier
- `DomainAdaptationModule`: Complete GRL pipeline

#### **models/gait_model.py**
- `GaitRecognitionModel`: Complete end-to-end model
- Combines backbone + GRL + classifiers
- Handles training/evaluation modes
- Supports GRL toggle

#### **models/losses.py**
- `TripletLoss`: Batch hard triplet loss
- `CenterLoss`: Intra-class compactness
- `CombinedLoss`: Multi-task loss aggregation

#### **utils/metrics.py**
- `compute_distance_matrix`: Pairwise distances
- `evaluate_rank`: Rank-k accuracy
- `compute_cmc`: CMC curve
- `compute_map`: Mean Average Precision
- `AverageMeter`: Training metric tracking

#### **utils/device.py**
- `get_device`: Auto-detect best device
- `setup_seed`: Set random seeds for reproducibility
- `print_system_info`: Display environment info

#### **utils/visualization.py**
- `plot_cmc_curve`: CMC visualization
- `plot_tsne`: Feature space visualization
- `plot_confusion_matrix`: Confusion matrix
- `plot_training_curves`: Training progress

#### **train.py**
- Main training script
- Implements training loop
- Handles checkpointing
- Supports resume from checkpoint
- Periodic evaluation

#### **scripts/evaluate.py**
- Standalone evaluation script
- Loads trained model
- Comprehensive evaluation on test set
- Generates visualizations

---

## 7. Usage Instructions

### 7.1 Installation

```bash
# Clone or navigate to project directory
cd gait_biometric_identification

# Install dependencies
pip install -r requirements.txt
```

### 7.2 Data Preparation

Ensure CASIA-B dataset is organized as:

```
CASIA-B/casiab-128-end2end/
├── 001/
│   ├── nm-01/
│   │   ├── 000/
│   │   │   ├── 000-sils.pkl
│   │   │   └── ...
│   │   ├── 018/
│   │   └── ...
│   ├── nm-02/
│   ├── bg-01/
│   ├── cl-01/
│   └── ...
├── 002/
└── ...
```

Update `config.yaml` with your dataset path:

```yaml
dataset:
  data_root: "/path/to/CASIA-B/casiab-128-end2end"
```

### 7.3 Training

**Option 1: Using shell script**

```bash
cd scripts
./train.sh
```

**Option 2: Direct Python command**

```bash
python train.py --config configs/config.yaml
```

**Training with custom config**

```bash
python train.py --config configs/my_config.yaml
```

**Resume from checkpoint**

```bash
python train.py \
    --config configs/config.yaml \
    --resume output/checkpoint_epoch_50.pth
```

### 7.4 Evaluation

**Evaluate trained model**

```bash
python scripts/evaluate.py \
    --config configs/config.yaml \
    --checkpoint output/best_model.pth \
    --output_dir evaluation_results \
    --visualize
```

**Evaluation only (no training)**

```bash
python train.py \
    --config configs/config.yaml \
    --resume output/best_model.pth \
    --eval_only
```

### 7.5 Monitoring Training

**TensorBoard**

```bash
tensorboard --logdir output/tensorboard
```

Then open http://localhost:6006 in your browser.

**Metrics displayed**:
- Training loss (total, identity, triplet, center, view)
- Learning rate
- Validation metrics (Rank-1, Rank-5, mAP)

---

## 8. Technical Details

### 8.1 Mathematical Foundations

#### 8.1.1 Set-Based Learning

Traditional sequence models (RNN, LSTM) assume temporal order matters. For gait, the order of frames is less important than the overall pattern.

**Set Function Properties**:
```
f(S) = f(permute(S))  # Order-invariant
f(S ∪ {x}) depends on f(S) and x  # Permutation-equivariant per-element processing
```

**Implementation**:
```python
# Process each frame independently
frame_features = CNN(frames)  # [N×T, C, H, W]

# Aggregate using symmetric function
max_pool = max(frame_features, dim=time)
mean_pool = mean(frame_features, dim=time)
set_features = concat([max_pool, mean_pool])
```

#### 8.1.2 Metric Learning Objective

**Goal**: Learn embedding space where:
```
d(x_i, x_j) < d(x_i, x_k)  if y_i = y_j and y_i ≠ y_k
```

**Triplet Loss**:
```
L = max(||f(a) - f(p)||² - ||f(a) - f(n)||² + α, 0)

where:
- a: anchor
- p: positive (same identity)
- n: negative (different identity)
- α: margin
```

**Batch Hard Mining**:
```python
# For each anchor
for a in anchors:
    # Hardest positive: furthest same-identity sample
    p_hard = argmax(distance(a, positives))
    
    # Hardest negative: closest different-identity sample
    n_hard = argmin(distance(a, negatives))
    
    loss += max(d(a, p_hard) - d(a, n_hard) + margin, 0)
```

#### 8.1.3 Domain Adaptation via GRL

**Standard Training**:
```
min L_task(θ)  # Minimize task loss
```

**With Domain Adaptation**:
```
min L_task(θ_f, θ_y) + λ L_domain(θ_f, θ_d)
    θ_f, θ_y

max L_domain(θ_f, θ_d)
    θ_d
```

**Unified via GRL**:
```
min L_task(θ_f, θ_y) - λ L_domain(θ_f, θ_d)
    θ_f, θ_y, θ_d
```

The negative sign is implemented by reversing gradients:

```python
∂L/∂θ_f = ∂L_task/∂θ_f - λ ∂L_domain/∂θ_f
```

### 8.2 Implementation Optimizations

#### 8.2.1 Memory Efficiency

**Data Caching**:
```python
if cache_enabled:
    # Load all data into RAM at start
    self._cache = {}
    for pkl_path in data_files:
        self._cache[str(pkl_path)] = load_pickle(pkl_path)
```

**Benefits**:
- Avoid repeated disk I/O
- Faster epoch iteration
- Trade memory for speed

**When to Use**:
- Dataset fits in RAM (~15-20GB for CASIA-B)
- Multiple epochs of training
- Fast GPU training bottlenecked by I/O

#### 8.2.2 Mixed Precision Training

Can be enabled for faster training on modern GPUs:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():  # Mixed precision forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Benefits**:
- 2-3x faster training
- Reduced memory usage
- Minimal accuracy impact

#### 8.2.3 DataLoader Optimization

```python
DataLoader(
    dataset,
    batch_size=128,
    num_workers=8,  # Parallel data loading
    pin_memory=True,  # Faster GPU transfer
    prefetch_factor=2,  # Pre-load batches
    persistent_workers=True,  # Keep workers alive
)
```

### 8.3 Debugging and Troubleshooting

#### Common Issues

**Issue 1: CUDA Out of Memory**

```
RuntimeError: CUDA out of memory
```

**Solutions**:
- Reduce batch size (person_num or sample_num)
- Reduce frame_num
- Disable data caching
- Use gradient accumulation:
  ```python
  for i, batch in enumerate(dataloader):
      loss = model(batch) / accumulation_steps
      loss.backward()
      
      if (i + 1) % accumulation_steps == 0:
          optimizer.step()
          optimizer.zero_grad()
  ```

**Issue 2: NaN Loss**

```
Loss becomes NaN after few iterations
```

**Solutions**:
- Reduce learning rate
- Enable gradient clipping (already enabled by default)
- Check for division by zero in custom code
- Verify data normalization

**Issue 3: Poor Rank-1 Accuracy**

**Possible Causes**:
- Insufficient training epochs
- Learning rate too high/low
- GRL lambda too high (overpowering identity loss)
- Data loading issues (verify samples are correct identities)

**Debugging Steps**:
```python
# 1. Check if model can overfit small subset
# Train on 10 subjects only - should reach 100% accuracy

# 2. Visualize features with t-SNE
# Should see clear clusters per identity

# 3. Print distance statistics
pos_distances = distances_within_identity
neg_distances = distances_between_identities
print(f"Positive: {pos_distances.mean()}")
print(f"Negative: {neg_distances.mean()}")
# Negative should be > Positive + margin
```

---

## 9. Advanced Topics

### 9.1 Extending the System

#### Adding New Loss Functions

```python
# In models/losses.py

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, embeddings, labels):
        # Implement contrastive loss
        distances = compute_pairwise_distances(embeddings)
        is_positive = (labels.unsqueeze(0) == labels.unsqueeze(1))
        
        pos_loss = (distances * is_positive).mean()
        neg_loss = (torch.clamp(self.margin - distances, min=0) * ~is_positive).mean()
        
        return pos_loss + neg_loss
```

#### Adding New Backbones

```python
# In models/backbone.py

class GaitGL(nn.Module):
    """GaitGL: Another gait recognition architecture"""
    
    def __init__(self, ...):
        # Implement GaitGL architecture
        pass
    
    def forward(self, x):
        # Forward pass
        pass
```

### 9.2 Hyperparameter Tuning

**Key Hyperparameters**:

| Hyperparameter | Impact | Recommended Range |
|----------------|--------|-------------------|
| Learning Rate | Convergence speed | 1e-5 to 1e-3 |
| Batch Size (P×K) | Gradient stability | 64-256 |
| GRL Lambda | View invariance | 0.5-2.0 |
| Triplet Margin | Embedding separation | 0.1-0.5 |
| Frame Number | Sequence information | 20-40 |
| Embedding Dim | Capacity | 128-512 |

**Tuning Strategy**:

1. **Start with learning rate**:
   ```python
   lrs = [1e-5, 1e-4, 1e-3]
   # Train for 10 epochs each, pick best
   ```

2. **Tune batch composition**:
   ```python
   # Try different P×K combinations
   (P=4, K=32), (P=8, K=16), (P=16, K=8)
   # Keep total batch size constant
   ```

3. **Optimize GRL lambda**:
   ```python
   lambdas = [0.0, 0.5, 1.0, 1.5, 2.0]
   # Evaluate cross-view performance
   ```

### 9.3 Cross-Dataset Evaluation

To evaluate generalization to other datasets:

```python
# 1. Train on CASIA-B
python train.py --config configs/casiab_config.yaml

# 2. Evaluate on different dataset (e.g., OU-MVLP)
# Modify config to point to new dataset

python scripts/evaluate.py \
    --config configs/oumvlp_config.yaml \
    --checkpoint output/casiab_best_model.pth
```

---

## 10. Datasets

- **CASIA-B**: http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp
- **OU-MVLP**: http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitMVLP.html

---

## 11. Conclusion

This system implements a complete, production-ready gait biometric identification pipeline with:

✓ State-of-the-art architecture  
✓ Optional GRL for view-invariant learning  
✓ Comprehensive training and evaluation tools  
✓ Multi-device support (CUDA/MPS/CPU)  
✓ Extensive documentation and examples  

**Next Steps**:
1. Train baseline model without GRL
2. Train model with GRL
3. Compare cross-view performance
4. Experiment with different hyperparameters
5. Evaluate on real-world scenarios

For questions or issues, refer to the troubleshooting section or examine the detailed code comments.

---

