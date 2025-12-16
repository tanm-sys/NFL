# Configuration Guide

> Complete reference for all hyperparameters, training settings, and optimization options.

## 📁 Configuration Files

| Config File | Use Case | Description |
|-------------|----------|-------------|
| `max_accuracy_rtx3050.yaml` | Lowest ADE/FDE | Ultimate accuracy for competition |
| `high_accuracy.yaml` | High accuracy | Balanced accuracy config |
| `production.yaml` | Production | Speed/accuracy balance |
| `sanity.yaml` | Testing | Quick sanity checks |

---

## 🎯 Ultimate Accuracy Configuration

The recommended config for competition-grade results:

```yaml
# configs/max_accuracy_rtx3050.yaml

experiment_name: "nfl_ultimate_accuracy"

# Data
weeks: [1-18]           # All data
future_seq_len: 10      # 1 second prediction (10 fps)
history_len: 5          # 0.5 second history

# Model Architecture
hidden_dim: 256         # Maximum for 4GB VRAM
num_gnn_layers: 8       # Deep network
heads: 8                # Multi-head attention
dropout: 0.10           # Moderate dropout
droppath_rate: 0.12     # Stochastic depth
num_modes: 8            # GMM modes (critical for minADE)

# Training
batch_size: 32          # Smaller for better gradients
accumulate_grad_batches: 5  # Effective batch: 160
learning_rate: 0.0008   # Lower for stability
max_epochs: 100         # Long training
early_stopping_patience: 20

# SOTA Losses
use_social_nce: true
use_wta_loss: true
use_diversity_loss: true
use_endpoint_focal: true
```

---

## 📊 Parameter Reference

### Model Architecture

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `input_dim` | int | 9 | - | Node feature dimension |
| `hidden_dim` | int | 256 | 64-512 | Hidden layer size |
| `num_gnn_layers` | int | 8 | 4-12 | GNN depth |
| `heads` | int | 8 | 4-16 | Attention heads |
| `dropout` | float | 0.10 | 0.0-0.3 | Dropout rate |
| `droppath_rate` | float | 0.12 | 0.0-0.2 | Stochastic depth |
| `probabilistic` | bool | true | - | GMM decoder |
| `num_modes` | int | 8 | 1-12 | GMM modes |

### Training Settings

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `batch_size` | int | 32 | 16-64 | Batch size |
| `accumulate_grad_batches` | int | 5 | 1-8 | Gradient accumulation |
| `learning_rate` | float | 0.0008 | 1e-4 to 1e-2 | Initial LR |
| `weight_decay` | float | 0.03 | 0.01-0.1 | L2 regularization |
| `max_epochs` | int | 100 | 10-200 | Maximum epochs |
| `warmup_epochs` | int | 8 | 3-15 | LR warmup |
| `min_lr` | float | 1e-7 | 1e-8 to 1e-6 | Minimum LR |

### Loss Weights

| Loss | Default Weight | Range | Purpose |
|------|----------------|-------|---------|
| `trajectory_weight` | 1.0 | 0.5-2.0 | Primary MSE |
| `velocity_weight` | 0.5 | 0.1-1.0 | Smooth motion |
| `acceleration_weight` | 0.3 | 0.1-0.5 | Physical plausibility |
| `collision_weight` | 0.15 | 0.05-0.3 | Collision avoidance |
| `coverage_weight` | 0.6 | 0.2-1.0 | Zone coverage |

### SOTA Contrastive Losses

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `social_nce_weight` | 0.15 | 0.05-0.3 | Social-NCE loss weight |
| `social_nce_temperature` | 0.07 | 0.05-0.15 | Contrastive temperature |
| `wta_k_best` | 2 | 1-4 | WTA top-k modes |
| `diversity_weight` | 0.04 | 0.01-0.1 | Diversity penalty |
| `diversity_min_distance` | 2.0 | 1.0-3.0 | Min mode separation (yards) |
| `endpoint_focal_weight` | 0.25 | 0.1-0.5 | Endpoint focal weight |
| `endpoint_focal_gamma` | 2.5 | 1.5-3.0 | Focal gamma |

---

## ⚡ Hardware Settings

### GPU Optimization

```yaml
# Mixed precision for Tensor Cores
precision: "16-mixed"

# CUDA settings
accelerator: "gpu"
devices: 1
benchmark: true       # cuDNN autotuning
deterministic: false  # Speed over reproducibility
```

### Memory Management

```yaml
# RTX 3050 4GB optimized
batch_size: 32                    # Fits in 4GB with 8-mode GMM
in_memory_cache_size: 200         # RAM cache for graphs
num_workers: 0                    # Use pre-cached graphs
pin_memory: true                  # Fast CPU→GPU transfer
```

### Pre-Caching (Required)

```bash
# Build graph cache before training
python -c "
from src.data_loader import *
from pathlib import Path

loader = DataLoader('.')
play_meta = build_play_metadata(loader, list(range(1,19)), 5, 10)
tuples = expand_play_tuples(play_meta)

cache_dir = Path('cache/finetune/train')
cache_dir.mkdir(parents=True, exist_ok=True)

ds = GraphDataset(loader, tuples, 20.0, 10, 5, 
                  cache_dir=cache_dir, persist_cache=True)
for i in range(len(ds)):
    _ = ds[i]
    if i % 1000 == 0: print(f'{i}/{len(ds)}')
"
```

---

## 🔄 Callbacks

### Early Stopping

```yaml
early_stopping_patience: 20       # Epochs without improvement
early_stopping_min_delta: 0.0001  # Minimum improvement
monitor_metric: "val_minADE"      # Metric to monitor
```

### Model Checkpointing

```yaml
save_top_k: 5         # Keep best 5 checkpoints
save_last: true       # Always save last epoch
monitor: "val_minADE" # Sort by this metric
mode: "min"           # Lower is better
```

### Stochastic Weight Averaging (SWA)

```yaml
swa_enabled: true
swa_epoch_start: 0.75     # Start at 75% of training
swa_lrs: 3e-6             # SWA learning rate
swa_annealing_epochs: 10  # Annealing period
```

---

## 🎮 CLI Usage

### Training Commands

```bash
# Ultimate accuracy training
python finetune_best_model.py --config configs/max_accuracy_rtx3050.yaml

# Quick sanity check
python finetune_best_model.py --config configs/sanity.yaml

# Production training
python finetune_best_model.py --config configs/production.yaml
```

### Monitoring

```bash
# TensorBoard
tensorboard --logdir lightning_logs/

# GPU monitoring
watch -n 1 nvidia-smi
```

---

## 📈 Tuning Recommendations

### For Lower ADE/FDE

1. **Increase `num_modes`** (8 → 10)
2. **Lower `learning_rate`** (0.0008 → 0.0005)
3. **Increase `wta_k_best`** (2 → 3)
4. **Raise `endpoint_focal_weight`** (0.25 → 0.35)

### For Faster Training

1. **Increase `batch_size`** (32 → 48)
2. **Reduce `num_gnn_layers`** (8 → 6)
3. **Lower `max_epochs`** (100 → 50)
4. **Use fewer weeks** (18 → 9)

### For Better Generalization

1. **Increase `dropout`** (0.10 → 0.15)
2. **Raise `droppath_rate`** (0.12 → 0.18)
3. **Enable more data augmentation**
4. **Use SWA** (already enabled)

---

## 🧪 Debugging Settings

```yaml
# For debugging only
limit_train_batches: 10    # Run only 10 batches
limit_val_batches: 5       # 5 validation batches
fast_dev_run: true         # Single batch test
detect_anomaly: true       # NaN/Inf detection
```
