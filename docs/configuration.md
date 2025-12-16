# Configuration Guide

> Complete reference for all hyperparameters and training settings.

## 📁 Configuration Files

```mermaid
flowchart LR
    subgraph Configs["Configuration Files"]
        C1["max_accuracy_rtx3050.yaml<br/>12M params"]
        C2["high_accuracy.yaml<br/>5.4M params"]
        C3["production.yaml<br/>3.2M params"]
        C4["sanity.yaml<br/>1M params"]
    end
    
    C1 --> T1["Competition"]
    C2 --> T2["High Quality"]
    C3 --> T3["Balanced"]
    C4 --> T4["Testing"]
    
    style C1 fill:#c8e6c9
```

---

## 🎯 Maximum Parameters Configuration

The recommended config for competition-winning results:

```mermaid
flowchart TB
    subgraph Model["Model Architecture"]
        M1["hidden_dim: 384"]
        M2["num_gnn_layers: 8"]
        M3["heads: 12"]
        M4["num_modes: 16"]
        M5["dropout: 0.08"]
        M6["droppath: 0.18"]
    end
    
    subgraph Training["Training Settings"]
        T1["batch_size: 16"]
        T2["accumulate: 12"]
        T3["lr: 8e-5"]
        T4["epochs: 200"]
        T5["patience: 30"]
    end
    
    subgraph Losses["SOTA Losses"]
        L1["social_nce: 0.25"]
        L2["wta_k: 4"]
        L3["diversity: 0.08"]
        L4["endpoint: 0.40"]
    end
```

### Full Configuration

```yaml
# configs/max_accuracy_rtx3050.yaml

experiment_name: "nfl_max_parameters"

# Data
weeks: [1-18]              # All data
radius: 30.0               # Maximum context
future_seq_len: 10         # 1 second prediction

# Model (12M parameters)
hidden_dim: 384            # 1.5× default
num_gnn_layers: 8          # Deep network
heads: 12                  # Multi-head attention
num_modes: 16              # Maximum multi-modal
dropout: 0.08
droppath_rate: 0.18

# Training
batch_size: 16
accumulate_grad_batches: 12  # Effective: 192
learning_rate: 0.00008
max_epochs: 200
weight_decay: 0.08
```

---

## 📊 Parameter Reference

### Model Architecture

```mermaid
flowchart LR
    subgraph Size["Model Size Impact"]
        S1["hidden_dim↑"] --> P1["Params ↑↑"]
        S2["num_layers↑"] --> P2["Params ↑"]
        S3["heads↑"] --> P3["Minimal"]
        S4["num_modes↑"] --> P4["Params ↑"]
    end
```

| Parameter | Min | Default | Max (4GB) | Impact |
|-----------|-----|---------|-----------|--------|
| `hidden_dim` | 64 | 256 | **384** | Quadratic |
| `num_gnn_layers` | 2 | 6 | **8** | Linear |
| `heads` | 2 | 8 | **12** | Minimal |
| `num_modes` | 1 | 8 | **16** | Linear |
| `dropout` | 0.0 | 0.1 | 0.2 | N/A |
| `droppath_rate` | 0.0 | 0.12 | **0.18** | N/A |

### Training Settings

| Parameter | Range | Recommended | Notes |
|-----------|-------|-------------|-------|
| `batch_size` | 8-64 | **16** | Lower for large model |
| `accumulate_grad_batches` | 1-16 | **12** | Effective batch = 192 |
| `learning_rate` | 1e-5 to 1e-3 | **8e-5** | Lower for large model |
| `weight_decay` | 0.01-0.1 | **0.08** | Stronger for large model |
| `max_epochs` | 50-300 | **200** | Long training |
| `warmup_epochs` | 5-20 | **15** | Longer for stability |

---

## 🎯 Loss Weight Configuration

```mermaid
flowchart TB
    subgraph Weights["Loss Weights (Total: 4.78)"]
        subgraph Primary["Primary: 2.2"]
            W1["trajectory: 1.0"]
            W2["velocity: 0.7"]
            W3["acceleration: 0.5"]
        end
        
        subgraph SOTA["SOTA: 1.73"]
            W4["social_nce: 0.25"]
            W5["wta: 1.0"]
            W6["diversity: 0.08"]
            W7["endpoint_focal: 0.40"]
        end
        
        subgraph Aux["Auxiliary: 0.85"]
            W8["collision: 0.25"]
            W9["coverage: 0.60"]
        end
    end
```

### Loss Tuning Guide

| Loss | ↓ When | ↑ When |
|------|--------|--------|
| `trajectory_weight` | Overfitting | Underfitting |
| `velocity_weight` | Jerky motion | Too smooth |
| `social_nce_weight` | Collisions ok | Social modeling |
| `wta_weight` | Mode collapse | Multi-modal |
| `diversity_weight` | Modes similar | Memory issues |
| `endpoint_focal_weight` | FDE ok | FDE too high |

---

## ⚡ Hardware Settings

```mermaid
flowchart LR
    subgraph GPU["GPU Optimization"]
        G1["precision: 16-mixed"]
        G2["benchmark: true"]
        G3["Tensor Cores: ON"]
    end
    
    subgraph Memory["Memory Management"]
        M1["batch_size: 16"]
        M2["gradient_clip: 0.25"]
        M3["cache_size: 100"]
    end
    
    GPU --> Performance["2× Faster"]
    Memory --> Fits["Fits 4GB"]
```

### RTX 3050 4GB Settings

```yaml
# Hardware
precision: "16-mixed"       # Tensor Cores
accelerator: "gpu"
benchmark: true             # cuDNN autotuning
num_workers: 0              # Pre-cached graphs

# Memory
batch_size: 16              # Fits with 16 modes
in_memory_cache_size: 100   # RAM cache
gradient_clip_val: 0.25     # Stability
```

---

## 🔄 Callback Configuration

```mermaid
flowchart TB
    subgraph Callbacks["Training Callbacks"]
        CB1["Early Stopping<br/>patience: 30"]
        CB2["Model Checkpoint<br/>save_top_k: 15"]
        CB3["LR Monitor<br/>log every step"]
        CB4["SWA<br/>start: 65%"]
    end
    
    CB1 --> Action1["Stop if no improvement"]
    CB2 --> Action2["Save best minADE"]
    CB3 --> Action3["TensorBoard logging"]
    CB4 --> Action4["Weight averaging"]
```

### Callback Settings

```yaml
# Early Stopping
early_stopping_patience: 30
early_stopping_min_delta: 0.00002
monitor_metric: "val_minADE"

# Checkpointing
save_top_k: 15
save_last: true

# SWA
swa_enabled: true
swa_epoch_start: 0.65       # Start at 65%
swa_lrs: 5e-7
swa_annealing_epochs: 20
```

---

## 🎮 CLI Usage

### Training Commands

```bash
# Maximum parameters (competition)
python finetune_best_model.py --config configs/max_accuracy_rtx3050.yaml

# Quick test
python finetune_best_model.py --config configs/sanity.yaml

# Custom config
python finetune_best_model.py --config my_config.yaml
```

### Monitoring

```bash
# GPU usage
watch -n 1 nvidia-smi

# TensorBoard
tensorboard --logdir lightning_logs/
```

---

## 📈 Tuning Recommendations

### For Lower minADE

```mermaid
flowchart LR
    A["Increase num_modes"] --> B["Lower minADE"]
    C["Increase wta_k_best"] --> B
    D["Lower diversity_min_distance"] --> B
```

### For Lower FDE

```mermaid
flowchart LR
    A["Increase endpoint_focal_weight"] --> B["Lower FDE"]
    C["Higher endpoint_focal_gamma"] --> B
    D["More epochs"] --> B
```

### For Faster Training

```mermaid
flowchart LR
    A["Increase batch_size"] --> B["Faster"]
    C["Fewer GNN layers"] --> B
    D["Lower hidden_dim"] --> B
```

---

## 🧪 Debug Settings

```yaml
# For debugging only
fast_dev_run: true          # Single batch
limit_train_batches: 10
detect_anomaly: true        # NaN detection
log_every_n_steps: 1
```
