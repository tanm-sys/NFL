# Usage Guide

> Complete workflow guide for training, inference, and competition submission.

## 🚀 Quick Start

### 1. Pre-Cache Graphs (Required)

```bash
# Build cache for all 18 weeks (~30 min)
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
for i, _ in enumerate(ds):
    if i % 1000 == 0: print(f'{i}/{len(ds)}')
"
```

### 2. Train Model

```bash
# Ultimate accuracy
python finetune_best_model.py --config configs/max_accuracy_rtx3050.yaml

# Quick test
python finetune_best_model.py --config configs/sanity.yaml
```

### 3. Monitor Training

```bash
# GPU usage
watch -n 1 nvidia-smi

# TensorBoard
tensorboard --logdir lightning_logs/
```

---

## 📊 Training Configurations

| Config | Use Case | Time/Epoch | Target ADE |
|--------|----------|------------|------------|
| `max_accuracy_rtx3050.yaml` | Competition | ~45 min | < 0.32 |
| `high_accuracy.yaml` | High quality | ~40 min | < 0.38 |
| `production.yaml` | Balanced | ~35 min | < 0.45 |
| `sanity.yaml` | Quick test | ~2 min | N/A |

---

## 🎯 Training Workflow

### Full Training Pipeline

```bash
# Step 1: Pre-cache (one-time, ~30 min)
python scripts/precache_graphs.py

# Step 2: Train (~8 hours for 100 epochs)
python finetune_best_model.py --config configs/max_accuracy_rtx3050.yaml

# Step 3: Find best checkpoint
ls checkpoints_finetuned/

# Step 4: Generate submission
python -m src.competition_output \
    --checkpoint checkpoints_finetuned/best.ckpt \
    --output submission.csv
```

---

## 🔮 Inference

### Load Trained Model

```python
from src.train import NFLGraphPredictor
import torch

# Load checkpoint
model = NFLGraphPredictor.load_from_checkpoint(
    "checkpoints_finetuned/best.ckpt",
    map_location="cuda"
)
model.eval()

# Prepare batch
from torch_geometric.data import Data, Batch

data = Data(
    x=torch.randn(22, 9),
    edge_index=torch.randint(0, 22, (2, 100)),
    edge_attr=torch.randn(100, 5),
    current_pos=torch.randn(22, 2),
    context=torch.randn(1, 3),
)
batch = Batch.from_data_list([data]).to("cuda")

# Predict
with torch.no_grad():
    predictions, coverage, _ = model.model(batch)
    
print(predictions.shape)  # [22, 10, 2]
```

### Probabilistic Prediction (Multi-Modal)

```python
# Get all modes
with torch.no_grad():
    params, probs, cov, _ = model.model(batch, return_distribution=True)
    
# params: [N, T, K, 5] - mu_x, mu_y, sigma_x, sigma_y, rho
# probs: [N, K] - mode probabilities

# Best mode
best_mode = probs.argmax(dim=-1)
mu = params[..., :2]  # positions

for i in range(22):
    best = mu[i, :, best_mode[i], :]  # [10, 2]
```

---

## 📤 Competition Submission

### Generate Predictions

```bash
python -m src.competition_output \
    --checkpoint checkpoints_finetuned/best.ckpt \
    --data-dir . \
    --weeks 1 2 3 4 5 6 7 8 9 \
    --output submission.csv \
    --batch-size 64
```

### Output Format

```csv
game_id,play_id,node_idx,frame_id,predicted_x,predicted_y,confidence_lower_x,confidence_upper_x,confidence_lower_y,confidence_upper_y
2023090700,1,0,0,45.23,26.15,44.12,46.34,25.04,27.26
...
```

---

## 📈 Metrics Tracking

### Log Metrics During Training

```python
# Automatically logged:
# - train_loss, val_loss
# - val_ade, val_fde
# - val_minADE, val_minFDE
# - val_miss_rate
# - learning_rate
```

### Custom Evaluation

```python
from src.metrics import calculate_ade, calculate_fde

# predictions: [N, T, 2]
# targets: [N, T, 2]

ade = calculate_ade(predictions, targets)
fde = calculate_fde(predictions, targets)

print(f"ADE: {ade:.4f} yards")
print(f"FDE: {fde:.4f} yards")
```

---

## 🎨 Visualization

### Trajectory Plot

```python
from src.visualization import plot_trajectories

plot_trajectories(
    predictions=predictions,   # [N, T, 2]
    ground_truth=targets,      # [N, T, 2]
    current_pos=current_pos,   # [N, 2]
    output_path="trajectories.png"
)
```

### Attention Map

```python
# Get attention weights
with torch.no_grad():
    _, _, attn = model.model(batch, return_attention_weights=True)

from src.visualization import plot_attention_map
plot_attention_map(attn, output_path="attention.png")
```

### Play Animation

```python
from src.visualization import animate_play

animate_play(
    play_df=tracking_df,
    predictions=predictions,
    output_path="play.gif",
    fps=10
)
```

---

## 🔧 Advanced Usage

### Custom Loss Weights

```python
model = NFLGraphPredictor(
    trajectory_weight=1.0,
    velocity_weight=0.5,
    social_nce_weight=0.15,
    wta_k_best=2,
    diversity_weight=0.04,
    endpoint_focal_weight=0.25,
)
```

### Resume Training

```bash
# From latest checkpoint
python finetune_best_model.py \
    --config configs/max_accuracy_rtx3050.yaml \
    --resume checkpoints_finetuned/last.ckpt
```

### Hyperparameter Tuning

```bash
# Optuna optimization
python -m src.train --mode tune --n-trials 100
```
