# Usage Guide

This guide covers how to train, evaluate, visualize, and use the NFL Analytics Engine.

## Quick Start

### Sanity Check (Recommended First Step)

Run a quick verification to ensure everything works:

```bash
python -m src.train --mode train --sanity
```

**What it does:**
- Loads a single game from Week 1
- Processes only ~500 frames
- Trains for 10 batches
- Reports validation metrics

**Expected output:**
```
Loading data...
Creating graphs...
Training...
Epoch 0: train_loss=2.45, val_loss=2.31, val_ade=1.85
✓ Sanity check passed!
```

### Production Run (full dataset, reproducible)

```bash
python train_production.py --config configs/production.yaml
```

- Saves config snapshot to `outputs/nfl_production_v2_config.json`
- Reuses deterministic play splits from `outputs/splits_production.json` (created on first run)
- Exports TorchScript/ONNX artifacts to `outputs/exported_models/` when enabled
- Uses graph caching in `cache/graphs` to speed up subsequent epochs
---

## Training

### Production Training (default path)

Full-data training with the production configuration:

```bash
python train_production.py --config configs/production.yaml
```

- Probabilistic 8-mode decoder enabled by default
- bf16 mixed precision + SWA + cosine warmup
- AdamW (wd=0.05) with gradient accumulation (effective batch 128)
- Deterministic splits stored at `outputs/splits_production.json`
- Config snapshot and exported models saved under `outputs/`

**Common overrides (CLI > YAML):**

```bash
python train_production.py \
  --config configs/production.yaml \
  --batch-size 32 \
  --limit-train-batches 0.25 \
  --limit-val-batches 0.5 \
  --enable-sample-batch-warmup \
  --resume-from checkpoints/last.ckpt
```

- `--enable-sample-batch-warmup` warms callbacks with a validation batch (default off)
- `--limit-*` is great for smoke testing on a subset
- `--probabilistic` flag forces GMM decoding when overriding YAML
- `--data-dir` allows external data locations

### Research Training (lightweight)

For quick experiments or custom research runs:

```bash
python -m src.train --mode train
```

**Training process (src.train):**
1. Loads Week 1 tracking data (~150-200 plays)
2. Constructs graph objects with radius=20.0
3. Trains for default epochs (configurable)
4. Saves checkpoints to `lightning_logs/`
5. Logs metrics (loss, ADE, FDE, coverage accuracy)

**Common flags (src.train):**

```bash
# Probabilistic mode (GMM decoder)
python -m src.train --mode train --probabilistic

# Multi-week training
python -m src.train --mode train --weeks 1 2 3 4 5

# Combined
python -m src.train --mode train --probabilistic --weeks 1 2 3
```

**Adjust hyperparameters (src.train):**

```python
# In train_model() function
model = NFLGraphPredictor(
    input_dim=9,
    hidden_dim=64,          # Increase to 128 for more capacity
    lr=1e-3,                # Learning rate
    future_seq_len=10,
    probabilistic=True,     # Enable GMM decoder
    num_modes=6,            # Number of trajectory modes
    velocity_weight=0.3,    # Velocity loss weight
    coverage_weight=0.5,    # Coverage loss weight
    use_augmentation=True   # Enable data augmentation
)

trainer = pl.Trainer(
    max_epochs=50,      # Increase for longer training
    accelerator="gpu",  # or "cpu"
    devices=1,
    log_every_n_steps=10
)
```

### Monitoring Training

**TensorBoard (if configured):**
```bash
tensorboard --logdir lightning_logs/
```

**Weights & Biases (if configured):**
```python
# In src/train.py
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(project="nfl-analytics")
trainer = pl.Trainer(logger=wandb_logger, ...)
```

---

## Hyperparameter Tuning

Use Optuna to find optimal hyperparameters:

```bash
python -m src.train --mode tune
```

> `train_production.py --mode tune` currently prints a placeholder; use `src.train` for tuning or edit `configs/production.yaml` directly.

**Tuning process:**
1. Runs 5 trials (configurable in `tune_model()`)
2. Optimizes: learning rate, hidden dimensions
3. Evaluates on validation set
4. Reports best configuration

**Output:**
```
Trial 0: lr=0.001, hidden_dim=64, val_loss=2.15
Trial 1: lr=0.0005, hidden_dim=128, val_loss=1.98
...
Best trial: lr=0.0005, hidden_dim=128
```

**Customize tuning:**

```python
def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 2, 6)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    
    # Train and return validation loss
    ...
```

---

## Inference

### Load Trained Model

```python
import torch
from src.models.gnn import NFLGraphTransformer
from src.train import NFLGraphPredictor

# Load checkpoint
checkpoint_path = "lightning_logs/version_0/checkpoints/epoch=49-step=1000.ckpt"
model = NFLGraphPredictor.load_from_checkpoint(checkpoint_path)
model.eval()
```

### Predict on New Data

```python
from src.data_loader import DataLoader
from src.features import create_graph_data
import polars as pl

# Load data
loader = DataLoader(data_dir=".")
df = loader.load_week_data(1)

# Standardize
df = loader.standardize_tracking_directions(df)

# Create graphs
graphs = create_graph_data(df, radius=20.0, future_seq_len=10)

# Predict
with torch.no_grad():
    for graph in graphs[:5]:  # First 5 plays
        traj_pred, cov_pred, attn = model(graph, return_attention_weights=True)
        
        print(f"Trajectory shape: {traj_pred.shape}")  # [N, 10, 2]
        print(f"Coverage prediction: {torch.sigmoid(cov_pred).item():.3f}")
```

### Batch Prediction

```python
from torch_geometric.loader import DataLoader as PyGDataLoader

# Create dataloader
test_loader = PyGDataLoader(graphs, batch_size=32, shuffle=False)

predictions = []
for batch in test_loader:
    with torch.no_grad():
        traj_pred, cov_pred, _ = model(batch)
        predictions.append({
            'trajectory': traj_pred.cpu(),
            'coverage': torch.sigmoid(cov_pred).cpu()
        })
```

---

## Visualization

### Field Plots

Plot a single frame with player positions:

```python
from src.visualization import plot_play_frame, create_football_field
import matplotlib.pyplot as plt

# Filter to specific play and frame
play_df = df.filter(
    (pl.col("game_id") == 2022091108) & 
    (pl.col("play_id") == 56) &
    (pl.col("frame_id") == 1)
)

# Create field and plot
fig, ax = create_football_field(linenumbers=True)
plot_play_frame(play_df.to_pandas(), frame_id=1, ax=ax)
plt.show()
```

### Animate Plays

Create MP4 animation of entire play:

```python
from src.visualization import animate_play

# Filter to specific play
play_df = df.filter(
    (pl.col("game_id") == 2022091108) & 
    (pl.col("play_id") == 56)
)

# Animate
animate_play(play_df.to_pandas(), output_path="play_animation.mp4")
```

### Attention Visualization

Visualize what the model is "looking at":

```python
from src.visualization import plot_attention_map

# Get predictions with attention weights
with torch.no_grad():
    traj_pred, cov_pred, attn_weights = model(
        graph, 
        return_attention_weights=True
    )

# Extract frame data
frame_df = df.filter(
    (pl.col("game_id") == game_id) & 
    (pl.col("play_id") == play_id) &
    (pl.col("frame_id") == frame_id)
).to_pandas()

# Plot attention for specific player
fig, ax = create_football_field()
plot_attention_map(
    frame_df, 
    attn_weights, 
    target_nfl_id=12345,  # Focus on specific player
    ax=ax
)
plt.title("Attention Map - QB Perspective")
plt.show()
```

**Interpretation:**
- **Thick lines**: Strong attention (model focuses here)
- **Thin lines**: Weak attention
- **Direction**: From observer to attended player

---

## Evaluation Metrics

### Compute Metrics Manually

```python
import torch
import torch.nn.functional as F

def compute_ade(pred, target):
    """Average Displacement Error"""
    # pred, target: [N, 10, 2]
    errors = torch.norm(pred - target, dim=2)  # [N, 10]
    ade = errors.mean()
    return ade.item()

def compute_fde(pred, target):
    """Final Displacement Error"""
    # pred, target: [N, 10, 2]
    final_error = torch.norm(pred[:, -1] - target[:, -1], dim=1)  # [N]
    fde = final_error.mean()
    return fde.item()

def compute_coverage_accuracy(pred_logits, target):
    """Coverage classification accuracy"""
    # pred_logits: [B, 1], target: [B]
    pred_labels = (torch.sigmoid(pred_logits) > 0.5).long().squeeze()
    accuracy = (pred_labels == target).float().mean()
    return accuracy.item()

# Example usage
ade = compute_ade(traj_pred, graph.y)
fde = compute_fde(traj_pred, graph.y)
cov_acc = compute_coverage_accuracy(cov_pred, graph.y_coverage)

print(f"ADE: {ade:.3f} yards")
print(f"FDE: {fde:.3f} yards")
print(f"Coverage Accuracy: {cov_acc:.3f}")
```

### Advanced Metrics

```python
from src.metrics import (
    calculate_zone_collapse_speed,
    calculate_defensive_reaction_time
)

# Zone collapse speed
play_df = df.filter(
    (pl.col("game_id") == game_id) & 
    (pl.col("play_id") == play_id)
)

collapse_df = calculate_zone_collapse_speed(
    play_df, 
    defensive_team="KC"
)

print(collapse_df)
# frame_id | hull_area | hull_area_rate
# 1        | 850.2     | NaN
# 2        | 845.1     | -51.0  (sq yards/sec)

# Defensive reaction time
reaction_time = calculate_defensive_reaction_time(
    play_df,
    ball_start_frame=5,
    defensive_team="KC"
)

print(f"Avg reaction time: {reaction_time:.3f} seconds")

# Novel metrics (NEW)
from src.metrics import (
    calculate_matchup_difficulty,
    calculate_separation_at_target,
    calculate_coverage_pressure_index
)

# Matchup difficulty
matchup = calculate_matchup_difficulty(
    play_df,
    receiver_nfl_id=12345,
    defender_nfl_id=67890
)
print(f"Matchup difficulty: {matchup['matchup_difficulty']:.3f}")
print(f"Speed advantage: {matchup['speed_advantage']:.2f} mph")
print(f"Avg separation: {matchup['avg_separation']:.2f} yards")

# Separation at target
separation = calculate_separation_at_target(
    play_df,
    target_nfl_id=12345,
    target_frame=15
)
print(f"Separation at catch: {separation:.2f} yards")

# Coverage pressure index
pressure = calculate_coverage_pressure_index(
    play_df,
    defensive_team="KC",
    target_frame=10
)
print(f"Pressure index: {pressure['pressure_index']:.3f}")
print(f"Tight coverage count: {pressure['tight_coverage_count']}")
```

---

## Common Workflows

### Workflow 1: Train and Evaluate

```bash
# 1. Sanity check
python -m src.train --mode train --sanity

# 2. Full training
python -m src.train --mode train

# 3. Evaluate (in Python)
python
>>> from src.train import NFLGraphPredictor
>>> model = NFLGraphPredictor.load_from_checkpoint("lightning_logs/.../checkpoint.ckpt")
>>> # Run evaluation...
```

### Workflow 2: Hyperparameter Optimization

```bash
# 1. Run tuning
python -m src.train --mode tune

# 2. Update src/train.py with best params
# 3. Retrain with optimal configuration
python -m src.train --mode train
```

### Workflow 3: Analyze Specific Play

```python
# Load data
from src.data_loader import DataLoader
import polars as pl

loader = DataLoader(data_dir=".")
df = loader.load_week_data(1)
df = loader.standardize_tracking_directions(df)

# Filter play
play_df = df.filter(
    (pl.col("game_id") == 2022091108) & 
    (pl.col("play_id") == 56)
)

# Visualize
from src.visualization import animate_play
animate_play(play_df.to_pandas(), "play_56.mp4")

# Get predictions
from src.features import create_graph_data
graphs = create_graph_data(play_df, radius=20.0)

model = NFLGraphPredictor.load_from_checkpoint("checkpoint.ckpt")
traj, cov, attn = model(graphs[0], return_attention_weights=True)

# Analyze
print(f"Predicted coverage: {'Zone' if torch.sigmoid(cov) > 0.5 else 'Man'}")

# For probabilistic model
if model.model.probabilistic:
    params, mode_probs, cov, attn = model.model(
        graphs[0], 
        return_distribution=True
    )
    print(f"Mode probabilities: {mode_probs[0]}")
    print(f"Most likely mode: {mode_probs[0].argmax().item()}")
```

### Workflow 4: Batch Processing

```python
# Process all plays in a week
from src.data_loader import DataLoader
from src.features import create_graph_data
from torch_geometric.loader import DataLoader as PyGDataLoader
import torch

loader = DataLoader(data_dir=".")
df = loader.load_week_data(1)
df = loader.standardize_tracking_directions(df)

# Create all graphs
graphs = create_graph_data(df, radius=20.0)

# Batch predict
test_loader = PyGDataLoader(graphs, batch_size=32)
model = NFLGraphPredictor.load_from_checkpoint("checkpoint.ckpt")
model.eval()

all_predictions = []
with torch.no_grad():
    for batch in test_loader:
        traj, cov, _ = model(batch)
        all_predictions.append({
            'trajectory': traj.cpu(),
            'coverage': torch.sigmoid(cov).cpu()
        })

# Save predictions
torch.save(all_predictions, "week1_predictions.pt")
```

---

## Jupyter Notebooks

### Exploratory Data Analysis

```bash
jupyter notebook notebooks/01_eda.ipynb
```

**Contents:**
- Data loading and inspection
- Feature distributions
- Play statistics
- Visualization examples

### Baseline Model

```bash
jupyter notebook notebooks/02_baseline_model.ipynb
```

**Contents:**
- XGBoost baseline
- Feature importance
- Performance comparison

### Insights and Metrics

```bash
jupyter notebook notebooks/03_insights.ipynb
```

**Contents:**
- Zone collapse speed analysis
- Defensive reaction time
- Strategic pattern discovery

### Final Submission

```bash
jupyter notebook notebooks/04_submission.ipynb
```

**Contents:**
- Complete pipeline demonstration
- Model predictions
- Submission formatting

---

## Debugging Tips

### Enable Detailed Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# In your code
logger.debug(f"Graph shape: {graph.x.shape}")
logger.debug(f"Edge count: {graph.edge_index.shape[1]}")
```

### Inspect Model Internals

```python
# Print model architecture
print(model.model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm().item():.4f}")
```

### Visualize Intermediate Outputs

```python
# Hook to capture intermediate activations
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Register hooks
model.model.encoder.gat_layers[0].register_forward_hook(get_activation('gat1'))
model.model.encoder.gat_layers[-1].register_forward_hook(get_activation('gat4'))

# Forward pass
output = model(graph)

# Inspect
print(f"GAT1 output shape: {activations['gat1'].shape}")
print(f"GAT4 output shape: {activations['gat4'].shape}")
```

---

## Performance Optimization

### GPU Utilization

```python
# Monitor GPU usage
import torch

print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Clear cache if needed
torch.cuda.empty_cache()
```

### Data Loading Speed

```python
# Use multiple workers
from torch_geometric.loader import DataLoader

loader = DataLoader(
    graphs,
    batch_size=32,
    num_workers=4,      # Parallel loading
    pin_memory=True,    # Faster GPU transfer
    persistent_workers=True  # Keep workers alive
)
```

### Mixed Precision Training

```python
# In src/train.py
trainer = pl.Trainer(
    precision="16-mixed",  # Use mixed precision
    accelerator="gpu",
    devices=1
)
```

---

## Export and Deployment

### Save Model for Production

```python
# Save entire model
torch.save(model.state_dict(), "nfl_model_weights.pt")

# Save with metadata
torch.save({
    'model_state_dict': model.state_dict(),
    'hyperparameters': model.hparams,
    'epoch': 50,
    'metrics': {'ade': 1.85, 'fde': 3.21}
}, "nfl_model_full.pt")
```

### Load in Production

```python
# Load model
checkpoint = torch.load("nfl_model_full.pt")
model = NFLGraphTransformer(**checkpoint['hyperparameters'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
with torch.no_grad():
    predictions = model(graph)
```

### Standalone Inference Script

Use the lightweight runner for CSV inputs and optional visualization:

```bash
python -m src.inference \
  --checkpoint path/to/ckpt.ckpt \
  --input-csv train/input_2023_w01.csv \
  --output-csv outputs/preds.csv \
  --visualize
```

### ONNX Export (Production)

The production trainer now exports TorchScript and ONNX artifacts to `outputs/exported_models/` when `export_onnx` or `export_torchscript` are enabled in the config.

---

## Next Steps

- **Configuration:** See [configuration.md](configuration.md) for hyperparameter details
- **Testing:** See [testing.md](testing.md) for verification procedures
- **API Reference:** See [api_reference.md](api_reference.md) for detailed API docs
- **Performance:** See [performance.md](performance.md) for benchmarks and optimization

---

## Troubleshooting

**Issue: Training is slow**
- Reduce batch size
- Use fewer workers
- Enable mixed precision
- Check GPU utilization

**Issue: Out of memory**
- Reduce batch size
- Reduce hidden_dim
- Use gradient accumulation
- Clear CUDA cache

**Issue: Poor predictions**
- Train longer (more epochs)
- Tune hyperparameters
- Check data quality
- Verify standardization

**Issue: Attention weights all similar**
- Increase model capacity
- Train longer
- Check edge construction
- Verify edge attributes
