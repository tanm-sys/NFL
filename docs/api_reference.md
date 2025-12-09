# API Reference

Complete API documentation for all modules in the NFL Analytics Engine.

## Module Overview

| Module | Description | Key Classes/Functions |
|--------|-------------|----------------------|
| `src.data_loader` | Data ingestion with Polars | `DataLoader` |
| `src.features` | Graph construction | `create_graph_data`, `prepare_tensor_data` |
| `src.models.gnn` | GNN architecture | `NFLGraphTransformer`, `GraphPlayerEncoder` |
| `src.models.transformer` | Legacy transformer | `NFLTransformer` |
| `src.metrics` | Custom metrics | `calculate_zone_collapse_speed`, `calculate_defensive_reaction_time` |
| `src.visualization` | Plotting tools | `plot_play_frame`, `animate_play`, `plot_attention_map` |
| `src.train` | Training loop | `NFLGraphPredictor`, `train_model`, `tune_model` |
| `train_production` | Production trainer (YAML + logging) | `ProductionConfig`, `ProductionTrainer`, `DataValidator` |

---

## src.data_loader

### DataLoader

**Class for loading and preprocessing NFL tracking data.**

```python
class DataLoader:
    def __init__(self, data_dir: str = ".")
```

**Parameters:**
- `data_dir` (str): Root directory containing `train/` folder and `supplementary_data.csv`

#### Methods

##### load_week_data

```python
def load_week_data(self, week: int, standard_cols: bool = True) -> pl.DataFrame
```

Load tracking data for a specific week and merge with play context.

**Parameters:**
- `week` (int): Week number (1-18)
- `standard_cols` (bool): Whether to standardize column names to snake_case

**Returns:**
- `pl.DataFrame`: Merged tracking and play data with features

**Raises:**
- `FileNotFoundError`: If tracking file doesn't exist

**Example:**
```python
loader = DataLoader(data_dir=".")
df = loader.load_week_data(1)
print(df.shape)  # (500000, 50+)
```

**Features added:**
- `role_id`: Player role encoding (0-4)
- `formation_id`: Formation encoding (0-7)
- `alignment_id`: Alignment encoding (0-9)
- `coverage_label`: Coverage type (0=Man, 1=Zone)
- `weight_norm`: Normalized player weight
- `defenders_box_norm`: Normalized defenders in box

##### load_plays

```python
def load_plays(self) -> pl.DataFrame
```

Load supplementary play data.

**Returns:**
- `pl.DataFrame`: Play-level context data

##### standardize_tracking_directions

```python
def standardize_tracking_directions(self, df: pl.DataFrame) -> pl.DataFrame
```

Adjust coordinates so all plays move left-to-right.

**Parameters:**
- `df` (pl.DataFrame): Raw tracking data

**Returns:**
- `pl.DataFrame`: Data with `std_x`, `std_y`, `std_dir`, `std_o` columns

**Transformations:**
- If `play_direction == 'left'`:
  - `std_x = 120 - x`
  - `std_y = 53.3 - y`
  - `std_dir = (dir + 180) % 360`
  - `std_o = (o + 180) % 360`

---

## src.features

### create_graph_data

```python
def create_graph_data(
    df: pl.DataFrame,
    radius: float = 20.0,
    future_seq_len: int = 10
) -> List[Data]
```

Convert DataFrame into list of PyG Data objects (graphs).

**Parameters:**
- `df` (pl.DataFrame): Preprocessed tracking data
- `radius` (float): Edge creation radius in yards (default: 20.0)
- `future_seq_len` (int): Number of future frames to predict (default: 10)

**Returns:**
- `List[Data]`: List of PyTorch Geometric Data objects

**Edge Attribute Shape:** `[Num_Edges, 5]`

**Data Object Structure:**
```python
Data(
    x=[N, 7],              # Node features
    edge_index=[2, E],     # Edge connectivity
    edge_attr=[E, 5],      # Edge features (5D)
    y=[N, 10, 2],          # Target trajectory
    history=[N, T, 4],     # Motion history (vel_x, vel_y, acc_x, acc_y)
    role=[N],              # Role IDs
    side=[N],              # Side IDs
    formation=[1],         # Formation ID
    alignment=[1],         # Alignment ID
    context=[1, 3],        # Context vector
    frame_t=[1],           # Normalized frame position
    y_coverage=[1]         # Coverage label
)
```

**Example:**
```python
from src.features import create_graph_data

graphs = create_graph_data(df, radius=20.0)
print(f"Created {len(graphs)} graphs")
print(f"First graph: {graphs[0]}")
```

### prepare_tensor_data

```python
def prepare_tensor_data(
    df: pl.DataFrame,
    seq_len: int = 10
) -> torch.Tensor
```

Prepare data for Transformer (legacy).

**Parameters:**
- `df` (pl.DataFrame): Tracking data
- `seq_len` (int): Sequence length

**Returns:**
- `torch.Tensor`: Shape `[Batch, Seq, Agents, Features]`

### calculate_distance_to_ball

```python
def calculate_distance_to_ball(df: pl.DataFrame) -> pl.DataFrame
```

Calculate distance from each player to the football.

**Parameters:**
- `df` (pl.DataFrame): Tracking data with `std_x`, `std_y`

**Returns:**
- `pl.DataFrame`: Data with `dist_to_ball` column

---

## src.models.gnn

### NFLGraphTransformer

**Main model architecture combining GNN and Transformer.**

```python
class NFLGraphTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 64,
        heads: int = 4,
        future_seq_len: int = 10,
        edge_dim: int = 5,         # 5D edge features
        num_gnn_layers: int = 4,   # Configurable: 4-8
        probabilistic: bool = False,
        num_modes: int = 6,
        use_scene_encoder: bool = True  # P3 Scene Flow Encoder
    )
```

**Parameters:**
- `input_dim` (int): Node feature dimension (default: 7)
- `hidden_dim` (int): Hidden embedding dimension (default: 64)
- `heads` (int): Number of attention heads (default: 4)
- `future_seq_len` (int): Prediction horizon in frames (default: 10)
- `edge_dim` (int): Edge feature dimension (default: 5)
- `num_gnn_layers` (int): Number of GNN layers (default: 4, range: 4-8)
- `probabilistic` (bool): Use GMM decoder (default: False)
- `num_modes` (int): Number of trajectory modes for GMM (default: 6)
- `use_scene_encoder` (bool): Enable P3 Scene Flow Encoder (default: True)

#### Methods

##### forward

```python
def forward(
    self,
    data: Data,
    return_attention_weights: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]
```

Forward pass through the model.

**Parameters:**
- `data` (Data): PyG Data object with graph
- `return_attention_weights` (bool): Whether to return attention weights

**Returns:**
- `predictions` (Tensor): Trajectory predictions `[N, 10, 2]`
- `coverage` (Tensor): Coverage logits `[B, 1]`
- `attention_weights` (Optional[Tuple]): `(edge_index, alpha)` if requested

**Example:**
```python
model = NFLGraphTransformer(input_dim=7, hidden_dim=64)
traj, cov, attn = model(graph, return_attention_weights=True)

print(f"Trajectory: {traj.shape}")  # [23, 10, 2]
print(f"Coverage: {torch.sigmoid(cov).item():.3f}")  # 0.723
```

### GraphPlayerEncoder

**GNN encoder with strategic embeddings.**

```python
class GraphPlayerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 64,
        heads: int = 4,
        context_dim: int = 3,
        edge_dim: int = 2,
        num_layers: int = 4,
        dropout: float = 0.1
    )
```

**Parameters:**
- `input_dim` (int): Node feature dimension
- `hidden_dim` (int): Hidden dimension
- `heads` (int): Attention heads
- `context_dim` (int): Context feature dimension
- `edge_dim` (int): Edge feature dimension (default: 5)
- `num_layers` (int): Number of GATv2 layers (default: 4)
- `dropout` (float): Dropout rate
- `droppath_rate` (float): Stochastic Depth drop rate (default: 0.1)

#### Methods

##### forward

```python
def forward(
    self,
    x: Tensor,
    edge_index: Tensor,
    edge_attr: Tensor,
    context: Optional[Tensor] = None,
    batch: Optional[Tensor] = None,
    role: Optional[Tensor] = None,
    side: Optional[Tensor] = None,
    formation: Optional[Tensor] = None,
    alignment: Optional[Tensor] = None,
    return_attention_weights: bool = False
) -> Tuple[Tensor, Optional[Tuple]]
```

**Returns:**
- `node_embeddings` (Tensor): `[N, hidden_dim]`
- `attention_weights` (Optional[Tuple]): Attention weights if requested

### TrajectoryDecoder

**Transformer decoder for trajectory prediction.**

```python
class TrajectoryDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        future_seq_len: int
    )
```

#### Methods

##### forward

```python
def forward(self, context_emb: Tensor) -> Tensor
```

**Parameters:**
- `context_emb` (Tensor): Node embeddings `[N, hidden_dim]`

**Returns:**
- `predictions` (Tensor): Trajectory predictions `[N, future_seq_len, 2]`

---

## src.metrics

### calculate_zone_collapse_speed

```python
def calculate_zone_collapse_speed(
    play_df: pl.DataFrame,
    defensive_team: str
) -> pl.DataFrame
```

Calculate rate of change of defensive convex hull area.

**Parameters:**
- `play_df` (pl.DataFrame): Single play tracking data
- `defensive_team` (str): Team abbreviation (e.g., "KC")

**Returns:**
- `pl.DataFrame`: Columns: `frame_id`, `hull_area`, `hull_area_rate`

**Example:**
```python
from src.metrics import calculate_zone_collapse_speed

collapse_df = calculate_zone_collapse_speed(play_df, "KC")
print(collapse_df)
# frame_id | hull_area | hull_area_rate
# 1        | 850.2     | NaN
# 2        | 845.1     | -51.0  (sq yards/sec)
```

### calculate_defensive_reaction_time

```python
def calculate_defensive_reaction_time(
    play_df: pl.DataFrame,
    ball_start_frame: int,
    defensive_team: str
) -> float
```

Calculate average defender reaction time after ball release.

**Parameters:**
- `play_df` (pl.DataFrame): Single play tracking data
- `ball_start_frame` (int): Frame when ball is released
- `defensive_team` (str): Team abbreviation

**Returns:**
- `float`: Average reaction time in seconds

**Example:**
```python
reaction_time = calculate_defensive_reaction_time(
    play_df,
    ball_start_frame=5,
    defensive_team="KC"
)
print(f"Reaction time: {reaction_time:.3f}s")  # 0.234s
```

---

## src.visualization

### create_football_field

```python
def create_football_field(
    linenumbers: bool = True,
    endzones: bool = True,
    highlight_line: bool = False,
    highlight_line_number: int = 50,
    highlighted_name: str = 'Line of Scrimmage',
    fifty_is_los: bool = False,
    figsize: Tuple[int, int] = (12, 6.33)
) -> Tuple[Figure, Axes]
```

Create matplotlib figure with football field.

**Returns:**
- `fig` (Figure): Matplotlib figure
- `ax` (Axes): Matplotlib axes

**Example:**
```python
fig, ax = create_football_field(linenumbers=True)
plt.show()
```

### plot_play_frame

```python
def plot_play_frame(
    play_df: pd.DataFrame,
    frame_id: int,
    ax: Optional[Axes] = None
) -> Axes
```

Plot player positions for a specific frame.

**Parameters:**
- `play_df` (pd.DataFrame): Play tracking data
- `frame_id` (int): Frame to plot
- `ax` (Optional[Axes]): Existing axes (creates new if None)

**Returns:**
- `ax` (Axes): Matplotlib axes with plot

**Example:**
```python
fig, ax = create_football_field()
plot_play_frame(play_df, frame_id=1, ax=ax)
plt.title("Frame 1")
plt.show()
```

### animate_play

```python
def animate_play(
    play_df: pd.DataFrame,
    output_path: str = "play_animation.mp4"
) -> None
```

Create MP4 animation of entire play.

**Parameters:**
- `play_df` (pd.DataFrame): Play tracking data
- `output_path` (str): Output file path

**Example:**
```python
animate_play(play_df, "play_56.mp4")
```

### plot_attention_map

```python
def plot_attention_map(
    frame_df: pd.DataFrame,
    attn_data: Tuple[Tensor, Tensor],
    target_nfl_id: Optional[int] = None,
    ax: Optional[Axes] = None
) -> Axes
```

Visualize attention weights on the field.

**Parameters:**
- `frame_df` (pd.DataFrame): Single frame data
- `attn_data` (Tuple): `(edge_index, alpha)` from model
- `target_nfl_id` (Optional[int]): Focus on specific player
- `ax` (Optional[Axes]): Existing axes

**Returns:**
- `ax` (Axes): Matplotlib axes with attention visualization

**Example:**
```python
fig, ax = create_football_field()
plot_attention_map(frame_df, attn_weights, target_nfl_id=12345, ax=ax)
plt.title("QB Attention")
plt.show()
```

---

## src.train

### NFLGraphPredictor

**PyTorch Lightning module for training.**

```python
class NFLGraphPredictor(pl.LightningModule):
    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        future_seq_len: int = 10
    )
```

**Parameters:**
- `input_dim` (int): Node feature dimension
- `hidden_dim` (int): Hidden dimension
- `lr` (float): Learning rate
- `future_seq_len` (int): Prediction horizon

#### Methods

##### forward

```python
def forward(self, data: Data) -> Tuple[Tensor, Tensor, Optional[Tuple]]
```

##### training_step

```python
def training_step(self, batch: Data, batch_idx: int) -> Tensor
```

##### validation_step

```python
def validation_step(self, batch: Data, batch_idx: int) -> None
```

##### configure_optimizers

```python
def configure_optimizers(self) -> Dict
```

### train_model

```python
def train_model(sanity: bool = False) -> None
```

Main training function.

**Parameters:**
- `sanity` (bool): Run quick sanity check if True

**Example:**
```bash
python -m src.train --mode train
```

### tune_model

```python
def tune_model(num_trials: int = 5) -> None
```

Hyperparameter tuning with Optuna.

**Parameters:**
- `num_trials` (int): Number of Optuna trials

**Example:**
```bash
python -m src.train --mode tune
```

---

## Type Hints

Common type aliases used throughout the codebase:

```python
from typing import List, Tuple, Optional, Dict, Union
import torch
from torch import Tensor
from torch_geometric.data import Data
import polars as pl
import pandas as pd

# Common types
GraphList = List[Data]
TensorTuple = Tuple[Tensor, Tensor, Optional[Tuple]]
DataFrame = Union[pl.DataFrame, pd.DataFrame]
```

---

## Error Handling

### Common Exceptions

```python
# FileNotFoundError
try:
    df = loader.load_week_data(99)
except FileNotFoundError as e:
    print(f"Week data not found: {e}")

# Shape mismatch
try:
    model(invalid_graph)
except RuntimeError as e:
    print(f"Shape error: {e}")

# CUDA out of memory
try:
    trainer.fit(model, train_loader)
except torch.cuda.OutOfMemoryError:
    print("Reduce batch size or model size")
    torch.cuda.empty_cache()
```

---

## Best Practices

### Memory Management

```python
# Clear CUDA cache
torch.cuda.empty_cache()

# Use context managers
with torch.no_grad():
    predictions = model(graph)

# Delete large objects
del large_tensor
torch.cuda.empty_cache()
```

### Reproducibility

```python
import torch
import numpy as np
import random

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
```

### Device Handling

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)
```

---

## Examples

### Complete Workflow

```python
# 1. Load data
from src.data_loader import DataLoader
loader = DataLoader(data_dir=".")
df = loader.load_week_data(1)
df = loader.standardize_tracking_directions(df)

# 2. Create graphs
from src.features import create_graph_data
graphs = create_graph_data(df, radius=20.0)

# 3. Initialize model
from src.models.gnn import NFLGraphTransformer
model = NFLGraphTransformer(input_dim=7, hidden_dim=64)

# 4. Predict
import torch
model.eval()
with torch.no_grad():
    traj, cov, attn = model(graphs[0], return_attention_weights=True)

# 5. Visualize
from src.visualization import plot_attention_map, create_football_field
import matplotlib.pyplot as plt

fig, ax = create_football_field()
frame_df = df.filter(pl.col("frame_id") == 1).to_pandas()
plot_attention_map(frame_df, attn, ax=ax)
plt.show()
```

---

## Version Compatibility

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.11+ | Required |
| PyTorch | 2.2+ | GPU support recommended |
| PyTorch Geometric | Latest | Auto-installed |
| Polars | 0.20+ | Fast data processing |
| PyTorch Lightning | 2.2+ | Training framework |

---

## train_production

### ProductionConfig
Dataclass-style container built from CLI/YAML.

**Key fields (subset):**
- Data: `data_dir`, `weeks`, `train_split=0.8`, `val_split=0.1`, `validate_data`
- Model: `hidden_dim=128`, `num_gnn_layers=6`, `heads=8`, `probabilistic=True`, `num_modes=8`, `droppath_rate=0.08`
- Training: `batch_size=64`, `accumulate_grad_batches=2`, `max_epochs=100`, `precision='bf16-mixed'`
- Loss/regularization: `velocity_weight=0.3`, `acceleration_weight=0.1`, `collision_weight=0.05`, `coverage_weight=0.5`, `use_huber_loss=True`, `label_smoothing=0.05`
- Experiment: `experiment_name`, `use_mlflow`, `use_wandb`, `use_tensorboard`
- Production: `export_onnx`, `export_torchscript`, `enable_sample_batch_warmup`, `enable_profiling`

### ProductionTrainer

```python
trainer = ProductionTrainer(config)
trainer.train(resume_from=None)
```

**Responsibilities:**
- Validates data (`DataValidator`) and shows a per-week summary
- Builds deterministic play splits and persists them to `outputs/splits_production.json`
- Creates cached graph datasets (`cache/graphs`) and PyG dataloaders
- Instantiates `NFLGraphPredictor` with probabilistic or deterministic decoder
- Configures callbacks: EarlyStopping, ModelCheckpoint, LR Monitor, SWA, EpochSummary, optional AttentionVisualization
- Saves resolved config to `outputs/<experiment>_config.json` (e.g., `nfl_production_v2_config.json`)
- Exports TorchScript/ONNX to `outputs/exported_models/` when enabled

### DataValidator

```python
valid, issues = DataValidator.validate_dataframe(df, week=1)
```

Checks required columns, nulls, range violations, and duplicate frames before training.

---

## Further Reading

- [Architecture Documentation](architecture.md)
- [Usage Guide](usage.md)
- [Configuration Reference](configuration.md)
