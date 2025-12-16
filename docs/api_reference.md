# API Reference

> Complete API documentation for all modules, classes, and functions.

## 📦 Core Modules

---

## `src.models.gnn`

### NFLGraphTransformer

Main trajectory prediction model.

```python
class NFLGraphTransformer(nn.Module):
    """
    SOTA Graph Neural Network for NFL trajectory prediction.
    
    Combines GATv2 encoder with probabilistic GMM decoder.
    """
    
    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 256,
        num_layers: int = 8,
        heads: int = 8,
        future_seq_len: int = 10,
        probabilistic: bool = True,
        num_modes: int = 8,
        dropout: float = 0.1,
        droppath_rate: float = 0.12,
    ):
        """
        Args:
            input_dim: Node feature dimension (default: 9)
            hidden_dim: Hidden layer size (default: 256)
            num_layers: Number of GATv2 layers (default: 8)
            heads: Attention heads per layer (default: 8)
            future_seq_len: Prediction horizon in frames (default: 10)
            probabilistic: Use GMM decoder (default: True)
            num_modes: GMM modes (default: 8)
            dropout: Dropout rate (default: 0.1)
            droppath_rate: Stochastic depth rate (default: 0.12)
        """
    
    def forward(
        self,
        batch,
        return_distribution: bool = False,
        return_attention_weights: bool = False,
    ):
        """
        Forward pass.
        
        Args:
            batch: PyG Batch object with node features
            return_distribution: Return GMM parameters
            return_attention_weights: Return attention maps
            
        Returns:
            predictions: [N, T, 2] trajectory predictions
            coverage: [B, 1] coverage scores
            attention: Optional attention weights
        """
```

### GraphPlayerEncoder

```python
class GraphPlayerEncoder(nn.Module):
    """
    8-layer GATv2 encoder with strategic embeddings.
    
    Features:
    - Role, side, formation, alignment embeddings
    - Temporal position encoding
    - DropPath (stochastic depth)
    - Residual connections + LayerNorm
    """
    
    def forward(
        self,
        x,              # [N, input_dim] node features
        edge_index,     # [2, E] edge connectivity
        edge_attr,      # [E, 5] edge features
        context=None,   # [B, 3] play context
        batch=None,     # [N] batch assignment
        role=None,      # [N] player roles
        side=None,      # [N] offense/defense
        history=None,   # [N, T, 4] motion history
    ):
        """
        Returns:
            node_embs: [N, hidden_dim] encoded features
            attention_weights: Optional attention maps
        """
```

### LearnableGraphPooling

```python
class LearnableGraphPooling(nn.Module):
    """
    SOTA graph pooling using AttentionalAggregation.
    
    Replaces mean pooling with learned attention.
    """
    
    def __init__(self, hidden_dim: int = 64):
        """Gate + transform networks for attention."""
    
    def forward(self, x, batch=None):
        """
        Args:
            x: [N, D] node embeddings
            batch: [N] batch assignment
        Returns:
            [B, D] graph-level embeddings
        """
```

---

## `src.losses.contrastive_losses`

### SocialNCELoss

```python
class SocialNCELoss(nn.Module):
    """
    Contrastive learning for social interactions.
    
    Args:
        hidden_dim: Embedding dimension (default: 256)
        temperature: Contrastive temperature (default: 0.07)
    """
    
    def forward(self, node_embeddings, batch, edge_index):
        """
        Args:
            node_embeddings: [N, D] from encoder
            batch: [N] batch indices
            edge_index: [2, E] edges
        Returns:
            loss: Scalar NCE loss
        """
```

### WinnerTakesAllLoss

```python
class WinnerTakesAllLoss(nn.Module):
    """
    Multi-modal training with k-best selection.
    
    Args:
        k_best: Number of modes to backprop (default: 2)
    """
    
    def forward(self, predictions, targets, mode_probs):
        """
        Args:
            predictions: [N, T, K, 2] per-mode predictions
            targets: [N, T, 2] ground truth
            mode_probs: [N, K] mode probabilities
        Returns:
            loss: WTA loss (only best k modes)
        """
```

### DiversityLoss

```python
class DiversityLoss(nn.Module):
    """
    Encourages diverse trajectory modes.
    
    Args:
        min_distance: Minimum separation (default: 2.0 yards)
    """
    
    def forward(self, predictions):
        """
        Args:
            predictions: [N, T, K, 2] multi-modal predictions
        Returns:
            loss: Diversity penalty
        """
```

### EndpointFocalLoss

```python
class EndpointFocalLoss(nn.Module):
    """
    Focal loss for trajectory endpoints.
    
    Args:
        gamma: Focusing parameter (default: 2.5)
    """
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [N, 2] final positions
            targets: [N, 2] ground truth endpoints
        Returns:
            loss: Focal-weighted endpoint loss
        """
```

---

## `src.train`

### NFLGraphPredictor

```python
class NFLGraphPredictor(pl.LightningModule):
    """
    PyTorch Lightning module for training.
    
    Handles:
    - Model forward pass
    - Loss computation (all SOTA losses)
    - Optimizer configuration
    - Metric logging
    """
    
    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 256,
        lr: float = 0.0008,
        weight_decay: float = 0.03,
        probabilistic: bool = True,
        num_modes: int = 8,
        use_social_nce: bool = True,
        use_wta_loss: bool = True,
        use_diversity_loss: bool = True,
        use_endpoint_focal: bool = True,
    ):
        """Initialize model with all loss modules."""
    
    def training_step(self, batch, batch_idx):
        """Single training step with all losses."""
    
    def validation_step(self, batch, batch_idx):
        """Validation with metric computation."""
    
    def configure_optimizers(self):
        """AdamW with cosine LR schedule."""
```

---

## `src.data_loader`

### DataLoader

```python
class DataLoader:
    """
    Polars-based data loading for NFL tracking data.
    """
    
    def __init__(self, data_dir: str = "."):
        """
        Args:
            data_dir: Directory with CSV files
        """
    
    def load_week_data(self, week: int) -> pl.DataFrame:
        """Load tracking data for a week."""
    
    def load_plays(self) -> pl.DataFrame:
        """Load plays.csv."""
    
    def load_players(self) -> pl.DataFrame:
        """Load players.csv."""
```

### GraphDataset

```python
class GraphDataset(Dataset):
    """
    PyG Dataset with disk caching.
    """
    
    def __init__(
        self,
        loader: DataLoader,
        play_tuples: List[Tuple],
        radius: float = 20.0,
        future_seq_len: int = 10,
        history_len: int = 5,
        cache_dir: Path = None,
        persist_cache: bool = True,
        in_memory_cache_size: int = 100,
    ):
        """
        Args:
            loader: DataLoader instance
            play_tuples: List of (week, game_id, play_id, frame_id)
            radius: Edge connection radius
            future_seq_len: Prediction frames
            history_len: History frames
            cache_dir: Disk cache directory
            persist_cache: Save graphs to disk
            in_memory_cache_size: RAM cache size
        """
    
    def __getitem__(self, idx) -> Data:
        """Returns PyG Data object (cached if available)."""
```

### Helper Functions

```python
def build_play_metadata(
    loader: DataLoader,
    weeks: List[int],
    history_len: int,
    future_seq_len: int,
) -> List[dict]:
    """Build metadata for all valid plays."""

def expand_play_tuples(play_meta: List[dict]) -> List[Tuple]:
    """Expand to frame-level tuples."""
```

---

## `src.competition_metrics`

### Confidence Intervals

```python
def calculate_confidence_intervals(
    predictions: np.ndarray,      # [N, T, 2]
    std_predictions: np.ndarray,  # [N, T, 2] optional
    confidence_level: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        lower_bounds: [N, T, 2]
        upper_bounds: [N, T, 2]
    """
```

### Spatial Control

```python
def calculate_spatial_control_probability(
    player_positions: np.ndarray,   # [N, 2]
    player_velocities: np.ndarray,  # [N, 2]
    field_resolution: float = 1.0,
) -> np.ndarray:
    """
    Returns:
        control_map: [N, H, W] probability per player
    """
```

### Recovery Index

```python
def calculate_recovery_ability_index(
    defender_positions: np.ndarray,  # [T, 2]
    receiver_positions: np.ndarray,  # [T, 2]
) -> Dict[str, float]:
    """
    Returns:
        {
            'recovery_index': float,
            'closing_speed_yps': float,
            'separation_closed': float,
        }
    """
```

---

## Type Aliases

```python
# Defined in src/models/gnn.py
NodeFeatures = Tensor    # [N, D]
EdgeIndex = Tensor       # [2, E]
EdgeAttr = Tensor        # [E, D_edge]
BatchIndex = Tensor      # [N]
```
