# Data Pipeline

> ETL process, graph construction, feature engineering, and caching strategies.

## 🔄 Pipeline Overview

```mermaid
graph LR
    A[CSV Files] --> B[Polars Load]
    B --> C[Feature Engineering]
    C --> D[Graph Construction]
    D --> E[Disk Cache]
    E --> F[PyG DataLoader]
    F --> G[Training]
```

---

## 📥 Data Ingestion

### Source Files

| File | Description | Size (18 weeks) |
|------|-------------|-----------------|
| `tracking_week_*.csv` | Player positions @ 10 Hz | ~500 MB/week |
| `plays.csv` | Play metadata | ~5 MB |
| `players.csv` | Player info | ~200 KB |
| `games.csv` | Game schedule | ~50 KB |

### Loading with Polars

```python
from src.data_loader import DataLoader

loader = DataLoader(".")

# Load single week
df = loader.load_week_data(1)
print(df.schema)

# Key columns:
# - nfl_id: Player ID
# - x, y: Position (yards)
# - s: Speed (yards/sec)
# - a: Acceleration
# - dir, o: Direction, Orientation (degrees)
# - game_id, play_id, frame_id
```

---

## ⚙️ Feature Engineering

### Node Features (9D)

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | x | [0, 120] | X position (yards) |
| 1 | y | [0, 53.33] | Y position (yards) |
| 2 | speed | [0, ~12] | Speed (yards/sec) |
| 3 | acceleration | [-5, 5] | Acceleration |
| 4 | direction | [0, 360] | Movement direction |
| 5 | velocity_x | [-12, 12] | X velocity |
| 6 | velocity_y | [-12, 12] | Y velocity |
| 7 | is_ball_carrier | {0, 1} | Ball carrier flag |
| 8 | motion_intensity | [0, 15] | sqrt(vx² + vy²) + a |

### Edge Features (5D)

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | distance | [0, radius] | Euclidean distance |
| 1 | angle | [0, 2π] | Relative angle |
| 2 | relative_speed | [-15, 15] | Speed difference |
| 3 | relative_direction | [0, 360] | Direction difference |
| 4 | same_team | {0, 1} | Same team indicator |

### Strategic Embeddings

| Embedding | Vocab | Description |
|-----------|-------|-------------|
| Role | 5 | QB, RB, WR, TE, OL, DL, LB, DB |
| Side | 3 | Offense, Defense, Ball |
| Formation | 8 | I-Form, Shotgun, Pistol, etc. |
| Alignment | 10 | Left, Right, Center, Wide, etc. |

---

## 🔗 Graph Construction

### Algorithm

```python
def create_graph(players_df, frame_id, radius=20.0):
    """
    1. Extract player positions at frame_id
    2. Connect players within `radius` yards
    3. Compute node features
    4. Compute edge features
    5. Add context (down, distance, box)
    """
    
    # Node selection
    nodes = players_df.filter(pl.col("frame_id") == frame_id)
    
    # Edge creation (radius graph)
    from torch_geometric.nn import radius_graph
    positions = torch.tensor(nodes[["x", "y"]].to_numpy())
    edge_index = radius_graph(positions, r=radius)
    
    # Feature computation
    node_features = compute_node_features(nodes)
    edge_features = compute_edge_features(nodes, edge_index)
    
    return Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_features,
        ...
    )
```

### Graph Statistics

| Property | Value |
|----------|-------|
| Nodes per graph | 22 (players) |
| Edges (radius=20) | ~100-150 |
| Node features | 9D |
| Edge features | 5D |
| Graphs per week | ~10K |
| Total graphs | ~185K (18 weeks) |

---

## 💾 Caching System

### Disk Cache

```python
GraphDataset(
    cache_dir="cache/finetune/train",
    persist_cache=True,  # Save to disk
)
```

**Benefits:**
- Skip graph construction after first run
- 9x faster training
- ~1.8GB disk for 185K graphs

### RAM Cache

```python
GraphDataset(
    in_memory_cache_size=200,  # Keep 200 graphs in RAM
)
```

**Benefits:**
- Zero disk I/O for cached graphs
- Best for repeated epochs

### Pre-Caching Script

```python
# cache/precache_graphs.py
from src.data_loader import *
from pathlib import Path

loader = DataLoader('.')
play_meta = build_play_metadata(loader, list(range(1,19)), 5, 10)
tuples = expand_play_tuples(play_meta)

cache_dir = Path('cache/finetune/train')
cache_dir.mkdir(parents=True, exist_ok=True)

ds = GraphDataset(loader, tuples, 20.0, 10, 5,
                  cache_dir=cache_dir, persist_cache=True)

# Build cache
for i, _ in enumerate(ds):
    if i % 1000 == 0: print(f'{i}/{len(ds)}')
```

---

## 📊 Data Splits

### Train/Val Split

```python
# 80/20 split by play
from sklearn.model_selection import train_test_split

train_plays, val_plays = train_test_split(
    all_plays, test_size=0.2, random_state=42
)

# Result:
# Train: ~147K samples
# Val: ~38K samples
```

### Week-Based Split

```python
# Alternative: temporal split
train_weeks = list(range(1, 15))   # Weeks 1-14
val_weeks = list(range(15, 19))    # Weeks 15-18
```

---

## ⚡ Performance Optimization

| Optimization | Impact |
|--------------|--------|
| Polars (vs Pandas) | 3-5x faster loading |
| Graph pre-caching | 9x faster training |
| RAM cache | Zero disk I/O |
| Lazy evaluation | Lower memory |
