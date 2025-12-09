# Testing & Verification Strategy

This document outlines the comprehensive testing approach for the NFL Analytics Engine.

## Testing Philosophy

The project employs a **multi-tiered verification strategy**:

1. **Sanity Checks** - Fast end-to-end smoke tests
2. **Phase Verification** - Feature-specific validation scripts
3. **Unit Tests** - Component-level testing
4. **Integration Tests** - Full pipeline validation
5. **Visual Verification** - Explainability through visualization

---

## 1. Sanity Checks

Quick, end-to-end runs designed to crash early if something is broken.

### Quick Sanity Check

```bash
python -m src.train --mode train --sanity
```

**What it does:**
1. Loads a single game from Week 1
2. Processes only ~500 frames
3. Constructs ~400 graphs
4. Trains for 10 batches
5. Reports validation metrics

**Expected output:**
```
Loading data...
Loaded 1 game, 523 frames
Creating graphs...
Created 412 graphs
Training...
Epoch 0: train_loss=2.45, val_loss=2.31, val_ade=1.85, val_fde=3.21
✓ Sanity check passed!
```

**Pass criteria:**
- ✅ Exit code 0
- ✅ No exceptions raised
- ✅ Validation ADE < 5.0 yards
- ✅ Coverage accuracy > 0%

**Typical runtime:** 30-60 seconds (GPU), 2-3 minutes (CPU)

### Production Smoke Test (deterministic splits)

```bash
python train_production.py \
  --config configs/production.yaml \
  --limit-train-batches 0.1 \
  --limit-val-batches 0.2 \
  --no-validation \
  --enable-sample-batch-warmup
```

- Reuses splits from `outputs/splits_production.json` for repeatability
- Writes the resolved config to `outputs/nfl_production_v2_config.json` even in smoke mode
- Expect <10 minutes on GPU with the above limits

---

## 2. Phase Verification Scripts

Specific scripts verify features added in each development phase.

### Available Verification Scripts

| Script | Phase | Purpose | Key Checks |
|--------|-------|---------|------------|
| `verify_rules.py` | 1 | Data integrity | Nullified plays removed? |
| `verify_phase1.py` | 1 | Basic pipeline | Data loads correctly? |
| `verify_phase8.py` | 8 | Context fusion | Context tensor shape `[B, 3]`? |
| `verify_phase9.py` | 9 | Multi-task | Both outputs present? |
| `verify_phase10.py` | 10 | Attention | Attention weights extracted? |
| `verify_phase11.py` | 11 | Strategic embeddings | All embeddings fuse correctly? |

### Running Verification Scripts

```bash
# Run all verification scripts
python tests/verify_rules.py
python tests/verify_phase8.py
python tests/verify_phase10.py
python tests/verify_phase11.py

# Or run individually
cd tests
python verify_phase11.py
```

### Example: verify_phase11.py

Tests strategic embedding integration:

```python
"""
Verify Phase 11: Strategic Embeddings
- Role embeddings (5 categories)
- Side embeddings (3 categories)
- Formation embeddings (8 categories)
- Alignment embeddings (10 categories)
"""

def test_strategic_embeddings():
    # Create mock data
    data = create_mock_graph()
    
    # Initialize model
    model = NFLGraphTransformer(
        input_dim=7,
        hidden_dim=64,
        num_gnn_layers=4
    )
    
    # Forward pass
    traj, cov, attn = model(data, return_attention_weights=True)
    
    # Assertions
    assert traj.shape == (data.x.shape[0], 10, 2), "Trajectory shape mismatch"
    assert cov.shape == (1, 1), "Coverage shape mismatch"
    assert attn is not None, "Attention weights not returned"
    
    print("✓ Strategic embeddings working correctly")

if __name__ == "__main__":
    test_strategic_embeddings()
```

**Expected output:**
```
Testing strategic embeddings...
✓ Role embeddings: (23, 64)
✓ Formation embeddings: (1, 64)
✓ Alignment embeddings: (1, 64)
✓ Context fusion successful
✓ Forward pass successful
✓ Strategic embeddings working correctly
```

---

## 3. Unit Tests

Component-level tests using pytest.

### Test Structure

```
tests/
├── test_gnn.py          # Model architecture tests
├── test_metrics.py      # Metrics computation tests
├── test_viz.py          # Visualization tests
└── conftest.py          # Pytest fixtures
```

### Running Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_gnn.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Example: test_gnn.py

```python
import pytest
import torch
from torch_geometric.data import Data
from src.models.gnn import NFLGraphTransformer

@pytest.fixture
def mock_graph():
    \"\"\"Create mock graph for testing\"\"\"
    return Data(
        x=torch.randn(23, 7),
        edge_index=torch.randint(0, 23, (2, 100)),
        edge_attr=torch.randn(100, 5),  # 5D edge features
        role=torch.randint(0, 5, (23,)),
        side=torch.randint(0, 3, (23,)),
        formation=torch.tensor([0]),
        alignment=torch.tensor([1]),
        context=torch.randn(1, 3),
        y=torch.randn(23, 10, 2),
        y_coverage=torch.tensor([1]),
        history=torch.randn(23, 5, 4)  # Motion history
    )

def test_model_forward(mock_graph):
    """Test forward pass"""
    model = NFLGraphTransformer(input_dim=7, hidden_dim=64)
    traj, cov, attn = model(mock_graph, return_attention_weights=True)
    
    assert traj.shape == (23, 10, 2)
    assert cov.shape == (1, 1)
    assert attn is not None

def test_model_shapes(mock_graph):
    """Test output shapes for different configurations"""
    configs = [
        {'hidden_dim': 32, 'heads': 2},
        {'hidden_dim': 64, 'heads': 4},
        {'hidden_dim': 128, 'heads': 8},
    ]
    
    for config in configs:
        model = NFLGraphTransformer(input_dim=7, **config)
        traj, cov, _ = model(mock_graph)
        assert traj.shape == (23, 10, 2)
        assert cov.shape == (1, 1)

def test_gradient_flow(mock_graph):
    """Test that gradients flow through model"""
    model = NFLGraphTransformer(input_dim=7, hidden_dim=64)
    traj, cov, _ = model(mock_graph)
    
    # Compute loss
    loss = traj.mean() + cov.mean()
    loss.backward()
    
    # Check gradients exist
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
```

### Example: test_metrics.py

```python
import pytest
import torch
from src.metrics import calculate_zone_collapse_speed
import polars as pl

def test_ade_computation():
    """Test Average Displacement Error"""
    pred = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])  # [1, 2, 2]
    target = torch.tensor([[[1.5, 1.5], [2.5, 2.5]]])
    
    errors = torch.norm(pred - target, dim=2)
    ade = errors.mean()
    
    expected = (0.707 + 0.707) / 2  # sqrt(0.5) ≈ 0.707
    assert abs(ade.item() - expected) < 0.01

def test_fde_computation():
    """Test Final Displacement Error"""
    pred = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])
    target = torch.tensor([[[1.5, 1.5], [3.0, 3.0]]])
    
    final_error = torch.norm(pred[:, -1] - target[:, -1], dim=1)
    fde = final_error.mean()
    
    expected = 1.414  # sqrt(2)
    assert abs(fde.item() - expected) < 0.01

def test_zone_collapse_speed():
    """Test zone collapse metric"""
    # Create mock play data
    df = pl.DataFrame({
        'frame_id': [1, 1, 1, 2, 2, 2],
        'club': ['KC', 'KC', 'KC', 'KC', 'KC', 'KC'],
        'std_x': [50, 55, 60, 51, 54, 59],
        'std_y': [20, 25, 30, 21, 24, 29]
    })
    
    result = calculate_zone_collapse_speed(df, 'KC')
    
    assert 'hull_area' in result.columns
    assert 'hull_area_rate' in result.columns
    assert result.height == 2  # 2 frames
```

---

## 4. Integration Tests

Test the complete pipeline from data loading to prediction.

### Full Pipeline Test

```python
def test_full_pipeline():
    """Test complete workflow"""
    from src.data_loader import DataLoader
    from src.features import create_graph_data
    from src.models.gnn import NFLGraphTransformer
    
    # 1. Load data
    loader = DataLoader(data_dir=".")
    df = loader.load_week_data(1)
    df = loader.standardize_tracking_directions(df)
    
    # 2. Create graphs
    graphs = create_graph_data(df, radius=20.0)
    assert len(graphs) > 0
    
    # 3. Initialize model
    model = NFLGraphTransformer(input_dim=7, hidden_dim=64)
    
    # 4. Forward pass
    graph = graphs[0]
    traj, cov, attn = model(graph, return_attention_weights=True)
    
    # 5. Verify outputs
    assert traj.shape[1] == 10  # 10 future frames
    assert traj.shape[2] == 2   # x, y coordinates
    assert cov.shape == (1, 1)  # Coverage logit
    
    print("✓ Full pipeline test passed")
```

---

## 5. Visual Verification (Explainability)

Trust the model by visualizing its decisions.

### Attention Map Visualization

```python
from src.visualization import plot_attention_map, create_football_field
import matplotlib.pyplot as plt

# Get predictions with attention
model.eval()
with torch.no_grad():
    traj, cov, attn_weights = model(graph, return_attention_weights=True)

# Extract frame data
frame_df = df.filter(
    (pl.col("game_id") == game_id) & 
    (pl.col("play_id") == play_id) &
    (pl.col("frame_id") == frame_id)
).to_pandas()

# Plot
fig, ax = create_football_field()
plot_attention_map(frame_df, attn_weights, target_nfl_id=qb_id, ax=ax)
plt.title("QB Attention Map")
plt.savefig("attention_qb.png")
```

**Interpretation:**
- **Thick lines**: Strong attention (model focuses here)
- **Thin lines**: Weak attention
- **Direction**: From observer to attended player

**Expected patterns:**
- QB → WR (strong attention to receivers)
- CB → WR (defenders tracking receivers)
- LB → RB (linebackers watching running backs)

### Trajectory Visualization

```python
from src.visualization import animate_play

# Animate actual vs predicted
play_df = df.filter(
    (pl.col("game_id") == game_id) & 
    (pl.col("play_id") == play_id)
).to_pandas()

animate_play(play_df, output_path="play_actual.mp4")

# TODO: Add predicted trajectory overlay
```

---

## 6. Performance Metrics

### Metrics to Track

| Metric | Target | Description |
|--------|--------|-------------|
| **ADE** | < 2.0 yards | Average Displacement Error |
| **FDE** | < 3.5 yards | Final Displacement Error |
| **Coverage Acc** | > 70% | Man vs Zone classification |
| **Training Time** | < 2 hours | On single GPU (RTX 3090) |
| **Inference Speed** | < 50 ms/play | Average prediction latency |

### Computing Metrics

```python
def evaluate_model(model, test_loader):
    """Comprehensive model evaluation"""
    model.eval()
    
    total_ade = 0
    total_fde = 0
    total_cov_acc = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            traj_pred, cov_pred, _ = model(batch)
            
            # ADE
            errors = torch.norm(traj_pred - batch.y, dim=2)
            total_ade += errors.mean().item()
            
            # FDE
            final_errors = torch.norm(traj_pred[:, -1] - batch.y[:, -1], dim=1)
            total_fde += final_errors.mean().item()
            
            # Coverage accuracy
            cov_labels = (torch.sigmoid(cov_pred) > 0.5).long().squeeze()
            total_cov_acc += (cov_labels == batch.y_coverage).float().mean().item()
            
            num_samples += 1
    
    return {
        'ade': total_ade / num_samples,
        'fde': total_fde / num_samples,
        'coverage_acc': total_cov_acc / num_samples
    }
```

---

## 7. Continuous Integration (Future)

### GitHub Actions Workflow

Create `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=src
    
    - name: Run verification scripts
      run: |
        python tests/verify_phase11.py
        python tests/verify_phase10.py
```

---

## 8. Debugging Tests

### Enable Verbose Output

```bash
# Pytest verbose mode
pytest tests/ -v -s

# Show print statements
pytest tests/ -s

# Stop on first failure
pytest tests/ -x
```

### Debug Specific Test

```bash
# Run single test
pytest tests/test_gnn.py::test_model_forward -v

# With debugger
pytest tests/test_gnn.py::test_model_forward --pdb
```

### Logging in Tests

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_with_logging():
    logger.debug("Starting test")
    # ... test code ...
    logger.debug("Test completed")
```

---

## 9. Test Coverage

### Generate Coverage Report

```bash
# Run with coverage
pytest tests/ --cov=src --cov-report=html

# View report
open htmlcov/index.html
```

**Target coverage:** > 80%

### Coverage by Module

| Module | Target Coverage | Description |
|--------|----------------|-------------|
| `data_loader.py` | 90% | Critical data pipeline |
| `features.py` | 85% | Graph construction |
| `models/gnn.py` | 80% | Model architecture |
| `metrics.py` | 95% | Metrics computation |
| `visualization.py` | 60% | Visualization (manual testing) |

---

## 10. Regression Testing

### Baseline Metrics

Record baseline performance to detect regressions:

```python
# baseline_metrics.json
{
    "version": "1.0.0",
    "date": "2025-12-07",
    "metrics": {
        "ade": 1.85,
        "fde": 3.21,
        "coverage_acc": 0.72
    }
}
```

### Regression Test

```python
import json

def test_no_regression():
    """Ensure performance doesn't degrade"""
    with open('baseline_metrics.json') as f:
        baseline = json.load(f)
    
    # Evaluate current model
    current_metrics = evaluate_model(model, test_loader)
    
    # Check for regression (allow 5% degradation)
    assert current_metrics['ade'] <= baseline['metrics']['ade'] * 1.05
    assert current_metrics['fde'] <= baseline['metrics']['fde'] * 1.05
    assert current_metrics['coverage_acc'] >= baseline['metrics']['coverage_acc'] * 0.95
```

---

## Testing Checklist

Before committing code:

- [ ] Run sanity check: `python -m src.train --mode train --sanity`
- [ ] Run all verification scripts
- [ ] Run unit tests: `pytest tests/ -v`
- [ ] Check test coverage: `pytest tests/ --cov=src`
- [ ] Visual verification: Check attention maps
- [ ] Performance check: Metrics within acceptable range
- [ ] No new warnings or errors in logs

---

## Troubleshooting Tests

**Issue: Tests fail with import errors**
```bash
# Ensure package is installed
pip install -e .
```

**Issue: GPU tests fail on CPU**
```python
# Add device handling
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)
```

**Issue: Flaky tests (random failures)**
```python
# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
```

---

## Next Steps

- **API Reference:** See [api_reference.md](api_reference.md) for detailed API docs
- **Performance:** See [performance.md](performance.md) for benchmarks
- **Usage:** See [usage.md](usage.md) for training workflows
