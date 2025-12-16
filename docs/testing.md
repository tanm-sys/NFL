# Testing Guide

> Verification scripts, unit tests, and quality assurance.

## 🧪 Test Overview

| Test Type | Purpose | Location |
|-----------|---------|----------|
| Unit Tests | Module-level testing | `tests/` |
| Integration | End-to-end validation | `tests/` |
| Verification | Architecture checks | `tests/verify_*.py` |
| Sanity | Quick training test | `configs/sanity.yaml` |

---

## 🚀 Quick Tests

### Sanity Training

```bash
# 2 epochs on small data
python finetune_best_model.py --config configs/sanity.yaml
```

### Module Verification

```bash
# Test model architecture
python -c "
from src.models.gnn import NFLGraphTransformer

model = NFLGraphTransformer(input_dim=9, hidden_dim=64)
print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')
print('✅ Model OK')
"
```

### SOTA Components Test

```bash
python -c "
from src.models.gnn import (
    LearnableGraphPooling,
    GoalConditionedDecoder,
    SceneFlowEncoder,
    EINOPS_AVAILABLE,
)
import torch

print(f'einops: {EINOPS_AVAILABLE}')

# Test LearnableGraphPooling
pooler = LearnableGraphPooling(64)
x = torch.randn(22, 64)
batch = torch.zeros(22, dtype=torch.long)
out = pooler(x, batch)
print(f'✅ LearnableGraphPooling: {out.shape}')

# Test decoder
dec = GoalConditionedDecoder(64, future_seq_len=10)
pred = dec(torch.randn(22, 64))
print(f'✅ GoalConditionedDecoder: {pred.shape}')

print('✅ All SOTA components working!')
"
```

### Loss Functions Test

```bash
python -c "
from src.losses.contrastive_losses import (
    SocialNCELoss,
    WinnerTakesAllLoss,
    DiversityLoss,
    EndpointFocalLoss,
)
import torch

N, T, K = 22, 10, 8

# Social-NCE
snce = SocialNCELoss(hidden_dim=256)
node_emb = torch.randn(N, 256)
edge_index = torch.randint(0, N, (2, 100))
batch = torch.zeros(N, dtype=torch.long)
loss = snce(node_emb, batch, edge_index)
print(f'✅ SocialNCE: {loss.item():.4f}')

# WTA
wta = WinnerTakesAllLoss(k_best=2)
preds = torch.randn(N, T, K, 2)
targets = torch.randn(N, T, 2)
probs = torch.softmax(torch.randn(N, K), dim=-1)
loss = wta(preds, targets, probs)
print(f'✅ WTA: {loss.item():.4f}')

# Diversity
div = DiversityLoss(min_distance=2.0)
loss = div(preds)
print(f'✅ Diversity: {loss.item():.4f}')

# Endpoint Focal
efl = EndpointFocalLoss(gamma=2.5)
loss = efl(preds[:, -1, 0, :], targets[:, -1, :])
print(f'✅ EndpointFocal: {loss.item():.4f}')

print('✅ All losses working!')
"
```

---

## 🔧 Unit Tests

### Run All Tests

```bash
python -m pytest tests/ -v
```

### Test Structure

```
tests/
├── test_data_loader.py     # Data loading tests
├── test_graph_dataset.py   # Graph construction
├── test_model.py           # Model forward pass
├── test_losses.py          # Loss functions
├── test_metrics.py         # Metric calculations
└── verify_*.py             # Phase verification
```

### Sample Test

```python
# tests/test_model.py
import torch
import pytest
from src.models.gnn import NFLGraphTransformer
from torch_geometric.data import Data, Batch

def test_forward_pass():
    model = NFLGraphTransformer(input_dim=9, hidden_dim=64)
    
    data = Data(
        x=torch.randn(22, 9),
        edge_index=torch.randint(0, 22, (2, 100)),
        edge_attr=torch.randn(100, 5),
        current_pos=torch.randn(22, 2),
        context=torch.randn(1, 3),
    )
    batch = Batch.from_data_list([data])
    
    pred, cov, _ = model(batch)
    
    assert pred.shape == (22, 10, 2)
    assert cov.shape == (1, 1)
```

---

## ✅ Verification Checklist

### Pre-Training

- [ ] GPU detected: `torch.cuda.is_available()`
- [ ] Model loads: `NFLGraphTransformer()`
- [ ] Data loads: `DataLoader.load_week_data(1)`
- [ ] Graph cache exists: `ls cache/finetune/train/`

### During Training

- [ ] GPU utilization > 50%: `nvidia-smi`
- [ ] Loss decreasing: TensorBoard
- [ ] No NaN/Inf: Check logs

### Post-Training

- [ ] Checkpoint saved: `ls checkpoints_finetuned/`
- [ ] Metrics computed: Check val_ade, val_fde
- [ ] Inference works: Load and predict

---

## 🐛 Debugging

### Enable Anomaly Detection

```python
torch.autograd.set_detect_anomaly(True)
```

### Gradient Checking

```python
# In training_step
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm()
        if grad_norm > 100:
            print(f"Large gradient: {name} = {grad_norm}")
```

### Memory Profiling

```bash
# Profile memory usage
python -c "
import torch
torch.cuda.memory._record_memory_history(True)

# Run model...

torch.cuda.memory._dump_snapshot('memory_profile.pickle')
"
```
