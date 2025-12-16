# Performance Guide

> Benchmarks, optimization techniques, and GPU utilization strategies for the NFL Analytics Engine.

## 📊 Performance Benchmarks

### Training Speed (RTX 3050 4GB)

| Configuration | Speed (it/s) | Epoch Time | GPU Util |
|---------------|-------------|------------|----------|
| Without Cache | 0.13 | ~6.5 hours | ~10% |
| **With Cache** | 1.2-1.5 | ~45 min | ~70-80% |
| Cache + FP16 | 1.5-2.0 | ~35 min | ~85% |

### Memory Usage

| Component | VRAM Usage |
|-----------|------------|
| Model (5.4M params) | ~500 MB |
| Batch (32 graphs) | ~1.5 GB |
| Gradients | ~1.2 GB |
| Optimizer states | ~800 MB |
| **Total** | **~3.5 GB / 4 GB** |

---

## ⚡ Optimization Techniques

### 1. Graph Pre-Caching (Critical)

**Problem**: Graph construction is CPU-bound. GPU idles waiting for data.

**Solution**: Pre-cache all graphs to disk.

```bash
# Pre-cache ~185K graphs (~1.8GB on disk)
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

**Impact**: 9x speedup (0.13 → 1.2 it/s)

### 2. Mixed Precision Training

**Problem**: FP32 operations don't utilize Tensor Cores.

**Solution**: Enable automatic mixed precision.

```python
# In trainer
precision="16-mixed"

# In code
torch.set_float32_matmul_precision('medium')
```

**Impact**: ~1.5x speedup, 50% less memory

### 3. Gradient Accumulation

**Problem**: Large effective batch size doesn't fit in VRAM.

**Solution**: Accumulate gradients over multiple forward passes.

```yaml
batch_size: 32
accumulate_grad_batches: 5
# Effective batch = 160
```

**Impact**: Better gradients without OOM

### 4. cuDNN Benchmark Mode

**Problem**: Default convolution algorithms not optimal.

**Solution**: Enable autotuning.

```python
trainer = Trainer(benchmark=True)
```

**Impact**: ~10% speedup after warmup

---

## 🎯 Target Metrics

### Competition Targets (Ultimate Config)

| Metric | Target | World-Class |
|--------|--------|-------------|
| **ADE** | < 0.32 | < 0.25 |
| **FDE** | < 0.50 | < 0.40 |
| **minADE** | < 0.22 | < 0.18 |
| **minFDE** | < 0.35 | < 0.28 |
| **Miss Rate** | < 2% | < 1% |

### Metric Definitions

| Metric | Formula | Description |
|--------|---------|-------------|
| **ADE** | Σ‖pred - gt‖ / (N×T) | Average error across all frames |
| **FDE** | ‖pred_T - gt_T‖ / N | Error at final frame |
| **minADE** | min_k(ADE_k) | Best of K modes |
| **minFDE** | min_k(FDE_k) | Best final frame |
| **Miss Rate** | %{FDE > 2} | Fraction of bad predictions |

---

## 🔧 Hardware Requirements

### Minimum (RTX 3050 4GB)

| Resource | Requirement |
|----------|-------------|
| GPU VRAM | 4 GB |
| RAM | 16 GB |
| Disk (cache) | 5 GB |
| CPU Cores | 4+ |

### Recommended (RTX 3060+)

| Resource | Requirement |
|----------|-------------|
| GPU VRAM | 8+ GB |
| RAM | 32 GB |
| Disk (SSD) | 20 GB |
| CPU Cores | 8+ |

---

## 📈 Monitoring

### Real-Time GPU Stats

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Detailed metrics
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv -l 1
```

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir lightning_logs/

# Metrics logged:
# - train_loss, val_loss
# - val_ade, val_fde, val_minADE
# - learning_rate
# - gpu_memory_usage
```

---

## 🚀 Speed Optimization Checklist

1. ✅ **Pre-cache graphs** (9x speedup)
2. ✅ **Enable FP16 mixed precision** (1.5x speedup)
3. ✅ **Use benchmark=True** (10% speedup)
4. ✅ **Set torch_matmul_precision('medium')** (Tensor Cores)
5. ✅ **Use SSD for cache** (faster disk I/O)
6. ✅ **Pin memory** (faster CPU→GPU transfer)
7. ⚠️ **num_workers=0** (with cache, CPU not bottleneck)

---

## 🔍 Bottleneck Diagnosis

### Symptom: Low GPU Utilization (< 50%)

**Cause 1**: CPU bottleneck (data loading)
```bash
# Check CPU usage
htop
# If CPU at 100%, graphs not cached
```

**Solution**: Pre-cache graphs

**Cause 2**: Small batch size
```yaml
# Increase batch size
batch_size: 48  # From 32
```

### Symptom: OOM Errors

**Cause**: Model + batch too large

**Solution**:
```yaml
batch_size: 24           # Reduce
num_modes: 6             # Reduce modes
accumulate_grad_batches: 6  # Compensate
```

### Symptom: Slow Training Speed

**Cause**: No mixed precision

**Solution**:
```python
precision="16-mixed"
```
