# Installation Guide

> Complete setup instructions for the NFL Analytics Engine.

## 📋 Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11+ | Required for latest PyTorch |
| CUDA | 11.8+ | For GPU acceleration |
| GPU | 4GB+ VRAM | RTX 3050 or better |
| RAM | 16GB+ | For graph caching |
| Disk | 10GB | For cache storage |

---

## 🚀 Quick Install

```bash
# Clone repository
git clone https://github.com/tanm-sys/nfl-analytics-engine.git
cd nfl-analytics-engine

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install package
pip install -e .
```

---

## 📦 Dependencies

### Core Libraries

```txt
# Core ML
torch>=2.2.0
pytorch-lightning>=2.2.0
torch_geometric>=2.5.0

# Data Processing
polars>=0.20.0
numpy>=1.24.0
pandas>=2.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Optimization
optuna>=3.5.0
```

### SOTA Libraries

```txt
# Advanced features
einops>=0.7.0          # Tensor operations
lion-pytorch>=0.1.2    # Lion optimizer
safetensors>=0.4.0     # Safe model saving
timm>=0.9.0            # Vision models
rich>=13.0.0           # Rich terminal output
```

### Optional (GPU-only)

```txt
# Uncomment if needed:
# mamba-ssm>=2.0.0     # Mamba temporal encoder
# causal-conv1d>=1.2.0 # For mamba-ssm
# triton>=2.1.0        # torch.compile optimization
```

---

## 🔧 PyTorch Geometric Installation

PyG requires separate installation of extensions:

```bash
# Install PyTorch first
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118

# Install PyG
pip install torch_geometric

# Install optional extensions (for faster operations)
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
```

---

## ✅ Verify Installation

```bash
# Quick verification
python -c "
import torch
import torch_geometric
from src.models.gnn import NFLGraphTransformer

print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'PyG: {torch_geometric.__version__}')

# Test model
model = NFLGraphTransformer(input_dim=9, hidden_dim=64)
print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')
print('✅ Installation successful!')
"
```

Expected output:
```
PyTorch: 2.2.0+cu118
CUDA: True
PyG: 2.5.0
Model params: 1,234,567
✅ Installation successful!
```

---

## 📂 Data Setup

### 1. Download NFL Data

Place tracking data in project root:
```
nfl-analytics-engine/
├── tracking_week_1.csv
├── tracking_week_2.csv
├── ...
├── tracking_week_18.csv
├── plays.csv
├── players.csv
└── games.csv
```

### 2. Pre-Cache Graphs (Required)

```bash
# Build cache (~30 min, ~2GB disk)
python -c "
from src.data_loader import *
from pathlib import Path

print('Pre-caching graphs...')
loader = DataLoader('.')
play_meta = build_play_metadata(loader, list(range(1,19)), 5, 10)
tuples = expand_play_tuples(play_meta)

cache_dir = Path('cache/finetune/train')
cache_dir.mkdir(parents=True, exist_ok=True)

ds = GraphDataset(loader, tuples, 20.0, 10, 5,
                  cache_dir=cache_dir, persist_cache=True)
for i, _ in enumerate(ds):
    if i % 1000 == 0: print(f'{i}/{len(ds)}')
print('Done!')
"
```

---

## 🐛 Troubleshooting

### CUDA Not Available

```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### PyG Import Errors

```bash
# Reinstall PyG extensions
pip uninstall torch_scatter torch_sparse torch_cluster
pip install pyg_lib torch_scatter torch_sparse \
    -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
```

### Memory Errors

```yaml
# Reduce batch size in config
batch_size: 24  # From 32
num_modes: 6    # From 8
```

---

## 🐳 Docker (Optional)

```dockerfile
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["python", "finetune_best_model.py"]
```

```bash
# Build and run
docker build -t nfl-engine .
docker run --gpus all nfl-engine
```

---

## ☁️ Google Colab

```python
# In Colab notebook
!pip install torch-geometric
!pip install polars einops lion-pytorch rich

# Clone repo
!git clone https://github.com/tanm-sys/nfl-analytics-engine.git
%cd nfl-analytics-engine

# Verify
!python -c "import torch; print(torch.cuda.is_available())"
```
