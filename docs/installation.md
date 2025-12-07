# Installation & Setup Guide

This guide covers the complete environment setup required for the NFL Analytics Engine.

## System Requirements

### Minimum Requirements

| Component | Requirement | Notes |
|-----------|-------------|-------|
| **OS** | Linux (Ubuntu 20.04+), macOS 11+, Windows 10+ | Linux recommended for best performance |
| **Python** | 3.11 or higher | **Required** - uses modern type hints |
| **RAM** | 8 GB | 16 GB recommended for full training |
| **Storage** | 10 GB free | For data, models, and dependencies |

### Recommended for Training

| Component | Specification | Purpose |
|-----------|---------------|---------|
| **GPU** | NVIDIA GPU with 8+ GB VRAM | RTX 3060, RTX 3070, or better |
| **CUDA** | 11.8 or 12.1 | For PyTorch GPU acceleration |
| **RAM** | 16 GB | Faster data loading |
| **CPU** | 8+ cores | Parallel data processing |

> [!NOTE]
> The model can run on CPU for inference and sanity checks, but GPU is strongly recommended for full training.

---

## Installation Steps

### 1. Clone the Repository

```bash
# Clone from GitHub
git clone https://github.com/your-org/nfl-analytics-engine.git
cd nfl-analytics-engine

# Verify structure
ls -la
# Should see: src/, docs/, tests/, train/, pyproject.toml, README.md
```

### 2. Create Virtual Environment

**Using venv (recommended):**

```bash
# Create virtual environment
python3.11 -m venv .venv

# Activate on Linux/macOS
source .venv/bin/activate

# Activate on Windows
.venv\Scripts\activate

# Verify Python version
python --version
# Should output: Python 3.11.x or higher
```

**Using conda (alternative):**

```bash
# Create conda environment
conda create -n nfl-analytics python=3.11
conda activate nfl-analytics
```

### 3. Upgrade pip

```bash
# Upgrade pip to latest version
pip install --upgrade pip setuptools wheel
```

### 4. Install Dependencies

The project uses `pyproject.toml` for dependency management.

```bash
# Install in editable mode (recommended for development)
pip install -e .

# Or install normally
pip install .
```

**Dependencies installed:**
- **Data Processing:** polars, pandas, numpy, scipy, duckdb
- **Machine Learning:** torch, pytorch-lightning, scikit-learn, xgboost
- **Graph Neural Networks:** torch-geometric (auto-installed)
- **Visualization:** matplotlib, seaborn, plotly
- **Experiment Tracking:** mlflow
- **Notebooks:** jupyter

### 5. Install PyTorch Geometric (if needed)

PyTorch Geometric should install automatically, but if you encounter issues:

```bash
# For CUDA 11.8
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

# For CUDA 12.1
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# For CPU only
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
```

### 6. Verify Installation

```bash
# Test imports
python -c "import torch; import torch_geometric; import polars; print('✓ All imports successful')"

# Check GPU availability
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Expected output:**
```
✓ All imports successful
GPU Available: True
GPU Name: NVIDIA GeForce RTX 3090
```

---

## Project Structure

Understanding the layout helps navigate the codebase.

```
nfl-analytics-engine/
├── src/                          # Source code
│   ├── __init__.py
│   ├── data_loader.py            # Polars-based data ingestion
│   ├── features.py               # Graph construction & feature engineering
│   ├── metrics.py                # Custom metrics (Zone Collapse, Reaction Time)
│   ├── train.py                  # PyTorch Lightning training loop
│   ├── visualization.py          # Field plots and animations
│   └── models/                   # Model architectures
│       ├── __init__.py
│       ├── gnn.py                # NFLGraphTransformer (main model)
│       └── transformer.py        # Legacy transformer implementation
│
├── docs/                         # Documentation
│   ├── architecture.md
│   ├── data_pipeline.md
│   ├── data_dictionary.md
│   ├── installation.md           # This file
│   ├── usage.md
│   ├── configuration.md
│   ├── testing.md
│   ├── api_reference.md
│   └── performance.md
│
├── tests/                        # Verification scripts
│   ├── verify_phase11.py         # Strategic embeddings test
│   ├── verify_phase10.py         # Multi-task learning test
│   ├── verify_phase8.py          # Context fusion test
│   ├── verify_rules.py           # Data integrity test
│   ├── test_gnn.py               # GNN unit tests
│   ├── test_metrics.py           # Metrics unit tests
│   └── test_viz.py               # Visualization tests
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_baseline_model.ipynb   # Baseline XGBoost model
│   ├── 03_insights.ipynb         # Metric demonstrations
│   └── 04_submission.ipynb       # Final submission notebook
│
├── train/                        # Training data (not in repo)
│   ├── input_2023_w01.csv        # Week 1 tracking data
│   ├── input_2023_w02.csv        # Week 2 tracking data
│   └── ...
│
├── supplementary_data.csv        # Play context data (not in repo)
├── pyproject.toml                # Project dependencies
├── README.md                     # Project overview
└── .gitignore
```

---

## Data Setup

### Download Data

1. **Register for NFL Big Data Bowl 2026** on Kaggle
2. **Download datasets:**
   - Tracking data: `input_2023_w*.csv` files
   - Supplementary data: `supplementary_data.csv`

3. **Place in project directory:**
   ```bash
   # Create train directory
   mkdir -p train
   
   # Move tracking files
   mv input_2023_w*.csv train/
   
   # Move supplementary data to root
   mv supplementary_data.csv .
   ```

### Verify Data

```bash
# Check data files exist
ls train/input_2023_w*.csv
ls supplementary_data.csv

# Quick data inspection
python -c "import polars as pl; df = pl.read_csv('train/input_2023_w01.csv'); print(df.shape); print(df.columns)"
```

---

## GPU Setup (Optional but Recommended)

### NVIDIA GPU Configuration

1. **Install NVIDIA Drivers**

```bash
# Ubuntu
sudo ubuntu-drivers autoinstall
sudo reboot

# Verify
nvidia-smi
```

2. **Install CUDA Toolkit** (if not already installed)

```bash
# Ubuntu 22.04 - CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

3. **Verify CUDA**

```bash
nvcc --version
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

## Troubleshooting

### Common Issues

#### Issue: `ModuleNotFoundError: No module named 'torch_geometric'`

**Solution:**
```bash
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
```

#### Issue: `ImportError: cannot import name 'DataLoader' from 'torch_geometric.loader'`

**Cause:** Outdated PyTorch Geometric version

**Solution:**
```bash
pip install --upgrade torch-geometric
```

#### Issue: Python version error

**Cause:** Python < 3.11

**Solution:**
```bash
# Install Python 3.11
sudo apt install python3.11 python3.11-venv

# Recreate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

#### Issue: CUDA out of memory

**Cause:** Batch size too large for GPU

**Solution:**
```python
# In src/train.py, reduce batch_size
# Change from 32 to 16 or 8
batch_size = 16
```

#### Issue: Polars installation fails

**Solution:**
```bash
# Install build dependencies
sudo apt install build-essential

# Reinstall polars
pip install --upgrade polars
```

#### Issue: `FileNotFoundError: train/input_2023_w01.csv`

**Cause:** Data not downloaded or in wrong location

**Solution:**
```bash
# Ensure data is in correct location
ls train/input_2023_w01.csv
ls supplementary_data.csv

# If missing, download from Kaggle
```

---

## Development Setup

### Install Development Dependencies

```bash
# Install with dev extras (if defined)
pip install -e ".[dev]"

# Or install manually
pip install pytest black flake8 mypy
```

### Pre-commit Hooks (Optional)

```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### IDE Configuration

**VS Code:**

Create `.vscode/settings.json`:
```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black"
}
```

**PyCharm:**

1. File → Settings → Project → Python Interpreter
2. Add Interpreter → Existing Environment
3. Select `.venv/bin/python`

---

## Quick Verification

Run this complete verification script:

```bash
#!/bin/bash

echo "=== NFL Analytics Engine Installation Verification ==="

# Check Python version
echo -n "Python version: "
python --version

# Check virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Virtual environment active: $VIRTUAL_ENV"
else
    echo "✗ No virtual environment detected"
fi

# Check imports
python -c "
import sys
try:
    import torch
    import torch_geometric
    import polars as pl
    import pytorch_lightning as pl_lightning
    import numpy as np
    import pandas as pd
    print('✓ All core dependencies imported successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
"

# Check GPU
python -c "
import torch
if torch.cuda.is_available():
    print(f'✓ GPU available: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠ No GPU detected (CPU-only mode)')
"

# Check data files
if [ -f "supplementary_data.csv" ]; then
    echo "✓ supplementary_data.csv found"
else
    echo "✗ supplementary_data.csv not found"
fi

if ls train/input_2023_w*.csv 1> /dev/null 2>&1; then
    echo "✓ Training data found in train/"
else
    echo "✗ No training data in train/"
fi

echo ""
echo "=== Installation verification complete ==="
```

Save as `verify_install.sh`, make executable, and run:

```bash
chmod +x verify_install.sh
./verify_install.sh
```

---

## Next Steps

After successful installation:

1. **Read Documentation:**
   - [Usage Guide](usage.md) - Learn how to train and use the model
   - [Architecture](architecture.md) - Understand the model design
   - [Configuration](configuration.md) - Customize hyperparameters

2. **Run Sanity Check:**
   ```bash
   python -m src.train --mode train --sanity
   ```

3. **Explore Notebooks:**
   ```bash
   jupyter notebook notebooks/01_eda.ipynb
   ```

4. **Start Training:**
   ```bash
   python -m src.train --mode train
   ```

---

## Additional Resources

- **PyTorch Documentation:** https://pytorch.org/docs/
- **PyTorch Geometric:** https://pytorch-geometric.readthedocs.io/
- **Polars Guide:** https://pola-rs.github.io/polars-book/
- **NFL Big Data Bowl:** https://www.kaggle.com/c/nfl-big-data-bowl-2026

---

## Getting Help

If you encounter issues:

1. Check this troubleshooting section
2. Review [GitHub Issues](https://github.com/your-org/nfl-analytics-engine/issues)
3. Open a new issue with:
   - Error message
   - Python version
   - OS and GPU info
   - Steps to reproduce
