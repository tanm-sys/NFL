# 🏈 NFL Analytics Engine: Production Training Guide

**Complete guide for production-ready model training on Google Colab**

---

## 📋 Overview

This guide covers the **production training pipeline** with:

- ✅ **Experiment Tracking**: MLflow + W&B + TensorBoard
- ✅ **Data Validation**: Automated quality checks
- ✅ **Robust Training**: Multi-GPU, mixed precision, gradient accumulation
- ✅ **Model Versioning**: Automatic checkpointing and export
- ✅ **Monitoring**: Real-time metrics and alerts
- ✅ **Deployment**: ONNX and TorchScript export

---

## 🚀 Quick Start

### 1. Setup Colab Environment

```python
# Enable GPU
# Runtime → Change runtime type → GPU (T4/A100)

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate to project
%cd /content/drive/MyDrive/NFL_Project
```

### 2. Install Dependencies

```python
# IMPORTANT: Ensure you are in the project directory
%cd /content/drive/MyDrive/NFL_Project

# Option A: Automated Setup (Recommended)
!bash setup_colab.sh

# Option B: Manual Installation (If automation fails)
# Step 1: Install torch-geometric (works without optional extensions in v2.7.0+)
!pip install torch-geometric

# Step 2: Install project requirements
!pip install -r requirements.txt
!pip install -e .

# Step 3 (Optional): Try to install optimized extensions if wheels exist
# Note: May fail for very new PyTorch versions - that's OK, core functionality works
import torch
torch_version = torch.__version__.split('+')[0]
cuda_version = torch.version.cuda.replace('.', '')
!pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-{torch_version}+cu{cuda_version}.html || echo "Optional extensions not available - continuing without them"
```

### 3. Run Production Training

```python
# Sanity check first (5 minutes)
!python train_production.py --mode train --config configs/sanity.yaml

# Full production training (1-2 hours)
!python train_production.py --mode train --config configs/production.yaml
```

---

## 📁 Project Structure

```
NFL_Project/
├── train_production.py          # Main production training script
├── configs/
│   ├── production.yaml          # Full production config
│   ├── sanity.yaml              # Quick sanity check
│   └── probabilistic.yaml       # GMM decoder config
├── src/
│   ├── train.py                 # Training module
│   ├── models/gnn.py            # Model architecture
│   ├── data_loader.py           # Data loading
│   └── features.py              # Feature engineering
├── train/                       # Training data (week CSVs)
├── checkpoints/                 # Model checkpoints
├── logs/                        # Training logs
├── outputs/                     # Exported models
└── mlruns/                      # MLflow experiments
```

---

## ⚙️ Configuration Files

### Production Config ([configs/production.yaml](file:///home/tanmay/Desktop/NFL/configs/production.yaml))

**Use for:** Final model training with all data

**Features:**
- All 18 weeks of data
- 100 epochs with early stopping
- Mixed precision (FP16)
- Full experiment tracking
- Model export enabled

**Expected Time:** 1-2 hours on T4 GPU

### Sanity Config ([configs/sanity.yaml](file:///home/tanmay/Desktop/NFL/configs/sanity.yaml))

**Use for:** Quick validation before full training

**Features:**
- Week 1 only
- 5 epochs
- Smaller model (hidden_dim=32)
- Minimal logging

**Expected Time:** 5-10 minutes

### Probabilistic Config ([configs/probabilistic.yaml](file:///home/tanmay/Desktop/NFL/configs/probabilistic.yaml))

**Use for:** Uncertainty-aware predictions

**Features:**
- GMM decoder with 6 modes
- NLL loss
- Longer training (120 epochs)
- Better accuracy potential

**Expected Time:** 2-3 hours

---

## 🎯 Training Modes

### Mode 1: Standard Training

```bash
python train_production.py \
  --mode train \
  --data-dir . \
  --weeks 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 \
  --batch-size 32 \
  --max-epochs 100 \
  --precision 16-mixed \
  --experiment-name my_production_run
```

### Mode 2: Resume from Checkpoint

```bash
python train_production.py \
  --mode train \
  --resume-from checkpoints/last.ckpt \
  --max-epochs 150
```

### Mode 3: Probabilistic GMM

```bash
python train_production.py \
  --mode train \
  --probabilistic \
  --batch-size 24 \
  --max-epochs 120
```

### Mode 4: Quick Sanity Check

```bash
python train_production.py \
  --mode train \
  --weeks 1 \
  --batch-size 16 \
  --max-epochs 5 \
  --no-validation
```

---

## 📊 Experiment Tracking

### MLflow

**Access UI:**
```bash
# In Colab
!mlflow ui --port 5000

# Then use ngrok for public URL
!pip install pyngrok
from pyngrok import ngrok
public_url = ngrok.connect(5000)
print(f"MLflow UI: {public_url}")
```

**Features:**
- Automatic metric logging
- Hyperparameter tracking
- Model versioning
- Artifact storage

### Weights & Biases

**Setup:**
```python
import wandb
wandb.login()  # Enter your API key
```

**Features:**
- Real-time metric plots
- System monitoring (GPU, CPU, memory)
- Model checkpoints
- Collaborative dashboards

**Access:** https://wandb.ai/your-username/nfl-analytics-production

### TensorBoard

**Access:**
```python
%load_ext tensorboard
%tensorboard --logdir logs/
```

**Features:**
- Training curves
- Model graph visualization
- Histogram tracking
- Profiling data

---

## 📈 Monitoring Metrics

### Trajectory Metrics

| Metric | Description | Target | Excellent |
|--------|-------------|--------|-----------|
| **val_ade** | Average Displacement Error (yards) | <3.0 | <2.5 |
| **val_fde** | Final Displacement Error (yards) | <5.0 | <4.0 |
| **val_miss_rate_2yd** | % predictions >2 yards off | <0.40 | <0.30 |

### Coverage Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **val_cov_acc** | Coverage classification accuracy | >0.75 |
| **val_cov_loss** | Coverage BCE loss | <0.5 |

### Training Metrics

| Metric | Description | Good Range |
|--------|-------------|------------|
| **train_loss** | Overall training loss | Decreasing |
| **train_vel_loss** | Velocity consistency | <1.0 |
| **train_acc_loss** | Acceleration loss | <0.5 |
| **learning_rate** | Current LR | 1e-6 to 1e-3 |

---

## 🔍 Data Validation

The production script automatically validates:

### Structural Checks
- ✅ Required columns present
- ✅ No null values in critical fields
- ✅ Correct data types

### Range Checks
- ✅ X coordinates: 0-120 yards
- ✅ Y coordinates: 0-53.3 yards
- ✅ Speed: 0-25 yards/second
- ✅ Acceleration: -10 to 10 yards/s²

### Quality Checks
- ✅ No duplicate frames
- ✅ Consistent play sequences
- ✅ Valid player IDs

**Disable validation** (for speed):
```bash
python train_production.py --no-validation
```

---

## 🚨 Troubleshooting

### Issue 1: Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size:**
```yaml
# In config file
training:
  batch_size: 16  # or 8
```

2. **Use gradient accumulation:**
```yaml
training:
  batch_size: 16
  accumulate_grad_batches: 2  # Effective batch size = 32
```

3. **Reduce model size:**
```yaml
model:
  hidden_dim: 32
  num_gnn_layers: 2
```

4. **Use CPU offloading:**
```yaml
hardware:
  precision: "16-mixed"
  strategy: "deepspeed_stage_2"
```

### Issue 2: Slow Training

**Symptoms:** <10 iterations/second

**Solutions:**

1. **Verify GPU usage:**
```python
!nvidia-smi
# Should show 80-100% GPU utilization
```

2. **Reduce num_workers:**
```yaml
training:
  num_workers: 2  # or 0
```

3. **Enable benchmarking:**
```yaml
reproducibility:
  deterministic: false
  benchmark: true
```

4. **Use mixed precision:**
```yaml
hardware:
  precision: "16-mixed"  # 2x speedup
```

### Issue 3: NaN Loss

**Symptoms:**
```
train_loss: nan
```

**Solutions:**

1. **Reduce learning rate:**
```yaml
training:
  learning_rate: 0.0005  # or lower
```

2. **Enable gradient clipping:**
```yaml
regularization:
  gradient_clip_val: 0.5
```

3. **Check data quality:**
```bash
python train_production.py --validate-data
```

4. **Use Huber loss:**
```yaml
regularization:
  use_huber_loss: true
  huber_delta: 1.0
```

### Issue 4: Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'rich'
```

**Solution:**
```bash
pip install -r requirements.txt --force-reinstall
pip install -e . --force-reinstall
```

### Issue 5: Colab Disconnects

**Symptoms:** Session timeout after 90 minutes

**Solutions:**

1. **Upgrade to Colab Pro** (24-hour sessions)

2. **Auto-save checkpoints:**
```yaml
callbacks:
  model_checkpoint:
    save_last: true  # Saves every epoch
```

3. **Resume training:**
```bash
python train_production.py --resume-from checkpoints/last.ckpt
```

4. **Use Google Drive for persistence:**
```python
# All outputs automatically saved to Drive
%cd /content/drive/MyDrive/NFL_Project
```

---

## 📦 Model Export

### Automatic Export

After training completes, models are automatically exported to:

```
outputs/exported_models/
├── {experiment_name}_torchscript.pt
└── {experiment_name}_onnx.onnx
```

### Manual Export

```python
import torch
from src.train import NFLGraphPredictor

# Load checkpoint
model = NFLGraphPredictor.load_from_checkpoint("checkpoints/best.ckpt")
model.eval()

# Export to TorchScript
scripted = torch.jit.script(model.model)
torch.jit.save(scripted, "model_torchscript.pt")

# Export to ONNX
dummy_input = ...  # Create dummy graph input
torch.onnx.export(
    model.model,
    dummy_input,
    "model.onnx",
    opset_version=14,
    input_names=['x', 'edge_index', 'edge_attr'],
    output_names=['predictions', 'coverage']
)
```

---

## 🎯 Production Best Practices

### 1. Always Run Sanity Check First

```bash
# Quick validation (5 min)
python train_production.py --config configs/sanity.yaml

# If successful, run full training
python train_production.py --config configs/production.yaml
```

### 2. Use Experiment Tracking

```yaml
logging:
  use_mlflow: true
  use_wandb: true
  use_tensorboard: true
```

**Benefits:**
- Compare multiple runs
- Track hyperparameters
- Visualize metrics
- Share results with team

### 3. Enable Data Validation

```yaml
data:
  validate_data: true
```

**Catches:**
- Missing data
- Corrupted files
- Out-of-range values
- Duplicate records

### 4. Save Configuration

```yaml
production:
  save_config: true
```

**Ensures:**
- Reproducibility
- Experiment tracking
- Easy debugging

### 5. Monitor Resource Usage

```python
# In Colab
!nvidia-smi -l 1  # Update every second

# Check memory
import psutil
print(f"RAM: {psutil.virtual_memory().percent}%")
```

### 6. Use Version Control

```bash
# Save experiment metadata
git add configs/production.yaml
git commit -m "Production run with hidden_dim=64, lr=1e-3"
git tag v1.0-production
```

### 7. Test on Subset First

```yaml
data:
  weeks: [1, 2, 3]  # Test on 3 weeks first
```

### 8. Enable Checkpointing

```yaml
callbacks:
  model_checkpoint:
    save_top_k: 3  # Keep best 3 models
    save_last: true  # Always save latest
```

---

## 🔄 Hyperparameter Tuning

### Manual Grid Search

```bash
# Try different learning rates
for lr in 0.0005 0.001 0.002; do
  python train_production.py \
    --learning-rate $lr \
    --experiment-name "lr_${lr}"
done
```

### Optuna Integration (Coming Soon)

```python
import optuna

def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    
    # Run training
    # Return validation ADE
    return val_ade

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)
```

---

## 📊 Expected Results

### After Full Training (18 weeks, 100 epochs)

| Metric | Expected | Best Possible |
|--------|----------|---------------|
| **Training Time** | 1-2 hours (T4) | 30-45 min (A100) |
| **val_ade** | 2.5-3.0 yards | <2.3 yards |
| **val_fde** | 4.0-5.0 yards | <3.8 yards |
| **val_miss_rate** | 0.30-0.40 | <0.25 |
| **val_cov_acc** | 0.78-0.82 | >0.84 |
| **Model Size** | ~810K params | - |
| **Inference Speed** | ~100 graphs/sec | - |

### Probabilistic GMM Results

| Metric | Deterministic | Probabilistic |
|--------|---------------|---------------|
| **val_ade** | 2.5 yards | **2.2 yards** |
| **val_fde** | 4.0 yards | **3.6 yards** |
| **Uncertainty** | No | **Yes** |
| **Training Time** | 1-2 hours | 2-3 hours |

---

## 🚀 Deployment

### 1. Export Final Model

```bash
python train_production.py \
  --config configs/production.yaml \
  --export-onnx \
  --export-torchscript
```

### 2. Create Inference API

```python
# inference_api.py
from fastapi import FastAPI
import torch

app = FastAPI()
model = torch.jit.load("model_torchscript.pt")

@app.post("/predict")
def predict(data: dict):
    # Process input
    # Run inference
    # Return predictions
    return {"predictions": predictions}
```

### 3. Deploy to Cloud

```bash
# Google Cloud Run
gcloud run deploy nfl-predictor \
  --source . \
  --platform managed \
  --region us-central1
```

---

## 📝 Checklist

### Pre-Training
- [ ] GPU enabled in Colab
- [ ] All data files uploaded
- [ ] Dependencies installed
- [ ] Sanity check passed
- [ ] Configuration reviewed

### During Training
- [ ] Metrics decreasing
- [ ] No NaN losses
- [ ] GPU utilization >80%
- [ ] Checkpoints saving
- [ ] Logs being written

### Post-Training
- [ ] Final metrics meet targets
- [ ] Model exported successfully
- [ ] Checkpoints saved to Drive
- [ ] Experiment logged
- [ ] Results documented

---

## 🆘 Support

**Common Issues:**
1. Check [Troubleshooting](#troubleshooting) section
2. Review error messages carefully
3. Verify configuration file
4. Test with sanity config first

**Resources:**
- [PyTorch Lightning Docs](https://lightning.ai/docs/pytorch/stable/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [W&B Guides](https://docs.wandb.ai/)

---

**🏈 Ready for Production Training!**

*Last Updated: December 2025*
