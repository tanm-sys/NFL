# NFL Analytics Engine: Context-Aware Trajectory Prediction

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.5+-3C2179.svg)](https://pytorch-geometric.readthedocs.io/)
[![Parameters](https://img.shields.io/badge/Parameters-12M-green.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🏈 Overview

A **competition-grade** deep learning system for NFL player trajectory prediction, designed for the **NFL Big Data Bowl 2026**. This engine combines **Graph Neural Networks (GNN)**, **Transformer decoders**, **SOTA contrastive losses**, and **probabilistic multi-modal prediction** to achieve state-of-the-art accuracy.

> **🏆 Model Size**: 12M parameters - optimized for RTX 3050 4GB VRAM

---

## ⭐ Key Features

```mermaid
mindmap
  root((NFL Analytics Engine))
    Architecture
      8-Layer GATv2
      12 Attention Heads
      384 Hidden Dim
      16 GMM Modes
    SOTA Losses
      Social-NCE
      Winner-Takes-All
      Diversity Loss
      Endpoint Focal
    Optimizations
      Mixed Precision FP16
      Graph Pre-Caching
      Gradient Accumulation
      SWA Ensemble
    Outputs
      Multi-Modal Trajectories
      Uncertainty Estimates
      Coverage Classification
      Competition Metrics
```

---

## 🏗️ System Architecture

### High-Level Pipeline

```mermaid
flowchart TB
    subgraph Input["📥 Data Input"]
        A[("Tracking CSV<br/>18 Weeks")] --> B["Polars DataLoader"]
        B --> C["Graph Cache<br/>185K Graphs"]
    end
    
    subgraph Encoder["🧠 Graph Encoder"]
        C --> D["Node Embedding<br/>9D → 384D"]
        D --> E["Strategic Embeddings<br/>Role + Formation"]
        E --> F["8× GATv2 Blocks<br/>+ DropPath"]
        F --> G["Social Pooling<br/>Pairwise Interactions"]
    end
    
    subgraph Scene["🌍 Scene Understanding"]
        G --> H["Scene Flow Encoder<br/>Set Transformer"]
        H --> I["Attentional Aggregation<br/>Graph Pooling"]
    end
    
    subgraph Decoder["🎯 Trajectory Decoder"]
        I --> J["Temporal History<br/>LSTM + Attention"]
        J --> K["GMM Decoder<br/>16 Modes"]
        K --> L["μ, σ, π<br/>Per Mode"]
    end
    
    subgraph Output["📤 Outputs"]
        L --> M["Trajectory Predictions<br/>[N, 10, 2]"]
        L --> N["Uncertainty<br/>[N, 10, 16, 2]"]
        I --> O["Coverage Class<br/>[B, 1]"]
    end
    
    style Input fill:#e1f5fe
    style Encoder fill:#fff3e0
    style Scene fill:#f3e5f5
    style Decoder fill:#e8f5e9
    style Output fill:#fce4ec
```

### Detailed Model Architecture

```mermaid
flowchart LR
    subgraph GATv2["GATv2 Block (×8)"]
        direction TB
        GA1["Input: [N, 384]"] --> GA2["Multi-Head Attention<br/>12 Heads"]
        GA2 --> GA3["DropPath<br/>p=0.18"]
        GA3 --> GA4["Residual Add"]
        GA4 --> GA5["LayerNorm"]
        GA5 --> GA6["Output: [N, 384]"]
    end
    
    subgraph GMM["GMM Decoder"]
        direction TB
        GM1["Context: [N, 384]"] --> GM2["Transformer<br/>6 Layers"]
        GM2 --> GM3["Mode Heads<br/>×16"]
        GM3 --> GM4["μ: [N, 10, 16, 2]"]
        GM3 --> GM5["σ: [N, 10, 16, 2]"]
        GM3 --> GM6["π: [N, 16]"]
    end
```

---

## 📊 Model Specifications

### Parameter Count

| Component | Parameters | % of Total |
|-----------|------------|------------|
| Graph Encoder (8 layers) | 8.2M | 68% |
| GMM Decoder (16 modes) | 2.8M | 23% |
| Scene Encoder | 0.5M | 4% |
| Embeddings | 0.4M | 3% |
| SOTA Losses | 0.1M | 2% |
| **Total** | **12.0M** | 100% |

### Architecture Details

```yaml
# Maximum Parameters Config
hidden_dim: 384           # 1.5× default
num_gnn_layers: 8         # Deep network
heads: 12                 # Multi-head attention
num_modes: 16             # Maximum multi-modal
dropout: 0.08
droppath_rate: 0.18
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/tanm-sys/nfl-analytics-engine.git
cd nfl-analytics-engine
pip install -e .
```

### 2. Pre-Cache Graphs (Required)

```bash
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

### 3. Train Maximum Parameters Model

```bash
python finetune_best_model.py --config configs/max_accuracy_rtx3050.yaml
```

---

## 🎯 Training Pipeline

```mermaid
flowchart LR
    subgraph Prep["Preparation"]
        P1["Download Data"] --> P2["Pre-Cache Graphs<br/>~30 min"]
    end
    
    subgraph Train["Training"]
        P2 --> T1["Load Config"]
        T1 --> T2["Initialize Model<br/>12M params"]
        T2 --> T3["Train Loop<br/>200 epochs"]
        T3 --> T4["Validation<br/>Every epoch"]
        T4 --> T5["Checkpoint<br/>Best minADE"]
    end
    
    subgraph Post["Post-Training"]
        T5 --> PT1["SWA Averaging"]
        PT1 --> PT2["Model Ensemble<br/>Top-3"]
        PT2 --> PT3["Export for<br/>Competition"]
    end
    
    style Prep fill:#e3f2fd
    style Train fill:#fff8e1
    style Post fill:#e8f5e9
```

---

## 📈 Loss Functions

```mermaid
flowchart TB
    subgraph Primary["Primary Losses"]
        L1["Trajectory MSE<br/>weight: 1.0"]
        L2["Velocity Loss<br/>weight: 0.7"]
        L3["Acceleration<br/>weight: 0.5"]
    end
    
    subgraph SOTA["SOTA Contrastive"]
        L4["Social-NCE<br/>weight: 0.25"]
        L5["WTA (k=4)<br/>weight: 1.0"]
        L6["Diversity<br/>weight: 0.08"]
        L7["Endpoint Focal<br/>weight: 0.40"]
    end
    
    subgraph Aux["Auxiliary"]
        L8["Collision<br/>weight: 0.25"]
        L9["Coverage BCE<br/>weight: 0.6"]
    end
    
    Primary --> Total["Total Loss"]
    SOTA --> Total
    Aux --> Total
```

---

## 🎛️ Configuration Options

| Config | Parameters | Batch | Epochs | Use Case |
|--------|------------|-------|--------|----------|
| `max_accuracy_rtx3050.yaml` | **12.0M** | 16 | 200 | Competition |
| `high_accuracy.yaml` | 5.4M | 32 | 100 | High quality |
| `production.yaml` | 3.2M | 48 | 80 | Fast training |
| `sanity.yaml` | 1.0M | 64 | 2 | Quick test |

---

## 📂 Project Structure

```
nfl-analytics-engine/
├── src/
│   ├── models/
│   │   └── gnn.py              # NFLGraphTransformer (12M params)
│   ├── losses/
│   │   └── contrastive_losses.py  # SOTA losses
│   ├── data_loader.py          # Polars + PyG caching
│   ├── train.py                # Lightning module
│   ├── competition_metrics.py  # Novel metrics
│   └── competition_output.py   # Submission generator
├── configs/
│   ├── max_accuracy_rtx3050.yaml  # Maximum parameters
│   └── *.yaml                  # Other configs
├── docs/                       # Documentation
├── cache/                      # Graph cache (1.8GB)
└── checkpoints_finetuned/      # Model checkpoints
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | Model design & tensor flows |
| [Configuration](docs/configuration.md) | All hyperparameters |
| [Data Pipeline](docs/data_pipeline.md) | ETL & caching |
| [API Reference](docs/api_reference.md) | Module documentation |
| [Performance](docs/performance.md) | Benchmarks |
| [Installation](docs/installation.md) | Setup guide |
| [Usage](docs/usage.md) | Training & inference |
| [Testing](docs/testing.md) | Verification |

---

## 🏆 Competition Metric: RMSE

> **Evaluation**: Submissions are evaluated using **Root Mean Squared Error (RMSE)** between predicted and observed target positions.

```
RMSE = √(mean((pred - target)²))
```

### Target Performance (12M Model)

| Metric | Target | World-Class | Description |
|--------|--------|-------------|-------------|
| **RMSE** | < 0.35 | < 0.28 | Competition metric |
| **minRMSE** | < 0.25 | < 0.20 | Best mode RMSE |
| ADE | < 0.22 | < 0.18 | Secondary metric |
| FDE | < 0.35 | < 0.28 | Secondary metric |
| Miss Rate | < 0.5% | < 0.2% | Secondary metric |

---

## 📖 Citation

```bibtex
@misc{nfl-analytics-engine,
  title={NFL Analytics Engine: Context-Aware Trajectory Prediction},
  author={Tanmay},
  year={2025},
  howpublished={\url{https://github.com/tanm-sys/nfl-analytics-engine}}
}
```

---

**Built with ❤️ for NFL Big Data Bowl 2026**

*Powered by PyTorch • PyTorch Geometric • PyTorch Lightning*
