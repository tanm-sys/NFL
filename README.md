# NFL Analytics Engine: Context-Aware Trajectory Prediction

## Overview
A state-of-the-art implementation for the NFL Big Data Bowl, using **Graph Neural Networks (GNN)**, **Transformers**, and **Strategic Embeddings** to predict player movements and classify defensive coverage.

## 📚 Documentation
*   [**System Architecture**](docs/architecture.md): Deep dive into the GNN+Transformer design.
*   [**Data Dictionary**](docs/data_dictionary.md): Explanation of all input features (Physics + Strategy).
*   [**Usage Guide**](docs/usage.md): Instructions for Training, Tuning, and Visualization.

## 🚀 Quick Start
```bash
# Run a quick sanity check
python -m src.train --mode train --sanity
```

## Project Status
*   **Phase 1-6**: Baseline Setup & Physics (Completed).
*   **Phase 7-10**: Context, Multi-Task, Attention (Completed).
*   **Phase 11**: Strategic Embeddings (Completed).
  - `data_loader.py`: Efficient Polars-based tracking data loader.
  - `features.py`: H3 spatial indexing and feature engineering.
  - `metrics.py`: Implementation of novel metrics (Zone Collapse, Reaction Time).
  - `models/`: PyTorch Lightning Transformer implementation.
  - `visualization.py`: Field plotting and animation tools.
- `notebooks/`: Jupyter notebooks for analysis.
  - `01_eda.ipynb`: Exploratory Data Analysis.
  - `02_baseline_model.ipynb`: Baseline XGBoost model.
  - `03_insights.ipynb`: Metric demonstrations.
  - `04_submission.ipynb`: **Final Submission**.
- `tests/`: Verification scripts.

## Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install .
   ```
   *Note: Requires Python 3.11+*

## Usage
Run the submission notebook:
```bash
jupyter notebook notebooks/04_submission.ipynb
```
