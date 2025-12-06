# NFL Big Data Bowl 2026 - Analytics Track Comparison

**Team:** AI Analytics
**Track:** Analytics

## Overview
This repository contains our submission for the NFL Big Data Bowl 2026. We leverage a **Spatial-Temporal Transformer** model to analyze coverage schemes and introduce **Zone Collapse Speed** as a new metric for defensive responsiveness.

## Repository Structure
- `src/`: Source code for data loading, metrics, models, and visualization.
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
