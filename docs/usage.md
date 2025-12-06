# Usage Guide

## Installation
Ensure you have Python 3.9+ and the required inputs (`train/` and `supplementary_data.csv`).

```bash
# Install dependencies
pip install -r requirements.txt
```

## Running the Model

### 1. Sanity Check
Run a quick verification on <1% of the data to ensure the pipeline works and get baseline metrics.
```bash
python -m src.train --mode train --sanity
```

### 2. Full Training
Train the model on the full dataset (Week 1 by default in current loader).
```bash
python -m src.train --mode train
```
*   **Checkpoints**: Saved automatically by PyTorch Lightning.
*   **Logs**: Viewable via Weights & Biases (if configured) or TensorBoard.

### 3. Hyperparameter Tuning
Run Optuna to find the best learning rate and hidden dimensions.
```bash
python -m src.train --mode tune
```

## Visualization
You can visualize the model's "Attention" (what players are looking at) using `src.visualization`.

```python
from src.visualization import plot_attention_map
from src.models.gnn import NFLGraphTransformer

# Assuming you have a loaded model and data
batch = next(iter(val_loader))
model.eval()
pred, cov, attn_weights = model(batch, return_attention_weights=True)

# Plot
plot_attention_map(frame_df, attn_weights, target_nfl_id=None)
```

## Project Structure
*   `src/data_loader.py`: Ingestion and Cleaning.
*   `src/features.py`: Graph construction and Feature Engineering.
*   `src/models/gnn.py`: PyTorch Model Architecture.
*   `src/train.py`: Training Loop (Lightning).
*   `src/visualization.py`: Plotting tools.
