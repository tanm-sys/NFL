
import sys
import os
import torch
from src.models.gnn import NFLGraphTransformer
from src.features import create_graph_data
from src.data_loader import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
import polars as pl

def debug_train():
    print("Debug: Loading Data...")
    loader = DataLoader(".")
    df = loader.load_week_data(1)
    # Filter 1 game
    game_ids = df["game_id"].unique().head(1)
    df = df.filter(pl.col("game_id").is_in(game_ids))
    df = loader.standardize_tracking_directions(df)
    
    print("Debug: Creating Graphs...")
    graphs = create_graph_data(df, radius=20.0, future_seq_len=10)
    print(f"Generated {len(graphs)} graphs.")
    
    graphs = graphs[:10]
    train_loader = PyGDataLoader(graphs, batch_size=2, shuffle=True)
    
    model = NFLGraphTransformer(input_dim=7, hidden_dim=64, future_seq_len=10)
    
    print("Debug: Running Manual Sanity Check (10 Batches)...")
    
    criterion_traj = torch.nn.MSELoss()
    criterion_cov = torch.nn.BCEWithLogitsLoss()
    
    total_traj_loss = 0
    total_cov_acc = 0
    batches = 0
    
    model.eval() # Eval mode for sanity check of "untrained" or "init" performance (or should we train?)
    # User asked "What would be the prediction accuracy".
    # For untrained model, it should be random.
    # To see potential, we'd need to train. 
    # But training for 2-3 mins is feasible here.
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    
    print("Training 10 batches...")
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        pred, cov_pred, attn = model(batch)
        
        # Trajectory Loss
        y = batch.y
        loss_traj = criterion_traj(pred, y)
        
        # Coverage Loss
        loss_cov = 0.0
        if hasattr(batch, 'y_coverage') and batch.y_coverage is not None:
             target_cov = batch.y_coverage.view(-1, 1)
             loss_cov = criterion_cov(cov_pred, target_cov)
             
        loss = loss_traj + 0.5 * loss_cov
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_traj_loss += loss_traj.item()
        
        # Cov Acc
        if hasattr(batch, 'y_coverage'):
            probs = torch.sigmoid(cov_pred)
            preds = (probs > 0.5).float()
            acc = (preds == target_cov).float().mean().item()
            total_cov_acc += acc
        
        batches += 1
        if batches >= 10: break
        
    print(f"\n=== Sanity Check Results (10 Batches) ===")
    print(f"Avg Trajectory MSE: {total_traj_loss / batches:.4f}")
    print(f"Avg Coverage Accuracy: {total_cov_acc / batches:.2%}")
    print("=========================================")

if __name__ == "__main__":
    debug_train()
