
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import polars as pl
from src.models.gnn import NFLGraphTransformer
from src.visualization import plot_attention_map
from torch_geometric.data import Data, Batch
import matplotlib.pyplot as plt

def verify_phase10():
    print("Verifying Phase 10: Attention Explainability...")
    
    # 1. Mock Data (Graph)
    num_nodes = 5
    x = torch.randn(num_nodes, 6)
    # Fully connected edge index (excluding self loops for GATv2 usually adds them or handles them)
    # Let's make explicit edges
    source = [0, 1, 2, 3, 4]
    target = [0, 0, 0, 0, 0] # Everyone looks at Node 0
    edge_index = torch.tensor([source, target], dtype=torch.long)
    edge_attr = torch.randn(5, 2)
    y_traj = torch.randn(num_nodes, 2, 2)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_traj)
    data.context = torch.randn(1, 2)
    
    batch = Batch.from_data_list([data])
    
    # 2. Mock Model & Forward
    model = NFLGraphTransformer(input_dim=6, hidden_dim=32, heads=2, future_seq_len=2)
    
    # Run with flag
    traj, cov, attn_weights = model(batch, return_attention_weights=True)
    
    print(f"Model returned attention weights: {attn_weights is not None}")
    if attn_weights is not None:
        edge_idx, alpha = attn_weights
        print(f"Edge Index Shape: {edge_idx.shape}, Alpha Shape: {alpha.shape}")
        # Alpha should be [Num_Edges, Heads]
        
    # 3. Mock DataFrame for Viz
    # Needs: frame_id, nfl_id, std_x, std_y, club, player_name, player_position
    df = pl.DataFrame({
        "frame_id": [1]*num_nodes,
        "nfl_id": [10, 11, 12, 13, 14], # Sorted IDs match Node Index 0.4
        "std_x": [50, 52, 48, 55, 45],
        "std_y": [26, 28, 24, 30, 20],
        "club": ["KC", "KC", "KC", "DET", "DET"],
        "player_name": ["Mahomes", "Kelce", "Rice", "Hutchinson", "Campbell"],
        "player_position": ["QB", "TE", "WR", "DE", "LB"]
    })
    
    # 4. Test Plot
    try:
        ax = plot_attention_map(df, attn_weights, target_nfl_id=10) # 10 is Mahomes (Node 0)
        print("PASS: Plot generated successfully.")
        plt.close()
    except Exception as e:
        print(f"FAIL: Plotting error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_phase10()
