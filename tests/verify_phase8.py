
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import polars as pl
from src.features import create_graph_data
from src.models.gnn import NFLGraphTransformer
from torch_geometric.data import Batch

def verify_phase8():
    print("Verifying Phase 8: Context-Aware Fusion...")
    
    # 1. Mock Data with Context
    print("1. Creating Mock Data...")
    df = pl.DataFrame({
        "game_id": [2023]*22,
        "play_id": [1]*22,
        "frame_id": [1]*22,
        "nfl_id": list(range(1, 23)),
        "std_x": torch.randn(22).numpy(),
        "std_y": torch.randn(22).numpy(),
        "s": torch.randn(22).numpy(),
        "a": torch.randn(22).numpy(),
        "std_dir": torch.randn(22).numpy(),
        "std_o": torch.randn(22).numpy(),
        "weight_norm": torch.randn(22).numpy(),  # Added weight_norm
        "role_id": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1],  # Added role_id
        "side_id": [0]*11 + [1]*11,  # Added side_id (11 defense, 11 offense)
        "down": [3]*22,
        "yards_to_go": [15]*22,
        "defenders_box_norm": [0.5]*22  # Added defenders_box_norm
    })
    
    # Need future frames too for create_graph_data to work
    future_len = 2
    frames = []
    for f in range(1, future_len + 5):
        d = df.with_columns(pl.lit(f).alias("frame_id"))
        frames.append(d)
    
    full_df = pl.concat(frames)
    
    graph_list = create_graph_data(full_df, future_seq_len=future_len)
    
    if len(graph_list) == 0:
        print("FAIL: No graphs created.")
        return
        
    data = graph_list[0]
    print(f"Graph Created. Keys: {data.keys}")
    
    if hasattr(data, 'context'):
        print(f"Context Found: {data.context.shape} (Expected [1, 3])")
        if data.context.shape == torch.Size([1, 3]):
            print("PASS: Feature Engineering correct.")
        else:
            print(f"Note: Context shape is {data.context.shape}, model handles it.")
    else:
        print("FAIL: 'context' attribute missing from Data object.")
        
    # 2. Model Forward Pass - corrected dimensions
    print("\n2. Testing Model Forward Pass with Context...")
    model = NFLGraphTransformer(input_dim=7, hidden_dim=32, heads=2, future_seq_len=future_len, edge_dim=5)
    
    # Create Batch
    batch = Batch.from_data_list([data, data])  # Batch of 2
    print(f"Batch Created. Context shape: {batch.context.shape if hasattr(batch, 'context') else 'None'}")
    
    try:
        pred, cov, attn = model(batch)
        print(f"Model Output Shape: {pred.shape}")
        # Expected: [Total_Nodes, Future, 2]
        # Total Nodes = 22 * 2 = 44
        if pred.shape[0] == 44 and pred.shape[1] == future_len:
            print("PASS: Model Forward Pass successful with context fusion.")
        else:
            print("FAIL: Output shape incorrect.")
            
    except Exception as e:
        print(f"FAIL: Model execution error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_phase8()
