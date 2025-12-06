
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from src.models.gnn import NFLGraphTransformer
from src.train import NFLGraphPredictor
from torch_geometric.data import Data, Batch

def verify_phase9():
    print("Verifying Phase 9: Multi-Task Learning (Man vs Zone)...")
    
    # 1. Mock Data with Coverage Label
    # Create 2 Graphs
    # Graph 1: Man (0)
    # Graph 2: Zone (1)
    
    x = torch.randn(11, 6) # 11 players
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.randn(2, 2)
    y_traj = torch.randn(11, 2, 2) # Future len 2
    
    data1 = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_traj)
    data1.context = torch.randn(1, 2) # Down/Dist
    data1.y_coverage = torch.tensor([0.0]) # Man
    
    data2 = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_traj)
    data2.context = torch.randn(1, 2)
    data2.y_coverage = torch.tensor([1.0]) # Zone
    
    batch = Batch.from_data_list([data1, data2])
    print(f"Batch Created. y_coverage: {batch.y_coverage}")
    
    # 2. Model Forward
    model = NFLGraphPredictor(input_dim=6, hidden_dim=32, future_seq_len=2)
    
    try:
        traj_pred, cov_pred = model(batch)
        print(f"Predictions: Traj {traj_pred.shape}, Cov {cov_pred.shape}")
        
        if cov_pred.shape == (2, 1):
             print("PASS: Coverage Prediction Shape Correct.")
        else:
             print(f"FAIL: Coverage Shape {cov_pred.shape}")
             
        # 3. Loss Calculation
        loss = model.training_step(batch, 0)
        print(f"Training Step Loss: {loss:.4f}")
        
        if loss > 0:
            print("PASS: Loss Calculation functional.")
            
    except Exception as e:
        print(f"FAIL: Execution Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_phase9()
