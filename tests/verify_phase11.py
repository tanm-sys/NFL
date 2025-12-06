
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.models.gnn import NFLGraphTransformer
from torch_geometric.data import Data, Batch

def verify_phase11():
    print("Verifying Phase 11: Strategic Embeddings...")
    
    # 1. Mock Data (Strategic)
    # 5 Nodes, 7 Input Features (last is Weight)
    # Role: [0, 1, 2, 3, 4]
    # Formation: [0]
    # Alignment: [1]
    # Context: [Down, Dist, Box] = [3]
    
    x = torch.randn(5, 7) 
    role = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.randn(2, 2)
    
    # Context
    context = torch.tensor([[3.0, 10.0, 7.0]], dtype=torch.float) # Dim 3
    formation = torch.tensor([0], dtype=torch.long)
    alignment = torch.tensor([1], dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.context = context
    data.role = role
    data.formation = formation
    data.alignment = alignment
    data.batch = torch.zeros(5, dtype=torch.long) # Single graph batch
    
    # 2. Model
    model = NFLGraphTransformer(input_dim=7, hidden_dim=32, future_seq_len=2)
    # Note: GNN init creates Context Encoder for dim=3 by default?
    # No, GraphPlayerEncoder defaults context_dim=3 now.
    
    # 3. Forward
    try:
        # Check defaults
        print(f"Encoder Context Dim: {model.encoder.context_encoder.in_features}")
        if model.encoder.context_encoder.in_features != 3:
             print("FAIL: Context Dim default not updated to 3?")
        
        # Run
        pred, cov, attn = model(data, return_attention_weights=True)
        print(f"Prediction Shape: {pred.shape}")
        
        if pred.shape[0] == 5:
            print("PASS: Forward pass with Strategic Embeddings successful.")
        else:
            print("FAIL: Output shape mismatch.")
            
    except Exception as e:
        print(f"FAIL: Execution Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_phase11()
