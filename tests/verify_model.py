import torch
import sys
from pathlib import Path
sys.path.append('src')

from models.transformer import NFLTransformer
from train import NFLPredictor

def test_transformer_shapes():
    # Batch=2, Seq=10, Agents=23, Feats=6
    dummy_input = torch.randn(2, 10, 23, 6)
    
    model = NFLTransformer(input_dim=6, hidden_dim=32, num_heads=2)
    output = model(dummy_input)
    
    # Expected Output: [Batch, Future=10, Agents=23, 2]
    expected_shape = (2, 10, 23, 2)
    print(f"Output shape: {output.shape}")
    assert output.shape == expected_shape
    print("Transformer Forward Pass: OK")

def test_predictor_step():
    dummy_x = torch.randn(2, 10, 23, 6)
    dummy_y = torch.randn(2, 10, 23, 2)
    
    model = NFLPredictor(input_dim=6, hidden_dim=32)
    loss = model.training_step((dummy_x, dummy_y), 0)
    print(f"Training Loss: {loss.item()}")
    assert not torch.isnan(loss)
    print("Predictor Training Step: OK")

if __name__ == "__main__":
    test_transformer_shapes()
    test_predictor_step()
