"""
Model Architecture Configuration
Centralizes all model hyperparameters for easy experimentation.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for NFL trajectory prediction models."""
    
    # Input/Output dimensions
    input_dim: int = 7
    hidden_dim: int = 64
    future_seq_len: int = 10
    
    # GNN Architecture
    num_gnn_layers: int = 4
    num_heads: int = 4
    edge_dim: int = 5
    dropout: float = 0.1
    
    # Probabilistic Mode (GMM)
    probabilistic: bool = False
    num_modes: int = 6
    
    # History Encoding (P1)
    history_len: int = 5
    history_input_dim: int = 4  # vel_x, vel_y, acc_x, acc_y
    history_lstm_layers: int = 2
    
    # Embeddings
    num_roles: int = 5
    num_sides: int = 3
    num_formations: int = 8
    num_alignments: int = 10
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.num_gnn_layers > 0, "num_gnn_layers must be positive"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
        
    @property
    def total_params_estimate(self) -> int:
        """Rough estimate of model parameters."""
        # Simplified estimate
        gnn_params = self.num_gnn_layers * self.hidden_dim * self.hidden_dim * self.num_heads
        decoder_params = self.hidden_dim * self.future_seq_len * 2
        embedding_params = (self.num_roles + self.num_sides + self.num_formations + self.num_alignments) * self.hidden_dim
        return gnn_params + decoder_params + embedding_params
