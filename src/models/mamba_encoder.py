"""
Mamba Temporal Encoder for NFL Trajectory Prediction
=====================================================

State Space Model (SSM) based temporal encoder using Mamba architecture.
Provides linear complexity O(n) vs quadratic O(n²) for attention-based models.

Falls back to LSTM if mamba-ssm is not installed (GPU-only library).

References:
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)
- Trajectory Mamba (Tamba) for efficient trajectory forecasting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Mamba (GPU-only)
MAMBA_AVAILABLE = False
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    pass


class MambaBlock(nn.Module):
    """
    Single Mamba block with residual connection and layer normalization.
    """
    def __init__(self, hidden_dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            # Fallback: Use a simple gated RNN-like structure
            self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
            self.transform = nn.Linear(hidden_dim, hidden_dim)
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """
        Args:
            x: [B, L, D] input sequence
        Returns:
            out: [B, L, D] output sequence
        """
        residual = x
        x = self.norm(x)
        
        if MAMBA_AVAILABLE:
            x = self.mamba(x)
        else:
            # Simple gated fallback
            B, L, D = x.shape
            # Shift for causality
            x_shifted = F.pad(x[:, :-1, :], (0, 0, 1, 0))
            gate_input = torch.cat([x, x_shifted], dim=-1)
            gate = torch.sigmoid(self.gate(gate_input))
            x = gate * self.transform(x)
        
        x = self.dropout(x)
        return x + residual


class MambaTemporalEncoder(nn.Module):
    """
    Mamba-based temporal encoder for trajectory history.
    
    Uses selective state spaces for efficient long-range dependency modeling.
    Superior to LSTM for longer sequences with O(n) complexity.
    
    Args:
        input_dim: Input feature dimension (default: 4 for vel_x, vel_y, acc_x, acc_y)
        hidden_dim: Hidden dimension
        num_layers: Number of Mamba blocks
        d_state: SSM state dimension (higher = more capacity)
        d_conv: Convolution kernel size
    """
    def __init__(
        self, 
        input_dim: int = 4, 
        hidden_dim: int = 64, 
        num_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.using_mamba = MAMBA_AVAILABLE
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(hidden_dim, d_state=d_state, d_conv=d_conv)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, history):
        """
        Args:
            history: [N, T, input_dim] past motion features
            
        Returns:
            temporal_emb: [N, hidden_dim] temporal context embedding
        """
        if history is None or history.numel() == 0:
            return None
            
        # Project input
        x = self.input_proj(history)  # [N, T, H]
        
        # Apply Mamba blocks
        for layer in self.layers:
            x = layer(x)
        
        # Use final timestep as summary (like LSTM)
        final_hidden = x[:, -1, :]  # [N, H]
        
        return self.output_proj(final_hidden)


class BidirectionalMambaEncoder(nn.Module):
    """
    Bidirectional Mamba encoder for richer temporal representations.
    
    Combines forward and backward Mamba passes for full context.
    Inspired by BiLSTM but with Mamba efficiency.
    """
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2
    ):
        super().__init__()
        
        # Each direction gets half the hidden dim
        self.forward_encoder = MambaTemporalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim // 2,
            num_layers=num_layers
        )
        
        self.backward_encoder = MambaTemporalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim // 2,
            num_layers=num_layers
        )
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, history):
        """
        Args:
            history: [N, T, input_dim] past motion features
            
        Returns:
            temporal_emb: [N, hidden_dim] bidirectional temporal embedding
        """
        if history is None or history.numel() == 0:
            return None
            
        # Forward pass
        fwd_emb = self.forward_encoder(history)  # [N, H/2]
        
        # Backward pass (reverse sequence)
        bwd_history = history.flip(dims=[1])
        bwd_emb = self.backward_encoder(bwd_history)  # [N, H/2]
        
        # Combine
        combined = torch.cat([fwd_emb, bwd_emb], dim=-1)  # [N, H]
        
        return self.output_proj(combined)


class HybridMambaTransformer(nn.Module):
    """
    Hybrid Mamba-Transformer encoder combining efficiency with precision.
    
    Uses Mamba for initial sequence processing (efficiency) and
    Transformer self-attention for final refinement (precision).
    
    This is the SOTA approach for trajectory forecasting in 2024.
    """
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_mamba_layers: int = 2,
        num_attn_heads: int = 4
    ):
        super().__init__()
        
        # Mamba for efficient sequence encoding
        self.mamba_encoder = MambaTemporalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mamba_layers
        )
        
        # Single transformer layer for precision refinement
        self.attn_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attn_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, history):
        """
        Args:
            history: [N, T, input_dim] past motion features
            
        Returns:
            temporal_emb: [N, hidden_dim] hybrid temporal embedding
        """
        if history is None or history.numel() == 0:
            return None
            
        # Get Mamba summary
        mamba_emb = self.mamba_encoder(history)  # [N, H]
        
        # Project history for attention
        h = self.input_proj(history)  # [N, T, H]
        
        # Inject Mamba context
        h = h + mamba_emb.unsqueeze(1)  # Broadcast Mamba embedding
        
        # Attention refinement
        h = self.attn_layer(h)  # [N, T, H]
        
        # Final summary
        final = h[:, -1, :]  # [N, H]
        
        return self.output_proj(final)


def get_temporal_encoder(encoder_type: str = "lstm", **kwargs):
    """
    Factory function to get temporal encoder by type.
    
    Args:
        encoder_type: One of "lstm", "mamba", "bimamba", "hybrid"
        **kwargs: Arguments passed to encoder constructor
        
    Returns:
        Temporal encoder module
    """
    from src.models.gnn import TemporalHistoryEncoder  # Existing LSTM
    
    encoders = {
        "lstm": TemporalHistoryEncoder,
        "mamba": MambaTemporalEncoder,
        "bimamba": BidirectionalMambaEncoder,
        "hybrid": HybridMambaTransformer,
    }
    
    if encoder_type not in encoders:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Choose from {list(encoders.keys())}")
    
    # Log if Mamba is being used
    if encoder_type in ["mamba", "bimamba", "hybrid"]:
        if MAMBA_AVAILABLE:
            print(f"[INFO] Using Mamba-based encoder: {encoder_type}")
        else:
            print(f"[WARN] mamba-ssm not installed, using fallback for {encoder_type}")
    
    return encoders[encoder_type](**kwargs)
