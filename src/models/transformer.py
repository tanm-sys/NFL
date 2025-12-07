import torch
import torch.nn as nn
import torch.nn.functional as F

class PlayerEncoder(nn.Module):
    """
    Encodes spatial state of each player using a shared MLP + Self-Attention.
    Input: [Batch, Seq, Agents, Feats]
    Output: [Batch, Seq, Agents, Hidden_Dim]
    """
    def __init__(self, input_dim=6, hidden_dim=64, num_heads=4):
        super().__init__()
        self.input_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.spatial_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # x: [B, S, A, F]
        B, S, A, F = x.shape
        
        # Merge Batch and Seq for spatial attention (frame by frame independent initially)
        x_flat = x.view(B * S, A, F) # [B*S, A, F]
        
        # Shared MLP
        emb = self.input_mlp(x_flat) # [B*S, A, H]
        
        # Spatial Attention (Agents interacting within a frame)
        # Query=Key=Value=emb
        attn_out, _ = self.spatial_attn(emb, emb, emb) 
        
        # Residual + Norm
        out = self.norm(emb + attn_out)
        
        # Reshape back to sequence
        out = out.view(B, S, A, -1)
        return out

class TrajectoryDecoder(nn.Module):
    """
    Temporal Transformer to predict future trajectory.
    Input: [Batch, Seq, Agents, Hidden_Dim] (Encoded history)
    Output: [Batch, Future_Seq, Agents, 2] (Predicted X,Y)
    """
    def __init__(self, hidden_dim=64, num_heads=4, future_seq_len=10):
        super().__init__()
        # Simple approach: Flatten agents or treat agents as batch?
        # We want to predict trajectory for EACH agent.
        # Temporal attention per agent.
        
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True),
            num_layers=2
        )
        
        self.future_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, future_seq_len * 2) # Predict all future steps at once (Non-autoregressive for speed)
        )
        
        self.future_seq_len = future_seq_len
        
    def forward(self, x):
        # x: [B, S, A, H]
        B, S, A, H = x.shape
        
        # Treat Agents as independent for temporal dynamics (after spatial interaction)
        # Or separate batch dimension
        x_flat = x.permute(0, 2, 1, 3).reshape(B * A, S, H) # [B*A, S, H]
        
        # Temporal encoding
        temp_emb = self.temporal_encoder(x_flat) # [B*A, S, H]
        
        # Take last state as summary of history
        last_state = temp_emb[:, -1, :] # [B*A, H]
        
        # Predict future
        pred = self.future_head(last_state) # [B*A, Future*2]
        
        # Reshape
        pred = pred.view(B, A, self.future_seq_len, 2) # [B, A, Future, 2]
        pred = pred.permute(0, 2, 1, 3) # [B, Future, A, 2]
        
        return pred

class NFLTransformer(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_heads=4, future_seq_len=10):
        super().__init__()
        self.encoder = PlayerEncoder(input_dim, hidden_dim, num_heads)
        self.decoder = TrajectoryDecoder(hidden_dim, num_heads, future_seq_len)
        
    def forward(self, x):
        # x: [B, S, A, F]
        enc = self.encoder(x)
        pred = self.decoder(enc)
        return pred # [B, Future, A, 2]
