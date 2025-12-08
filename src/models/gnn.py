import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GlobalAttention, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import scatter

# Try to import einops for cleaner tensor operations
try:
    from einops import rearrange, repeat
    EINOPS_AVAILABLE = True
except ImportError:
    EINOPS_AVAILABLE = False


class DropPath(nn.Module):
    """
    Stochastic Depth (DropPath) - SOTA regularization technique.
    Randomly drops entire residual paths during training.
    Used in Vision Transformers, ConvNeXt, and modern architectures.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Work with different tensor shapes
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        output = x.div(keep_prob) * random_tensor
        return output


class TemporalHistoryEncoder(nn.Module):
    """
    Encodes past motion history (velocity, acceleration) using LSTM.
    Captures temporal patterns that inform future trajectory prediction.
    """
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2):
        super().__init__()
        # Input: [velocity_x, velocity_y, accel_x, accel_y] per timestep
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, history):
        """
        Args:
            history: [N, T, 4] past motion features (vel_x, vel_y, acc_x, acc_y)
            
        Returns:
            temporal_emb: [N, hidden_dim] temporal context embedding
        """
        if history is None or history.numel() == 0:
            return None
            
        # LSTM encoding
        _, (h_n, _) = self.lstm(history)  # h_n: [num_layers, N, hidden]
        
        # Use final layer hidden state
        final_hidden = h_n[-1]  # [N, hidden]
        
        return self.output_proj(final_hidden)


class SceneFlowEncoder(nn.Module):
    """
    P3: Global scene understanding via Set Transformer.
    Captures overall play dynamics beyond individual player interactions.
    """
    def __init__(self, hidden_dim=64, num_heads=4, num_inducing=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Inducing points for efficient set attention (like PMA in Set Transformer)
        self.inducing_points = nn.Parameter(torch.randn(num_inducing, hidden_dim))
        
        # Multi-head attention: inducing points attend to all nodes
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Aggregate inducing points to single scene embedding
        self.pool_attention = nn.Linear(hidden_dim, 1)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, node_embs, batch_idx):
        """
        Args:
            node_embs: [N, H] node embeddings from GNN
            batch_idx: [N] batch assignment
            
        Returns:
            scene_emb: [B, H] global scene embeddings per graph
        """
        if batch_idx is None:
            # Single graph
            nodes = node_embs.unsqueeze(0)  # [1, N, H]
            queries = self.inducing_points.unsqueeze(0)  # [1, K, H]
            
            # Cross attention
            attended, _ = self.cross_attention(queries, nodes, nodes)  # [1, K, H]
            
            # Weighted pooling
            weights = F.softmax(self.pool_attention(attended), dim=1)  # [1, K, 1]
            scene = (weights * attended).sum(dim=1)  # [1, H]
            
            return self.output_proj(scene)
        
        # Process each graph in batch
        num_graphs = batch_idx.max().item() + 1
        scene_embs = []
        
        for b in range(num_graphs):
            mask = batch_idx == b
            nodes = node_embs[mask].unsqueeze(0)  # [1, n, H]
            queries = self.inducing_points.unsqueeze(0)  # [1, K, H]
            
            # Cross attention
            attended, _ = self.cross_attention(queries, nodes, nodes)  # [1, K, H]
            
            # Weighted pooling
            weights = F.softmax(self.pool_attention(attended), dim=1)
            scene = (weights * attended).sum(dim=1)  # [1, H]
            scene_embs.append(scene)
        
        return self.output_proj(torch.cat(scene_embs, dim=0))  # [B, H]


class GoalConditionedDecoder(nn.Module):
    """
    P3: Goal-conditioned trajectory prediction.
    Predicts trajectories conditioned on potential target locations.
    """
    def __init__(self, hidden_dim=64, num_heads=4, future_seq_len=10, num_goals=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.future_seq_len = future_seq_len
        self.num_goals = num_goals
        
        # Goal proposals from context
        self.goal_proposer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_goals * 2)  # x, y for each goal
        )
        
        # Goal attention - which goal to follow
        self.goal_attention = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Trajectory decoder given goal
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.query_pos_emb = nn.Parameter(torch.randn(1, future_seq_len, hidden_dim))
        self.goal_emb = nn.Linear(2, hidden_dim)
        self.output_head = nn.Linear(hidden_dim, 2)
        
    def forward(self, context_emb, return_goals=False):
        """
        Args:
            context_emb: [N, H] node context embeddings
            return_goals: If True, also return goal proposals
            
        Returns:
            predictions: [N, T, 2] trajectory predictions
            goals: Optional [N, num_goals, 2] proposed goal positions
        """
        num_nodes = context_emb.shape[0]
        device = context_emb.device
        
        # Propose goals
        goal_params = self.goal_proposer(context_emb)  # [N, num_goals * 2]
        goals = goal_params.view(num_nodes, self.num_goals, 2)  # [N, K, 2]
        
        # Compute goal attention weights
        context_expanded = context_emb.unsqueeze(1).expand(-1, self.num_goals, -1)  # [N, K, H]
        goal_context = torch.cat([context_expanded, goals], dim=-1)  # [N, K, H+2]
        goal_weights = F.softmax(self.goal_attention(goal_context), dim=1)  # [N, K, 1]
        
        # Weighted goal
        selected_goal = (goal_weights * goals).sum(dim=1)  # [N, 2]
        
        # Encode goal
        goal_emb = self.goal_emb(selected_goal)  # [N, H]
        
        # Combine context and goal
        combined = context_emb + goal_emb  # [N, H]
        
        # Decode trajectory
        x = combined.unsqueeze(1).repeat(1, self.future_seq_len, 1)  # [N, T, H]
        x = x + self.query_pos_emb.repeat(num_nodes, 1, 1)
        
        out = self.transformer(x)  # [N, T, H]
        predictions = self.output_head(out)  # [N, T, 2]
        
        if return_goals:
            return predictions, goals
        return predictions


class HierarchicalDecoder(nn.Module):
    """
    P3: Hierarchical coarse-to-fine trajectory prediction.
    First predicts waypoints, then refines intermediate positions.
    """
    def __init__(self, hidden_dim=64, num_heads=4, future_seq_len=10, num_waypoints=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.future_seq_len = future_seq_len
        self.num_waypoints = num_waypoints
        
        # Coarse prediction (waypoints at key frames)
        self.coarse_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_waypoints * 2)
        )
        
        # Fine prediction (interpolate between waypoints)
        self.refine_encoder = nn.Linear(hidden_dim + 4, hidden_dim)  # context + start/end waypoint
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.refine_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.refine_head = nn.Linear(hidden_dim, 2)
        
    def forward(self, context_emb, return_waypoints=False):
        """
        Args:
            context_emb: [N, H] node context embeddings
            return_waypoints: If True, return coarse waypoints
            
        Returns:
            predictions: [N, T, 2] refined trajectory predictions
        """
        num_nodes = context_emb.shape[0]
        device = context_emb.device
        
        # Coarse waypoints
        waypoint_params = self.coarse_decoder(context_emb)  # [N, num_waypoints * 2]
        waypoints = waypoint_params.view(num_nodes, self.num_waypoints, 2)  # [N, W, 2]
        
        # Compute frames per segment
        frames_per_segment = self.future_seq_len // (self.num_waypoints - 1)
        
        # Refine each segment
        all_refined = []
        
        for i in range(self.num_waypoints - 1):
            start_wp = waypoints[:, i, :]  # [N, 2]
            end_wp = waypoints[:, i+1, :]  # [N, 2]
            
            # Conditioning on segment endpoints
            segment_context = torch.cat([context_emb, start_wp, end_wp], dim=-1)  # [N, H+4]
            segment_emb = self.refine_encoder(segment_context)  # [N, H]
            
            # Expand for frames in this segment
            x = segment_emb.unsqueeze(1).repeat(1, frames_per_segment, 1)  # [N, F, H]
            
            # Refine
            refined = self.refine_transformer(x)
            refined = self.refine_head(refined)  # [N, F, 2]
            
            # Add linear interpolation as residual
            t = torch.linspace(0, 1, frames_per_segment, device=device).view(1, -1, 1)
            linear_interp = start_wp.unsqueeze(1) + t * (end_wp - start_wp).unsqueeze(1)
            
            all_refined.append(refined + linear_interp)
        
        predictions = torch.cat(all_refined, dim=1)  # [N, T, 2]
        
        # Pad if needed
        if predictions.shape[1] < self.future_seq_len:
            pad = predictions[:, -1:, :].repeat(1, self.future_seq_len - predictions.shape[1], 1)
            predictions = torch.cat([predictions, pad], dim=1)
        
        if return_waypoints:
            return predictions[:, :self.future_seq_len], waypoints
        return predictions[:, :self.future_seq_len]

class GraphPlayerEncoder(nn.Module):
    """
    Encodes player spatial state + Strategic Features.
    Enhanced with GATv2 layers, residual connections, DropPath, and dropout.
    SOTA architecture used in competition-winning models.
    """
    def __init__(self, input_dim=7, hidden_dim=64, heads=4, context_dim=3, edge_dim=5, 
                 num_layers=4, dropout=0.1, droppath_rate=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Initial feature embedding
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Strategic Embeddings
        # Roles: 0-3, 4=Unknown
        self.role_emb = nn.Embedding(5, hidden_dim)
        # Side: 0=Defense, 1=Offense, 2=Unknown
        self.side_emb = nn.Embedding(3, hidden_dim // 2)
        
        # Context Encoder (Down, Dist, Box)
        self.context_encoder = nn.Linear(context_dim, hidden_dim)
        
        # Strategic Context Embeddings
        # Formation: 0-6, 7=Unknown
        self.formation_emb = nn.Embedding(8, hidden_dim)
        # Alignment: 0-8, 9=Unknown
        self.alignment_emb = nn.Embedding(10, hidden_dim)
        
        # Temporal Encoding (frame position in play)
        self.temporal_emb = nn.Embedding(100, hidden_dim)
        
        # Multi-Layer Graph Attention Network with residual connections
        self.gat_layers = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False, edge_dim=edge_dim)
            for _ in range(num_layers)
        ])
        
        # Layer normalization for each GAT layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # DropPath (Stochastic Depth) - SOTA regularization
        # Linearly increase drop probability through layers
        drop_rates = [droppath_rate * i / (num_layers - 1) for i in range(num_layers)]
        self.drop_paths = nn.ModuleList([
            DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
            for drop_rate in drop_rates
        ])
        
        # Social Pooling Layer
        self.social_pooling = SocialPoolingLayer(hidden_dim)
        
        # Temporal History Encoder (P1)
        self.history_encoder = TemporalHistoryEncoder(input_dim=4, hidden_dim=hidden_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Final normalization
        self.final_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, edge_index, edge_attr, context=None, batch=None, role=None, side=None, 
                formation=None, alignment=None, frame_t=None, history=None, return_attention_weights=False):
        # x: [Num_Nodes, Input_Dim]
        
        h = self.embedding(x)
        h = F.relu(h)
        
        # Fuse Role (Node-Level Strategy)
        if role is not None:
             # role: [Num_Nodes]
             r_emb = self.role_emb(role)
             h = h + r_emb
             
        # Fuse Side (Node-Level Team Awareness)
        if side is not None:
            s_emb = self.side_emb(side)
            # Pad to match hidden_dim (side_emb is hidden_dim//2)
            s_emb = F.pad(s_emb, (0, h.size(1) - s_emb.size(1)))
            h = h + s_emb
        
        # Fuse Temporal Context (NEW)
        if frame_t is not None:
            # frame_t is normalized [0, 1], convert to index [0, 99]
            t_idx = (frame_t * 99).long().clamp(0, 99)
            if t_idx.dim() == 1:
                t_idx = t_idx[0] if len(t_idx) > 0 else torch.tensor(0)
            t_emb = self.temporal_emb(t_idx.expand(h.size(0)))
            h = h + t_emb
        
        # Fuse Context (Graph-Level Strategy)
        if context is not None:
            c_emb = self.context_encoder(context)  # [Batch, Hidden]
            c_emb = F.relu(c_emb)
            
            # Fuse categorical context
            if formation is not None:
                 c_emb = c_emb + self.formation_emb(formation)
                 
            if alignment is not None:
                 c_emb = c_emb + self.alignment_emb(alignment)
            
            # Broadcast to nodes
            if batch is not None:
                c_nodes = c_emb[batch]
            else:
                c_nodes = c_emb.expand(h.size(0), -1)
                
            h = h + c_nodes
        
        # Fuse Temporal History (P1 - LSTM encoded motion history)
        if history is not None and history.numel() > 0:
            history_emb = self.history_encoder(history)  # [N, hidden]
            if history_emb is not None:
                h = h + history_emb
        
        # Multi-Layer GNN with Residual Connections + DropPath
        attn_weights = None
        for i, (gat, norm, drop_path) in enumerate(zip(self.gat_layers, self.layer_norms, self.drop_paths)):
            h_res = h
            
            # Last layer can return attention weights
            if i == len(self.gat_layers) - 1 and return_attention_weights:
                h, (edge_index_attn, alpha) = gat(h, edge_index, edge_attr=edge_attr, return_attention_weights=True)
                attn_weights = (edge_index_attn, alpha)
            else:
                h = gat(h, edge_index, edge_attr=edge_attr)
            
            h = F.relu(h)
            h = self.dropout(h)
            h = drop_path(h)  # Stochastic depth
            h = norm(h + h_res)  # Residual + LayerNorm
        
        # Apply Social Pooling (NEW)
        if edge_index.numel() > 0:
            social_context = self.social_pooling(h, edge_index)
            h = h + social_context
        
        h = self.final_norm(h)
        return h, attn_weights


class SocialPoolingLayer(nn.Module):
    """
    Models explicit pairwise interactions between players.
    Aggregates neighbor information weighted by interaction strength.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.interaction_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, node_emb, edge_index):
        src, dst = edge_index
        
        # Compute pairwise interaction features
        pair_emb = torch.cat([node_emb[src], node_emb[dst]], dim=-1)
        interaction = self.interaction_mlp(pair_emb)
        
        # Gated aggregation
        gate_values = torch.sigmoid(self.gate(interaction))
        gated_interaction = interaction * gate_values
        
        # Aggregate per node using scatter
        num_nodes = node_emb.size(0)
        pooled = scatter(gated_interaction, src, dim=0, dim_size=num_nodes, reduce='mean')
        
        return pooled


class TrajectoryDecoder(nn.Module):
    """
    Temporal Transformer for deterministic future path prediction.
    Kept for backwards compatibility.
    """
    def __init__(self, hidden_dim, num_heads, future_seq_len):
        super().__init__()
        self.future_seq_len = future_seq_len
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, 20, hidden_dim)) # Max history 20
        self.query_pos_emb = nn.Parameter(torch.randn(1, future_seq_len, hidden_dim))
        
        self.output_head = nn.Linear(hidden_dim, 2) # dx, dy output
        
    def forward(self, context_emb):
        num_nodes, hidden = context_emb.shape
        x = context_emb.unsqueeze(1).repeat(1, self.future_seq_len, 1) # [Nodes, Seq, Hidden]
        
        # Add temporal queries
        queries = self.query_pos_emb.repeat(num_nodes, 1, 1)
        x = x + queries
        
        # Transformer
        out = self.transformer(x)
        
        # Head
        pred = self.output_head(out) # [Nodes, Seq, 2]
        return pred


class ProbabilisticTrajectoryDecoder(nn.Module):
    """
    GMM-based trajectory prediction with uncertainty quantification.
    Predicts multiple trajectory modes with associated probabilities.
    
    Each mode outputs:
    - mu_x, mu_y: Mean position
    - sigma_x, sigma_y: Standard deviations  
    - rho: Correlation coefficient
    """
    def __init__(self, hidden_dim, num_heads, future_seq_len, num_modes=6):
        super().__init__()
        self.num_modes = num_modes
        self.future_seq_len = future_seq_len
        self.hidden_dim = hidden_dim
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Positional embeddings
        self.query_pos_emb = nn.Parameter(torch.randn(1, future_seq_len, hidden_dim))
        
        # Mode embedding (helps differentiate predictions)
        self.mode_emb = nn.Embedding(num_modes, hidden_dim)
        
        # GMM parameters: [mu_x, mu_y, sigma_x, sigma_y, rho] per mode per timestep
        self.gmm_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5)  # mu_x, mu_y, log_sigma_x, log_sigma_y, rho_raw
        )
        
        # Mode probabilities (per node, across all modes)
        self.mode_prob_head = nn.Linear(hidden_dim, num_modes)
        
    def forward(self, context_emb, return_best_mode=False):
        """
        Args:
            context_emb: [N, H] node embeddings from GNN
            return_best_mode: If True, return only the most likely trajectory
            
        Returns:
            If return_best_mode:
                predictions: [N, T, 2] best trajectory
            Else:
                params: [N, T, K, 5] GMM parameters (mu_x, mu_y, sigma_x, sigma_y, rho)
                probs: [N, K] mode probabilities
        """
        num_nodes, hidden = context_emb.shape
        device = context_emb.device
        
        # Compute mode probabilities from context
        mode_probs = F.softmax(self.mode_prob_head(context_emb), dim=-1)  # [N, K]
        
        all_mode_params = []
        
        for m in range(self.num_modes):
            # Add mode embedding to context
            mode_context = context_emb + self.mode_emb.weight[m].unsqueeze(0)  # [N, H]
            
            # Expand for sequence
            x = mode_context.unsqueeze(1).repeat(1, self.future_seq_len, 1)  # [N, T, H]
            
            # Add temporal queries
            x = x + self.query_pos_emb.repeat(num_nodes, 1, 1)
            
            # Transformer
            out = self.transformer(x)  # [N, T, H]
            
            # GMM parameters
            raw_params = self.gmm_head(out)  # [N, T, 5]
            
            # Process parameters
            mu = raw_params[..., :2]  # [N, T, 2]
            log_sigma = raw_params[..., 2:4]  # [N, T, 2]
            rho_raw = raw_params[..., 4:5]  # [N, T, 1]
            
            # Ensure valid sigma and rho
            sigma = torch.exp(log_sigma).clamp(min=0.01, max=10.0)  # [N, T, 2]
            rho = torch.tanh(rho_raw)  # [N, T, 1] in (-1, 1)
            
            mode_params = torch.cat([mu, sigma, rho], dim=-1)  # [N, T, 5]
            all_mode_params.append(mode_params)
        
        # Stack all modes: [N, T, K, 5]
        params = torch.stack(all_mode_params, dim=2)
        
        if return_best_mode:
            # Return trajectory from most likely mode
            best_mode = mode_probs.argmax(dim=-1)  # [N]
            batch_idx = torch.arange(num_nodes, device=device)
            best_params = params[batch_idx, :, best_mode, :]  # [N, T, 5]
            return best_params[..., :2]  # Return just mu_x, mu_y
        
        return params, mode_probs
    
    def nll_loss(self, params, probs, target):
        """
        Compute negative log-likelihood loss for GMM.
        
        Args:
            params: [N, T, K, 5] GMM parameters
            probs: [N, K] mode probabilities  
            target: [N, T, 2] ground truth positions
            
        Returns:
            loss: scalar NLL loss
        """
        num_nodes, seq_len, num_modes, _ = params.shape
        
        # Expand target for all modes: [N, T, K, 2]
        target_expanded = target.unsqueeze(2).expand(-1, -1, num_modes, -1)
        
        # Extract GMM parameters
        mu = params[..., :2]  # [N, T, K, 2]
        sigma = params[..., 2:4]  # [N, T, K, 2]
        rho = params[..., 4:5].squeeze(-1)  # [N, T, K]
        
        # Compute bivariate Gaussian log-likelihood
        diff = target_expanded - mu  # [N, T, K, 2]
        
        # Normalized differences
        z_x = diff[..., 0] / sigma[..., 0]
        z_y = diff[..., 1] / sigma[..., 1]
        
        # Log determinant of covariance matrix
        log_det = torch.log(sigma[..., 0]) + torch.log(sigma[..., 1]) + 0.5 * torch.log(1 - rho**2 + 1e-6)
        
        # Mahalanobis distance for bivariate Gaussian
        z = (z_x**2 + z_y**2 - 2 * rho * z_x * z_y) / (1 - rho**2 + 1e-6)
        
        # Log-likelihood per mode per timestep
        log_likelihood = -0.5 * z - log_det - torch.log(torch.tensor(2 * 3.14159))  # [N, T, K]
        
        # Sum over timesteps
        log_likelihood_seq = log_likelihood.sum(dim=1)  # [N, K]
        
        # Add log mode probabilities
        log_probs = torch.log(probs + 1e-6)  # [N, K]
        log_mixture = log_likelihood_seq + log_probs  # [N, K]
        
        # Log-sum-exp over modes
        nll = -torch.logsumexp(log_mixture, dim=-1)  # [N]
        
        return nll.mean()


class NFLGraphTransformer(nn.Module):
    """
    Main model combining GNN encoder with trajectory decoder and coverage classifier.
    P3: Now includes SceneFlowEncoder for global play understanding.
    """
    def __init__(self, input_dim=7, hidden_dim=64, heads=4, future_seq_len=10, 
                 edge_dim=5, num_gnn_layers=4, probabilistic=False, num_modes=6,
                 use_scene_encoder=True):
        super().__init__()
        self.probabilistic = probabilistic
        self.use_scene_encoder = use_scene_encoder
        
        self.encoder = GraphPlayerEncoder(
            input_dim, hidden_dim, heads, 
            edge_dim=edge_dim, num_layers=num_gnn_layers
        )
        
        # P3: Scene Flow Encoder for global context
        if use_scene_encoder:
            self.scene_encoder = SceneFlowEncoder(hidden_dim, heads)
        
        if probabilistic:
            self.decoder = ProbabilisticTrajectoryDecoder(
                hidden_dim, heads, future_seq_len, num_modes
            )
        else:
            self.decoder = TrajectoryDecoder(hidden_dim, heads, future_seq_len)
        
        # Multi-Task Head: Coverage Classification (Binary: Man vs Zone)
        self.classifier = nn.Linear(hidden_dim, 1)
        
    def forward(self, data, return_attention_weights=False, return_distribution=False):
        """
        Args:
            data: PyG Batch Data object
            return_attention_weights: Return GNN attention weights
            return_distribution: For probabilistic mode, return full GMM params
            
        Returns:
            predictions: [N, T, 2] trajectory predictions
            cov_pred: [B, 1] coverage logits
            attn_weights: Optional attention weights
        """
        # PyG Batch Data object
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else None
        
        # Context extraction (handle missing safely)
        context = data.context if hasattr(data, 'context') else None
        
        # Strategic Features
        role = data.role if hasattr(data, 'role') else None
        side = data.side if hasattr(data, 'side') else None
        formation = data.formation if hasattr(data, 'formation') else None
        alignment = data.alignment if hasattr(data, 'alignment') else None
        frame_t = data.frame_t if hasattr(data, 'frame_t') else None
        
        # Temporal History (P1)
        history = data.history if hasattr(data, 'history') else None
        
        # Encode Spatial (GNN)
        node_embs, attn_weights = self.encoder(
            x, edge_index, edge_attr, context, batch, 
            role=role, side=side, formation=formation, alignment=alignment,
            frame_t=frame_t, history=history, return_attention_weights=return_attention_weights
        )
        
        # P3: Add global scene context
        if self.use_scene_encoder:
            scene_emb = self.scene_encoder(node_embs, batch)  # [B, H]
            # Broadcast scene embedding to all nodes
            if batch is not None:
                scene_nodes = scene_emb[batch]  # [N, H]
            else:
                scene_nodes = scene_emb.expand(node_embs.size(0), -1)
            node_embs = node_embs + 0.5 * scene_nodes  # Weighted addition
        
        # Decode Temporal (Trajectory Prediction)
        if self.probabilistic:
            if return_distribution:
                predictions, mode_probs = self.decoder(node_embs)
            else:
                predictions = self.decoder(node_embs, return_best_mode=True)
                mode_probs = None
        else:
            predictions = self.decoder(node_embs)
            mode_probs = None
        
        # Multi-Task: Coverage Classification
        if batch is None:
            global_emb = torch.mean(node_embs, dim=0, keepdim=True)
        else:
            global_emb = global_mean_pool(node_embs, batch)
            
        cov_pred = self.classifier(global_emb)
        
        if return_distribution and self.probabilistic:
            return predictions, mode_probs, cov_pred, attn_weights
        
        return predictions, cov_pred, attn_weights
