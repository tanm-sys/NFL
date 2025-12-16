"""
Contrastive Losses for NFL Trajectory Prediction
=================================================

State-of-the-art contrastive learning techniques for trajectory forecasting:
- Social-NCE: Collision-aware trajectory learning via negative sampling
- Trajectory Contrastive: History-future alignment (TrajCLIP-style)
- Diversity Loss: Encourages multi-modal predictions

References:
- Social NCE: Contrastive Learning of Socially-aware Motion Representations (CVPR 2021)
- TrajCLIP: Contrastive Learning for Trajectory Representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SocialNCELoss(nn.Module):
    """
    Social Noise Contrastive Estimation Loss.
    
    Creates negative samples by generating collision-prone trajectories,
    then learns to distinguish safe trajectories from dangerous ones.
    
    This encourages the model to produce socially-aware predictions that
    avoid collisions with other agents.
    
    Args:
        temperature: Contrastive temperature (lower = sharper distinctions)
        num_negatives: Number of negative samples per positive
        collision_threshold: Distance in yards below which collision occurs
    """
    def __init__(
        self,
        temperature: float = 0.1,
        num_negatives: int = 8,
        collision_threshold: float = 1.5
    ):
        super().__init__()
        self.temperature = temperature
        self.num_negatives = num_negatives
        self.collision_threshold = collision_threshold
        
        # Projection head for contrastive learning
        self.proj = nn.Sequential(
            nn.Linear(128, 64),  # Will be set dynamically
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self._proj_initialized = False
        
    def _init_proj(self, hidden_dim: int):
        """Lazily initialize projection head with correct dimensions."""
        if not self._proj_initialized:
            self.proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 32)
            ).to(next(self.parameters()).device if list(self.parameters()) else 'cpu')
            self._proj_initialized = True
    
    def generate_negative_samples(
        self,
        trajectories: torch.Tensor,
        batch_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate collision-prone negative trajectory samples.
        
        Creates negatives by:
        1. Perturbing toward other agents (collision)
        2. Adding unrealistic acceleration (physics violation)
        3. Random perturbations with high variance
        
        Args:
            trajectories: [N, T, 2] predicted trajectories
            batch_idx: [N] batch assignment for each node
            
        Returns:
            negatives: [N, num_negatives, T, 2] negative samples
        """
        N, T, _ = trajectories.shape
        device = trajectories.device
        negatives = []
        
        for _ in range(self.num_negatives):
            neg_type = torch.randint(0, 3, (1,)).item()
            
            if neg_type == 0:
                # Collision-prone: Move toward random other agent
                random_idx = torch.randint(0, N, (N,), device=device)
                other_traj = trajectories[random_idx]
                # Interpolate toward other agent
                alpha = torch.rand(N, 1, 1, device=device) * 0.8 + 0.1
                neg = trajectories + alpha * (other_traj - trajectories)
                
            elif neg_type == 1:
                # Physics violation: Unrealistic acceleration
                accel = torch.randn(N, T, 2, device=device) * 5.0
                neg = trajectories + accel.cumsum(dim=1) * 0.1
                
            else:
                # Random perturbation with high variance
                noise = torch.randn(N, T, 2, device=device) * 3.0
                neg = trajectories + noise
            
            negatives.append(neg)
        
        return torch.stack(negatives, dim=1)  # [N, num_neg, T, 2]
    
    def forward(
        self,
        pred_traj: torch.Tensor,
        node_embeddings: torch.Tensor,
        batch_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Social NCE loss.
        
        Args:
            pred_traj: [N, T, 2] predicted trajectories (positive samples)
            node_embeddings: [N, H] node embeddings from encoder
            batch_idx: [N] batch assignment
            
        Returns:
            loss: Scalar contrastive loss
        """
        N, T, _ = pred_traj.shape
        H = node_embeddings.shape[1]
        device = pred_traj.device
        
        # Initialize projection if needed
        self._init_proj(H)
        
        # Encode trajectory into embedding
        traj_flat = pred_traj.reshape(N, -1)  # [N, T*2]
        traj_proj = F.linear(traj_flat, torch.randn(H, T*2, device=device) * 0.01)  # Simple projection
        combined = node_embeddings + traj_proj
        z_pos = self.proj.to(device)(combined)  # [N, 32]
        z_pos = F.normalize(z_pos, dim=-1)
        
        # Generate negatives
        neg_trajs = self.generate_negative_samples(pred_traj, batch_idx)  # [N, K, T, 2]
        K = neg_trajs.shape[1]
        
        # Encode negatives
        neg_flat = neg_trajs.reshape(N * K, -1)  # [N*K, T*2]
        node_emb_exp = node_embeddings.unsqueeze(1).expand(-1, K, -1).reshape(N * K, -1)
        neg_proj = F.linear(neg_flat, torch.randn(H, T*2, device=device) * 0.01)
        neg_combined = node_emb_exp + neg_proj
        z_neg = self.proj.to(device)(neg_combined).reshape(N, K, -1)  # [N, K, 32]
        z_neg = F.normalize(z_neg, dim=-1)
        
        # InfoNCE loss
        pos_sim = (z_pos.unsqueeze(1) * z_pos.unsqueeze(0)).sum(-1)  # [N, N]
        neg_sim = (z_pos.unsqueeze(1) * z_neg).sum(-1)  # [N, K]
        
        # Positive: self-similarity, Negatives: collision-prone
        logits = torch.cat([
            torch.diag(pos_sim).unsqueeze(1),  # [N, 1] positive
            neg_sim  # [N, K] negatives
        ], dim=1) / self.temperature
        
        # Target: first column is positive
        targets = torch.zeros(N, dtype=torch.long, device=device)
        
        return F.cross_entropy(logits, targets)


class TrajectoryContrastiveLoss(nn.Module):
    """
    Trajectory Contrastive Learning Loss (TrajCLIP-style).
    
    Aligns history and future trajectory representations:
    - Positive pairs: history and corresponding future from same agent
    - Negative pairs: history from one agent, future from different agent
    
    This ensures the model learns meaningful motion patterns that
    connect past observations to future predictions.
    
    Args:
        temperature: Contrastive temperature
        hidden_dim: Embedding dimension
    """
    def __init__(self, temperature: float = 0.07, hidden_dim: int = 64):
        super().__init__()
        self.temperature = temperature
        
        # History encoder (lightweight)
        self.history_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Future encoder
        self.future_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(
        self,
        history_emb: torch.Tensor,
        future_emb: torch.Tensor,
        batch_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute trajectory contrastive loss.
        
        Args:
            history_emb: [N, H] history embeddings
            future_emb: [N, H] future trajectory embeddings
            batch_idx: [N] batch assignment
            
        Returns:
            loss: Scalar contrastive loss
        """
        # Project and normalize
        z_hist = F.normalize(self.history_proj(history_emb), dim=-1)  # [N, H]
        z_fut = F.normalize(self.future_proj(future_emb), dim=-1)  # [N, H]
        
        # Compute similarities
        sim_matrix = torch.mm(z_hist, z_fut.t()) / self.temperature  # [N, N]
        
        # Labels: diagonal elements are positives
        N = z_hist.shape[0]
        labels = torch.arange(N, device=z_hist.device)
        
        # Symmetric loss (history->future and future->history)
        loss_h2f = F.cross_entropy(sim_matrix, labels)
        loss_f2h = F.cross_entropy(sim_matrix.t(), labels)
        
        return (loss_h2f + loss_f2h) / 2


class DiversityLoss(nn.Module):
    """
    Trajectory Diversity Loss for multi-modal predictions.
    
    Encourages diversity in predicted modes by penalizing
    mode collapse (all modes predicting same trajectory).
    
    Args:
        min_distance: Minimum desired distance between modes
    """
    def __init__(self, min_distance: float = 2.0):
        super().__init__()
        self.min_distance = min_distance
        
    def forward(
        self,
        multimodal_preds: torch.Tensor,
        mode_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute diversity loss.
        
        Args:
            multimodal_preds: [N, T, K, 2] or [N, T, K, 5] multi-modal predictions
            mode_probs: [N, K] mode probabilities
            
        Returns:
            loss: Scalar diversity loss
        """
        # Handle GMM format (take just means)
        if multimodal_preds.shape[-1] == 5:
            preds = multimodal_preds[..., :2]  # [N, T, K, 2]
        else:
            preds = multimodal_preds
            
        N, T, K, _ = preds.shape
        
        if K <= 1:
            return torch.tensor(0.0, device=preds.device)
        
        # Compute pairwise mode distances
        # [N, T, K, 2] -> [N, K, T*2]
        preds_flat = preds.permute(0, 2, 1, 3).reshape(N, K, -1)
        
        # Pairwise L2 distance between modes
        diff = preds_flat.unsqueeze(2) - preds_flat.unsqueeze(1)  # [N, K, K, T*2]
        distances = torch.norm(diff, dim=-1)  # [N, K, K]
        
        # Mask diagonal (self-distance)
        mask = ~torch.eye(K, dtype=torch.bool, device=preds.device).unsqueeze(0)
        distances = distances * mask
        
        # Weight by mode probabilities
        prob_weights = mode_probs.unsqueeze(2) * mode_probs.unsqueeze(1)  # [N, K, K]
        prob_weights = prob_weights * mask
        
        # Penalize if distance < min_distance
        diversity_penalty = F.relu(self.min_distance - distances) * prob_weights
        
        return diversity_penalty.sum() / (N * K * (K - 1) + 1e-6)


class WinnerTakesAllLoss(nn.Module):
    """
    Winner-Takes-All (WTA) Loss for multi-modal trajectory prediction.
    
    Only penalizes the best-matching mode, allowing other modes to
    capture alternative futures without gradient interference.
    
    Standard in SOTA trajectory forecasting models (e.g., Trajectron++, LaneGCN).
    
    Args:
        k_best: Number of best modes to consider (default: 1 for pure WTA)
    """
    def __init__(self, k_best: int = 1):
        super().__init__()
        self.k_best = k_best
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mode_probs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute WTA loss.
        
        Args:
            predictions: [N, T, K, 2] multi-modal predictions
            targets: [N, T, 2] ground truth trajectories
            mode_probs: [N, K] optional mode probabilities
            
        Returns:
            wta_loss: Scalar WTA regression loss
            best_mode_idx: [N] indices of best modes
        """
        N, T, K, _ = predictions.shape
        
        # Expand targets for comparison
        targets_exp = targets.unsqueeze(2).expand(-1, -1, K, -1)  # [N, T, K, 2]
        
        # Compute per-mode ADE
        errors = torch.norm(predictions - targets_exp, dim=-1)  # [N, T, K]
        mode_ade = errors.mean(dim=1)  # [N, K]
        
        # Find best mode per sample
        if self.k_best == 1:
            best_mode_idx = mode_ade.argmin(dim=1)  # [N]
            best_errors = mode_ade[torch.arange(N, device=predictions.device), best_mode_idx]
        else:
            # Top-k averaging
            topk_errors, topk_idx = mode_ade.topk(self.k_best, dim=1, largest=False)
            best_errors = topk_errors.mean(dim=1)
            best_mode_idx = topk_idx[:, 0]
        
        wta_loss = best_errors.mean()
        
        # Optional: mode probability regularization
        if mode_probs is not None:
            # Encourage probability mass on best mode
            target_probs = F.one_hot(best_mode_idx, K).float()
            prob_loss = F.cross_entropy(mode_probs.log(), best_mode_idx)
            wta_loss = wta_loss + 0.1 * prob_loss
        
        return wta_loss, best_mode_idx


class EndpointFocalLoss(nn.Module):
    """
    Focal Loss applied to trajectory endpoints.
    
    Gives higher weight to hard-to-predict final positions,
    which are often the most important for downstream tasks.
    
    Args:
        gamma: Focal loss gamma (higher = more focus on hard examples)
        alpha: Balance factor for endpoint vs full trajectory
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute endpoint focal loss.
        
        Args:
            predictions: [N, T, 2] predicted trajectories
            targets: [N, T, 2] ground truth trajectories
            mask: [N, T] optional validity mask
            
        Returns:
            loss: Scalar focal endpoint loss
        """
        # Endpoint error
        endpoint_pred = predictions[:, -1, :]  # [N, 2]
        endpoint_target = targets[:, -1, :]  # [N, 2]
        
        endpoint_error = torch.norm(endpoint_pred - endpoint_target, dim=-1)  # [N]
        
        # Normalize error for focal weighting (soft clip)
        normalized_error = torch.tanh(endpoint_error / 5.0)  # [0, 1]
        
        # Focal weight: higher for larger errors
        focal_weight = (normalized_error ** self.gamma)
        
        # Weighted loss
        weighted_error = focal_weight * endpoint_error
        
        if mask is not None:
            # Use mask for final timestep
            final_mask = mask[:, -1]
            weighted_error = weighted_error * final_mask
            return weighted_error.sum() / (final_mask.sum() + 1e-6)
        
        return weighted_error.mean()
