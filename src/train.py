import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import optuna
from torch_geometric.loader import DataLoader as PyGDataLoader
from src.models.gnn import NFLGraphTransformer
from src.features import create_graph_data
import numpy as np


def velocity_loss(pred, target):
    """
    Penalize unrealistic trajectory velocity changes.
    Encourages smooth, physically plausible trajectories.
    
    Args:
        pred: [N, T, 2] predicted positions
        target: [N, T, 2] ground truth positions
    
    Returns:
        loss: scalar velocity consistency loss
    """
    pred_vel = pred[:, 1:] - pred[:, :-1]  # [N, T-1, 2]
    target_vel = target[:, 1:] - target[:, :-1]  # [N, T-1, 2]
    return F.mse_loss(pred_vel, target_vel)


def acceleration_loss(pred, target):
    """
    Penalize unrealistic acceleration patterns (P1 enhancement).
    Second-order smoothness constraint for more realistic motion.
    
    Args:
        pred: [N, T, 2] predicted positions
        target: [N, T, 2] ground truth positions
    
    Returns:
        loss: scalar acceleration consistency loss
    """
    if pred.shape[1] < 3:
        return torch.tensor(0.0, device=pred.device)
    
    pred_vel = pred[:, 1:] - pred[:, :-1]  # [N, T-1, 2]
    target_vel = target[:, 1:] - target[:, :-1]
    
    pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]  # [N, T-2, 2]
    target_acc = target_vel[:, 1:] - target_vel[:, :-1]
    
    return F.mse_loss(pred_acc, target_acc)


def collision_avoidance_loss(pred_positions, batch_idx, min_distance=1.0):
    """
    Penalize predictions where players occupy the same space (P1 enhancement).
    Encourages physically plausible non-overlapping trajectories.
    
    Args:
        pred_positions: [N, T, 2] predicted positions
        batch_idx: [N] batch assignment for each node
        min_distance: Minimum allowed distance in yards
        
    Returns:
        loss: scalar collision penalty
    """
    if batch_idx is None or pred_positions.shape[0] < 2:
        return torch.tensor(0.0, device=pred_positions.device)
    
    # Compute pairwise distances within each graph
    total_loss = 0.0
    n_graphs = 0
    
    for b in batch_idx.unique():
        mask = batch_idx == b
        pos = pred_positions[mask]  # [n_nodes, T, 2]
        
        if pos.shape[0] < 2:
            continue
        
        # Average position across time for efficiency
        avg_pos = pos.mean(dim=1)  # [n_nodes, 2]
        
        # Pairwise distances
        dist = torch.cdist(avg_pos.unsqueeze(0), avg_pos.unsqueeze(0)).squeeze(0)
        
        # Penalize close distances (excluding self-distance)
        dist = dist + torch.eye(dist.shape[0], device=dist.device) * 100  # Mask diagonal
        violations = F.relu(min_distance - dist)
        
        total_loss += violations.sum()
        n_graphs += 1
    
    return total_loss / max(n_graphs, 1)


def augment_graph(data, flip_horizontal=True, add_noise=True, noise_std=0.1):
    """
    Apply data augmentation to graph data.
    
    Args:
        data: PyG Data object
        flip_horizontal: Mirror play horizontally
        add_noise: Add Gaussian noise to positions
        noise_std: Standard deviation of noise
    
    Returns:
        Augmented data object
    """
    data = data.clone()
    
    if flip_horizontal and torch.rand(1).item() > 0.5:
        # Flip y coordinate (field is 53.3 yards wide)
        data.x[:, 1] = 53.3 - data.x[:, 1]  # Flip y position
        if data.x.size(1) > 4:
            data.x[:, 4] = (360 - data.x[:, 4]) % 360  # Flip direction
        if data.x.size(1) > 5:
            data.x[:, 5] = (360 - data.x[:, 5]) % 360  # Flip orientation
        
        # Flip targets
        if hasattr(data, 'y') and data.y is not None:
            data.y[:, :, 1] = 53.3 - data.y[:, :, 1]
    
    if add_noise:
        # Add small noise to positions
        noise = torch.randn_like(data.x[:, :2]) * noise_std
        data.x[:, :2] = data.x[:, :2] + noise
    
    return data


class NFLGraphPredictor(pl.LightningModule):
    """
    PyTorch Lightning module for NFL trajectory prediction.
    Supports both deterministic and probabilistic (GMM) modes.
    """
    def __init__(self, input_dim=7, hidden_dim=64, lr=1e-3, future_seq_len=10,
                 probabilistic=False, num_modes=6, velocity_weight=0.3, 
                 acceleration_weight=0.1, coverage_weight=0.5, collision_weight=0.05,
                 use_augmentation=True, use_huber_loss=False, huber_delta=1.0):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = NFLGraphTransformer(
            input_dim, hidden_dim, 
            future_seq_len=future_seq_len,
            probabilistic=probabilistic,
            num_modes=num_modes
        )
        self.lr = lr
        self.probabilistic = probabilistic
        self.velocity_weight = velocity_weight
        self.acceleration_weight = acceleration_weight
        self.coverage_weight = coverage_weight
        self.collision_weight = collision_weight
        self.use_augmentation = use_augmentation
        self.use_huber_loss = use_huber_loss
        self.huber_delta = huber_delta

    def forward(self, data):
        return self.model(data)
    
    def _augment_batch(self, batch):
        """Apply augmentation during training if enabled."""
        if self.use_augmentation and self.training:
            return augment_graph(batch, flip_horizontal=True, add_noise=True)
        return batch
        
    def training_step(self, batch, batch_idx):
        # Apply augmentation
        batch = self._augment_batch(batch)
        
        # Forward pass
        if self.probabilistic:
            params, mode_probs, cov_pred, _ = self.model(batch, return_distribution=True)
            
            # GMM NLL Loss
            y = batch.y  # [N, T, 2]
            loss_traj = self.model.decoder.nll_loss(params, mode_probs, y)
            
            # Best mode prediction for metrics
            predictions = params[..., :2].mean(dim=2)  # Average over modes for velocity loss
        else:
            predictions, cov_pred, _ = self(batch)
            
            # Loss: MSE or Huber (P2)
            y = batch.y
            if self.use_huber_loss:
                loss_traj = F.huber_loss(predictions, y, delta=self.huber_delta)
            else:
                loss_traj = F.mse_loss(predictions, y)
        
        self.log("train_traj_loss", loss_traj, batch_size=batch.num_graphs)
        
        # Velocity Loss
        loss_vel = velocity_loss(predictions, y)
        self.log("train_vel_loss", loss_vel, batch_size=batch.num_graphs)
        
        # Acceleration Loss (P1)
        loss_acc = acceleration_loss(predictions, y)
        self.log("train_acc_loss", loss_acc, batch_size=batch.num_graphs)
        
        # Collision Avoidance Loss (P1)
        batch_idx = batch.batch if hasattr(batch, 'batch') else None
        loss_collision = collision_avoidance_loss(predictions, batch_idx)
        self.log("train_collision_loss", loss_collision, batch_size=batch.num_graphs)
        
        # Coverage Loss (BCE) - filter out missing coverage labels (sentinel value -1.0)
        loss_cov = torch.tensor(0.0, device=self.device)
        if hasattr(batch, 'y_coverage') and batch.y_coverage is not None:
            target_cov = batch.y_coverage.view(-1, 1)
            valid_mask = (target_cov >= 0).squeeze()  # -1.0 = missing
            if valid_mask.any():
                loss_cov = F.binary_cross_entropy_with_logits(
                    cov_pred[valid_mask], target_cov[valid_mask]
                )
                self.log("train_cov_loss", loss_cov, batch_size=valid_mask.sum())
             
        # Total Loss (Weighted) - All P0/P1/P2 losses
        loss = (loss_traj + 
                self.velocity_weight * loss_vel + 
                self.acceleration_weight * loss_acc +
                self.collision_weight * loss_collision +
                self.coverage_weight * loss_cov)
        self.log("train_loss", loss, batch_size=batch.num_graphs)
        return loss
        
    def validation_step(self, batch, batch_idx):
        # Forward pass
        if self.probabilistic:
            predictions = self.model(batch, return_distribution=False)[0]  # Best mode prediction
        else:
            predictions, cov_pred, _ = self(batch)
        
        y = batch.y  # Relative targets
        
        # Convert predictions to absolute positions for metrics
        if hasattr(batch, 'current_pos'):
            current_pos = batch.current_pos.unsqueeze(1)  # [N, 1, 2]
            pred_abs = predictions + current_pos  # Cumulative prediction
            target_abs = y + current_pos
        else:
            pred_abs = predictions
            target_abs = y
        
        # Traj Validation (on relative for loss, absolute for metrics)
        loss_traj = F.mse_loss(predictions, y)  # Loss on relative predictions
        disp = torch.sqrt(torch.sum((pred_abs - target_abs)**2, dim=-1))  # [N, T] - ADE/FDE on absolute
        
        # ADE: Average Displacement Error
        ade = torch.mean(disp)
        
        # FDE: Final Displacement Error
        fde = torch.mean(disp[:, -1])
        
        # Miss Rate @ 2 yards
        miss_threshold = 2.0
        miss_rate = (disp[:, -1] > miss_threshold).float().mean()
        
        self.log("val_loss_traj", loss_traj, batch_size=batch.num_graphs)
        self.log("val_ade", ade, batch_size=batch.num_graphs)
        self.log("val_fde", fde, batch_size=batch.num_graphs)
        self.log("val_miss_rate_2yd", miss_rate, batch_size=batch.num_graphs)
        
        # Coverage Validation - filter out missing coverage labels (sentinel value -1.0)
        if self.probabilistic:
            _, _, cov_pred, _ = self.model(batch, return_distribution=True)
            
        if hasattr(batch, 'y_coverage') and batch.y_coverage is not None:
            target_cov = batch.y_coverage.view(-1, 1)
            valid_mask = (target_cov >= 0).squeeze()  # -1.0 = missing
            
            if valid_mask.any():
                loss_cov = F.binary_cross_entropy_with_logits(
                    cov_pred[valid_mask], target_cov[valid_mask]
                )
                self.log("val_loss_cov", loss_cov, batch_size=valid_mask.sum())
                 
                # Accuracy
                probs = torch.sigmoid(cov_pred[valid_mask])
                preds = (probs > 0.5).float()
                acc = (preds == target_cov[valid_mask]).float().mean()
                self.log("val_cov_acc", acc, batch_size=valid_mask.sum())
             
        return loss_traj
        
    def configure_optimizers(self):
        """
        Configure optimizer with Lion (SOTA) or AdamW fallback.
        Lion uses ~3x lower LR and higher weight decay than AdamW.
        """
        # Try Lion optimizer (faster convergence, less memory)
        try:
            from lion_pytorch import Lion
            # Lion uses lower LR (0.3x) and higher weight decay (10x) than AdamW
            optimizer = Lion(
                self.parameters(), 
                lr=self.lr * 0.3,  # Scale down LR for Lion
                weight_decay=0.01,  # Higher weight decay for Lion
                betas=(0.9, 0.99)
            )
            print("🦁 Using Lion optimizer (SOTA)")
        except ImportError:
            # Fallback to AdamW
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=1e-4
            )
            print("📈 Using AdamW optimizer")
        
        # Linear Warmup + Cosine Annealing with Restarts
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR
        
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=0.1, 
            end_factor=1.0,
            total_iters=5
        )
        
        cosine_scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,
            T_mult=2,
            eta_min=1e-7
        )
        
        combined_scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[5]
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": combined_scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }


def tune_model(num_trials=5):
    """
    Optuna hyperparameter tuning loop.
    """
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
        velocity_weight = trial.suggest_float("velocity_weight", 0.1, 0.5)
        coverage_weight = trial.suggest_float("coverage_weight", 0.3, 0.7)
        
        model = NFLGraphPredictor(
            hidden_dim=hidden_dim, 
            lr=lr,
            velocity_weight=velocity_weight,
            coverage_weight=coverage_weight
        )
        
        # Dummy Data for tuning
        return 0.5  # Dummy validation score
        
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=num_trials)
    
    print("Best params:", study.best_params)
    return study.best_params


def train_model(sanity=False, weeks=None, probabilistic=False):
    """
    Main training function.
    
    Args:
        sanity: Quick sanity check mode
        weeks: List of weeks to train on (default: [1])
        probabilistic: Use GMM probabilistic decoder
    """
    print(f"Starting training run... (Sanity Mode: {sanity}, Probabilistic: {probabilistic})")
    
    from src.data_loader import DataLoader
    from src.features import create_graph_data
    import polars as polars_df
    
    if weeks is None:
        weeks = list(range(1, 19))  # All 18 weeks by default for best accuracy
    
    loader = DataLoader(".")
    all_graphs = []
    
    try:
        for week in weeks:
            try:
                print(f"Loading week {week}...")
                df = loader.load_week_data(week) 
                df = loader.standardize_tracking_directions(df)
                
                if sanity and week == weeks[0]:
                    # Filter to first game for sanity check
                    game_ids = df["game_id"].unique().head(1).to_list()
                    df = df.filter(polars_df.col("game_id").is_in(game_ids))
                    print(f"Sanity: Filtered to {df.shape[0]} rows (1 game).")
                
                graphs = create_graph_data(df, radius=20.0, future_seq_len=10)
                all_graphs.extend(graphs)
                print(f"Week {week}: Generated {len(graphs)} frames of graph data.")
                
            except FileNotFoundError:
                print(f"Week {week} not found, skipping...")
                continue
        
        if len(all_graphs) == 0:
            print("No graph data generated. Exiting.")
            return
            
        print(f"Total: {len(all_graphs)} frames of graph data.")
        
        if sanity:
            all_graphs = all_graphs[:500]
            print(f"Sanity: Truncated to 500 frames.")
        
        # Split by play_id to prevent data leakage
        play_ids = []
        for g in all_graphs:
            if hasattr(g, 'game_id') and hasattr(g, 'play_id'):
                play_ids.append((int(g.game_id), int(g.play_id)))
            else:
                play_ids.append(None)
        
        unique_plays = list(set([p for p in play_ids if p is not None]))
        
        if len(unique_plays) > 0:
            np.random.seed(42)
            np.random.shuffle(unique_plays)
            split_idx = int(0.8 * len(unique_plays))
            train_plays = set(unique_plays[:split_idx])
            val_plays = set(unique_plays[split_idx:])
            
            train_data = [g for g, pid in zip(all_graphs, play_ids) if pid in train_plays]
            val_data = [g for g, pid in zip(all_graphs, play_ids) if pid in val_plays]
            print(f"Split by play_id: {len(train_plays)} train plays, {len(val_plays)} val plays")
            print(f"Train frames: {len(train_data)}, Val frames: {len(val_data)}")
        else:
            print("Warning: play_id not found, using simple split")
            train_len = int(0.8 * len(all_graphs))
            train_data = all_graphs[:train_len]
            val_data = all_graphs[train_len:]
        
        train_loader = PyGDataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
        val_loader = PyGDataLoader(val_data, batch_size=32, num_workers=4)
        
        # Model
        model = NFLGraphPredictor(
            input_dim=7, 
            hidden_dim=64, 
            lr=1e-3,
            probabilistic=probabilistic,
            num_modes=6 if probabilistic else 1,
            use_augmentation=not sanity  # Disable augmentation for sanity check
        )
        
        # Trainer
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        max_epochs = 50 if not sanity else 1
        limit_train_batches = 10 if sanity else 1.0
        
        callbacks = [
            EarlyStopping(
                monitor='val_ade',
                patience=7,
                mode='min',
                verbose=True
            ),
            ModelCheckpoint(
                monitor='val_ade',
                mode='min',
                save_top_k=1,
                filename='best-{epoch:02d}-{val_ade:.3f}'
            )
        ]
        
        trainer = pl.Trainer(
            max_epochs=max_epochs, 
            accelerator=accelerator, 
            log_every_n_steps=1 if sanity else 10,
            limit_train_batches=limit_train_batches,
            gradient_clip_val=1.0,
            callbacks=callbacks if not sanity else [],
            enable_progress_bar=True
        )
        
        trainer.fit(model, train_loader, val_loader)
        
        print("Training complete. Validating...")
        metrics = trainer.validate(model, val_loader)
        print("\n=== Final Metrics ===")
        print(metrics)
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "tune"], help="Mode: train or tune")
    parser.add_argument("--sanity", action="store_true", help="Run quick sanity check")
    parser.add_argument("--probabilistic", action="store_true", help="Use probabilistic GMM decoder")
    parser.add_argument("--weeks", type=int, nargs="+", default=[1], help="Weeks to train on")
    args = parser.parse_args()
    
    if args.mode == "tune":
        print("Starting Optuna Tuning...")
        tune_model(num_trials=5)
    else:
        train_model(sanity=args.sanity, weeks=args.weeks, probabilistic=args.probabilistic)
