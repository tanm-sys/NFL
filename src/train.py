import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
import optuna
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Batch
from src.models.gnn import NFLGraphTransformer
from src.features import create_graph_data, build_edge_index_and_attr
from src.data_loader import (
    DataLoader,
    GraphDataset,
    build_play_metadata,
    expand_play_tuples,
)
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt


class AttentionVisualizationCallback(pl.Callback):
    """
    Periodically logs attention weight histograms to TensorBoard-compatible loggers.
    """

    def __init__(self, sample_batch, log_every_n_epochs: int = 10, tag: str = "attention/histogram"):
        super().__init__()
        self.sample_batch = sample_batch
        self.log_every_n_epochs = log_every_n_epochs
        self.tag = tag

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        if self.sample_batch is None:
            return

        batch = self.sample_batch.to(pl_module.device)
        with torch.no_grad():
            if pl_module.probabilistic:
                _, _, _, attn = pl_module.model(batch, return_distribution=True, return_attention_weights=True)
            else:
                _, _, attn = pl_module.model(batch, return_attention_weights=True)

        if attn is None:
            return

        edge_index_attn, alpha = attn
        alpha_cpu = alpha.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(alpha_cpu, bins=30, color="teal", alpha=0.8)
        ax.set_title("Attention Weights")
        ax.set_xlabel("alpha")
        ax.set_ylabel("count")
        ax.grid(True, linestyle="--", alpha=0.4)

        logger = trainer.logger
        experiment = getattr(logger, "experiment", None) if logger is not None else None
        if experiment is not None and hasattr(experiment, "add_figure"):
            experiment.add_figure(self.tag, fig, global_step=trainer.global_step)
        plt.close(fig)


def velocity_loss(pred, target, mask=None):
    """
    Penalize unrealistic trajectory velocity changes.
    Encourages smooth, physically plausible trajectories.
    
    Args:
        pred: [N, T, 2] predicted positions
        target: [N, T, 2] ground truth positions
    
    Returns:
        loss: scalar velocity consistency loss
    """
    if mask is not None:
        mask = mask.bool()
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        pred = pred[mask]
        target = target[mask]

    pred_vel = pred[:, 1:] - pred[:, :-1]  # [N, T-1, 2]
    target_vel = target[:, 1:] - target[:, :-1]  # [N, T-1, 2]
    return F.mse_loss(pred_vel, target_vel)


def acceleration_loss(pred, target, mask=None):
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
    
    if mask is not None:
        mask = mask.bool()
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        pred = pred[mask]
        target = target[mask]

    pred_vel = pred[:, 1:] - pred[:, :-1]  # [N, T-1, 2]
    target_vel = target[:, 1:] - target[:, :-1]
    
    pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]  # [N, T-2, 2]
    target_acc = target_vel[:, 1:] - target_vel[:, :-1]
    
    return F.mse_loss(pred_acc, target_acc)


def collision_avoidance_loss(pred_positions, batch_idx, min_distance=1.0, valid_mask=None):
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
        if valid_mask is not None:
            mask = mask & valid_mask

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
    Apply data augmentation and rebuild graph geometry to keep edges consistent.
    """
    data = data.clone()
    pos = data.x[:, :2].clone()
    speed = data.x[:, 2].clone() if data.x.size(1) > 2 else None
    acc = data.x[:, 3].clone() if data.x.size(1) > 3 else None
    weight = data.x[:, 8].clone() if data.x.size(1) > 8 else None
    dir_raw = getattr(data, "dir_raw", None)
    ori_raw = getattr(data, "ori_raw", None)
    valid_mask = getattr(data, "valid_mask", None)
    side = data.side if hasattr(data, "side") else None

    if flip_horizontal and torch.rand(1).item() > 0.5:
        pos[:, 1] = 53.3 - pos[:, 1]
        if dir_raw is not None:
            dir_raw = (360 - dir_raw) % 360
        if ori_raw is not None:
            ori_raw = (360 - ori_raw) % 360
        if hasattr(data, "y") and data.y is not None:
            data.y[:, :, 1] = -data.y[:, :, 1]

    if add_noise:
        noise = torch.randn_like(pos) * noise_std
        pos = pos + noise
        if hasattr(data, "y") and data.y is not None:
            data.y = data.y - noise.unsqueeze(1)

    # Recompute angular features (sin/cos)
    if dir_raw is not None:
        dir_sin = torch.sin(torch.deg2rad(dir_raw))
        dir_cos = torch.cos(torch.deg2rad(dir_raw))
    else:
        if data.x.size(1) > 5:
            dir_sin = data.x[:, 4]
            dir_cos = data.x[:, 5]
        else:
            dir_sin = torch.zeros_like(pos[:, 0])
            dir_cos = torch.zeros_like(pos[:, 0])

    if ori_raw is not None:
        ori_sin = torch.sin(torch.deg2rad(ori_raw))
        ori_cos = torch.cos(torch.deg2rad(ori_raw))
    else:
        ori_sin = data.x[:, 6] if data.x.size(1) > 6 else torch.zeros_like(pos[:, 0])
        ori_cos = data.x[:, 7] if data.x.size(1) > 7 else torch.zeros_like(pos[:, 0])

    weight = weight if weight is not None else torch.zeros_like(pos[:, 0])
    acc = acc if acc is not None else torch.zeros_like(pos[:, 0])
    speed = speed if speed is not None else torch.zeros_like(pos[:, 0])

    data.x = torch.stack(
        [pos[:, 0], pos[:, 1], speed, acc, dir_sin, dir_cos, ori_sin, ori_cos, weight],
        dim=-1,
    )

    data.pos = pos
    data.current_pos = pos
    radius = float(getattr(data, "edge_radius", 20.0))
    direction_for_edges = dir_raw if dir_raw is not None else torch.zeros_like(speed)
    edge_index, edge_attr = build_edge_index_and_attr(
        pos, speed, direction_for_edges, side, radius, valid_mask=valid_mask if isinstance(valid_mask, torch.Tensor) else None
    )
    data.edge_index = edge_index
    data.edge_attr = edge_attr
    if dir_raw is not None:
        data.dir_raw = dir_raw
    if ori_raw is not None:
        data.ori_raw = ori_raw

    return data


class NFLGraphPredictor(pl.LightningModule):
    """
    PyTorch Lightning module for NFL trajectory prediction.
    Supports both deterministic and probabilistic (GMM) modes.
    """
    def __init__(self, input_dim=9, hidden_dim=64, lr=1e-3, future_seq_len=10,
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
            if isinstance(batch, Batch):
                data_list = batch.to_data_list()
                aug_list = [augment_graph(b, flip_horizontal=True, add_noise=True) for b in data_list]
                return Batch.from_data_list(aug_list)
            return augment_graph(batch, flip_horizontal=True, add_noise=True)
        return batch
        
    def training_step(self, batch, batch_idx):
        # Apply augmentation
        batch = self._augment_batch(batch)
        valid_mask = batch.valid_mask if hasattr(batch, "valid_mask") else None
        
        # Forward pass
        if self.probabilistic:
            params, mode_probs, cov_pred, _ = self.model(batch, return_distribution=True)
            
            # GMM NLL Loss
            y = batch.y  # [N, T, 2]
            if valid_mask is not None and valid_mask.sum() > 0:
                mask_nodes = valid_mask.bool()
                loss_traj = self.model.decoder.nll_loss(params[mask_nodes], mode_probs[mask_nodes], y[mask_nodes])
            elif valid_mask is None:
                loss_traj = self.model.decoder.nll_loss(params, mode_probs, y)
            else:
                loss_traj = torch.tensor(0.0, device=self.device)
            # Probability-weighted expectation for auxiliary losses
            mu = params[..., :2]  # [N, T, K, 2]
            probs_expanded = mode_probs.unsqueeze(1).unsqueeze(-1)  # [N, 1, K, 1]
            predictions = (mu * probs_expanded).sum(dim=2)
        else:
            predictions, cov_pred, _ = self(batch)
            
            # Loss: MSE or Huber (P2)
            y = batch.y
            if valid_mask is not None and valid_mask.sum() == 0:
                loss_traj = torch.tensor(0.0, device=self.device)
            else:
                target = y if valid_mask is None else y[valid_mask]
                pred_target = predictions if valid_mask is None else predictions[valid_mask]
                if self.use_huber_loss:
                    loss_traj = F.huber_loss(pred_target, target, delta=self.huber_delta)
                else:
                    loss_traj = F.mse_loss(pred_target, target)
        
        self.log("train_traj_loss", loss_traj, batch_size=batch.num_graphs)
        
        # Velocity Loss
        loss_vel = velocity_loss(predictions, y, mask=valid_mask)
        self.log("train_vel_loss", loss_vel, batch_size=batch.num_graphs)
        
        # Acceleration Loss (P1)
        loss_acc = acceleration_loss(predictions, y, mask=valid_mask)
        self.log("train_acc_loss", loss_acc, batch_size=batch.num_graphs)
        
        # Collision Avoidance Loss (P1)
        batch_idx = batch.batch if hasattr(batch, 'batch') else None
        loss_collision = collision_avoidance_loss(predictions, batch_idx, valid_mask=valid_mask)
        self.log("train_collision_loss", loss_collision, batch_size=batch.num_graphs)
        
        # Coverage Loss (BCE) - filter out missing coverage labels (sentinel value -1.0)
        loss_cov = torch.tensor(0.0, device=self.device)
        if hasattr(batch, 'y_coverage') and batch.y_coverage is not None:
            target_cov = batch.y_coverage.view(-1, 1)
            cov_mask = (target_cov >= 0).squeeze()  # -1.0 = missing
            if cov_mask.any():
                loss_cov = F.binary_cross_entropy_with_logits(
                    cov_pred[cov_mask], target_cov[cov_mask]
                )
                self.log("train_cov_loss", loss_cov, batch_size=cov_mask.sum())
             
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
            params, mode_probs, cov_pred, _ = self.model(batch, return_distribution=True)
        else:
            predictions, cov_pred, _ = self(batch)
        
        y = batch.y  # Relative targets
        valid_mask = batch.valid_mask if hasattr(batch, "valid_mask") else None
        if valid_mask is not None:
            valid_mask = valid_mask.bool()
        
        # Determine base predictions (MAP or deterministic) for primary metrics
        if self.probabilistic:
            mu = params[..., :2]  # [N, T, K, 2]
            best_mode = mode_probs.argmax(dim=-1)  # [N]
            batch_idx_nodes = torch.arange(mu.shape[0], device=mu.device)
            predictions = mu[batch_idx_nodes, :, best_mode, :]  # [N, T, 2]
        # For deterministic path predictions already defined

        # Convert predictions to absolute positions for metrics
        if hasattr(batch, 'current_pos'):
            current_pos = batch.current_pos.unsqueeze(1)  # [N, 1, 2]
            if valid_mask is not None:
                current_pos = current_pos[valid_mask]
            pred_abs = (predictions if valid_mask is None else predictions[valid_mask]) + current_pos
            target_abs = (y if valid_mask is None else y[valid_mask]) + current_pos
        else:
            pred_abs = predictions if valid_mask is None else predictions[valid_mask]
            target_abs = y if valid_mask is None else y[valid_mask]
        
        # Traj Validation (on relative for loss, absolute for metrics)
        if valid_mask is not None and valid_mask.sum() == 0:
            loss_traj = torch.tensor(0.0, device=self.device)
            disp = torch.tensor([], device=self.device)
        else:
            rel_pred = predictions if valid_mask is None else predictions[valid_mask]
            rel_target = y if valid_mask is None else y[valid_mask]
            loss_traj = F.mse_loss(rel_pred, rel_target)
            disp = torch.sqrt(torch.sum((pred_abs - target_abs)**2, dim=-1))
        
        if disp.numel() == 0:
            ade = torch.tensor(0.0, device=self.device)
            fde = torch.tensor(0.0, device=self.device)
            miss_rate = torch.tensor(0.0, device=self.device)
        else:
            ade = torch.mean(disp)
            fde = torch.mean(disp[:, -1])
            miss_threshold = 2.0
            miss_rate = (disp[:, -1] > miss_threshold).float().mean()
        
        self.log("val_loss_traj", loss_traj, batch_size=batch.num_graphs)
        self.log("val_ade", ade, batch_size=batch.num_graphs)
        self.log("val_fde", fde, batch_size=batch.num_graphs)
        self.log("val_miss_rate_2yd", miss_rate, batch_size=batch.num_graphs)

        # Stratified metrics by role and down
        if disp.numel() > 0:
            disp_local = disp  # already aligned to valid_mask
            # Role-based
            if hasattr(batch, "role"):
                roles_all = batch.role
                roles_filtered = roles_all if valid_mask is None else roles_all[valid_mask]
                for rid, name in zip([0, 1, 2, 3], ["def_cov", "other_rr", "passer", "targeted"]):
                    mask_role = roles_filtered == rid
                    if mask_role.any():
                        ade_role = disp_local[mask_role].mean()
                        fde_role = disp_local[mask_role][:, -1].mean()
                        self.log(f"val_ade_role_{name}", ade_role, batch_size=mask_role.sum())
                        self.log(f"val_fde_role_{name}", fde_role, batch_size=mask_role.sum())
            # Down-based
            if hasattr(batch, "context") and hasattr(batch, "batch"):
                down_vals = batch.context[:, 0]  # [num_graphs]
                down_nodes = down_vals[batch.batch].long()
                down_nodes = down_nodes if valid_mask is None else down_nodes[valid_mask]
                for down in [1, 2, 3, 4]:
                    mask_down = down_nodes == down
                    if mask_down.any():
                        ade_down = disp_local[mask_down].mean()
                        fde_down = disp_local[mask_down][:, -1].mean()
                        self.log(f"val_ade_down_{down}", ade_down, batch_size=mask_down.sum())
                        self.log(f"val_fde_down_{down}", fde_down, batch_size=mask_down.sum())

        # Probabilistic metrics: NLL and minADE/minFDE across modes
        if self.probabilistic:
            if valid_mask is not None and valid_mask.sum() > 0:
                mask_nodes = valid_mask.bool()
                nll_loss = self.model.decoder.nll_loss(params[mask_nodes], mode_probs[mask_nodes], y[mask_nodes])
            else:
                nll_loss = self.model.decoder.nll_loss(params, mode_probs, y)
            self.log("val_nll_loss", nll_loss, batch_size=batch.num_graphs)

            # minADE/minFDE across modes
            if hasattr(batch, 'current_pos'):
                current_pos_full = batch.current_pos.unsqueeze(1)  # [N,1,2]
                mu_abs = mu + current_pos_full.unsqueeze(2)  # [N,T,K,2]
                target_abs_full = y + current_pos_full
            else:
                mu_abs = mu
                target_abs_full = y

            if valid_mask is not None:
                mu_abs = mu_abs[valid_mask]
                target_abs_full = target_abs_full[valid_mask]

            if mu_abs.numel() > 0:
                disp_modes = torch.sqrt(torch.sum((mu_abs - target_abs_full.unsqueeze(2)) ** 2, dim=-1))  # [N,T,K]
                ade_modes = disp_modes.mean(dim=1)  # [N,K]
                fde_modes = disp_modes[:, -1, :]  # [N,K]
                min_ade = ade_modes.min(dim=1).values.mean()
                min_fde = fde_modes.min(dim=1).values.mean()
                self.log("val_minADE", min_ade, batch_size=batch.num_graphs)
                self.log("val_minFDE", min_fde, batch_size=batch.num_graphs)
        
        # Coverage Validation - filter out missing coverage labels (sentinel value -1.0)
        if self.probabilistic:
            _, _, cov_pred, _ = self.model(batch, return_distribution=True)
            
        if hasattr(batch, 'y_coverage') and batch.y_coverage is not None:
            target_cov = batch.y_coverage.view(-1, 1)
            cov_mask = (target_cov >= 0).squeeze()  # -1.0 = missing
            
            if cov_mask.any():
                loss_cov = F.binary_cross_entropy_with_logits(
                    cov_pred[cov_mask], target_cov[cov_mask]
                )
                self.log("val_loss_cov", loss_cov, batch_size=cov_mask.sum())
                 
                # Accuracy
                probs = torch.sigmoid(cov_pred[cov_mask])
                preds = (probs > 0.5).float()
                acc = (preds == target_cov[cov_mask]).float().mean()
                self.log("val_cov_acc", acc, batch_size=cov_mask.sum())
             
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
        probabilistic = trial.suggest_categorical("probabilistic", [False, True])
        num_modes = 6 if probabilistic else 1

        # Build small streaming datasets for tuning
        loader = DataLoader(".")
        history_len = 5
        future_seq_len = 10
        play_meta = build_play_metadata(loader, weeks=[1, 2], history_len=history_len, future_seq_len=future_seq_len)
        if len(play_meta) == 0:
            return float("inf")

        play_keys = [(w, g, p) for (w, g, p, n) in play_meta if n > 0]
        play_keys = sorted(play_keys)
        rng = np.random.default_rng(123)
        play_keys_shuffled = play_keys.copy()
        rng.shuffle(play_keys_shuffled)
        n_total = len(play_keys_shuffled)
        n_train = max(1, int(0.8 * n_total))
        train_keys = set(play_keys_shuffled[:n_train])
        val_keys = set(play_keys_shuffled[n_train:])

        train_tuples = expand_play_tuples(play_meta, allowed_plays=train_keys)
        val_tuples = expand_play_tuples(play_meta, allowed_plays=val_keys)
        if len(train_tuples) == 0 or len(val_tuples) == 0:
            return float("inf")

        cache_dir = Path("./cache/graphs_optuna")
        train_ds = GraphDataset(
            loader,
            train_tuples,
            radius=20.0,
            future_seq_len=future_seq_len,
            history_len=history_len,
            cache_dir=cache_dir,
            persist_cache=True,
            in_memory_cache_size=3,
        )
        val_ds = GraphDataset(
            loader,
            val_tuples,
            radius=20.0,
            future_seq_len=future_seq_len,
            history_len=history_len,
            cache_dir=cache_dir,
            persist_cache=True,
            in_memory_cache_size=2,
        )

        train_loader = PyGDataLoader(train_ds, batch_size=24, shuffle=True, num_workers=2)
        val_loader = PyGDataLoader(val_ds, batch_size=24, shuffle=False, num_workers=2)
        
        model = NFLGraphPredictor(
            hidden_dim=hidden_dim, 
            lr=lr,
            velocity_weight=velocity_weight,
            coverage_weight=coverage_weight,
            input_dim=9,
            probabilistic=probabilistic,
            num_modes=num_modes,
            use_augmentation=False,
        )
        
        trainer = pl.Trainer(
            max_epochs=5,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            limit_train_batches=0.5,
            limit_val_batches=1.0,
        )

        trainer.fit(model, train_loader, val_loader)
        metrics = trainer.validate(model, val_loader, verbose=False)
        val_ade = metrics[0].get("val_ade", float("inf")) if metrics else float("inf")
        return val_ade
        
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=num_trials)
    
    print("Best params:", study.best_params)
    return study.best_params


def train_model(sanity=False, weeks=None, probabilistic=False, split_path: str = "./outputs/splits.json"):
    """
    Main training function.
    
    Args:
        sanity: Quick sanity check mode
        weeks: List of weeks to train on (default: [1])
        probabilistic: Use GMM probabilistic decoder
    """
    print(f"Starting training run... (Sanity Mode: {sanity}, Probabilistic: {probabilistic})")
    
    if weeks is None:
        weeks = list(range(1, 19))  # All 18 weeks by default for best accuracy
    
    loader = DataLoader(".")
    history_len = 5
    future_seq_len = 10
    split_path = Path(split_path)

    try:
        play_meta = build_play_metadata(loader, weeks, history_len, future_seq_len)
        if len(play_meta) == 0:
            print("No play metadata found. Exiting.")
            return

        # Build deterministic play list and split
        unique_play_keys = [(w, g, p) for (w, g, p, n) in play_meta if n > 0]
        unique_play_keys = sorted(unique_play_keys)

        if split_path.exists():
            with open(split_path, "r") as f:
                split_data = json.load(f)
            train_keys = {tuple(x) for x in split_data.get("train", [])}
            val_keys = {tuple(x) for x in split_data.get("val", [])}
            test_keys = {tuple(x) for x in split_data.get("test", [])}
            print(f"Loaded splits from {split_path}")
        else:
            rng = np.random.default_rng(42)
            play_keys_shuffled = unique_play_keys.copy()
            rng.shuffle(play_keys_shuffled)
            n_total = len(play_keys_shuffled)
            n_train = int(0.8 * n_total)
            n_val = int(0.1 * n_total)
            train_keys = set(play_keys_shuffled[:n_train])
            val_keys = set(play_keys_shuffled[n_train : n_train + n_val])
            test_keys = set(play_keys_shuffled[n_train + n_val :])
            split_path.parent.mkdir(parents=True, exist_ok=True)
            with open(split_path, "w") as f:
                json.dump(
                    {
                        "train": [list(x) for x in train_keys],
                        "val": [list(x) for x in val_keys],
                        "test": [list(x) for x in test_keys],
                    },
                    f,
                    indent=2,
                )
            print(f"Saved splits to {split_path}")

        if sanity:
            # limit to small subset of plays for speed
            train_keys = set(list(train_keys)[:20])
            val_keys = set(list(val_keys)[:5])

        train_tuples = expand_play_tuples(play_meta, allowed_plays=train_keys)
        val_tuples = expand_play_tuples(play_meta, allowed_plays=val_keys)

        if len(train_tuples) == 0 or len(val_tuples) == 0:
            print("No graph data generated for splits. Exiting.")
            return

        cache_dir = Path("./cache/graphs")
        train_dataset = GraphDataset(
            loader,
            train_tuples,
            radius=20.0,
            future_seq_len=future_seq_len,
            history_len=history_len,
            cache_dir=cache_dir,
            persist_cache=True,
            in_memory_cache_size=4,
        )
        val_dataset = GraphDataset(
            loader,
            val_tuples,
            radius=20.0,
            future_seq_len=future_seq_len,
            history_len=history_len,
            cache_dir=cache_dir,
            persist_cache=True,
            in_memory_cache_size=2,
        )

        train_loader = PyGDataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = PyGDataLoader(val_dataset, batch_size=32, num_workers=4)
        sample_batch = next(iter(val_loader)) if len(val_dataset) > 0 else None
        
        # Model
        model = NFLGraphPredictor(
            input_dim=9, 
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
            ),
            StochasticWeightAveraging(
                swa_lrs=1e-5,
                swa_epoch_start=int(max_epochs * 0.75),
                annealing_epochs=5,
            ),
        ]
        if not sanity and sample_batch is not None:
            callbacks.append(AttentionVisualizationCallback(sample_batch, log_every_n_epochs=10))
        
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
