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

class NFLGraphPredictor(pl.LightningModule):
    def __init__(self, input_dim=7, hidden_dim=64, lr=1e-3, future_seq_len=10):
        super().__init__()
        self.save_hyperparameters()
        self.model = NFLGraphTransformer(input_dim, hidden_dim, future_seq_len=future_seq_len)
        self.lr = lr

    def forward(self, data):
        return self.model(data)
        
    def training_step(self, batch, batch_idx):
        # batch is a PyG Batch object
        predictions, cov_pred, _ = self(batch)
        
        # 1. Trajectory Loss (MSE)
        y = batch.y # [Total_Nodes, Future_Seq, 2]
        loss_traj = F.mse_loss(predictions, y)
        self.log("train_traj_loss", loss_traj)
        
        # 2. Coverage Loss (BCE)
        loss_cov = 0.0
        if hasattr(batch, 'y_coverage') and batch.y_coverage is not None:
             # y_coverage: [Batch_Size, 1] (Float)
             target_cov = batch.y_coverage.view(-1, 1)
             loss_cov = F.binary_cross_entropy_with_logits(cov_pred, target_cov)
             self.log("train_cov_loss", loss_cov)
             
        # Total Loss (Weighted)
        # Weighting: Tune scalar? Start with 0.5 or 1.0 depending on magnitude.
        # MSE is approx 0.01~1.0. SCE is 0.6. Scale similar.
        loss = loss_traj + 0.5 * loss_cov
        self.log("train_loss", loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        predictions, cov_pred, _ = self(batch)
        y = batch.y
        
        # Traj Validation
        loss_traj = F.mse_loss(predictions, y)
        disp = torch.sqrt(torch.sum((predictions - y)**2, dim=-1))  # [N, T]
        
        # ADE: Average Displacement Error (mean over all timesteps)
        ade = torch.mean(disp)
        
        # FDE: Final Displacement Error (error at last timestep only)
        fde = torch.mean(disp[:, -1])
        
        # Miss Rate @ 2 yards: % of final predictions > 2 yards from ground truth
        miss_threshold = 2.0  # yards
        miss_rate = (disp[:, -1] > miss_threshold).float().mean()
        
        self.log("val_loss_traj", loss_traj)
        self.log("val_ade", ade)
        self.log("val_fde", fde)
        self.log("val_miss_rate_2yd", miss_rate)
        
        # Coverage Validation
        if hasattr(batch, 'y_coverage') and batch.y_coverage is not None:
             target_cov = batch.y_coverage.view(-1, 1)
             loss_cov = F.binary_cross_entropy_with_logits(cov_pred, target_cov)
             self.log("val_loss_cov", loss_cov)
             
             # Accuracy
             probs = torch.sigmoid(cov_pred)
             preds = (probs > 0.5).float()
             acc = (preds == target_cov).float().mean()
             self.log("val_cov_acc", acc)
             
        return loss_traj # Return primary metric or total loss? Lightning uses this for checkpointing if monitored.
        
    def configure_optimizers(self):
        # AdamW with weight decay for regularization
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=1e-4
        )
        
        # Cosine Annealing with Warm Restarts for better convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double the period after each restart
            eta_min=1e-6  # Minimum learning rate
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
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
        
        model = NFLGraphPredictor(hidden_dim=hidden_dim, lr=lr)
        
        # Dummy Data for tuning
        # In real scenario: Load data -> create_graph_data -> PyGDataLoader
        # loader = PyGDataLoader(graph_list, batch_size=32)
        # trainer.fit(model, loader)
        
        return 0.5 # Dummy validation score
        
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=num_trials)
    
    print("Best params:", study.best_params)
    return study.best_params


def train_model(sanity=False):
    print(f"Starting training run... (Sanity Mode: {sanity})")
    # Real implementation would load data here
    # Week 1 for demo
    from src.data_loader import DataLoader
    from src.features import create_graph_data
    import polars as pl # Added for sanity filtering
    
    loader = DataLoader(".")
    try:
        df = loader.load_week_data(1) 
        df = loader.standardize_tracking_directions(df)
        
        # Create Graphs
        print("Generating Graph Data (Vectorized)...")
        # If sanity, maybe load less data initially?
        # But data_loader loads whole file. We slice after creation or try to slice DF?
        # Slicing DF is risky for sequences. Let's create all graph data then slice list.
        # Week 1 is big, creating ALL graph data might take > 2-3 mins.
        # fast-path: Filter df to first N plays if sanity.
        
        if sanity:
            # Filter to first 5 games or plays
            game_ids = df["game_id"].unique().head(1)
            df = df.filter(pl.col("game_id").is_in(game_ids))
            print(f"Sanity: Filtered to {df.shape[0]} rows (1 game).")
            
        graphs = create_graph_data(df, radius=20.0, future_seq_len=10)
        print(f"Generated {len(graphs)} frames of graph data.")
        
        if sanity:
             graphs = graphs[:500]
             print(f"Sanity: Truncated to 500 frames.")
        
        # Split by play_id to prevent data leakage (frames from same play stay together)
        # Extract unique play identifiers from graphs
        play_ids = []
        for g in graphs:
            # Each graph should have game_id and play_id attributes
            if hasattr(g, 'game_id') and hasattr(g, 'play_id'):
                play_ids.append((int(g.game_id), int(g.play_id)))
            else:
                play_ids.append(None)
        
        unique_plays = list(set([p for p in play_ids if p is not None]))
        
        if len(unique_plays) > 0:
            # Shuffle plays and split 80/20
            np.random.seed(42)
            np.random.shuffle(unique_plays)
            split_idx = int(0.8 * len(unique_plays))
            train_plays = set(unique_plays[:split_idx])
            val_plays = set(unique_plays[split_idx:])
            
            train_data = [g for g, pid in zip(graphs, play_ids) if pid in train_plays]
            val_data = [g for g, pid in zip(graphs, play_ids) if pid in val_plays]
            print(f"Split by play_id: {len(train_plays)} train plays, {len(val_plays)} val plays")
            print(f"Train frames: {len(train_data)}, Val frames: {len(val_data)}")
        else:
            # Fallback to simple split if play_id not available
            print("Warning: play_id not found in graphs, using simple split (may have data leakage)")
            train_len = int(0.8 * len(graphs))
            train_data = graphs[:train_len]
            val_data = graphs[train_len:]
        
        train_loader = PyGDataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = PyGDataLoader(val_data, batch_size=32)
        
        # Model
        model = NFLGraphPredictor(input_dim=7, hidden_dim=64, lr=1e-3)
        
        # Trainer with improved settings
        # Check if GPU available
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        # Sanity: 1 epoch or even fewer steps?
        limit_train_batches = 1.0
        max_epochs = 50 if not sanity else 1
        if sanity: limit_train_batches = 10 
        
        # Callbacks for early stopping and checkpointing
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
            limit_train_batches=limit_train_batches if sanity else 1.0,
            gradient_clip_val=1.0,  # Gradient clipping for stability
            callbacks=callbacks if not sanity else [],
            enable_progress_bar=True
        )
        
        trainer.fit(model, train_loader, val_loader)
        
        print("Training complete. Validating...")
        metrics = trainer.validate(model, val_loader)
        print("\n=== Final Sanity Check Metrics ===")
        print(metrics)
        
    except Exception as e:
        print(f"Training setup failed (likely due to missing data in env): {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "tune"], help="Mode: train or tune")
    parser.add_argument("--sanity", action="store_true", help="Run quick sanity check")
    args = parser.parse_args()
    
    if args.mode == "tune":
        print("Starting Optuna Tuning...")
        tune_model(num_trials=5)
    else:
        train_model(sanity=args.sanity)
