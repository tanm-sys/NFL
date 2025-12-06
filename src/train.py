import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
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

    # ...

    # In train_model
        # Model
        model = NFLGraphPredictor(input_dim=7, hidden_dim=64, lr=1e-3)
        
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
        disp = torch.sqrt(torch.sum((predictions - y)**2, dim=-1))
        ade = torch.mean(disp)
        
        self.log("val_loss_traj", loss_traj)
        self.log("val_ade", ade)
        
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
        return torch.optim.Adam(self.parameters(), lr=self.lr)

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
        
        # Split
        train_len = int(0.8 * len(graphs))
        train_data = graphs[:train_len]
        val_data = graphs[train_len:]
        
        train_loader = PyGDataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = PyGDataLoader(val_data, batch_size=32)
        
        # Model
        model = NFLGraphPredictor(input_dim=7, hidden_dim=64, lr=1e-3)
        
        # Trainer
        # Check if GPU available
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        # Sanity: 1 epoch or even fewer steps?
        limit_train_batches = 1.0
        if sanity: limit_train_batches = 10 
        
        trainer = pl.Trainer(max_epochs=1, 
                             accelerator=accelerator, 
                             log_every_n_steps=1 if sanity else 10,
                             limit_train_batches=limit_train_batches if sanity else 1.0)
        
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
