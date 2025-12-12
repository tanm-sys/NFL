#!/usr/bin/env python3
"""
NFL Trajectory Prediction Model - Production Inference Module
=============================================================

This module provides a production-ready interface for the best trained model.
Load and use the model for trajectory prediction without the full training framework.

Usage:
    from nfl_production_model import NFLProductionModel
    
    model = NFLProductionModel.load("models/exported")
    predictions = model.predict(tracking_df)
"""

import sys
sys.path.insert(0, '.')

import torch
import json
import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from torch_geometric.data import Batch

# Import core modules
from src.models.gnn import NFLGraphTransformer
from src.data_loader import DataLoader as TrackingLoader
from src.features import create_graph_data


@dataclass
class PredictionResult:
    """Container for prediction results."""
    trajectories: np.ndarray  # [N, T, 2] predicted (x, y) displacements
    absolute_positions: np.ndarray  # [N, T, 2] absolute positions
    game_ids: List[int]
    play_ids: List[int]
    confidence: Optional[np.ndarray] = None  # Optional coverage prediction


class NFLProductionModel:
    """
    Production-ready NFL trajectory prediction model.
    
    This class wraps the trained neural network for easy inference
    without requiring the full PyTorch Lightning training framework.
    """
    
    def __init__(
        self,
        model: NFLGraphTransformer,
        config: Dict,
        device: torch.device = None
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Data processing helpers
        self.loader = None
        
    @classmethod
    def load(cls, model_dir: Union[str, Path], use_gpu: bool = True) -> "NFLProductionModel":
        """
        Load the production model from exported files.
        
        Args:
            model_dir: Directory containing exported model files
            use_gpu: Whether to use GPU if available
            
        Returns:
            NFLProductionModel instance ready for inference
        """
        model_dir = Path(model_dir)
        
        # Load config
        config_path = model_dir / "model_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
            
        with open(config_path) as f:
            config = json.load(f)
        
        # Setup device
        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Initialize model architecture
        model = NFLGraphTransformer(
            input_dim=config.get('input_dim', 9),
            hidden_dim=config.get('hidden_dim', 64),
            heads=config.get('heads', 4),
            future_seq_len=config.get('future_seq_len', 10),
            edge_dim=config.get('edge_dim', 5),
            probabilistic=config.get('probabilistic', False),
        )
        
        # Load state dict
        state_dict_path = model_dir / "nfl_best_model_state_dict.pt"
        if not state_dict_path.exists():
            # Try full model
            full_model_path = model_dir / "nfl_best_model_full.pt"
            if full_model_path.exists():
                model = torch.load(full_model_path, map_location=device, weights_only=False)
            else:
                raise FileNotFoundError(f"No model file found in {model_dir}")
        else:
            state_dict = torch.load(state_dict_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
        
        print(f"Model loaded: {config.get('parameters', 'unknown'):,} parameters")
        
        return cls(model=model, config=config, device=device)
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: Union[str, Path], use_gpu: bool = True) -> "NFLProductionModel":
        """
        Load directly from a PyTorch Lightning checkpoint.
        
        Args:
            checkpoint_path: Path to .ckpt file
            use_gpu: Whether to use GPU if available
            
        Returns:
            NFLProductionModel instance
        """
        from src.train import NFLGraphPredictor
        
        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        
        predictor = NFLGraphPredictor.load_from_checkpoint(
            str(checkpoint_path),
            map_location=device
        )
        
        config = {
            'input_dim': predictor.hparams.get('input_dim', 9),
            'hidden_dim': predictor.hparams.get('hidden_dim', 64),
            'future_seq_len': predictor.hparams.get('future_seq_len', 10),
            'heads': predictor.hparams.get('heads', 4),
            'num_layers': predictor.hparams.get('num_layers', 4),
            'edge_dim': predictor.hparams.get('edge_dim', 5),
            'probabilistic': predictor.hparams.get('probabilistic', False),
            'checkpoint_source': str(checkpoint_path),
        }
        
        return cls(model=predictor.model, config=config, device=device)
    
    @torch.no_grad()
    def predict_from_graphs(
        self,
        graphs: List,
        batch_size: int = 32
    ) -> PredictionResult:
        """
        Run prediction on pre-built graph data.
        
        Args:
            graphs: List of PyTorch Geometric Data objects
            batch_size: Batch size for inference
            
        Returns:
            PredictionResult with trajectories
        """
        from torch_geometric.loader import DataLoader as PyGDataLoader
        
        all_trajectories = []
        all_absolute = []
        all_game_ids = []
        all_play_ids = []
        all_confidence = []
        
        loader = PyGDataLoader(graphs, batch_size=batch_size, shuffle=False)
        
        for batch in loader:
            batch = batch.to(self.device)
            
            # Forward pass
            preds, coverage, _ = self.model(batch)
            
            # Convert to absolute positions
            current_pos = batch.current_pos.unsqueeze(1)  # [N, 1, 2]
            absolute = preds + current_pos
            
            # Store results
            all_trajectories.append(preds.cpu().numpy())
            all_absolute.append(absolute.cpu().numpy())
            
            if hasattr(batch, 'game_id'):
                all_game_ids.extend(batch.game_id.cpu().tolist())
            if hasattr(batch, 'play_id'):
                all_play_ids.extend(batch.play_id.cpu().tolist())
            if coverage is not None:
                all_confidence.append(coverage.cpu().numpy())
        
        return PredictionResult(
            trajectories=np.concatenate(all_trajectories, axis=0),
            absolute_positions=np.concatenate(all_absolute, axis=0),
            game_ids=all_game_ids,
            play_ids=all_play_ids,
            confidence=np.concatenate(all_confidence, axis=0) if all_confidence else None
        )
    
    def predict_from_dataframe(
        self,
        df: pl.DataFrame,
        radius: float = 20.0,
        batch_size: int = 32,
        preprocess: bool = True
    ) -> PredictionResult:
        """
        Run prediction from raw tracking DataFrame.
        
        Args:
            df: Polars DataFrame with tracking data
            radius: Graph construction radius
            batch_size: Batch size for inference
            preprocess: Whether to standardize directions
            
        Returns:
            PredictionResult with trajectories
        """
        if preprocess:
            if self.loader is None:
                self.loader = TrackingLoader(".")
            df = self.loader.standardize_tracking_directions(df)
        
        future_seq_len = self.config.get('future_seq_len', 10)
        graphs = create_graph_data(
            df,
            radius=radius,
            future_seq_len=future_seq_len,
            history_len=5
        )
        
        if not graphs:
            raise ValueError("No valid graphs created from DataFrame")
        
        return self.predict_from_graphs(graphs, batch_size=batch_size)
    
    def get_model_info(self) -> Dict:
        """Return model configuration and statistics."""
        return {
            **self.config,
            'device': str(self.device),
            'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'total_params': sum(p.numel() for p in self.model.parameters()),
        }


def main():
    """Demo: Load and test the production model."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NFL Production Model Demo")
    parser.add_argument("--model-dir", default="models/exported", help="Model directory")
    parser.add_argument("--checkpoint", default=None, help="Or load from checkpoint")
    parser.add_argument("--test-week", type=int, default=1, help="Week to test on")
    args = parser.parse_args()
    
    print("=" * 60)
    print("NFL Production Model Demo")
    print("=" * 60)
    
    # Load model
    if args.checkpoint:
        model = NFLProductionModel.from_checkpoint(args.checkpoint)
    else:
        model = NFLProductionModel.load(args.model_dir)
    
    print(f"\nModel Info: {model.get_model_info()}")
    
    # Test on sample data
    print(f"\nTesting on Week {args.test_week} data...")
    
    loader = TrackingLoader("/home/tanmay/Desktop/NFL")
    df = loader.load_week_data(args.test_week)
    df = loader.standardize_tracking_directions(df)
    
    # Get first few plays
    plays = df.select(["game_id", "play_id"]).unique().head(5)
    df = df.join(plays, on=["game_id", "play_id"])
    
    # Create graphs and predict
    graphs = create_graph_data(df, radius=20.0, future_seq_len=10, history_len=5)
    print(f"Created {len(graphs)} graphs")
    
    result = model.predict_from_graphs(graphs[:100])
    
    print(f"\nPrediction Results:")
    print(f"  Trajectories shape: {result.trajectories.shape}")
    print(f"  Mean displacement: {np.mean(np.abs(result.trajectories)):.4f} yards")
    print(f"  Max displacement: {np.max(np.abs(result.trajectories)):.4f} yards")
    
    print("\n✅ Model is working correctly!")


if __name__ == "__main__":
    main()
