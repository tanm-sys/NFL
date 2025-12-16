"""
Competition Output Generator for NFL Big Data Bowl 2026
========================================================

Main entry point for generating all competition deliverables:
1. Trajectory predictions with confidence intervals
2. Novel football metrics
3. Aggregated statistics
4. Submission-ready DataFrames
"""

import torch
import polars as pl
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.train import NFLGraphPredictor
from src.data_loader import DataLoader as TrackingLoader, GraphDataset, build_play_metadata, expand_play_tuples
from src.competition_metrics import (
    calculate_confidence_intervals,
    calculate_recovery_ability_index,
    predict_coverage_bust,
)
from src.metrics import (
    calculate_zone_collapse_speed,
    calculate_defensive_reaction_time,
    calculate_coverage_pressure_index,
)
from torch_geometric.loader import DataLoader as PyGDataLoader


@dataclass
class PlayPrediction:
    """Complete prediction output for a single play."""
    game_id: int
    play_id: int
    trajectories: List[Dict]  # Per-player trajectory predictions
    zone_collapse_speed: Optional[pl.DataFrame]
    reaction_time: Optional[float]
    coverage_pressure: Optional[Dict]
    route_anticipation_scores: Optional[Dict[int, float]]
    recovery_indices: Optional[Dict[int, Dict]]
    bust_predictions: Optional[Dict[int, Dict]]


def generate_trajectory_predictions(
    model: NFLGraphPredictor,
    batch,
    device: torch.device
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Generate trajectory predictions with uncertainty estimates.
    
    Args:
        model: Trained NFLGraphPredictor
        batch: PyG batch
        device: Compute device
        
    Returns:
        predictions: [N, T, 2] absolute positions
        std_predictions: [N, T, 2] optional uncertainty estimates
        current_pos: [N, 2] starting positions
    """
    model.eval()
    batch = batch.to(device)
    
    with torch.no_grad():
        if model.probabilistic:
            # Get GMM predictions with uncertainty
            params, mode_probs, _, _ = model.model(batch, return_distribution=True)
            
            # mu and sigma from GMM
            mu = params[..., :2]  # [N, T, K, 2]
            sigma = params[..., 2:4]  # [N, T, K, 2]
            
            # Best mode selection
            best_mode = mode_probs.argmax(dim=-1)  # [N]
            idx = torch.arange(mu.size(0), device=device)
            
            predictions = mu[idx, :, best_mode, :]  # [N, T, 2]
            std_predictions = sigma[idx, :, best_mode, :]  # [N, T, 2]
            
        else:
            # Deterministic predictions
            predictions, _, _ = model(batch)
            std_predictions = None
    
    # Convert to absolute positions
    current_pos = batch.current_pos  # [N, 2]
    pred_abs = predictions + current_pos.unsqueeze(1)
    
    # Convert to numpy
    pred_np = pred_abs.cpu().numpy()
    std_np = std_predictions.cpu().numpy() if std_predictions is not None else None
    pos_np = current_pos.cpu().numpy()
    
    return pred_np, std_np, pos_np


def generate_play_output(
    model: NFLGraphPredictor,
    play_df: pl.DataFrame,
    batch,
    device: torch.device,
    defensive_team: str = "",
    ball_release_frame: int = 0
) -> PlayPrediction:
    """
    Generate complete competition output for a single play.
    
    Args:
        model: Trained model
        play_df: Play tracking data
        batch: PyG batch for the play
        device: Compute device
        defensive_team: Defensive team abbreviation
        ball_release_frame: Frame of ball release
        
    Returns:
        PlayPrediction with all metrics
    """
    # Get game/play IDs
    game_id = int(play_df["game_id"][0]) if "game_id" in play_df.columns else -1
    play_id = int(play_df["play_id"][0]) if "play_id" in play_df.columns else -1
    
    # Generate trajectory predictions
    predictions, std_predictions, current_pos = generate_trajectory_predictions(
        model, batch, device
    )
    
    # Confidence intervals
    lower_bounds, upper_bounds = calculate_confidence_intervals(
        predictions, std_predictions
    )
    
    # Get player IDs from data
    player_ids = play_df["nfl_id"].unique().to_list()[:len(predictions)]
    
    # Format trajectories
    trajectories = []
    for i, pid in enumerate(player_ids):
        for t in range(predictions.shape[1]):
            traj = {
                "player_id": int(pid),
                "frame_id": ball_release_frame + t,
                "predicted_x": float(predictions[i, t, 0]),
                "predicted_y": float(predictions[i, t, 1]),
                "predicted_speed": 0.0,  # TODO: derive from trajectory
                "predicted_direction": 0.0,  # TODO: derive from trajectory
                "confidence_lower_x": float(lower_bounds[i, t, 0]),
                "confidence_upper_x": float(upper_bounds[i, t, 0]),
                "confidence_lower_y": float(lower_bounds[i, t, 1]),
                "confidence_upper_y": float(upper_bounds[i, t, 1]),
            }
            trajectories.append(traj)
    
    # Zone collapse speed
    try:
        zone_collapse = calculate_zone_collapse_speed(play_df, defensive_team)
    except Exception:
        zone_collapse = None
    
    # Reaction time
    try:
        reaction_time = calculate_defensive_reaction_time(
            play_df, ball_release_frame, defensive_team
        )
    except Exception:
        reaction_time = None
    
    # Coverage pressure
    try:
        coverage_pressure = calculate_coverage_pressure_index(
            play_df, defensive_team, ball_release_frame
        )
    except Exception:
        coverage_pressure = None
    
    # Route anticipation (defender vs receiver pairs)
    route_anticipation = {}
    # Would need receiver-defender matching logic here
    
    # Recovery indices
    recovery_indices = {}
    for i, pid in enumerate(player_ids):
        try:
            # Use predicted trajectory as defender
            recovery = calculate_recovery_ability_index(
                predictions[i],
                np.zeros_like(predictions[i])  # Placeholder receiver
            )
            recovery_indices[int(pid)] = recovery
        except Exception:
            pass
    
    # Bust predictions
    bust_predictions = {}
    for i, pid in enumerate(player_ids):
        try:
            bust = predict_coverage_bust(
                predictions[i],
                coverage_type="zone"
            )
            bust_predictions[int(pid)] = bust
        except Exception:
            pass
    
    return PlayPrediction(
        game_id=game_id,
        play_id=play_id,
        trajectories=trajectories,
        zone_collapse_speed=zone_collapse,
        reaction_time=reaction_time,
        coverage_pressure=coverage_pressure,
        route_anticipation_scores=route_anticipation if route_anticipation else None,
        recovery_indices=recovery_indices if recovery_indices else None,
        bust_predictions=bust_predictions if bust_predictions else None,
    )


def generate_submission(
    checkpoint_path: str,
    data_dir: str = ".",
    weeks: List[int] = [1],
    output_path: str = "submission.csv",
    batch_size: int = 32,
) -> pl.DataFrame:
    """
    Generate competition submission file.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        data_dir: Directory with tracking data
        weeks: Weeks to process
        output_path: Where to save submission
        batch_size: Batch size for inference
        
    Returns:
        Submission DataFrame
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = NFLGraphPredictor.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model.to(device)
    
    future_seq_len = model.hparams.get("future_seq_len", 10)
    history_len = 5
    
    # Load data
    loader = TrackingLoader(data_dir)
    play_meta = build_play_metadata(loader, weeks, history_len, future_seq_len)
    
    if len(play_meta) == 0:
        print("No plays found.")
        return pl.DataFrame()
    
    # Process all plays
    all_trajectories = []
    
    play_tuples = expand_play_tuples(play_meta)
    
    dataset = GraphDataset(
        loader=loader,
        play_tuples=play_tuples,
        radius=20.0,
        future_seq_len=future_seq_len,
        history_len=history_len,
    )
    
    dataloader = PyGDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Single-threaded for inference
    )
    
    print(f"Processing {len(dataset)} samples...")
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            # Get predictions
            predictions, std_predictions, _ = generate_trajectory_predictions(
                model, batch, device
            )
            
            # Confidence intervals
            lower, upper = calculate_confidence_intervals(predictions, std_predictions)
            
            # Format output
            for i in range(len(predictions)):
                game_id = int(batch.game_id[i]) if hasattr(batch, "game_id") else -1
                play_id = int(batch.play_id[i]) if hasattr(batch, "play_id") else -1
                
                for t in range(predictions.shape[1]):
                    all_trajectories.append({
                        "game_id": game_id,
                        "play_id": play_id,
                        "node_idx": i,
                        "frame_id": t,
                        "predicted_x": float(predictions[i, t, 0]),
                        "predicted_y": float(predictions[i, t, 1]),
                        "confidence_lower_x": float(lower[i, t, 0]),
                        "confidence_upper_x": float(upper[i, t, 0]),
                        "confidence_lower_y": float(lower[i, t, 1]),
                        "confidence_upper_y": float(upper[i, t, 1]),
                    })
    
    # Create DataFrame
    submission_df = pl.DataFrame(all_trajectories)
    
    # Save
    submission_df.write_csv(output_path)
    print(f"Submission saved to {output_path}")
    print(f"Total predictions: {len(submission_df)}")
    
    return submission_df


def generate_metrics_report(
    play_predictions: List[PlayPrediction],
    output_path: str = "metrics_report.csv"
) -> pl.DataFrame:
    """
    Generate aggregated metrics report from play predictions.
    
    Args:
        play_predictions: List of PlayPrediction objects
        output_path: Where to save report
        
    Returns:
        Metrics DataFrame
    """
    metrics_rows = []
    
    for pred in play_predictions:
        row = {
            "game_id": pred.game_id,
            "play_id": pred.play_id,
            "reaction_time": pred.reaction_time,
            "num_trajectories": len(pred.trajectories),
        }
        
        if pred.coverage_pressure:
            row["pressure_index"] = pred.coverage_pressure.get("pressure_index")
            row["avg_separation"] = pred.coverage_pressure.get("avg_separation")
        
        if pred.bust_predictions:
            bust_probs = [v.get("bust_probability", 0) for v in pred.bust_predictions.values()]
            row["avg_bust_probability"] = np.mean(bust_probs) if bust_probs else None
        
        if pred.recovery_indices:
            recovery_vals = [v.get("recovery_index", 0) for v in pred.recovery_indices.values()]
            row["avg_recovery_index"] = np.mean(recovery_vals) if recovery_vals else None
        
        metrics_rows.append(row)
    
    df = pl.DataFrame(metrics_rows)
    df.write_csv(output_path)
    print(f"Metrics report saved to {output_path}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate NFL competition outputs")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--data-dir", type=str, default=".", help="Data directory")
    parser.add_argument("--weeks", type=int, nargs="+", default=[1], help="Weeks to process")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output file")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    
    generate_submission(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        weeks=args.weeks,
        output_path=args.output,
        batch_size=args.batch_size,
    )
