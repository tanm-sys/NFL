"""
Competition Metrics for NFL Big Data Bowl 2026
===============================================

Novel football analytics metrics for defensive coverage evaluation.
Implements the expected ML outputs for the analytics competition.

Metrics:
- Spatial Control Probability: Field control heatmaps
- Route Anticipation Score: Defender-receiver path correlation
- Recovery Ability Index: Closing speed after initial break
- Coverage Bust Predictor: Likelihood of coverage failure
- Coverage Response Time: Delay between ball release and defender reaction
"""

import numpy as np
import polars as pl
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import pearsonr

try:
    import h3
    H3_AVAILABLE = True
except ImportError:
    H3_AVAILABLE = False


@dataclass
class TrajectoryPrediction:
    """Single player trajectory prediction with uncertainty."""
    player_id: int
    frame_id: int
    predicted_x: float
    predicted_y: float
    predicted_speed: float
    predicted_direction: float
    confidence_lower_x: float
    confidence_upper_x: float
    confidence_lower_y: float
    confidence_upper_y: float


def calculate_confidence_intervals(
    predictions: np.ndarray,
    std_predictions: Optional[np.ndarray] = None,
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate confidence intervals for trajectory predictions.
    
    Args:
        predictions: [N, T, 2] predicted coordinates
        std_predictions: [N, T, 2] optional prediction std devs
        confidence_level: Confidence level (default 95%)
        
    Returns:
        lower_bounds: [N, T, 2] lower confidence bounds
        upper_bounds: [N, T, 2] upper confidence bounds
    """
    from scipy import stats
    
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    if std_predictions is None:
        # Estimate std from prediction variance (heuristic: 5% of range)
        std_predictions = np.abs(predictions) * 0.1 + 0.5  # Min 0.5 yards
    
    lower_bounds = predictions - z_score * std_predictions
    upper_bounds = predictions + z_score * std_predictions
    
    # Clip to field bounds
    lower_bounds[..., 0] = np.clip(lower_bounds[..., 0], 0, 120)
    lower_bounds[..., 1] = np.clip(lower_bounds[..., 1], 0, 53.33)
    upper_bounds[..., 0] = np.clip(upper_bounds[..., 0], 0, 120)
    upper_bounds[..., 1] = np.clip(upper_bounds[..., 1], 0, 53.33)
    
    return lower_bounds, upper_bounds


def calculate_spatial_control_probability(
    player_positions: np.ndarray,
    player_velocities: np.ndarray,
    field_resolution: float = 1.0,
    time_horizon: float = 1.0
) -> np.ndarray:
    """
    Calculate spatial control probability for each player.
    
    Based on pitch control models (Fernandez & Bornn, 2018).
    Each field location is assigned a probability based on which player
    can reach it fastest.
    
    Args:
        player_positions: [N, 2] current x, y positions
        player_velocities: [N, 2] current vx, vy velocities
        field_resolution: Grid resolution in yards (default 1)
        time_horizon: Time horizon for reachability (seconds)
        
    Returns:
        control_map: [N, 120//res, 54//res] control probability per player
    """
    n_players = len(player_positions)
    
    # Create field grid
    x_grid = np.arange(0, 120, field_resolution)
    y_grid = np.arange(0, 53.33, field_resolution)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=-1)  # [G, 2]
    n_points = len(grid_points)
    
    # Player max speed (typical NFL: 8-10 yds/s)
    max_speed = 9.0  # yards per second
    
    # Calculate time to reach each grid point for each player
    # Simple model: time = distance / max_speed - velocity_advantage
    
    control_probs = np.zeros((n_players, n_points))
    
    for i in range(n_players):
        pos = player_positions[i]
        vel = player_velocities[i] if player_velocities is not None else np.zeros(2)
        
        # Distance to each grid point
        distances = np.linalg.norm(grid_points - pos, axis=-1)
        
        # Velocity advantage (dot product with direction to point)
        directions = grid_points - pos
        directions_norm = directions / (np.linalg.norm(directions, axis=-1, keepdims=True) + 1e-8)
        vel_advantage = np.sum(directions_norm * vel, axis=-1)
        
        # Effective time to reach
        time_to_reach = distances / max_speed - vel_advantage * 0.2
        time_to_reach = np.maximum(time_to_reach, 0.01)
        
        # Sigmoid probability based on time
        control_probs[i] = 1.0 / (1.0 + np.exp(5 * (time_to_reach - time_horizon)))
    
    # Normalize across players (softmax-like)
    control_probs = control_probs / (control_probs.sum(axis=0, keepdims=True) + 1e-8)
    
    # Reshape to field dimensions
    h, w = len(y_grid), len(x_grid)
    control_maps = control_probs.reshape(n_players, h, w)
    
    return control_maps


def spatial_control_to_h3(
    control_map: np.ndarray,
    resolution: int = 8,
    threshold: float = 0.1
) -> Dict[str, float]:
    """
    Convert spatial control map to H3 hexagon format.
    
    Args:
        control_map: [H, W] control probabilities
        resolution: H3 resolution (7-9 typical for football)
        threshold: Minimum probability to include
        
    Returns:
        Dictionary mapping H3 index to probability
    """
    if not H3_AVAILABLE:
        return {}
    
    h3_probs = {}
    h, w = control_map.shape
    
    for yi in range(h):
        for xi in range(w):
            prob = control_map[yi, xi]
            if prob >= threshold:
                # Convert field coords to lat/lng (simplified)
                # Football field: 100 yards play + 10 each end zone
                # Using arbitrary location for demo
                lat = 40.0 + yi * 0.0001  # ~11 yards per 0.001 deg
                lng = -74.0 + xi * 0.0001
                
                h3_idx = h3.latlng_to_cell(lat, lng, resolution)
                if h3_idx in h3_probs:
                    h3_probs[h3_idx] = max(h3_probs[h3_idx], prob)
                else:
                    h3_probs[h3_idx] = prob
    
    return h3_probs


def calculate_route_anticipation_score(
    defender_trajectory: np.ndarray,
    receiver_trajectory: np.ndarray
) -> float:
    """
    Calculate how well a defender's path anticipates the receiver's route.
    
    Uses path correlation and direction alignment metrics.
    
    Args:
        defender_trajectory: [T, 2] defender x, y positions over time
        receiver_trajectory: [T, 2] receiver x, y positions over time
        
    Returns:
        Score from -1 to 1 (higher = better anticipation)
    """
    if len(defender_trajectory) < 3 or len(receiver_trajectory) < 3:
        return 0.0
    
    # Calculate movement directions
    def_dirs = np.diff(defender_trajectory, axis=0)
    rec_dirs = np.diff(receiver_trajectory, axis=0)
    
    # Normalize
    def_dirs = def_dirs / (np.linalg.norm(def_dirs, axis=-1, keepdims=True) + 1e-8)
    rec_dirs = rec_dirs / (np.linalg.norm(rec_dirs, axis=-1, keepdims=True) + 1e-8)
    
    # Direction alignment (dot product)
    alignments = np.sum(def_dirs * rec_dirs, axis=-1)
    
    # Path correlation (Pearson correlation of trajectories)
    try:
        corr_x, _ = pearsonr(defender_trajectory[:, 0], receiver_trajectory[:, 0])
        corr_y, _ = pearsonr(defender_trajectory[:, 1], receiver_trajectory[:, 1])
        path_corr = (corr_x + corr_y) / 2
    except:
        path_corr = 0.0
    
    # Combine: 60% direction alignment, 40% path correlation
    anticipation_score = 0.6 * np.mean(alignments) + 0.4 * path_corr
    
    return float(np.clip(anticipation_score, -1, 1))


def calculate_recovery_ability_index(
    defender_positions: np.ndarray,
    receiver_positions: np.ndarray,
    break_frame: int = 5
) -> Dict[str, float]:
    """
    Calculate defender's ability to close separation after initial route break.
    
    Args:
        defender_positions: [T, 2] defender trajectory
        receiver_positions: [T, 2] receiver trajectory
        break_frame: Frame where route breaks (typically 0.5s after snap)
        
    Returns:
        Dictionary with recovery metrics
    """
    T = len(defender_positions)
    if T < break_frame + 5:
        return {"recovery_index": 0.0, "closing_speed_yps": 0.0, "separation_closed": 0.0}
    
    # Calculate separation over time
    separations = np.linalg.norm(
        defender_positions - receiver_positions, axis=-1
    )
    
    # Separation at break and final
    sep_at_break = separations[break_frame]
    sep_final = separations[-1]
    
    # Separation closed (positive = defender caught up)
    sep_closed = sep_at_break - sep_final
    
    # Time elapsed after break
    time_elapsed = (T - break_frame) * 0.1  # 10 Hz data
    
    # Closing speed (yards per second)
    closing_speed = sep_closed / time_elapsed if time_elapsed > 0 else 0.0
    
    # Recovery index: normalized closing ability (0-1 scale)
    # Based on typical recovery ranges (0-5 yards closed per second)
    recovery_index = np.clip(closing_speed / 5.0, -1, 1)
    
    return {
        "recovery_index": float(recovery_index),
        "closing_speed_yps": float(closing_speed),
        "separation_closed": float(sep_closed),
        "initial_separation": float(sep_at_break),
        "final_separation": float(sep_final)
    }


def predict_coverage_bust(
    defender_trajectory: np.ndarray,
    assigned_zone: Optional[Tuple[float, float, float, float]] = None,
    receiver_trajectory: Optional[np.ndarray] = None,
    coverage_type: str = "zone"
) -> Dict[str, float]:
    """
    Predict likelihood of coverage bust (defender missing assignment).
    
    Args:
        defender_trajectory: [T, 2] predicted defender positions
        assigned_zone: (x_min, y_min, x_max, y_max) for zone coverage
        receiver_trajectory: [T, 2] for man coverage
        coverage_type: "zone" or "man"
        
    Returns:
        Dictionary with bust prediction and confidence
    """
    T = len(defender_trajectory)
    
    if coverage_type == "zone" and assigned_zone is not None:
        x_min, y_min, x_max, y_max = assigned_zone
        
        # Count frames where defender is in zone
        in_zone = (
            (defender_trajectory[:, 0] >= x_min) & 
            (defender_trajectory[:, 0] <= x_max) &
            (defender_trajectory[:, 1] >= y_min) & 
            (defender_trajectory[:, 1] <= y_max)
        )
        
        zone_coverage_pct = in_zone.mean()
        
        # Bust if not in zone for significant time
        bust_prob = 1.0 - zone_coverage_pct
        bust_prediction = int(bust_prob > 0.5)
        
    elif coverage_type == "man" and receiver_trajectory is not None:
        # Man coverage: bust if separation exceeds threshold
        separations = np.linalg.norm(
            defender_trajectory - receiver_trajectory, axis=-1
        )
        
        # Average separation
        avg_sep = separations.mean()
        max_sep = separations.max()
        
        # Bust probability increases with separation
        bust_prob = np.clip((avg_sep - 3.0) / 5.0, 0, 1)  # >3 yards = increasing bust risk
        bust_prediction = int(bust_prob > 0.5)
        
    else:
        bust_prob = 0.5
        bust_prediction = 0
    
    return {
        "bust_prediction": bust_prediction,
        "bust_probability": float(bust_prob),
        "confidence": 1.0 - abs(0.5 - bust_prob) * 2  # Higher when not near 0.5
    }


def calculate_coverage_response_time(
    defender_acceleration: np.ndarray,
    ball_release_frame: int = 0
) -> float:
    """
    Calculate delay between ball release and peak defender acceleration change.
    
    Args:
        defender_acceleration: [T] acceleration magnitude over time
        ball_release_frame: Frame when ball was released
        
    Returns:
        Response time in seconds
    """
    if len(defender_acceleration) < ball_release_frame + 5:
        return np.nan
    
    # Look at acceleration after ball release
    post_release = defender_acceleration[ball_release_frame:]
    
    # Find frame of peak acceleration (max jerk)
    jerk = np.abs(np.diff(post_release))
    
    if len(jerk) == 0:
        return np.nan
    
    peak_frame = np.argmax(jerk)
    
    # Convert to seconds (10 Hz data)
    response_time = peak_frame * 0.1
    
    return float(response_time)


def generate_competition_metrics(
    play_df: pl.DataFrame,
    predictions: np.ndarray,
    prediction_std: Optional[np.ndarray] = None,
    defensive_team: str = "",
    ball_release_frame: int = 0
) -> Dict[str, any]:
    """
    Generate all competition metrics for a single play.
    
    Args:
        play_df: Play tracking data
        predictions: [N, T, 2] trajectory predictions
        prediction_std: [N, T, 2] optional prediction uncertainties
        defensive_team: Team abbreviation for defense
        ball_release_frame: Frame of ball release
        
    Returns:
        Dictionary with all competition metrics
    """
    results = {
        "trajectory_predictions": [],
        "spatial_control": {},
        "novel_metrics": {}
    }
    
    # Get player IDs
    player_ids = play_df.filter(pl.col("club") == defensive_team)["nfl_id"].unique().to_list()
    
    # Confidence intervals
    lower_bounds, upper_bounds = calculate_confidence_intervals(predictions, prediction_std)
    
    # Format trajectory predictions
    for i, pid in enumerate(player_ids[:len(predictions)]):
        for t in range(predictions.shape[1]):
            pred = TrajectoryPrediction(
                player_id=pid,
                frame_id=ball_release_frame + t,
                predicted_x=float(predictions[i, t, 0]),
                predicted_y=float(predictions[i, t, 1]),
                predicted_speed=0.0,  # Would need velocity predictions
                predicted_direction=0.0,  # Would need direction predictions
                confidence_lower_x=float(lower_bounds[i, t, 0]),
                confidence_upper_x=float(upper_bounds[i, t, 0]),
                confidence_lower_y=float(lower_bounds[i, t, 1]),
                confidence_upper_y=float(upper_bounds[i, t, 1])
            )
            results["trajectory_predictions"].append(pred)
    
    # Spatial control (for first timestep as example)
    if len(predictions) > 0:
        current_positions = predictions[:, 0, :]
        current_velocities = np.diff(predictions[:, :2, :], axis=1)[:, 0, :] * 10  # Approx velocity
        control_maps = calculate_spatial_control_probability(
            current_positions, current_velocities
        )
        results["spatial_control"]["maps"] = control_maps
        
        if H3_AVAILABLE:
            results["spatial_control"]["h3"] = [
                spatial_control_to_h3(control_maps[i]) 
                for i in range(len(control_maps))
            ]
    
    return results


def format_for_submission(
    metrics_dict: Dict[str, any],
    include_spatial: bool = True
) -> pl.DataFrame:
    """
    Format competition metrics as a Polars DataFrame for submission.
    
    Args:
        metrics_dict: Output from generate_competition_metrics
        include_spatial: Whether to include spatial control data
        
    Returns:
        Submission-ready DataFrame
    """
    rows = []
    
    for pred in metrics_dict.get("trajectory_predictions", []):
        row = {
            "player_id": pred.player_id,
            "frame_id": pred.frame_id,
            "predicted_x": pred.predicted_x,
            "predicted_y": pred.predicted_y,
            "predicted_speed": pred.predicted_speed,
            "predicted_direction": pred.predicted_direction,
            "confidence_lower_x": pred.confidence_lower_x,
            "confidence_upper_x": pred.confidence_upper_x,
            "confidence_lower_y": pred.confidence_lower_y,
            "confidence_upper_y": pred.confidence_upper_y,
        }
        rows.append(row)
    
    if not rows:
        return pl.DataFrame()
    
    return pl.DataFrame(rows)
