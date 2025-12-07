import polars as pl
import numpy as np
from scipy.spatial import ConvexHull
from typing import List, Optional

def calculate_zone_collapse_speed(play_df: pl.DataFrame, defensive_team: str) -> pl.DataFrame:
    """
    Calculates the rate of change of the defensive Convex Hull area.
    
    Args:
        play_df: DataFrame containing tracking data for a single play.
        defensive_team: Abbreviation of the defensive team (e.g., 'KC').
        
    Returns:
        DataFrame with frame_id and hull_area_rate (sq yards / sec).
    """
    # Filter for defensive players
    # Assumption: 'club' column exists and matches defensive_team
    defenders = play_df.filter(pl.col("club") == defensive_team)
    
    frames = defenders["frame_id"].unique().sort()
    
    results = []
    
    # Iterate frames (ConvexHull is fast enough for per-frame on 11 pts)
    # Optimization: could group by frame
    
    # We need pandas/numpy for Scipy
    # Group by frame
    # Note: Polars doesn't easily support arbitrary python func aggregation with scipy yet efficiently
    # converting to arrow or list might be better
    
    # Approach:
    # 1. Get list of x,y per frame
    # 2. Compute hull area
    # 3. Compute gradient
    
    pdf = defenders.select(["frame_id", "std_x", "std_y"]).to_pandas()
    
    frame_areas = {}
    
    grouped = pdf.groupby("frame_id")
    for frame, group in grouped:
        points = group[["std_x", "std_y"]].values
        if len(points) >= 3:
            try:
                hull = ConvexHull(points)
                area = hull.volume # In 2D, volume is area, area is perimeter
            except:
                area = 0.0
        else:
            area = 0.0
        frame_areas[frame] = area
        
    # Create DataFrame
    area_df = pl.DataFrame({
        "frame_id": list(frame_areas.keys()),
        "hull_area": list(frame_areas.values())
    }).sort("frame_id")
    
    # Calculate Rate of Change (derivative)
    # 10Hz data -> dt = 0.1s
    area_df = area_df.with_columns(
        (pl.col("hull_area").diff() / 0.1).alias("hull_area_rate")
    )
    
    return area_df

def calculate_defensive_reaction_time(play_df: pl.DataFrame, 
                                    ball_start_frame: int, 
                                    defensive_team: str) -> float:
    """
    Calculates average time (seconds) for defenders to reach peak acceleration change 
    after ball release.
    """
    # Filter for reaction window: 2 seconds after ball release
    reaction_window = play_df.filter(
        (pl.col("frame_id") >= ball_start_frame) & 
        (pl.col("frame_id") <= ball_start_frame + 20) &
        (pl.col("club") == defensive_team)
    )
    
    if reaction_window.height == 0:
        return np.nan
        
    # Calculate Jerk approx (diff of acceleration)
    # Using 'a' (acceleration) provided in data
    # We want magnitude of change in 'a' or just peak 'a'?
    # Reaction = change in movement state.
    # Let's look for peak `jerk` (da/dt)
    
    # Sort
    reaction_window = reaction_window.sort(["nfl_id", "frame_id"])
    
    # Calc diff of 'a'
    reaction_window = reaction_window.with_columns(
        pl.col("a").diff().abs().over("nfl_id").alias("jerk_mag")
    )
    
    # Drop nulls (first frame diff)
    reaction_window = reaction_window.drop_nulls("jerk_mag")
    
    # Find frame of max jerk for each player
    # peak_frames = reaction_window.group_by("nfl_id").agg(
    #    pl.col("jerk_mag").arg_max().alias("peak_idx") # Index within window
    # )
    
    # We need the actual frame_id corresponding to that index, or just convert index to time
    # This is tricky with arg_max in polars for relative index.
    # Simpler: Filter to where jerk_mag is max
    
    # Let's get the frame_id where jerk_mag is max per player
    # Sort by jerk_mag desc, take first
    
    max_jerks = reaction_window.sort("jerk_mag", descending=True).unique("nfl_id")
    
    # Calculate time delay
    # frame_diff = max_jerk_frame - ball_start_frame
    # time = frame_diff * 0.1
    
    # Debug print if calc yields 0 surprisingly
    # print(f"Max Jerk Frames: {max_jerks['frame_id']}")
    
    delays = (max_jerks["frame_id"] - ball_start_frame) * 0.1
    
    # Ensure positive?
    delays = delays.filter(delays >= 0)
    
    return delays.mean()


def calculate_matchup_difficulty(play_df: pl.DataFrame, 
                                  receiver_nfl_id: int,
                                  defender_nfl_id: int) -> dict:
    """
    Quantify how difficult a defensive coverage assignment is.
    Higher scores = harder matchup for defender.
    
    Args:
        play_df: DataFrame containing tracking data for a single play.
        receiver_nfl_id: NFL ID of the receiver.
        defender_nfl_id: NFL ID of the covering defender.
        
    Returns:
        Dictionary with matchup difficulty metrics.
    """
    receiver = play_df.filter(pl.col("nfl_id") == receiver_nfl_id)
    defender = play_df.filter(pl.col("nfl_id") == defender_nfl_id)
    
    if receiver.height == 0 or defender.height == 0:
        return {"matchup_difficulty": np.nan, "speed_advantage": np.nan, "separation": np.nan}
    
    # Join on frame_id
    merged = receiver.join(
        defender.select(["frame_id", "std_x", "std_y", "s"]).rename({
            "std_x": "def_x", "std_y": "def_y", "s": "def_s"
        }),
        on="frame_id",
        how="inner"
    )
    
    if merged.height == 0:
        return {"matchup_difficulty": np.nan, "speed_advantage": np.nan, "separation": np.nan}
    
    # Calculate separation over time
    merged = merged.with_columns([
        ((pl.col("std_x") - pl.col("def_x"))**2 + 
         (pl.col("std_y") - pl.col("def_y"))**2).sqrt().alias("separation"),
        (pl.col("s") - pl.col("def_s")).alias("speed_diff")
    ])
    
    # Metrics
    avg_separation = merged["separation"].mean()
    max_separation = merged["separation"].max()
    avg_speed_advantage = merged["speed_diff"].mean()
    
    # Matchup difficulty score (higher = harder for defender)
    # Combines separation advantage + speed advantage
    difficulty = (avg_separation / 5.0) + (avg_speed_advantage / 2.0)
    difficulty = np.clip(difficulty, 0, 1)
    
    return {
        "matchup_difficulty": float(difficulty),
        "speed_advantage": float(avg_speed_advantage) if avg_speed_advantage else 0.0,
        "avg_separation": float(avg_separation) if avg_separation else 0.0,
        "max_separation": float(max_separation) if max_separation else 0.0
    }


def calculate_separation_at_target(play_df: pl.DataFrame, 
                                    target_nfl_id: int,
                                    target_frame: int) -> float:
    """
    Calculate separation of targeted receiver from nearest defender at catch point.
    
    Args:
        play_df: DataFrame with tracking data.
        target_nfl_id: NFL ID of targeted receiver.
        target_frame: Frame when ball arrives/is caught.
        
    Returns:
        Separation in yards at target moment.
    """
    frame_data = play_df.filter(pl.col("frame_id") == target_frame)
    
    receiver = frame_data.filter(pl.col("nfl_id") == target_nfl_id)
    if receiver.height == 0:
        return np.nan
    
    rec_x = receiver["std_x"][0]
    rec_y = receiver["std_y"][0]
    
    # Get all defenders (side = Defense)
    defenders = frame_data.filter(pl.col("player_side") == "Defense")
    
    if defenders.height == 0:
        return np.nan
    
    # Calculate distances to all defenders
    defenders = defenders.with_columns([
        ((pl.col("std_x") - rec_x)**2 + (pl.col("std_y") - rec_y)**2).sqrt().alias("dist_to_rec")
    ])
    
    # Return minimum distance (nearest defender)
    return float(defenders["dist_to_rec"].min())


def calculate_coverage_pressure_index(play_df: pl.DataFrame,
                                       defensive_team: str,
                                       target_frame: int) -> dict:
    """
    Comprehensive coverage pressure metric combining multiple factors.
    
    Args:
        play_df: DataFrame with tracking data.
        defensive_team: Team abbreviation for defense.
        target_frame: Reference frame for measurement.
        
    Returns:
        Dictionary with pressure metrics.
    """
    frame_data = play_df.filter(pl.col("frame_id") == target_frame)
    defenders = frame_data.filter(pl.col("club") == defensive_team)
    receivers = frame_data.filter(
        (pl.col("club") != defensive_team) & 
        (pl.col("nfl_id").is_not_null())  # Exclude football
    )
    
    if defenders.height == 0 or receivers.height == 0:
        return {"pressure_index": np.nan}
    
    # For each receiver, find nearest defender
    pdf_def = defenders.select(["nfl_id", "std_x", "std_y"]).to_pandas()
    pdf_rec = receivers.select(["nfl_id", "std_x", "std_y"]).to_pandas()
    
    separations = []
    for _, rec in pdf_rec.iterrows():
        distances = np.sqrt(
            (pdf_def["std_x"] - rec["std_x"])**2 + 
            (pdf_def["std_y"] - rec["std_y"])**2
        )
        min_dist = distances.min()
        separations.append(min_dist)
    
    # Pressure metrics
    avg_separation = np.mean(separations)
    min_separation = np.min(separations)
    tight_coverage_count = sum(1 for s in separations if s < 3.0)  # < 3 yards
    
    # Pressure index (inverse of average separation, scaled)
    pressure_index = 1.0 / (1.0 + avg_separation / 5.0)
    
    return {
        "pressure_index": float(pressure_index),
        "avg_separation": float(avg_separation),
        "min_separation": float(min_separation),
        "tight_coverage_count": int(tight_coverage_count),
        "total_receivers": len(separations)
    }

