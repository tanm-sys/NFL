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
