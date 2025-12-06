import polars as pl
import numpy as np

def calculate_distance_to_ball(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculates distance from each player to the football.
    Requires 'std_x', 'std_y' for players and ball.
    """
    # This usually requires joining the ball position back to the player rows
    # Filter for ball
    ball_df = df.filter(pl.col("display_name") == "football").select(
        ["game_id", "play_id", "frame_id", "std_x", "std_y"]
    ).rename({"std_x": "ball_x", "std_y": "ball_y"})
    
    # Join back
    df = df.join(ball_df, on=["game_id", "play_id", "frame_id"], how="left")
    
    # Calc distance
    df = df.with_columns(
        (((pl.col("std_x") - pl.col("ball_x")) ** 2 + 
          (pl.col("std_y") - pl.col("ball_y")) ** 2).sqrt())
        .alias("dist_to_ball")
    )
    
    return df

def create_baseline_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Creates simple features for baseline model:
    - Speed (s)
    - Acceleration (a)
    - Distance to ball
    - Relative orientation to ball
    """
    df = calculate_distance_to_ball(df)
    
    # Relative angle to ball
    # angle = arctan2(ball_y - y, ball_x - x)
    # rel_angle = angle - o (orientation)
    
    df = df.with_columns(
        pl.map_batches(["std_x", "std_y", "ball_x", "ball_y"], 
                       lambda s: np.arctan2(s[3] - s[1], s[2] - s[0]) * 180 / np.pi)
        .alias("angle_to_ball")
    )
    
    df = df.with_columns(
        ((pl.col("angle_to_ball") - pl.col("std_o")).mod(360))
        .alias("rel_angle_to_ball")
    )
    
    return df
