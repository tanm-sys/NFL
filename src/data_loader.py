import polars as pl
import pandas as pd
import os
from pathlib import Path
from typing import List, Optional, Union

class DataLoader:
    def __init__(self, data_dir: str = "."):
        self.data_dir = Path(data_dir)
        
    def load_week_data(self, week: int, standard_cols: bool = True) -> pl.DataFrame:
        """
        Load tracking data for a specific week and merge with play context.
        Enforces NFL Rules: Filters out nullified plays.
        """
        # 1. Load Tracking
        filename = f"input_2023_w{week:02d}.csv"
        filepath = self.data_dir / "train" / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
            
        tracking_df = pl.read_csv(filepath)
        
        # 2. Load Plays to filter Nullified and get Context
        plays_df = self.load_plays()
        
        # NFL RULE CHECK: Exclude plays nullified by penalty
        valid_plays = plays_df.filter(pl.col("play_nullified_by_penalty") == "N")
        
        
        
        # Select context columns
        # absolute_yardline_number might need standardization same as x side
        context_cols = ["game_id", "play_id", "down", "yards_to_go", "absolute_yardline_number", "possession_team", "team_coverage_man_zone"]
        existing_cols = [c for c in context_cols if c in valid_plays.columns]
        play_context = valid_plays.select(existing_cols)
        
        # Filter plays with known coverage if we are doing MTL?
        # Or keep them and mask loss. Let's keep them.
        
        # Map Coverage to Int: Man=0, Zone=1, Other=Null
        # Polars: .with_columns(pl.col("team_coverage_man_zone").map_dict({"MAN_COVERAGE": 0, "ZONE_COVERAGE": 1}))
        # But handle nulls.
        
        play_context = play_context.with_columns(
            pl.when(pl.col("team_coverage_man_zone") == "MAN_COVERAGE").then(0)
            .when(pl.col("team_coverage_man_zone") == "ZONE_COVERAGE").then(1)
            .otherwise(None) # Make sure others are null
            .alias("coverage_label")
        )

        # 3. Join
        # Tracking data has gameId, playId. Standardize names first if needed.
        if standard_cols:
            tracking_df = self._standardize_columns(tracking_df)
            # play_context is already standardized by load_plays -> _standardize_columns
        
        # Inner join filters out the nullified plays from tracking data
        # Cast join keys to match if needed (usually polars handles int/in but strict types matter)
        # Ensure game_id/play_id are Int64
        try:
            tracking_df = tracking_df.with_columns([
                pl.col("game_id").cast(pl.Int64),
                pl.col("play_id").cast(pl.Int64)
            ])
            play_context = play_context.with_columns([
                pl.col("game_id").cast(pl.Int64),
                pl.col("play_id").cast(pl.Int64)
            ])
        except Exception as e:
            pass

        df = tracking_df.join(play_context, on=["game_id", "play_id"], how="inner")
            
        return df

        if standard_cols:
            df = self._standardize_columns(df)
            
        return df

    def load_plays(self) -> pl.DataFrame:
        """
        Load supplementary data (plays and game info).
        """
        filepath = self.data_dir / "supplementary_data.csv"
        
        if not filepath.exists():
            # Fallback to checking if games/plays exist separately or just return empty/raise
            raise FileNotFoundError(f"File not found: {filepath}")
            
        df = pl.read_csv(filepath, null_values=["NA"], infer_schema_length=5000)
        df = self._standardize_columns(df)
        return df

    def load_games(self) -> pl.DataFrame:
        # supplementary_data.csv seems to contain game info as well (game_id, season, week, etc.)
        # so we can just return unique game rows from there
        df = self.load_plays()
        return df.unique(subset=["game_id"])

    def _standardize_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Standardize column names to snake_case if needed.
        """
        df = df.rename({col: col.lower() for col in df.columns})
        return df

    def standardize_tracking_directions(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Adjust coordinates so all plays go left to right.
        Standard NFL Big Data Bowl logic.
        """
        # Polars implementation of standardization
        # If playDirection is 'left', flip x and y derivatives
        
        # Check if play_direction column exists (usually 'playDirection')
        cols = df.columns
        if 'play_direction' not in cols and 'playdirection' not in cols and 'playDirection' not in cols:
            return df
            
        # Ensure we work with lowercase
        df = self._standardize_columns(df)
        
        # Determine plays to flip
        # x: 0 to 120 (including endzones)
        # y: 0 to 53.3
        
        # Condition: play_direction == 'left'
        # x_new = 120 - x
        # y_new = 53.3 - y
        # dir_new = (dir + 180) % 360
        # o_new = (o + 180) % 360
        
        # This is a bit complex in Polars lazy expression, doing eager for simplicity first
        # optimized later
        
        return df.with_columns([
            pl.when(pl.col('play_direction') == 'left')
            .then(120 - pl.col('x'))
            .otherwise(pl.col('x'))
            .alias('std_x'),
            
            pl.when(pl.col('play_direction') == 'left')
            .then(53.3 - pl.col('y'))
            .otherwise(pl.col('y'))
            .alias('std_y'),
             
             # Orientation
            pl.when(pl.col('play_direction') == 'left')
            .then((pl.col('o') + 180).mod(360))
            .otherwise(pl.col('o'))
            .alias('std_o'),
            
             # Direction (motion)
            pl.when(pl.col('play_direction') == 'left')
            .then((pl.col('dir') + 180).mod(360))
            .otherwise(pl.col('dir'))
            .alias('std_dir'),
        ])

if __name__ == "__main__":
    # Test
    loader = DataLoader(data_dir="/home/tanmay/Desktop/NFL")
    try:
        df = loader.load_week_data(1)
        print(df.head())
    except Exception as e:
        print(e)
