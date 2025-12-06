import sys
from pathlib import Path
sys.path.append('src')

try:
    from data_loader import DataLoader
    from features import create_baseline_features
    import polars as pl
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def test_data_loader():
    print("Testing DataLoader...")
    loader = DataLoader(".")
    df = loader.load_week_data(1)
    print(f"Loaded week 1 data: {df.height} rows")
    print(f"Columns: {df.columns}")
    assert df.height > 0
    
    # Standardize directions to get std_x/std_y
    df = loader.standardize_tracking_directions(df)
    print(f"Columns after std: {df.columns}")
    assert "std_x" in df.columns
    
    plays = loader.load_plays()
    print(f"Loaded plays data: {plays.height} rows")
    assert plays.height > 0

def test_features():
    print("Testing Features...")
    loader = DataLoader(".")
    df = loader.load_week_data(1)
    df = loader.standardize_tracking_directions(df)
    
    # Take small sample for speed
    df_sample = df.head(1000)
    # Mock football row if missing in sample (likely is)
    # Actually, let's just run it and see if it crashes on empty join if no football
    # Ideally we pick a game_id/play_id
    
    # Filter for first play
    first_game = df["game_id"][0]
    first_play = df["play_id"][0]
    play_df = df.filter((pl.col("game_id") == first_game) & (pl.col("play_id") == first_play))
    
    feats = create_baseline_features(play_df)
    print("Features created.")
    assert "dist_to_ball" in feats.columns
    assert "angle_to_ball" in feats.columns

if __name__ == "__main__":
    test_data_loader()
    test_features()
    print("Phase 1 Verification Passed!")
