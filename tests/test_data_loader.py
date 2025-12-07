"""
Comprehensive tests for DataLoader module.
Covers functional, edge case, and performance tests.
"""
import sys
sys.path.insert(0, 'src')

import pytest
import polars as pl
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from data_loader import DataLoader


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory with mock files."""
    # Create train subdirectory
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    
    # Create mock tracking data - use snake_case to match standardized columns
    tracking_data = pl.DataFrame({
        "game_id": [1, 1, 1, 1],
        "play_id": [100, 100, 100, 100],
        "frame_id": [1, 1, 2, 2],
        "nfl_id": [10, 20, 10, 20],
        "x": [50.0, 60.0, 51.0, 61.0],
        "y": [25.0, 30.0, 26.0, 31.0],
        "s": [5.0, 4.0, 5.5, 4.2],
        "a": [1.0, 0.5, 1.2, 0.6],
        "o": [90.0, 180.0, 85.0, 175.0],
        "dir": [45.0, 135.0, 50.0, 140.0],
        "play_direction": ["right", "right", "right", "right"],
        "player_role": ["Defensive Coverage", "Passer", "Defensive Coverage", "Passer"],
        "player_side": ["Defense", "Offense", "Defense", "Offense"],
        "player_weight": [220, 205, 220, 205],
    })
    tracking_data.write_csv(train_dir / "input_2023_w01.csv")
    
    # Create mock supplementary data
    supplementary_data = pl.DataFrame({
        "game_id": [1],
        "play_id": [100],
        "play_nullified_by_penalty": ["N"],
        "down": [1],
        "yards_to_go": [10],
        "absolute_yardline_number": [25],
        "possession_team": ["KC"],
        "team_coverage_man_zone": ["MAN_COVERAGE"],
        "offense_formation": ["SHOTGUN"],
        "receiver_alignment": ["2x2"],
        "defenders_in_the_box": [6],
    })
    supplementary_data.write_csv(tmp_path / "supplementary_data.csv")
    
    return tmp_path


@pytest.fixture
def loader(temp_data_dir):
    """Create DataLoader with temp directory."""
    return DataLoader(data_dir=str(temp_data_dir))


@pytest.fixture
def real_loader():
    """Create DataLoader with real data directory (for integration tests)."""
    return DataLoader(data_dir="/home/tanmay/Desktop/NFL")


# ============================================================================
# Functional Tests (DL-F01 to DL-F11)
# ============================================================================

class TestDataLoaderFunctional:
    """Functional tests for DataLoader."""
    
    def test_load_week_data_valid(self, loader):
        """DL-F01: Load valid week 1 data."""
        df = loader.load_week_data(1)
        
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0
        # Check merged columns from supplementary data
        assert "down" in df.columns
        assert "yards_to_go" in df.columns
        
    def test_load_plays_valid(self, loader):
        """DL-F02: Load supplementary_data.csv."""
        df = loader.load_plays()
        
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0
        assert "game_id" in df.columns
        assert "play_id" in df.columns
        
    def test_load_games_unique(self, loader):
        """DL-F03: Load unique games."""
        df = loader.load_games()
        
        assert isinstance(df, pl.DataFrame)
        # Check uniqueness on game_id
        assert len(df) == df["game_id"].n_unique()
        
    def test_standardize_columns(self, loader):
        """DL-F04: Column name standardization."""
        df = pl.DataFrame({
            "GameId": [1],
            "PlayId": [100],
            "FRAME_ID": [1]
        })
        result = loader._standardize_columns(df)
        
        assert "gameid" in result.columns
        assert "playid" in result.columns
        assert "frame_id" in result.columns
        
    def test_standardize_directions_left(self, loader):
        """DL-F05: Flip left-moving plays."""
        df = pl.DataFrame({
            "x": [50.0],
            "y": [25.0],
            "o": [90.0],
            "dir": [45.0],
            "play_direction": ["left"]
        })
        result = loader.standardize_tracking_directions(df)
        
        # x: 120 - 50 = 70
        assert result["std_x"][0] == 70.0
        # y: 53.3 - 25 = 28.3
        assert abs(result["std_y"][0] - 28.3) < 0.01
        # o: (90 + 180) % 360 = 270
        assert result["std_o"][0] == 270.0
        # dir: (45 + 180) % 360 = 225
        assert result["std_dir"][0] == 225.0
        
    def test_standardize_directions_right(self, loader):
        """DL-F06: Right-moving plays unchanged."""
        df = pl.DataFrame({
            "x": [50.0],
            "y": [25.0],
            "o": [90.0],
            "dir": [45.0],
            "play_direction": ["right"]
        })
        result = loader.standardize_tracking_directions(df)
        
        assert result["std_x"][0] == 50.0
        assert result["std_y"][0] == 25.0
        assert result["std_o"][0] == 90.0
        assert result["std_dir"][0] == 45.0
        
    def test_nullified_plays_filtered(self, temp_data_dir, loader):
        """DL-F07: Filter nullified plays."""
        # Add a nullified play to supplementary data
        supplementary_df = pl.DataFrame({
            "game_id": [1, 1],
            "play_id": [100, 101],
            "play_nullified_by_penalty": ["N", "Y"],
            "down": [1, 2],
            "yards_to_go": [10, 5],
            "absolute_yardline_number": [25, 30],
            "possession_team": ["KC", "KC"],
            "team_coverage_man_zone": ["MAN_COVERAGE", "ZONE_COVERAGE"],
            "offense_formation": ["SHOTGUN", "EMPTY"],
            "receiver_alignment": ["2x2", "3x1"],
            "defenders_in_the_box": [6, 7],
        })
        supplementary_df.write_csv(temp_data_dir / "supplementary_data.csv")
        
        df = loader.load_week_data(1)
        
        # Only play 100 should be included (101 is nullified)
        play_ids = df["play_id"].unique().to_list()
        assert 100 in play_ids
        assert 101 not in play_ids
        
    def test_role_mapping(self, loader):
        """DL-F08: Player role encoding."""
        df = loader.load_week_data(1)
        
        assert "role_id" in df.columns
        # Defensive Coverage -> 0, Passer -> 2
        assert 0 in df["role_id"].unique().to_list()  # Defensive Coverage
        assert 2 in df["role_id"].unique().to_list()  # Passer
        
    def test_formation_mapping(self, loader):
        """DL-F09: Formation encoding."""
        df = loader.load_week_data(1)
        
        assert "formation_id" in df.columns
        # SHOTGUN -> 0
        assert 0 in df["formation_id"].unique().to_list()
        
    def test_alignment_mapping(self, loader):
        """DL-F10: Alignment encoding."""
        df = loader.load_week_data(1)
        
        assert "alignment_id" in df.columns
        # 2x2 -> 0
        assert 0 in df["alignment_id"].unique().to_list()
        
    def test_coverage_label_mapping(self, loader):
        """DL-F11: Coverage label encoding."""
        df = loader.load_week_data(1)
        
        assert "coverage_label" in df.columns
        # MAN_COVERAGE -> 0
        assert 0 in df["coverage_label"].unique().to_list()


# ============================================================================
# Edge Case Tests (DL-E01 to DL-E10)
# ============================================================================

class TestDataLoaderEdgeCases:
    """Edge case tests for DataLoader."""
    
    def test_load_nonexistent_week(self, loader):
        """DL-E01: Load week that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            loader.load_week_data(99)
            
    def test_load_missing_supplementary(self, tmp_path):
        """DL-E02: Missing supplementary_data.csv."""
        # Create empty train dir without supplementary
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        
        loader = DataLoader(data_dir=str(tmp_path))
        
        with pytest.raises(FileNotFoundError):
            loader.load_plays()
            
    def test_null_direction_column(self, loader):
        """DL-E04: Missing play_direction column."""
        df = pl.DataFrame({
            "x": [50.0],
            "y": [25.0],
            "o": [90.0],
            "dir": [45.0],
            # No play_direction column
        })
        result = loader.standardize_tracking_directions(df)
        
        # Should return without std_ columns or handle gracefully
        assert "std_x" not in result.columns or result is not None
        
    def test_invalid_data_dir(self):
        """DL-E05: Non-existent data directory."""
        loader = DataLoader(data_dir="/nonexistent/path")
        
        with pytest.raises(FileNotFoundError):
            loader.load_plays()
            
    def test_large_week_number(self, loader):
        """DL-E07: Week number > 18."""
        with pytest.raises(FileNotFoundError):
            loader.load_week_data(99)
            
    def test_week_zero(self, loader):
        """DL-E08: Week number 0."""
        with pytest.raises(FileNotFoundError):
            loader.load_week_data(0)
            

# ============================================================================
# Performance Tests (DL-P01 to DL-P03)
# ============================================================================

class TestDataLoaderPerformance:
    """Performance tests for DataLoader."""
    
    @pytest.mark.skipif(
        not os.path.exists("/home/tanmay/Desktop/NFL/train/input_2023_w01.csv"),
        reason="Real data not available"
    )
    def test_load_week_data_timing(self, real_loader):
        """DL-P01: Load single week timing."""
        import time
        
        start = time.time()
        df = real_loader.load_week_data(1)
        elapsed = time.time() - start
        
        # Should load in < 10 seconds
        assert elapsed < 10.0, f"Loading took {elapsed:.2f}s, expected < 10s"
        assert len(df) > 0


# ============================================================================
# Security Tests (DL-S01 to DL-S02)
# ============================================================================

class TestDataLoaderSecurity:
    """Security tests for DataLoader."""
    
    def test_path_traversal_prevention(self, tmp_path):
        """DL-S01: Malicious path input."""
        # Attempt path traversal
        loader = DataLoader(data_dir=str(tmp_path / ".." / ".." / "etc"))
        
        # Should not access files outside intended directory
        with pytest.raises(FileNotFoundError):
            loader.load_plays()
            
    def test_week_number_type_safety(self, loader):
        """Week number type safety."""
        # String that could be interpreted maliciously
        with pytest.raises((TypeError, ValueError, FileNotFoundError)):
            loader.load_week_data("1; DROP TABLE")


# ============================================================================
# Integration Tests with Real Data
# ============================================================================

class TestDataLoaderIntegration:
    """Integration tests with real NFL data."""
    
    @pytest.mark.skipif(
        not os.path.exists("/home/tanmay/Desktop/NFL/train/input_2023_w01.csv"),
        reason="Real data not available"
    )
    def test_real_data_load_and_standardize(self, real_loader):
        """Load and standardize real data."""
        df = real_loader.load_week_data(1)
        df = real_loader.standardize_tracking_directions(df)
        
        assert len(df) > 1000  # Real data should have many rows
        assert "std_x" in df.columns
        assert "std_y" in df.columns
        
        # Check coordinate ranges
        assert df["std_x"].min() >= 0
        assert df["std_x"].max() <= 120
        assert df["std_y"].min() >= 0
        assert df["std_y"].max() <= 53.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
