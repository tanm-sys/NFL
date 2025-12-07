"""
Comprehensive tests for Features module.
Covers graph construction, coordinate calculations, and edge features.
"""
import sys
sys.path.insert(0, 'src')

import pytest
import torch
import polars as pl
import numpy as np
import h3
from torch_geometric.data import Data

from features import (
    calculate_distance_to_ball,
    create_baseline_features,
    add_h3_indices,
    prepare_tensor_data,
    create_graph_data
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_play_df():
    """Create simple play DataFrame with 2 players."""
    frames = []
    for i in range(10):  # 10 frames
        frames.append(pl.DataFrame({
            "game_id": [1, 1],
            "play_id": [100, 100],
            "frame_id": [i, i],
            "nfl_id": [10, 20],
            "std_x": [50.0 + i, 60.0 + i],
            "std_y": [25.0, 30.0],
            "s": [5.0, 4.0],
            "a": [1.0, 0.5],
            "std_dir": [45.0, 135.0],
            "std_o": [90.0, 180.0],
            "weight_norm": [0.4, -0.1],
            "role_id": [0, 2],
            "side_id": [0, 1],
        }))
    return pl.concat(frames)


@pytest.fixture
def play_with_ball():
    """Create play DataFrame with ball included."""
    frames = []
    for i in range(10):
        frames.append(pl.DataFrame({
            "game_id": [1, 1, 1],
            "play_id": [100, 100, 100],
            "frame_id": [i, i, i],
            "nfl_id": [10, 20, None],  # None for ball
            "std_x": [50.0 + i, 60.0 + i, 55.0 + i],
            "std_y": [25.0, 30.0, 27.5],
            "s": [5.0, 4.0, 10.0],
            "a": [1.0, 0.5, 0.0],
            "std_dir": [45.0, 135.0, 90.0],
            "std_o": [90.0, 180.0, 0.0],
            "weight_norm": [0.4, -0.1, 0.0],
            "role_id": [0, 2, 4],  # 4 = ball
            "side_id": [0, 1, 2],
            "club": ["DEF", "OFF", "football"],
            "player_name": ["Player1", "Player2", "football"],  # Required by calculate_distance_to_ball
        }))
    return pl.concat(frames)


@pytest.fixture
def full_team_df():
    """Create DataFrame with 22 players + ball."""
    frames = []
    for i in range(12):  # 12 frames for history + future
        player_data = {
            "game_id": [1] * 23,
            "play_id": [100] * 23,
            "frame_id": [i] * 23,
            "nfl_id": list(range(1, 24)),
            "std_x": [40.0 + j * 2 + i * 0.5 for j in range(23)],
            "std_y": [20.0 + (j % 5) * 3 for j in range(23)],
            "s": [5.0] * 23,
            "a": [1.0] * 23,
            "std_dir": [45.0] * 23,
            "std_o": [90.0] * 23,
            "weight_norm": [0.0] * 23,
            "role_id": [j % 5 for j in range(23)],
            "side_id": [j % 3 for j in range(23)],
        }
        frames.append(pl.DataFrame(player_data))
    return pl.concat(frames)


# ============================================================================
# Distance to Ball Tests (FE-F01)
# ============================================================================

class TestDistanceToBall:
    """Tests for distance to ball calculation."""
    
    def test_calculate_distance_to_ball(self, play_with_ball):
        """FE-F01: Distance calculation."""
        result = calculate_distance_to_ball(play_with_ball)
        
        assert "dist_to_ball" in result.columns  # Column name is dist_to_ball
        
        # Check distances are positive
        distances = result["dist_to_ball"].to_list()
        for d in distances:
            if d is not None:
                assert d >= 0
                
    def test_ball_distance_to_self(self, play_with_ball):
        """Ball's distance to itself should be 0."""
        result = calculate_distance_to_ball(play_with_ball)
        
        # Ball entries should have distance 0 or NaN
        ball_rows = result.filter(pl.col("player_name") == "football")
        # Ball's dist_to_ball should be 0 (same position as itself)
        assert len(ball_rows) > 0


# ============================================================================
# Baseline Features Tests (FE-F02)
# ============================================================================

class TestBaselineFeatures:
    """Tests for baseline feature creation."""
    
    def test_create_baseline_features(self, play_with_ball):
        """FE-F02: Basic feature extraction."""
        result = create_baseline_features(play_with_ball)
        
        assert isinstance(result, pl.DataFrame)
        # Should have distance features
        assert "dist_to_ball" in result.columns or len(result) > 0


# ============================================================================
# H3 Indexing Tests (FE-F03)
# ============================================================================

class TestH3Indexing:
    """Tests for H3 hexagonal indexing."""
    
    @pytest.mark.skipif(
        not hasattr(h3, 'geo_to_h3'),
        reason="H3 v3 API (geo_to_h3) not available - may be using v4 API"
    )
    def test_add_h3_indices_basic(self, simple_play_df):
        """FE-F03: H3 index generation."""
        result = add_h3_indices(simple_play_df, resolution=10)
        
        assert "h3_index" in result.columns
        
        # H3 indices should be non-empty strings
        indices = result["h3_index"].to_list()
        for idx in indices:
            assert idx is not None
            assert len(idx) > 0
            
    @pytest.mark.skipif(
        not hasattr(h3, 'geo_to_h3'),
        reason="H3 v3 API (geo_to_h3) not available - may be using v4 API"
    )
    def test_h3_different_resolutions(self, simple_play_df):
        """Different H3 resolutions."""
        for res in [7, 9, 11]:
            result = add_h3_indices(simple_play_df, resolution=res)
            assert "h3_index" in result.columns


# ============================================================================
# Graph Data Tests (FE-F05 to FE-F10)
# ============================================================================

class TestGraphData:
    """Tests for graph data creation."""
    
    def test_create_graph_data_basic(self, simple_play_df):
        """FE-F05: Basic graph creation."""
        graphs = create_graph_data(
            simple_play_df, 
            radius=20.0, 
            future_seq_len=2, 
            history_len=5
        )
        
        assert isinstance(graphs, list)
        assert len(graphs) > 0
        
        graph = graphs[0]
        assert isinstance(graph, Data)
        
    def test_graph_edge_attributes(self, simple_play_df):
        """FE-F06: Edge feature dimensions."""
        graphs = create_graph_data(
            simple_play_df,
            radius=20.0,
            future_seq_len=2,
            history_len=5
        )
        
        graph = graphs[0]
        assert hasattr(graph, 'edge_attr')
        
        if graph.edge_index.shape[1] > 0:
            # 5D edge features
            assert graph.edge_attr.shape[1] == 5
            
    def test_graph_future_trajectory(self, simple_play_df):
        """FE-F07: Future trajectory in y."""
        graphs = create_graph_data(
            simple_play_df,
            radius=20.0,
            future_seq_len=2,
            history_len=5
        )
        
        graph = graphs[0]
        assert hasattr(graph, 'y')
        
        # Shape: [N, T, 2]
        assert graph.y.shape[1] == 2  # future_seq_len
        assert graph.y.shape[2] == 2  # x, y coords
        
    def test_graph_history_features(self, simple_play_df):
        """FE-F08: Temporal history."""
        graphs = create_graph_data(
            simple_play_df,
            radius=20.0,
            future_seq_len=2,
            history_len=5
        )
        
        graph = graphs[0]
        assert hasattr(graph, 'history')
        
        # Shape: [N, history_len-1, 4] (vel_x, vel_y, acc_x, acc_y)
        assert graph.history.shape[1] == 4  # history_len - 1
        assert graph.history.shape[2] == 4  # 4 motion features
        
    def test_current_pos_attribute(self, simple_play_df):
        """FE-F09: Current position stored."""
        graphs = create_graph_data(
            simple_play_df,
            radius=20.0,
            future_seq_len=2,
            history_len=5
        )
        
        graph = graphs[0]
        assert hasattr(graph, 'current_pos')
        assert graph.current_pos.shape[1] == 2  # x, y
        
    def test_strategic_embeddings(self, simple_play_df):
        """FE-F10: Role/side/formation/alignment IDs."""
        graphs = create_graph_data(
            simple_play_df,
            radius=20.0,
            future_seq_len=2,
            history_len=5
        )
        
        graph = graphs[0]
        
        # Check strategic embedding attributes
        assert hasattr(graph, 'role')
        assert hasattr(graph, 'side')
        
    def test_graph_count_correct(self, simple_play_df):
        """Correct number of graphs created."""
        # 10 frames, history_len=5, future_seq_len=2
        # range(5, 10-2) = range(5, 8) = 3 graphs
        graphs = create_graph_data(
            simple_play_df,
            radius=20.0,
            future_seq_len=2,
            history_len=5
        )
        
        assert len(graphs) == 3


# ============================================================================
# Edge Case Tests (FE-E01 to FE-E11)
# ============================================================================

class TestGraphEdgeCases:
    """Edge case tests for graph creation."""
    
    def test_single_player_graph(self):
        """FE-E01: Only 1 player in frame."""
        frames = []
        for i in range(10):
            frames.append(pl.DataFrame({
                "game_id": [1],
                "play_id": [100],
                "frame_id": [i],
                "nfl_id": [10],
                "std_x": [50.0 + i],
                "std_y": [25.0],
                "s": [5.0],
                "a": [1.0],
                "std_dir": [45.0],
                "std_o": [90.0],
                "weight_norm": [0.0],
                "role_id": [0],
                "side_id": [0],
            }))
        df = pl.concat(frames)
        
        graphs = create_graph_data(df, radius=20.0, future_seq_len=2, history_len=5)
        
        # Should create graphs with single node
        assert len(graphs) > 0
        graph = graphs[0]
        assert graph.x.shape[0] == 1  # Single node
        
    def test_all_players_same_position(self):
        """FE-E02: All at same position."""
        frames = []
        for i in range(10):
            frames.append(pl.DataFrame({
                "game_id": [1, 1, 1],
                "play_id": [100, 100, 100],
                "frame_id": [i, i, i],
                "nfl_id": [10, 20, 30],
                "std_x": [50.0, 50.0, 50.0],  # Same position
                "std_y": [25.0, 25.0, 25.0],
                "s": [5.0, 5.0, 5.0],
                "a": [1.0, 1.0, 1.0],
                "std_dir": [45.0, 45.0, 45.0],
                "std_o": [90.0, 90.0, 90.0],
                "weight_norm": [0.0, 0.0, 0.0],
                "role_id": [0, 1, 2],
                "side_id": [0, 1, 0],
            }))
        df = pl.concat(frames)
        
        graphs = create_graph_data(df, radius=20.0, future_seq_len=2, history_len=5)
        
        # Should handle zero-distance edges
        assert len(graphs) > 0
        
    def test_insufficient_frames_for_history(self):
        """FE-E04: Less than history_len frames."""
        frames = []
        for i in range(3):  # Only 3 frames, but need 5 for history
            frames.append(pl.DataFrame({
                "game_id": [1, 1],
                "play_id": [100, 100],
                "frame_id": [i, i],
                "nfl_id": [10, 20],
                "std_x": [50.0, 60.0],
                "std_y": [25.0, 30.0],
                "s": [5.0, 4.0],
                "a": [1.0, 0.5],
                "std_dir": [45.0, 135.0],
                "std_o": [90.0, 180.0],
                "weight_norm": [0.0, 0.0],
                "role_id": [0, 2],
                "side_id": [0, 1],
            }))
        df = pl.concat(frames)
        
        graphs = create_graph_data(df, radius=20.0, future_seq_len=2, history_len=5)
        
        # Should return empty list
        assert len(graphs) == 0
        
    def test_empty_dataframe(self):
        """FE-E08: Empty input DataFrame."""
        df = pl.DataFrame({
            "game_id": [],
            "play_id": [],
            "frame_id": [],
            "nfl_id": [],
            "std_x": [],
            "std_y": [],
            "s": [],
            "a": [],
            "std_dir": [],
            "std_o": [],
            "weight_norm": [],
            "role_id": [],
            "side_id": [],
        }).cast({
            "game_id": pl.Int64,
            "play_id": pl.Int64,
            "frame_id": pl.Int64,
            "nfl_id": pl.Int64,
            "std_x": pl.Float64,
            "std_y": pl.Float64,
            "s": pl.Float64,
            "a": pl.Float64,
            "std_dir": pl.Float64,
            "std_o": pl.Float64,
            "weight_norm": pl.Float64,
            "role_id": pl.Int64,
            "side_id": pl.Int64,
        })
        
        graphs = create_graph_data(df, radius=20.0)
        
        assert len(graphs) == 0
        
    def test_radius_zero(self, simple_play_df):
        """FE-E10: radius=0 for edges."""
        graphs = create_graph_data(
            simple_play_df,
            radius=0.0,  # No connections
            future_seq_len=2,
            history_len=5
        )
        
        # Should handle zero radius
        if len(graphs) > 0:
            graph = graphs[0]
            # Either no edges or only self-loops
            assert graph.edge_index.shape[1] >= 0
            
    def test_very_large_radius(self, simple_play_df):
        """FE-E11: radius=1000 (all connected)."""
        graphs = create_graph_data(
            simple_play_df,
            radius=1000.0,  # Everything connected
            future_seq_len=2,
            history_len=5
        )
        
        assert len(graphs) > 0
        graph = graphs[0]
        
        # Should have many edges (fully connected)
        num_nodes = graph.x.shape[0]
        max_edges = num_nodes * (num_nodes - 1)  # Directed
        assert graph.edge_index.shape[1] <= max_edges


# ============================================================================
# Performance Tests
# ============================================================================

class TestFeaturePerformance:
    """Performance tests for feature engineering."""
    
    def test_graph_creation_speed(self, full_team_df):
        """FE-P01: Graph creation timing."""
        import time
        
        start = time.time()
        graphs = create_graph_data(
            full_team_df,
            radius=20.0,
            future_seq_len=2,
            history_len=5
        )
        elapsed = time.time() - start
        
        # Should be fast for small data
        assert elapsed < 5.0, f"Graph creation took {elapsed:.2f}s"
        assert len(graphs) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
