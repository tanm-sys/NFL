"""
Pytest configuration and shared fixtures for NFL Analytics Engine tests.
"""
import sys
sys.path.insert(0, 'src')

import pytest
import torch
import polars as pl
import os
from torch_geometric.data import Data, Batch


# ============================================================================
# Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests (deselect with '-m \"not integration\"')"
    )


# ============================================================================
# Global Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def device():
    """Get available device (GPU or CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture(scope="session")
def real_data_path():
    """Path to real NFL data."""
    return "/home/tanmay/Desktop/NFL"


@pytest.fixture(scope="session")
def real_data_available(real_data_path):
    """Check if real data is available."""
    return os.path.exists(os.path.join(real_data_path, "train/input_2023_w01.csv"))


# ============================================================================
# Mock Data Fixtures
# ============================================================================

@pytest.fixture
def mock_graph_data():
    """Create mock PyG Data object for testing."""
    num_nodes = 23
    num_edges = 100
    
    return Data(
        x=torch.randn(num_nodes, 7),
        edge_index=torch.randint(0, num_nodes, (2, num_edges)),
        edge_attr=torch.randn(num_edges, 5),
        y=torch.randn(num_nodes, 10, 2),
        y_coverage=torch.tensor([1.0]),
        role=torch.randint(0, 5, (num_nodes,)),
        side=torch.randint(0, 3, (num_nodes,)),
        formation=torch.tensor([0]),
        alignment=torch.tensor([1]),
        context=torch.randn(1, 3),
        current_pos=torch.randn(num_nodes, 2),
        history=torch.randn(num_nodes, 4, 4),
        batch=torch.zeros(num_nodes, dtype=torch.long),
    )


@pytest.fixture
def mock_graph_batch(mock_graph_data):
    """Create batched mock graph data."""
    return Batch.from_data_list([mock_graph_data.clone() for _ in range(3)])


@pytest.fixture
def mock_play_dataframe():
    """Create mock play DataFrame for testing."""
    frames = []
    for i in range(12):  # Enough frames for history + future
        player_data = {
            "game_id": [1] * 23,
            "play_id": [100] * 23,
            "frame_id": [i] * 23,
            "nfl_id": list(range(1, 24)),
            "std_x": [40.0 + j * 2 + i * 0.3 for j in range(23)],
            "std_y": [20.0 + (j % 5) * 3 for j in range(23)],
            "s": [5.0 + j * 0.1 for j in range(23)],
            "a": [1.0 + j * 0.05 for j in range(23)],
            "std_dir": [45.0 + j * 5 for j in range(23)],
            "std_o": [90.0 + j * 3 for j in range(23)],
            "weight_norm": [0.0] * 23,
            "role_id": [j % 5 for j in range(23)],
            "side_id": [j % 3 for j in range(23)],
        }
        frames.append(pl.DataFrame(player_data))
    return pl.concat(frames)


@pytest.fixture
def small_mock_dataframe():
    """Create small mock DataFrame for quick tests."""
    frames = []
    for i in range(10):
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
            "weight_norm": [0.0, 0.0],
            "role_id": [0, 2],
            "side_id": [0, 1],
        }))
    return pl.concat(frames)


# ============================================================================
# Model Fixtures
# ============================================================================

@pytest.fixture
def gnn_model():
    """Create GNN model for testing."""
    from models.gnn import NFLGraphTransformer
    return NFLGraphTransformer(
        input_dim=7,
        hidden_dim=32,
        future_seq_len=10,
        edge_dim=5
    )


@pytest.fixture
def predictor():
    """Create Lightning predictor for testing."""
    from train import NFLGraphPredictor
    return NFLGraphPredictor(
        input_dim=7,
        hidden_dim=32,
        future_seq_len=10,
        use_augmentation=False
    )


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory with mock files."""
    # Create train subdirectory
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    
    # Create mock tracking data
    tracking_data = pl.DataFrame({
        "gameId": [1, 1, 1, 1],
        "playId": [100, 100, 100, 100],
        "frameId": [1, 1, 2, 2],
        "nflId": [10, 20, 10, 20],
        "x": [50.0, 60.0, 51.0, 61.0],
        "y": [25.0, 30.0, 26.0, 31.0],
        "s": [5.0, 4.0, 5.5, 4.2],
        "a": [1.0, 0.5, 1.2, 0.6],
        "o": [90.0, 180.0, 85.0, 175.0],
        "dir": [45.0, 135.0, 50.0, 140.0],
        "playDirection": ["right", "right", "right", "right"],
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


# ============================================================================
# Utility Functions
# ============================================================================

def assert_tensor_shape(tensor, expected_shape, name="tensor"):
    """Assert tensor has expected shape with helpful error message."""
    assert tensor.shape == expected_shape, \
        f"{name} shape {tensor.shape} != expected {expected_shape}"


def assert_no_nan(tensor, name="tensor"):
    """Assert tensor has no NaN values."""
    assert not torch.isnan(tensor).any(), f"{name} contains NaN values"


def assert_no_inf(tensor, name="tensor"):
    """Assert tensor has no infinite values."""
    assert not torch.isinf(tensor).any(), f"{name} contains infinite values"
