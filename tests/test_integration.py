"""
Integration tests for end-to-end pipeline validation.
Tests complete workflow from data loading to prediction.
"""
import sys
sys.path.insert(0, 'src')

import pytest
import torch
import polars as pl
import os
from torch_geometric.data import Batch

from data_loader import DataLoader
from features import create_graph_data
from models.gnn import NFLGraphTransformer
from train import NFLGraphPredictor


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def real_data_available():
    """Check if real data is available."""
    return os.path.exists("/home/tanmay/Desktop/NFL/train/input_2023_w01.csv")


@pytest.fixture
def real_loader():
    """Create DataLoader with real data."""
    return DataLoader(data_dir="/home/tanmay/Desktop/NFL")


@pytest.fixture
def mock_pipeline_data():
    """Create mock data for pipeline testing."""
    frames = []
    for i in range(15):  # Enough frames for history + future
        player_data = {
            "game_id": [1] * 12,
            "play_id": [100] * 12,
            "frame_id": [i] * 12,
            "nfl_id": list(range(1, 13)),
            "std_x": [40.0 + j * 3 + i * 0.3 for j in range(12)],
            "std_y": [20.0 + (j % 4) * 5 for j in range(12)],
            "s": [5.0 + j * 0.2 for j in range(12)],
            "a": [1.0 + j * 0.1 for j in range(12)],
            "std_dir": [45.0 + j * 10 for j in range(12)],
            "std_o": [90.0 + j * 5 for j in range(12)],
            "weight_norm": [0.0] * 12,
            "role_id": [j % 5 for j in range(12)],
            "side_id": [j % 3 for j in range(12)],
        }
        frames.append(pl.DataFrame(player_data))
    return pl.concat(frames)


# ============================================================================
# Full Pipeline Tests (IN-F01 to IN-F06)
# ============================================================================

class TestFullPipeline:
    """End-to-end pipeline tests."""
    
    def test_data_to_graph_pipeline(self, mock_pipeline_data):
        """IN-F01: Load → Features → Graph."""
        # Skip data loading for mock test
        df = mock_pipeline_data
        
        # Create graphs
        graphs = create_graph_data(
            df,
            radius=20.0,
            future_seq_len=3,
            history_len=5
        )
        
        assert len(graphs) > 0
        
        # Create batch
        batch = Batch.from_data_list(graphs)
        
        assert batch is not None
        assert hasattr(batch, 'x')
        assert hasattr(batch, 'edge_index')
        
    def test_graph_to_prediction_pipeline(self, mock_pipeline_data):
        """IN-F02: Graph → Model → Prediction."""
        # Create graphs
        graphs = create_graph_data(
            mock_pipeline_data,
            radius=20.0,
            future_seq_len=3,
            history_len=5
        )
        batch = Batch.from_data_list(graphs)
        
        # Initialize model
        model = NFLGraphTransformer(
            input_dim=7,
            hidden_dim=32,
            future_seq_len=3,
            edge_dim=5
        )
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            traj, cov, attn = model(batch, return_attention_weights=True)
        
        # Validate outputs
        assert traj.shape[1] == 3  # Future seq len
        assert traj.shape[2] == 2  # x, y
        assert cov is not None
        
    def test_prediction_metrics_computation(self, mock_pipeline_data):
        """IN-X01: Model output → Metrics."""
        # Create graphs
        graphs = create_graph_data(
            mock_pipeline_data,
            radius=20.0,
            future_seq_len=3,
            history_len=5
        )
        batch = Batch.from_data_list(graphs)
        
        # Model prediction
        model = NFLGraphTransformer(
            input_dim=7,
            hidden_dim=32,
            future_seq_len=3
        )
        
        model.eval()
        with torch.no_grad():
            traj_pred, cov_pred, _ = model(batch)
        
        # Compute ADE (Average Displacement Error)
        traj_target = batch.y
        errors = torch.norm(traj_pred - traj_target, dim=2)
        ade = errors.mean()
        
        assert not torch.isnan(ade)
        assert ade >= 0
        
        # Compute FDE (Final Displacement Error)
        final_errors = torch.norm(traj_pred[:, -1] - traj_target[:, -1], dim=1)
        fde = final_errors.mean()
        
        assert not torch.isnan(fde)
        assert fde >= 0
        
    def test_full_training_loop_mock(self, mock_pipeline_data):
        """IN-F03: Complete training loop (mock data)."""
        # Create graphs
        graphs = create_graph_data(
            mock_pipeline_data,
            radius=20.0,
            future_seq_len=3,
            history_len=5
        )
        batch = Batch.from_data_list(graphs[:2])  # Small batch
        
        # Initialize predictor
        predictor = NFLGraphPredictor(
            input_dim=7,
            hidden_dim=32,
            future_seq_len=3,
            use_augmentation=False
        )
        
        # Training step
        loss1 = predictor.training_step(batch, batch_idx=0)
        
        # Validation step
        val_loss = predictor.validation_step(batch, batch_idx=0)
        
        assert not torch.isnan(loss1)
        assert not torch.isnan(val_loss)
        
    def test_model_gradient_update(self, mock_pipeline_data):
        """Test that gradients update parameters."""
        graphs = create_graph_data(
            mock_pipeline_data,
            radius=20.0,
            future_seq_len=3,
            history_len=5
        )
        batch = Batch.from_data_list(graphs[:2])
        
        predictor = NFLGraphPredictor(
            input_dim=7,
            hidden_dim=32,
            future_seq_len=3,
            lr=0.01
        )
        
        # Get initial parameters
        initial_param = next(predictor.model.parameters()).clone()
        
        # Training step with backward
        loss = predictor.training_step(batch, batch_idx=0)
        loss.backward()
        
        # Manually apply gradients
        optimizer = torch.optim.AdamW(predictor.parameters(), lr=0.01)
        optimizer.step()
        
        # Check parameters changed
        updated_param = next(predictor.model.parameters())
        assert not torch.allclose(initial_param, updated_param)


# ============================================================================
# Cross-Module Integration Tests (IN-X01 to IN-X03)
# ============================================================================

class TestCrossModuleIntegration:
    """Tests for cross-module interactions."""
    
    def test_config_to_model_consistency(self):
        """IN-X03: Config values propagate to model."""
        # Simulate config values
        hidden_dim = 128
        num_heads = 8
        future_seq_len = 5
        
        model = NFLGraphTransformer(
            input_dim=7,
            hidden_dim=hidden_dim,
            heads=num_heads,
            future_seq_len=future_seq_len
        )
        
        # Verify model has correct configuration
        assert model.encoder.hidden_dim == hidden_dim
        
    def test_batch_size_consistency(self, mock_pipeline_data):
        """Batch sizes remain consistent through pipeline."""
        graphs = create_graph_data(
            mock_pipeline_data,
            radius=20.0,
            future_seq_len=3,
            history_len=5
        )
        
        # Create batch of specific size
        batch_size = 2
        batch = Batch.from_data_list(graphs[:batch_size])
        
        model = NFLGraphTransformer(
            input_dim=7,
            hidden_dim=32,
            future_seq_len=3
        )
        
        _, cov_pred, _ = model(batch)
        
        # Coverage prediction should have batch_size rows
        assert cov_pred.shape[0] == batch_size


# ============================================================================
# Real Data Integration (Skipped if data unavailable)
# ============================================================================

class TestRealDataIntegration:
    """Integration tests with real NFL data."""
    
    @pytest.mark.skipif(
        not os.path.exists("/home/tanmay/Desktop/NFL/train/input_2023_w01.csv"),
        reason="Real data not available"
    )
    def test_full_pipeline_real_data(self, real_loader):
        """IN-F03: Full sanity check with real data."""
        # Load real data
        df = real_loader.load_week_data(1)
        df = real_loader.standardize_tracking_directions(df)
        
        # Limit to first few plays for speed
        unique_plays = df.select(["game_id", "play_id"]).unique()[:5]
        df_subset = df.join(unique_plays, on=["game_id", "play_id"])
        
        # Create graphs
        graphs = create_graph_data(
            df_subset,
            radius=20.0,
            future_seq_len=5,
            history_len=5
        )
        
        assert len(graphs) > 0
        
        # Model prediction
        batch = Batch.from_data_list(graphs[:10])
        
        model = NFLGraphTransformer(
            input_dim=7,
            hidden_dim=64,
            future_seq_len=5
        )
        
        model.eval()
        with torch.no_grad():
            traj, cov, _ = model(batch)
        
        # Basic sanity checks
        assert traj.shape[1] == 5
        assert traj.shape[2] == 2
        assert not torch.isnan(traj).any()
        
    @pytest.mark.skipif(
        not os.path.exists("/home/tanmay/Desktop/NFL/train/input_2023_w01.csv"),
        reason="Real data not available"
    )
    def test_trajectory_predictions_reasonable(self, real_loader):
        """Predictions are within reasonable bounds."""
        df = real_loader.load_week_data(1)
        df = real_loader.standardize_tracking_directions(df)
        
        unique_plays = df.select(["game_id", "play_id"]).unique()[:3]
        df_subset = df.join(unique_plays, on=["game_id", "play_id"])
        
        graphs = create_graph_data(
            df_subset,
            radius=20.0,
            future_seq_len=5,
            history_len=5
        )
        
        if len(graphs) == 0:
            pytest.skip("No graphs created")
            
        batch = Batch.from_data_list(graphs[:5])
        
        model = NFLGraphTransformer(
            input_dim=7,
            hidden_dim=64,
            future_seq_len=5
        )
        
        model.eval()
        with torch.no_grad():
            traj, _, _ = model(batch)
        
        # Predictions are relative displacements
        # They should be reasonable (< 50 yards per timestep)
        assert traj.abs().max() < 50


# ============================================================================
# Error Handling Integration
# ============================================================================

class TestErrorHandling:
    """Test error handling across modules."""
    
    def test_empty_batch_handling(self):
        """Empty batch should be handled gracefully."""
        # Create minimal valid data
        data = pl.DataFrame({
            "game_id": [1],
            "play_id": [100],
            "frame_id": [0],
            "nfl_id": [10],
            "std_x": [50.0],
            "std_y": [25.0],
            "s": [5.0],
            "a": [1.0],
            "std_dir": [45.0],
            "std_o": [90.0],
            "weight_norm": [0.0],
            "role_id": [0],
            "side_id": [0],
        })
        
        # Not enough frames for graph creation
        graphs = create_graph_data(data, future_seq_len=5, history_len=5)
        
        # Should return empty list, not crash
        assert graphs == []
        
    def test_model_with_varying_node_counts(self, mock_pipeline_data):
        """Model handles graphs with different node counts."""
        graphs = create_graph_data(
            mock_pipeline_data,
            radius=20.0,
            future_seq_len=3,
            history_len=5
        )
        
        # All graphs should have same node count for this data
        # But model should handle variable sizes
        model = NFLGraphTransformer(
            input_dim=7,
            hidden_dim=32,
            future_seq_len=3
        )
        
        for graph in graphs[:3]:
            batch = Batch.from_data_list([graph])
            traj, _, _ = model(batch)
            assert traj.shape[0] == graph.x.shape[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
