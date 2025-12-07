"""
Comprehensive tests for Training module.
Covers loss functions, augmentation, and training loop tests.
"""
import sys
sys.path.insert(0, 'src')

import pytest
import torch
import polars as pl
from torch_geometric.data import Data, Batch
import numpy as np
from unittest.mock import patch, MagicMock

from train import (
    velocity_loss, 
    acceleration_loss, 
    collision_avoidance_loss,
    augment_graph,
    NFLGraphPredictor
)
from features import create_graph_data


# Mock Lightning logging for unit tests
def mock_lightning_log(predictor):
    """Patch self.log to work without trainer."""
    predictor.log = MagicMock()


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_predictions():
    """Create sample trajectory predictions."""
    # Shape: [N=5, T=10, 2]
    pred = torch.randn(5, 10, 2)
    target = pred + torch.randn(5, 10, 2) * 0.1  # Similar with noise
    return pred, target


@pytest.fixture
def sample_batch_idx():
    """Create sample batch indices for collision loss."""
    # 5 nodes: first 3 in batch 0, next 2 in batch 1
    return torch.tensor([0, 0, 0, 1, 1])


@pytest.fixture
def mock_graph():
    """Create mock graph data for testing."""
    # Create graph with required attributes
    data = Data(
        x=torch.randn(23, 7),  # 23 players, 7 features
        edge_index=torch.randint(0, 23, (2, 100)),
        edge_attr=torch.randn(100, 5),
        y=torch.randn(23, 10, 2),  # Future trajectory targets
        y_coverage=torch.tensor([1.0]),  # Coverage label (float for BCE loss)
        role=torch.randint(0, 5, (23,)),
        side=torch.randint(0, 3, (23,)),
        formation=torch.tensor([0]),
        alignment=torch.tensor([1]),
        context=torch.randn(1, 3),
        current_pos=torch.randn(23, 2),
        history=torch.randn(23, 4, 4),
        batch=torch.zeros(23, dtype=torch.long),
    )
    return data


@pytest.fixture
def mock_graph_batch(mock_graph):
    """Create batched mock graph data."""
    # Create 3 copies
    graphs = [mock_graph.clone() for _ in range(3)]
    return Batch.from_data_list(graphs)


# ============================================================================
# Loss Function Tests (TR-F01 to TR-F05)
# ============================================================================

class TestLossFunctions:
    """Tests for custom loss functions."""
    
    def test_velocity_loss_basic(self, sample_predictions):
        """TR-F01: Velocity consistency loss."""
        pred, target = sample_predictions
        
        loss = velocity_loss(pred, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0  # Non-negative
        
    def test_velocity_loss_identical(self):
        """Velocity loss is zero for identical trajectories."""
        pred = torch.randn(5, 10, 2)
        target = pred.clone()
        
        loss = velocity_loss(pred, target)
        
        # Should be close to zero
        assert loss.item() < 1e-5
        
    def test_acceleration_loss_basic(self, sample_predictions):
        """TR-F02: Acceleration smoothness loss."""
        pred, target = sample_predictions
        
        loss = acceleration_loss(pred, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0
        
    def test_acceleration_loss_linear_trajectory(self):
        """Acceleration loss is zero for linear motion."""
        # Linear trajectory: constant velocity, zero acceleration
        t = torch.linspace(0, 1, 10).unsqueeze(-1)  # [10, 1]
        pred = t.expand(5, 10, 2)  # [5, 10, 2] linear motion
        target = pred.clone()
        
        loss = acceleration_loss(pred, target)
        
        # Should be close to zero for linear motion
        assert loss.item() < 1e-4
    
    def test_collision_avoidance_loss_basic(self, sample_batch_idx):
        """TR-F03: Player overlap penalty."""
        # Create positions with some collisions
        pred = torch.randn(5, 10, 2)
        
        loss = collision_avoidance_loss(pred, sample_batch_idx, min_distance=1.0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0
        
    def test_collision_avoidance_no_violations(self, sample_batch_idx):
        """TR-E02: No collisions should give zero loss."""
        # Create well-separated positions
        pred = torch.zeros(5, 10, 2)
        pred[0, :, 0] = 0   # Player 0 at x=0
        pred[1, :, 0] = 10  # Player 1 at x=10
        pred[2, :, 0] = 20  # Player 2 at x=20
        pred[3, :, 0] = 30  # Player 3 at x=30
        pred[4, :, 0] = 40  # Player 4 at x=40
        
        loss = collision_avoidance_loss(pred, sample_batch_idx, min_distance=1.0)
        
        # No collisions, loss should be zero
        assert loss.item() == 0.0 or loss.item() < 1e-6
        
    def test_velocity_loss_single_timestep(self):
        """TR-E01: Velocity loss with T=1 trajectory."""
        pred = torch.randn(5, 1, 2)
        target = torch.randn(5, 1, 2)
        
        loss = velocity_loss(pred, target)
        
        # Should handle gracefully (0 or valid value)
        assert isinstance(loss, torch.Tensor)
        

# ============================================================================
# Augmentation Tests (TR-F04 to TR-F05)
# ============================================================================

class TestAugmentation:
    """Tests for data augmentation functions."""
    
    def test_augment_graph_flip(self, mock_graph):
        """TR-F04: Horizontal flip augmentation."""
        # Set predictable positions
        original_y = mock_graph.x[:, 1].clone()  # Assuming y is feature index 1
        
        augmented = augment_graph(
            mock_graph.clone(), 
            flip_horizontal=True, 
            add_noise=False
        )
        
        assert isinstance(augmented, Data)
        # Y coordinates should be flipped (53.3 - y)
        
    def test_augment_graph_noise(self, mock_graph):
        """TR-F05: Gaussian noise addition."""
        original_x = mock_graph.x.clone()
        
        augmented = augment_graph(
            mock_graph.clone(),
            flip_horizontal=False,
            add_noise=True,
            noise_std=0.1
        )
        
        # Positions should be different (noise added)
        assert not torch.allclose(augmented.x, original_x, atol=1e-6)
        
    def test_augment_graph_both(self, mock_graph):
        """Both flip and noise together."""
        original_x = mock_graph.x.clone()
        
        augmented = augment_graph(
            mock_graph.clone(),
            flip_horizontal=True,
            add_noise=True,
            noise_std=0.05
        )
        
        assert isinstance(augmented, Data)
        # Should be different from original
        assert not torch.allclose(augmented.x, original_x, atol=1e-6)
        
    def test_augment_preserves_structure(self, mock_graph):
        """Augmentation preserves graph structure."""
        original_edge_index = mock_graph.edge_index.clone()
        
        augmented = augment_graph(
            mock_graph.clone(),
            flip_horizontal=True,
            add_noise=True
        )
        
        # Edge structure should be preserved
        assert augmented.edge_index.shape == original_edge_index.shape
        assert torch.equal(augmented.edge_index, original_edge_index)


# ============================================================================
# NFLGraphPredictor Tests (TR-F06 to TR-F10)
# ============================================================================

class TestNFLGraphPredictor:
    """Tests for PyTorch Lightning predictor module."""
    
    def test_predictor_initialization(self):
        """Basic predictor initialization."""
        predictor = NFLGraphPredictor(
            input_dim=7,
            hidden_dim=64,
            lr=1e-3,
            future_seq_len=10
        )
        
        assert predictor.model is not None
        assert predictor.lr == 1e-3
        
    def test_predictor_forward(self, mock_graph_batch):
        """TR-F06: Single forward pass."""
        predictor = NFLGraphPredictor(
            input_dim=7,
            hidden_dim=32,
            future_seq_len=10
        )
        
        traj, cov, attn = predictor(mock_graph_batch)
        
        # Check output shapes
        assert traj.shape[1] == 10  # Future seq len
        assert traj.shape[2] == 2  # x, y
        assert cov is not None
        
    def test_predictor_training_step(self, mock_graph_batch):
        """TR-F06: Single training step."""
        predictor = NFLGraphPredictor(
            input_dim=7,
            hidden_dim=32,
            future_seq_len=10,
            use_augmentation=False  # Disable for deterministic test
        )
        mock_lightning_log(predictor)  # Patch self.log for unit test
        
        loss = predictor.training_step(mock_graph_batch, batch_idx=0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert not torch.isnan(loss)
        
    def test_predictor_validation_step(self, mock_graph_batch):
        """TR-F07: Single validation step."""
        predictor = NFLGraphPredictor(
            input_dim=7,
            hidden_dim=32,
            future_seq_len=10
        )
        mock_lightning_log(predictor)  # Patch self.log for unit test
        
        loss = predictor.validation_step(mock_graph_batch, batch_idx=0)
        
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
        
    def test_configure_optimizers(self):
        """TR-F08: Optimizer setup."""
        predictor = NFLGraphPredictor(
            input_dim=7,
            hidden_dim=32,
            lr=1e-3
        )
        
        result = predictor.configure_optimizers()
        
        assert "optimizer" in result
        assert "lr_scheduler" in result
        
    def test_probabilistic_mode(self, mock_graph_batch):
        """TR-F09: GMM training mode."""
        predictor = NFLGraphPredictor(
            input_dim=7,
            hidden_dim=32,
            future_seq_len=10,
            probabilistic=True,
            num_modes=6
        )
        mock_lightning_log(predictor)  # Patch self.log for unit test
        
        loss = predictor.training_step(mock_graph_batch, batch_idx=0)
        
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
        
    def test_huber_loss_mode(self, mock_graph_batch):
        """TR-F10: Huber loss option."""
        predictor = NFLGraphPredictor(
            input_dim=7,
            hidden_dim=32,
            future_seq_len=10,
            use_huber_loss=True,
            huber_delta=1.0
        )
        mock_lightning_log(predictor)  # Patch self.log for unit test
        
        loss = predictor.training_step(mock_graph_batch, batch_idx=0)
        
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
        
    def test_different_loss_weights(self, mock_graph_batch):
        """Test with various loss weight configurations."""
        predictor = NFLGraphPredictor(
            input_dim=7,
            hidden_dim=32,
            velocity_weight=0.5,
            acceleration_weight=0.2,
            coverage_weight=0.3,
            collision_weight=0.1
        )
        mock_lightning_log(predictor)  # Patch self.log for unit test
        
        loss = predictor.training_step(mock_graph_batch, batch_idx=0)
        
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)


# ============================================================================
# Edge Case Tests (TR-E01 to TR-E05)
# ============================================================================

class TestTrainingEdgeCases:
    """Edge case tests for training."""
    
    def test_zero_learning_rate(self, mock_graph_batch):
        """TR-E03: Zero learning rate."""
        predictor = NFLGraphPredictor(
            input_dim=7,
            hidden_dim=32,
            lr=0.0
        )
        mock_lightning_log(predictor)  # Patch self.log for unit test
        
        # Get initial parameters
        initial_params = [p.clone() for p in predictor.parameters()]
        
        # Training step
        loss = predictor.training_step(mock_graph_batch, batch_idx=0)
        
        # With lr=0, params shouldn't change during actual training
        # (this just tests initialization works)
        assert loss is not None
        
    def test_gradient_flow(self, mock_graph_batch):
        """Test gradients flow through model."""
        predictor = NFLGraphPredictor(
            input_dim=7,
            hidden_dim=32
        )
        mock_lightning_log(predictor)  # Patch self.log for unit test
        
        loss = predictor.training_step(mock_graph_batch, batch_idx=0)
        loss.backward()
        
        # Check that gradients exist
        has_grad = False
        for param in predictor.parameters():
            if param.grad is not None:
                has_grad = True
                break
        
        assert has_grad, "No gradients found"
        
    def test_small_batch(self):
        """Test with minimal batch size."""
        # Single graph with 2 nodes
        data = Data(
            x=torch.randn(2, 7),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            edge_attr=torch.randn(2, 5),
            y=torch.randn(2, 10, 2),
            y_coverage=torch.tensor([0.0]),
            role=torch.randint(0, 5, (2,)),
            side=torch.randint(0, 3, (2,)),
            formation=torch.tensor([0]),
            alignment=torch.tensor([1]),
            context=torch.randn(1, 3),
            current_pos=torch.randn(2, 2),
            history=torch.randn(2, 4, 4),
            batch=torch.zeros(2, dtype=torch.long),
        )
        batch = Batch.from_data_list([data])
        
        predictor = NFLGraphPredictor(
            input_dim=7,
            hidden_dim=16
        )
        mock_lightning_log(predictor)  # Patch self.log for unit test
        
        loss = predictor.training_step(batch, batch_idx=0)
        
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)


# ============================================================================
# Performance Tests (TR-P01 to TR-P02)
# ============================================================================

class TestTrainingPerformance:
    """Performance tests for training."""
    
    def test_training_step_timing(self, mock_graph_batch):
        """TR-P01: Single step duration."""
        import time
        
        predictor = NFLGraphPredictor(
            input_dim=7,
            hidden_dim=64
        )
        mock_lightning_log(predictor)  # Patch self.log for unit test
        
        # Warmup
        _ = predictor.training_step(mock_graph_batch, batch_idx=0)
        
        # Timed run
        start = time.time()
        for _ in range(10):
            _ = predictor.training_step(mock_graph_batch, batch_idx=0)
        elapsed = (time.time() - start) / 10
        
        # Should complete in reasonable time (< 2s per step on CPU)
        assert elapsed < 2.0, f"Training step took {elapsed:.3f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
