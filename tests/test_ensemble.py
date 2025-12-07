"""
Comprehensive tests for Ensemble model.
Covers weighted averaging, uncertainty estimation, and checkpoint loading.
"""
import sys
sys.path.insert(0, 'src')

import pytest
import torch
import tempfile
import os
from torch_geometric.data import Data, Batch

from models.gnn import NFLGraphTransformer
from models.ensemble import EnsemblePredictor, load_ensemble


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_graph_batch():
    """Create mock graph batch for testing."""
    data = Data(
        x=torch.randn(10, 7),
        edge_index=torch.randint(0, 10, (2, 30)),
        edge_attr=torch.randn(30, 5),
        y=torch.randn(10, 5, 2),
        y_coverage=torch.tensor([1.0]),
        role=torch.randint(0, 5, (10,)),
        side=torch.randint(0, 3, (10,)),
        formation=torch.tensor([0]),
        alignment=torch.tensor([1]),
        context=torch.randn(1, 3),
        current_pos=torch.randn(10, 2),
        history=torch.randn(10, 4, 4),
        batch=torch.zeros(10, dtype=torch.long),
    )
    return Batch.from_data_list([data, data.clone()])


@pytest.fixture
def model_list():
    """Create list of models for ensemble."""
    models = []
    for _ in range(3):
        model = NFLGraphTransformer(
            input_dim=7,
            hidden_dim=32,
            future_seq_len=5,
            edge_dim=5
        )
        model.eval()
        models.append(model)
    return models


@pytest.fixture
def ensemble(model_list):
    """Create ensemble predictor."""
    return EnsemblePredictor(model_list)


@pytest.fixture
def weighted_ensemble(model_list):
    """Create ensemble with custom weights."""
    weights = [0.5, 0.3, 0.2]
    return EnsemblePredictor(model_list, weights=weights)


# ============================================================================
# Functional Tests (EN-F01 to EN-F05)
# ============================================================================

class TestEnsembleFunctional:
    """Functional tests for ensemble model."""
    
    def test_ensemble_forward(self, ensemble, mock_graph_batch):
        """EN-F01: Weighted average prediction."""
        pred, cov, _ = ensemble(mock_graph_batch)
        
        # Check output shapes
        assert pred.shape[1] == 5  # future_seq_len
        assert pred.shape[2] == 2  # x, y
        assert cov is not None
        
    def test_uncertainty_estimation(self, ensemble, mock_graph_batch):
        """EN-F02: Std dev from disagreement."""
        mean_pred, std_pred, cov_pred = ensemble.predict_with_uncertainty(mock_graph_batch)
        
        # Check shapes match
        assert mean_pred.shape == std_pred.shape
        assert std_pred.shape[1] == 5
        assert std_pred.shape[2] == 2
        
        # Std should be non-negative
        assert (std_pred >= 0).all()
        
    def test_individual_predictions(self, ensemble, mock_graph_batch):
        """EN-F03: Return all model outputs."""
        pred, cov, individual = ensemble(mock_graph_batch, return_individual=True)
        
        assert individual is not None
        assert len(individual) == 3  # 3 models in ensemble
        
        for ind_pred in individual:
            assert ind_pred.shape == pred.shape
            
    def test_weight_normalization(self, weighted_ensemble):
        """EN-F04: Weights sum to 1.0."""
        total = sum(weighted_ensemble.weights)
        assert abs(total - 1.0) < 1e-6
        
    def test_custom_weights_applied(self, weighted_ensemble, mock_graph_batch):
        """Custom weights affect predictions."""
        _, _, individual = weighted_ensemble(mock_graph_batch, return_individual=True)
        
        # Manually compute weighted average
        weights = weighted_ensemble.weights
        manual_pred = sum(w * p for w, p in zip(weights, individual))
        
        # Get ensemble prediction
        ensemble_pred, _, _ = weighted_ensemble(mock_graph_batch)
        
        # Should match
        assert torch.allclose(ensemble_pred, manual_pred, atol=1e-5)
        
    def test_ensemble_deterministic(self, ensemble, mock_graph_batch):
        """Ensemble predictions are deterministic in eval mode."""
        pred1, _, _ = ensemble(mock_graph_batch)
        pred2, _, _ = ensemble(mock_graph_batch)
        
        assert torch.allclose(pred1, pred2)


# ============================================================================
# Edge Case Tests (EN-E01 to EN-E04)
# ============================================================================

class TestEnsembleEdgeCases:
    """Edge case tests for ensemble."""
    
    def test_single_model_ensemble(self, mock_graph_batch):
        """EN-E01: Single model acts as passthrough."""
        model = NFLGraphTransformer(
            input_dim=7,
            hidden_dim=32,
            future_seq_len=5
        )
        model.eval()
        
        ensemble = EnsemblePredictor([model])
        
        # Get predictions from both
        with torch.no_grad():
            single_pred, single_cov, _ = model(mock_graph_batch)
            ensemble_pred, ensemble_cov, _ = ensemble(mock_graph_batch)
        
        # Should be identical
        assert torch.allclose(single_pred, ensemble_pred)
        
    def test_weights_not_sum_to_one(self, model_list):
        """EN-E02: Invalid weights raise error."""
        with pytest.raises(AssertionError):
            EnsemblePredictor(model_list, weights=[0.3, 0.3, 0.3])  # Sum = 0.9
            
    def test_empty_model_list(self):
        """EN-E03: Empty list raises error."""
        with pytest.raises((AssertionError, ValueError, IndexError, ZeroDivisionError)):
            EnsemblePredictor([])
            
    def test_mismatched_weights_count(self, model_list):
        """Wrong number of weights raises error."""
        with pytest.raises(AssertionError):
            EnsemblePredictor(model_list, weights=[0.5, 0.5])  # 2 weights, 3 models
            
    def test_uncertainty_single_model(self, mock_graph_batch):
        """Uncertainty is very small for single model."""
        model = NFLGraphTransformer(
            input_dim=7,
            hidden_dim=32,
            future_seq_len=5
        )
        model.eval()
        
        ensemble = EnsemblePredictor([model])
        _, std_pred, _ = ensemble.predict_with_uncertainty(mock_graph_batch)
        
        # With single model, std should be very small (from numerical precision + 1e-6 term)
        # The ensemble code has: variance = (weights * diff_sq).sum() + 1e-6 
        # so std will be sqrt(1e-6) = 1e-3
        assert std_pred.max() < 1e-2  # Allow for numerical precision


# ============================================================================
# Checkpoint Loading Tests (EN-F05)
# ============================================================================

class TestCheckpointLoading:
    """Tests for loading ensemble from checkpoints."""
    
    def test_save_and_load_model(self, mock_graph_batch):
        """Save model checkpoint and load into ensemble."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save models
            checkpoint_paths = []
            for i in range(2):
                model = NFLGraphTransformer(
                    input_dim=7,
                    hidden_dim=32,
                    future_seq_len=5
                )
                path = os.path.join(tmpdir, f"model_{i}.pt")
                torch.save(model.state_dict(), path)
                checkpoint_paths.append(path)
            
            # Load ensemble
            ensemble = load_ensemble(
                checkpoint_paths,
                NFLGraphTransformer,
                input_dim=7,
                hidden_dim=32,
                future_seq_len=5
            )
            
            # Test forward pass
            pred, cov, _ = ensemble(mock_graph_batch)
            assert pred.shape[1] == 5
            
    def test_load_lightning_checkpoint_format(self, mock_graph_batch):
        """Load from Lightning checkpoint format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create Lightning-style checkpoint
            model = NFLGraphTransformer(
                input_dim=7,
                hidden_dim=32,
                future_seq_len=5
            )
            
            # Lightning format has 'state_dict' key with 'model.' prefix
            checkpoint = {
                'state_dict': {f'model.{k}': v for k, v in model.state_dict().items()}
            }
            
            path = os.path.join(tmpdir, "lightning_model.ckpt")
            torch.save(checkpoint, path)
            
            # Load ensemble
            ensemble = load_ensemble(
                [path],
                NFLGraphTransformer,
                input_dim=7,
                hidden_dim=32,
                future_seq_len=5
            )
            
            pred, _, _ = ensemble(mock_graph_batch)
            assert pred is not None


# ============================================================================
# Performance Tests
# ============================================================================

class TestEnsemblePerformance:
    """Performance tests for ensemble."""
    
    def test_ensemble_inference_time(self, ensemble, mock_graph_batch):
        """Ensemble inference timing."""
        import time
        
        # Warmup
        _ = ensemble(mock_graph_batch)
        
        # Timed run
        start = time.time()
        for _ in range(10):
            _ = ensemble(mock_graph_batch)
        elapsed = (time.time() - start) / 10
        
        # Should be reasonably fast
        assert elapsed < 1.0, f"Ensemble inference took {elapsed:.3f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
