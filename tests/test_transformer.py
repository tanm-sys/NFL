"""
Comprehensive tests for Transformer model.
Covers player encoder, trajectory decoder, and full model.
"""
import sys
sys.path.insert(0, 'src')

import pytest
import torch
import torch.nn as nn

from models.transformer import PlayerEncoder, TrajectoryDecoder, NFLTransformer


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_input():
    """Create sample input tensor [B, S, A, F]."""
    batch_size = 4
    seq_len = 10
    num_agents = 23
    num_features = 6
    return torch.randn(batch_size, seq_len, num_agents, num_features)


@pytest.fixture
def player_encoder():
    """Create player encoder."""
    return PlayerEncoder(input_dim=6, hidden_dim=64, num_heads=4)


@pytest.fixture
def trajectory_decoder():
    """Create trajectory decoder."""
    return TrajectoryDecoder(hidden_dim=64, num_heads=4, future_seq_len=10)


@pytest.fixture
def full_model():
    """Create full transformer model."""
    return NFLTransformer(
        input_dim=6,
        hidden_dim=64,
        num_heads=4,
        future_seq_len=10
    )


# ============================================================================
# PlayerEncoder Tests (TF-F01, TF-F04)
# ============================================================================

class TestPlayerEncoder:
    """Tests for PlayerEncoder."""
    
    def test_forward(self, player_encoder, sample_input):
        """TF-F01: Encoder forward pass."""
        output = player_encoder(sample_input)
        
        # Output shape: [B, S, A, hidden_dim]
        assert output.shape[0] == sample_input.shape[0]  # Batch
        assert output.shape[1] == sample_input.shape[1]  # Sequence
        assert output.shape[2] == sample_input.shape[2]  # Agents
        assert output.shape[3] == 64  # hidden_dim
        
    def test_attention_applied(self, player_encoder, sample_input):
        """TF-F04: Spatial attention works."""
        # Different inputs should give different outputs
        output1 = player_encoder(sample_input)
        output2 = player_encoder(sample_input * 2)
        
        assert not torch.allclose(output1, output2)
        
    def test_different_batch_sizes(self, player_encoder):
        """Various batch sizes."""
        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, 10, 23, 6)
            output = player_encoder(x)
            assert output.shape[0] == batch_size
            
    def test_different_hidden_dims(self, sample_input):
        """Various hidden dimensions."""
        for hidden_dim in [32, 64, 128]:
            encoder = PlayerEncoder(input_dim=6, hidden_dim=hidden_dim)
            output = encoder(sample_input)
            assert output.shape[3] == hidden_dim


# ============================================================================
# TrajectoryDecoder Tests (TF-F02, TF-F05)
# ============================================================================

class TestTrajectoryDecoder:
    """Tests for TrajectoryDecoder."""
    
    def test_forward(self, trajectory_decoder):
        """TF-F02: Decoder forward pass."""
        # Input: encoded sequence [B, S, A, H]
        x = torch.randn(4, 10, 23, 64)
        output = trajectory_decoder(x)
        
        # Output: [B, Future, A, 2]
        assert output.shape[0] == 4  # Batch
        assert output.shape[1] == 10  # future_seq_len
        assert output.shape[2] == 23  # Agents
        assert output.shape[3] == 2  # x, y
        
    def test_temporal_dependencies(self, trajectory_decoder):
        """TF-F05: Temporal dependencies captured."""
        # Same spatial state but different temporal ordering
        x1 = torch.randn(4, 10, 23, 64)
        
        # Reverse temporal order
        x2 = x1.flip(dims=[1])
        
        output1 = trajectory_decoder(x1)
        output2 = trajectory_decoder(x2)
        
        # Different temporal order should give different predictions
        assert not torch.allclose(output1, output2)
        
    def test_different_future_lengths(self):
        """Various future sequence lengths."""
        x = torch.randn(4, 10, 23, 64)
        
        for future_len in [5, 10, 20]:
            decoder = TrajectoryDecoder(
                hidden_dim=64,
                num_heads=4,
                future_seq_len=future_len
            )
            output = decoder(x)
            assert output.shape[1] == future_len


# ============================================================================
# NFLTransformer E2E Tests (TF-F03)
# ============================================================================

class TestNFLTransformer:
    """Tests for full NFLTransformer model."""
    
    def test_forward(self, full_model, sample_input):
        """TF-F03: Full model forward."""
        output = full_model(sample_input)
        
        # Output: [B, Future, A, 2]
        assert output.shape[0] == sample_input.shape[0]
        assert output.shape[1] == 10  # future_seq_len
        assert output.shape[2] == sample_input.shape[2]  # Agents
        assert output.shape[3] == 2  # x, y
        
    def test_gradient_flow(self, full_model, sample_input):
        """Gradients flow through model."""
        output = full_model(sample_input)
        loss = output.mean()
        loss.backward()
        
        # Check gradients exist
        has_grad = False
        for param in full_model.parameters():
            if param.grad is not None:
                has_grad = True
                break
                
        assert has_grad
        
    def test_deterministic_output(self, full_model, sample_input):
        """Deterministic in eval mode."""
        full_model.eval()
        
        with torch.no_grad():
            output1 = full_model(sample_input)
            output2 = full_model(sample_input)
        
        assert torch.allclose(output1, output2)


# ============================================================================
# Edge Case Tests (TF-E01 to TF-E03)
# ============================================================================

class TestTransformerEdgeCases:
    """Edge case tests for transformer."""
    
    def test_single_agent(self, full_model):
        """TF-E01: Single agent input."""
        x = torch.randn(4, 10, 1, 6)
        output = full_model(x)
        
        assert output.shape[2] == 1
        
    def test_single_timestep(self, full_model):
        """TF-E02: Seq_len=1."""
        x = torch.randn(4, 1, 23, 6)
        output = full_model(x)
        
        assert output.shape[0] == 4
        assert output.shape[1] == 10  # Still predicts full future
        
    def test_batch_size_one(self, full_model):
        """TF-E03: Batch=1."""
        x = torch.randn(1, 10, 23, 6)
        output = full_model(x)
        
        assert output.shape[0] == 1
        
    def test_large_agent_count(self):
        """Large number of agents."""
        model = NFLTransformer(input_dim=6, hidden_dim=32)
        x = torch.randn(2, 10, 50, 6)  # 50 agents
        
        output = model(x)
        assert output.shape[2] == 50


# ============================================================================
# Performance Tests
# ============================================================================

class TestTransformerPerformance:
    """Performance tests for transformer."""
    
    def test_inference_timing(self, full_model, sample_input):
        """Inference timing."""
        import time
        
        full_model.eval()
        
        # Warmup
        with torch.no_grad():
            _ = full_model(sample_input)
        
        # Timed run
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = full_model(sample_input)
        elapsed = (time.time() - start) / 10
        
        # Should be fast
        assert elapsed < 0.5, f"Inference took {elapsed:.3f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
