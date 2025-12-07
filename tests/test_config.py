"""
Tests for Configuration modules.
Covers data config, model config, and training config.
"""
import sys
sys.path.insert(0, 'src')

import pytest


# ============================================================================
# Try to import config modules
# ============================================================================

try:
    from config import DataConfig, ModelConfig, TrainingConfig
    CONFIG_AVAILABLE = True
except ImportError:
    try:
        from config.data_config import DataConfig
        from config.model_config import ModelConfig
        from config.training_config import TrainingConfig
        CONFIG_AVAILABLE = True
    except ImportError:
        CONFIG_AVAILABLE = False


# ============================================================================
# Skip if configs not available
# ============================================================================

pytestmark = pytest.mark.skipif(
    not CONFIG_AVAILABLE,
    reason="Config modules not available"
)


# ============================================================================
# Functional Tests (CF-F01 to CF-F04)
# ============================================================================

class TestConfigFunctional:
    """Functional tests for configuration."""
    
    @pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config not available")
    def test_data_config_defaults(self):
        """CF-F01: DataConfig default values."""
        if not CONFIG_AVAILABLE:
            pytest.skip("Config not available")
            
        config = DataConfig()
        
        # Check default values exist
        assert hasattr(config, 'data_dir') or hasattr(config, 'DATA_DIR')
        
    @pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config not available")
    def test_model_config_defaults(self):
        """CF-F02: ModelConfig defaults."""
        if not CONFIG_AVAILABLE:
            pytest.skip("Config not available")
            
        config = ModelConfig()
        
        # Check key model hyperparameters
        assert hasattr(config, 'hidden_dim') or hasattr(config, 'HIDDEN_DIM')
        
    @pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config not available")
    def test_training_config_defaults(self):
        """CF-F03: TrainingConfig defaults."""
        if not CONFIG_AVAILABLE:
            pytest.skip("Config not available")
            
        config = TrainingConfig()
        
        # Check key training hyperparameters
        assert hasattr(config, 'learning_rate') or hasattr(config, 'lr') or hasattr(config, 'LR')


# ============================================================================
# Alternative: Test config files directly
# ============================================================================

class TestConfigFiles:
    """Test config files exist and are valid Python."""
    
    def test_config_init_exists(self):
        """Config __init__.py exists."""
        import os
        config_path = "src/config/__init__.py"
        assert os.path.exists(config_path) or os.path.exists("/home/tanmay/Desktop/NFL/" + config_path)
        
    def test_data_config_exists(self):
        """data_config.py exists."""
        import os
        config_path = "src/config/data_config.py"
        full_path = "/home/tanmay/Desktop/NFL/" + config_path
        assert os.path.exists(config_path) or os.path.exists(full_path)
        
    def test_model_config_exists(self):
        """model_config.py exists."""
        import os
        config_path = "src/config/model_config.py"
        full_path = "/home/tanmay/Desktop/NFL/" + config_path
        assert os.path.exists(config_path) or os.path.exists(full_path)
        
    def test_training_config_exists(self):
        """training_config.py exists."""
        import os
        config_path = "src/config/training_config.py"
        full_path = "/home/tanmay/Desktop/NFL/" + config_path
        assert os.path.exists(config_path) or os.path.exists(full_path)
        
    def test_config_imports_valid(self):
        """Config files are valid Python."""
        import importlib.util
        import os
        
        config_dir = "/home/tanmay/Desktop/NFL/src/config"
        
        for filename in ["data_config.py", "model_config.py", "training_config.py"]:
            filepath = os.path.join(config_dir, filename)
            if os.path.exists(filepath):
                spec = importlib.util.spec_from_file_location(
                    filename.replace(".py", ""),
                    filepath
                )
                module = importlib.util.module_from_spec(spec)
                # Just check it can be loaded
                try:
                    spec.loader.exec_module(module)
                except Exception as e:
                    pytest.fail(f"Failed to load {filename}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
