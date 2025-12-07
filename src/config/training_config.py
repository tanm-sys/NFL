"""
Training Configuration
Centralizes training hyperparameters and optimization settings.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Literal


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Optimizer
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: Literal["adam", "adamw", "sgd"] = "adamw"
    
    # Learning Rate Schedule
    lr_scheduler: Literal["cosine", "step", "plateau"] = "cosine"
    warmup_epochs: int = 5
    warmup_start_factor: float = 0.1
    cosine_t0: int = 10
    cosine_t_mult: int = 2
    min_lr: float = 1e-6
    
    # Training Loop
    max_epochs: int = 50
    batch_size: int = 32
    num_workers: int = 4
    gradient_clip_val: float = 1.0
    
    # Early Stopping
    early_stop_patience: int = 7
    early_stop_metric: str = "val_ade"
    early_stop_mode: Literal["min", "max"] = "min"
    
    # Loss Weights
    trajectory_weight: float = 1.0
    velocity_weight: float = 0.3
    acceleration_weight: float = 0.1
    coverage_weight: float = 0.5
    collision_weight: float = 0.05
    
    # Loss Type (P2)
    use_huber_loss: bool = False
    huber_delta: float = 1.0
    
    # Data Augmentation
    use_augmentation: bool = True
    flip_prob: float = 0.5
    noise_std: float = 0.1
    
    # Weeks to train on
    weeks: List[int] = field(default_factory=lambda: list(range(1, 19)))
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.max_epochs > 0, "max_epochs must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
