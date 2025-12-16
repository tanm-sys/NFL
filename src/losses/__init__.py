"""
Losses Module for NFL Trajectory Prediction
============================================

Advanced loss functions for world-class trajectory forecasting.
"""

from src.losses.contrastive_losses import (
    SocialNCELoss,
    TrajectoryContrastiveLoss,
    DiversityLoss,
    WinnerTakesAllLoss,
    EndpointFocalLoss,
)

__all__ = [
    "SocialNCELoss",
    "TrajectoryContrastiveLoss", 
    "DiversityLoss",
    "WinnerTakesAllLoss",
    "EndpointFocalLoss",
]
