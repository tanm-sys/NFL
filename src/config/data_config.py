"""
Data Processing Configuration
Centralizes data loading and feature engineering settings.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    
    # Data Paths
    data_dir: str = "."
    
    # Graph Construction
    radius: float = 20.0  # yards for edge creation
    
    # Sequence Lengths
    history_len: int = 5
    future_seq_len: int = 10
    
    # Feature Normalization
    normalize_positions: bool = True
    field_length: float = 120.0  # yards (including endzones)
    field_width: float = 53.3   # yards
    
    # Data Splits
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    split_by_play: bool = True  # Prevents data leakage
    random_seed: int = 42
    
    # Column Mappings
    role_map: dict = None
    side_map: dict = None
    formation_map: dict = None
    alignment_map: dict = None
    
    def __post_init__(self):
        """Set default mappings."""
        if self.role_map is None:
            self.role_map = {
                "Defensive Coverage": 0, 
                "Other Route Runner": 1, 
                "Passer": 2, 
                "Targeted Receiver": 3
            }
        if self.side_map is None:
            self.side_map = {"Defense": 0, "Offense": 1}
        if self.formation_map is None:
            self.formation_map = {
                "SHOTGUN": 0, "EMPTY": 1, "SINGLEBACK": 2, 
                "PISTOL": 3, "I_FORM": 4, "JUMBO": 5, "WILDCAT": 6
            }
        if self.alignment_map is None:
            self.alignment_map = {
                "2x2": 0, "3x1": 1, "3x2": 2, "2x1": 3, 
                "4x1": 4, "1x1": 5, "4x0": 6, "3x3": 7, "3x0": 8
            }
