"""
Ensemble Model for NFL Trajectory Prediction (P2)
Combines multiple models for improved prediction robustness.
"""
import torch
import torch.nn as nn
from typing import List, Optional


class EnsemblePredictor(nn.Module):
    """
    Ensemble of trajectory predictors.
    Combines predictions from multiple models using weighted averaging.
    """
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        """
        Args:
            models: List of trained NFLGraphTransformer models
            weights: Optional weights for each model (default: equal weights)
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = weights
        
        assert len(models) == len(weights), "Must have same number of models and weights"
        assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"
        
    def forward(self, data, return_individual=False):
        """
        Forward pass through ensemble.
        
        Args:
            data: PyG Batch Data object
            return_individual: If True, also return individual predictions
            
        Returns:
            predictions: [N, T, 2] weighted average trajectory predictions
            cov_pred: [B, 1] average coverage logits
            individual: Optional list of individual model predictions
        """
        all_predictions = []
        all_cov_preds = []
        
        for model in self.models:
            with torch.no_grad():
                pred, cov, _ = model(data)
            all_predictions.append(pred)
            all_cov_preds.append(cov)
        
        # Weighted average
        weighted_pred = sum(w * p for w, p in zip(self.weights, all_predictions))
        weighted_cov = sum(w * c for w, c in zip(self.weights, all_cov_preds))
        
        if return_individual:
            return weighted_pred, weighted_cov, all_predictions
        return weighted_pred, weighted_cov, None
    
    def predict_with_uncertainty(self, data):
        """
        Predict with uncertainty estimation from ensemble disagreement.
        
        Returns:
            mean_pred: [N, T, 2] mean prediction
            std_pred: [N, T, 2] prediction uncertainty (std dev)
            cov_pred: [B, 1] coverage prediction
        """
        all_predictions = []
        all_cov_preds = []
        
        for model in self.models:
            with torch.no_grad():
                pred, cov, _ = model(data)
            all_predictions.append(pred)
            all_cov_preds.append(cov)
        
        # Stack predictions: [num_models, N, T, 2]
        stacked = torch.stack(all_predictions, dim=0)
        
        # Weighted mean and standard deviation
        weights = torch.tensor(self.weights, device=stacked.device).view(-1, 1, 1, 1)
        mean_pred = (weights * stacked).sum(dim=0)
        
        # Compute weighted variance
        diff_sq = (stacked - mean_pred.unsqueeze(0)) ** 2
        variance = (weights * diff_sq).sum(dim=0)
        std_pred = torch.sqrt(variance + 1e-6)
        
        # Average coverage
        cov_pred = sum(w * c for w, c in zip(self.weights, all_cov_preds))
        
        return mean_pred, std_pred, cov_pred


def load_ensemble(checkpoint_paths: List[str], model_class, **model_kwargs):
    """
    Load an ensemble from saved checkpoints.
    
    Args:
        checkpoint_paths: List of paths to model checkpoints
        model_class: NFLGraphTransformer class
        **model_kwargs: Arguments for model constructor
        
    Returns:
        EnsemblePredictor with loaded models
    """
    models = []
    for path in checkpoint_paths:
        model = model_class(**model_kwargs)
        # weights_only=False needed for Lightning checkpoints (PyTorch 2.6+)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        # Handle Lightning checkpoint format
        if 'state_dict' in checkpoint:
            state_dict = {k.replace('model.', ''): v 
                         for k, v in checkpoint['state_dict'].items() 
                         if k.startswith('model.')}
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        models.append(model)
    
    return EnsemblePredictor(models)
