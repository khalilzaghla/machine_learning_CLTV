"""
Neural network models for CLTV prediction.
Includes baseline and advanced mixture-of-experts architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class BaselineNN(nn.Module):
    """Simple baseline neural network for CLTV prediction."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [256, 128, 64]):
        """
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (batch_size, input_dim)
            
        Returns:
            Predictions (batch_size, 1)
        """
        return F.relu(self.network(x))  # ReLU to ensure positive predictions


class ExpertNetwork(nn.Module):
    """Single expert network in mixture-of-experts."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GatingNetwork(nn.Module):
    """Gating network for routing inputs to experts."""
    
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 64):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            Gate weights (batch_size, num_experts) - softmax probabilities
        """
        return F.softmax(self.network(x), dim=1)


class MixtureOfExpertsCLTV(nn.Module):
    """
    Mixture-of-Experts model for CLTV prediction.
    Handles heterogeneous customer segments and zero-inflated distributions.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int = 4,
        expert_hidden_dim: int = 128,
        gate_hidden_dim: int = 64,
        use_zero_inflation: bool = True
    ):
        """
        Args:
            input_dim: Number of input features
            num_experts: Number of expert networks
            expert_hidden_dim: Hidden dimension for expert networks
            gate_hidden_dim: Hidden dimension for gating network
            use_zero_inflation: Whether to use zero-inflation modeling
        """
        super().__init__()
        
        self.num_experts = num_experts
        self.use_zero_inflation = use_zero_inflation
        
        # Create expert networks
        self.experts = nn.ModuleList([
            ExpertNetwork(input_dim, expert_hidden_dim)
            for _ in range(num_experts)
        ])
        
        # Gating network for routing
        self.gating = GatingNetwork(input_dim, num_experts, gate_hidden_dim)
        
        # Zero-inflation component
        if use_zero_inflation:
            self.zero_gate = nn.Sequential(
                nn.Linear(input_dim, gate_hidden_dim),
                nn.BatchNorm1d(gate_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(gate_hidden_dim, 1),
                nn.Sigmoid()  # Probability of being zero
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (batch_size, input_dim)
            
        Returns:
            Predictions (batch_size, 1)
        """
        # Get gating weights
        gate_weights = self.gating(x)  # (batch_size, num_experts)
        
        # Get predictions from all experts
        expert_outputs = []
        for expert in self.experts:
            expert_out = F.relu(expert(x))  # Ensure positive predictions
            expert_outputs.append(expert_out)
        
        # Stack expert outputs: (batch_size, num_experts, 1)
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # Weighted combination: (batch_size, 1, num_experts) @ (batch_size, num_experts, 1)
        gate_weights_expanded = gate_weights.unsqueeze(1)  # (batch_size, 1, num_experts)
        mixture_output = torch.bmm(gate_weights_expanded, expert_outputs).squeeze(1)
        
        # Apply zero-inflation if enabled
        if self.use_zero_inflation:
            zero_prob = self.zero_gate(x)  # (batch_size, 1)
            # Expected value: (1 - p_zero) * mixture_output
            mixture_output = (1 - zero_prob) * mixture_output
        
        return mixture_output
    
    def get_expert_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get gating weights for analysis."""
        return self.gating(x)
    
    def get_zero_probability(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Get zero-inflation probability for analysis."""
        if self.use_zero_inflation:
            return self.zero_gate(x)
        return None


class ZeroInflatedLoss(nn.Module):
    """
    Custom loss for zero-inflated data.
    Combines MSE with a zero-classification component.
    """
    
    def __init__(self, zero_weight: float = 0.3):
        """
        Args:
            zero_weight: Weight for zero-classification loss
        """
        super().__init__()
        self.zero_weight = zero_weight
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        zero_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            predictions: Model predictions (batch_size, 1)
            targets: Ground truth (batch_size, 1)
            zero_probs: Zero-inflation probabilities (batch_size, 1)
            
        Returns:
            Combined loss
        """
        # Regression loss
        mse_loss = self.mse(predictions, targets)
        
        # Zero-classification loss
        if zero_probs is not None:
            is_zero = (targets == 0).float()
            zero_loss = self.bce(zero_probs, is_zero)
            return (1 - self.zero_weight) * mse_loss + self.zero_weight * zero_loss
        
        return mse_loss


def create_model(
    model_type: str,
    input_dim: int,
    device: torch.device,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: 'baseline' or 'moe' (mixture-of-experts)
        input_dim: Number of input features
        device: Device to place model on
        **kwargs: Additional model-specific arguments
        
    Returns:
        Model on specified device
    """
    if model_type == 'baseline':
        model = BaselineNN(input_dim, **kwargs)
    elif model_type == 'moe':
        model = MixtureOfExpertsCLTV(input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nCreated {model_type} model with {num_params:,} parameters")
    
    return model
