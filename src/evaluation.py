"""
Evaluation metrics for CLTV prediction including ranking metrics.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)


def gini_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Gini coefficient for ranking quality.
    Measures how well predictions rank actual values.
    
    Returns:
        Gini coefficient (higher is better, max=1.0)
    """
    # Sort by predictions (descending)
    sorted_indices = np.argsort(y_pred.flatten())[::-1]
    y_true_sorted = y_true.flatten()[sorted_indices]
    
    # Calculate cumulative actual values
    cumsum = np.cumsum(y_true_sorted)
    cumsum_percent = cumsum / cumsum[-1] if cumsum[-1] > 0 else cumsum
    
    # Calculate area under Lorenz curve
    n = len(y_true)
    index = np.arange(1, n + 1)
    lorenz_area = np.sum(cumsum_percent) / n
    
    # Gini = 2 * (area between Lorenz and diagonal) = 2 * (0.5 - lorenz_area)
    gini = 2 * (0.5 - lorenz_area)
    
    return gini


def top_k_revenue_capture(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k_percent: float = 0.1
) -> float:
    """
    Top-K revenue capture rate.
    What percentage of total revenue is captured by targeting top K% of predicted customers.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        k_percent: Percentage of customers to target (e.g., 0.1 for top 10%)
        
    Returns:
        Percentage of total revenue captured
    """
    k = int(len(y_pred) * k_percent)
    
    # Sort by predictions (descending)
    sorted_indices = np.argsort(y_pred.flatten())[::-1]
    
    # Get actual values of top-k predicted customers
    top_k_actual = y_true.flatten()[sorted_indices[:k]]
    
    # Calculate capture rate
    total_revenue = y_true.sum()
    captured_revenue = top_k_actual.sum()
    
    return (captured_revenue / total_revenue * 100) if total_revenue > 0 else 0.0


def lift_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k_percent: float = 0.1
) -> float:
    """
    Lift at top K%.
    How much better than random targeting.
    
    Returns:
        Lift score (>1 is better than random)
    """
    capture_rate = top_k_revenue_capture(y_true, y_pred, k_percent)
    random_rate = k_percent * 100
    
    return capture_rate / random_rate if random_rate > 0 else 0.0


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Comprehensive evaluation of CLTV predictions.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name for printing
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'gini': gini_coefficient(y_true, y_pred),
        'top_10_capture': top_k_revenue_capture(y_true, y_pred, 0.1),
        'top_20_capture': top_k_revenue_capture(y_true, y_pred, 0.2),
        'lift_at_10': lift_at_k(y_true, y_pred, 0.1),
        'lift_at_20': lift_at_k(y_true, y_pred, 0.2),
    }
    
    print(f"\n{model_name} Evaluation Metrics:")
    print("=" * 50)
    print(f"RMSE:              ${metrics['rmse']:.2f}")
    print(f"MAE:               ${metrics['mae']:.2f}")
    print(f"Gini Coefficient:  {metrics['gini']:.4f}")
    print(f"\nRanking Performance:")
    print(f"Top 10% Capture:   {metrics['top_10_capture']:.1f}%")
    print(f"Top 20% Capture:   {metrics['top_20_capture']:.1f}%")
    print(f"Lift @ 10%:        {metrics['lift_at_10']:.2f}x")
    print(f"Lift @ 20%:        {metrics['lift_at_20']:.2f}x")
    
    return metrics


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: str = None
):
    """Plot prediction quality visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Scatter plot
    axes[0, 0].scatter(y_true, y_pred, alpha=0.3, s=10)
    axes[0, 0].plot([y_true.min(), y_true.max()], 
                    [y_true.min(), y_true.max()], 
                    'r--', lw=2, label='Perfect prediction')
    axes[0, 0].set_xlabel('Actual CLTV')
    axes[0, 0].set_ylabel('Predicted CLTV')
    axes[0, 0].set_title(f'{model_name}: Predictions vs Actual')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals
    residuals = y_true.flatten() - y_pred.flatten()
    axes[0, 1].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted CLTV')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Distribution comparison
    axes[1, 0].hist(y_true.flatten(), bins=50, alpha=0.5, label='Actual', density=True)
    axes[1, 0].hist(y_pred.flatten(), bins=50, alpha=0.5, label='Predicted', density=True)
    axes[1, 0].set_xlabel('CLTV')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Distribution Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cumulative revenue capture (Lorenz curve)
    sorted_indices = np.argsort(y_pred.flatten())[::-1]
    y_true_sorted = y_true.flatten()[sorted_indices]
    cumsum = np.cumsum(y_true_sorted)
    cumsum_percent = cumsum / cumsum[-1] if cumsum[-1] > 0 else cumsum
    
    x = np.linspace(0, 100, len(cumsum_percent))
    axes[1, 1].plot(x, cumsum_percent * 100, label='Model', lw=2)
    axes[1, 1].plot([0, 100], [0, 100], 'r--', label='Random', lw=2)
    axes[1, 1].set_xlabel('% of Customers (by predicted rank)')
    axes[1, 1].set_ylabel('% of Total Revenue Captured')
    axes[1, 1].set_title('Cumulative Revenue Capture')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_training_history(history: Dict[str, list], save_path: str = None):
    """Plot training history."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history['train_losses']) + 1)
    ax.plot(epochs, history['train_losses'], label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_losses'], label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training history to {save_path}")
    
    plt.show()
