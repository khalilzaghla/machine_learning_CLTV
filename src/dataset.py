"""
GPU-optimized PyTorch Dataset and DataLoader for CLTV prediction.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, List
import pickle
import os


class CLTVDataset(Dataset):
    """PyTorch Dataset for CLTV prediction."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'future_revenue',
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False
    ):
        """
        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Name of target column
            scaler: Pre-fitted scaler (for test set)
            fit_scaler: Whether to fit scaler on this data (for train set)
        """
        self.feature_cols = feature_cols
        self.target_col = target_col
        
        # Extract features and target
        X = df[feature_cols].values.astype(np.float32)
        y = df[target_col].values.astype(np.float32)
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        if fit_scaler:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        elif scaler is not None:
            self.scaler = scaler
            X = self.scaler.transform(X)
        else:
            self.scaler = None
        
        # Convert to tensors
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).unsqueeze(1)  # Shape: (n, 1)
        
        print(f"Dataset created: {len(self)} samples, {self.X.shape[1]} features")
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
    
    def get_scaler(self) -> Optional[StandardScaler]:
        return self.scaler


def create_dataloaders(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
    target_col: str = 'future_revenue'
) -> Tuple[DataLoader, DataLoader, StandardScaler, List[str]]:
    """
    Create GPU-optimized DataLoaders for train and test sets.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        target_col: Name of target column
        
    Returns:
        Tuple of (train_loader, test_loader, scaler, feature_cols)
    """
    # Identify feature columns (exclude CustomerID and target)
    feature_cols = [col for col in train_df.columns 
                   if col not in ['CustomerID', target_col]]
    
    print(f"\nCreating DataLoaders with {len(feature_cols)} features")
    print(f"Features: {feature_cols}")
    
    # Create train dataset with fitted scaler
    train_dataset = CLTVDataset(
        train_df,
        feature_cols=feature_cols,
        target_col=target_col,
        fit_scaler=True
    )
    
    # Create test dataset with train scaler
    test_dataset = CLTVDataset(
        test_df,
        feature_cols=feature_cols,
        target_col=target_col,
        scaler=train_dataset.get_scaler(),
        fit_scaler=False
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return train_loader, test_loader, train_dataset.get_scaler(), feature_cols


def save_scaler_and_features(scaler: StandardScaler, feature_cols: List[str], 
                             output_dir: str = 'experiments'):
    """Save scaler and feature columns for deployment."""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(os.path.join(output_dir, 'feature_cols.pkl'), 'wb') as f:
        pickle.dump(feature_cols, f)
    
    print(f"Saved scaler and feature columns to {output_dir}")
