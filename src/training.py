"""
GPU-accelerated training with mixed precision (AMP) for CLTV models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import time
from typing import Dict, Optional, Tuple
import numpy as np
from pathlib import Path


class Trainer:
    """Trainer with mixed precision support for CLTV models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        use_amp: bool = True,
        use_zero_inflated_loss: bool = False
    ):
        """
        Args:
            model: PyTorch model
            train_loader: Training DataLoader
            test_loader: Test DataLoader
            device: Device for training
            learning_rate: Learning rate
            weight_decay: L2 regularization
            use_amp: Whether to use automatic mixed precision
            use_zero_inflated_loss: Whether to use zero-inflated loss
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()
        self.use_zero_inflated_loss = use_zero_inflated_loss
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Loss function
        if use_zero_inflated_loss:
            from src.models import ZeroInflatedLoss
            self.criterion = ZeroInflatedLoss(zero_weight=0.3)
        else:
            self.criterion = nn.MSELoss()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        print(f"\nTrainer initialized:")
        print(f"  Device: {device}")
        print(f"  Mixed precision: {self.use_amp}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Zero-inflated loss: {use_zero_inflated_loss}")
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (features, targets) in enumerate(self.train_loader):
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.use_amp:
                with autocast():
                    predictions = self.model(features)
                    
                    # Handle zero-inflated loss
                    if self.use_zero_inflated_loss and hasattr(self.model, 'get_zero_probability'):
                        zero_probs = self.model.get_zero_probability(features)
                        loss = self.criterion(predictions, targets, zero_probs)
                    else:
                        loss = self.criterion(predictions, targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(features)
                
                if self.use_zero_inflated_loss and hasattr(self.model, 'get_zero_probability'):
                    zero_probs = self.model.get_zero_probability(features)
                    loss = self.criterion(predictions, targets, zero_probs)
                else:
                    loss = self.criterion(predictions, targets)
                
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """Evaluate on test set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        for features, targets in self.test_loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            if self.use_amp:
                with autocast():
                    predictions = self.model(features)
                    loss = self.criterion(predictions, targets)
            else:
                predictions = self.model(features)
                loss = self.criterion(predictions, targets)
            
            total_loss += loss.item()
            num_batches += 1
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
        
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        return total_loss / num_batches, predictions, targets
    
    def train(
        self,
        num_epochs: int,
        save_dir: str = 'experiments',
        model_name: str = 'model'
    ) -> Dict[str, list]:
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs
            save_dir: Directory to save model checkpoints
            model_name: Name for saved model
            
        Returns:
            Dictionary with training history
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"\nStarting training for {num_epochs} epochs...")
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Evaluate
            val_loss, predictions, targets = self.evaluate()
            self.val_losses.append(val_loss)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            epoch_time = time.time() - start_time
            
            # Print progress
            print(f"Epoch {epoch}/{num_epochs} ({epoch_time:.1f}s) - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                model_path = Path(save_dir) / f"{model_name}_best.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, model_path)
                print(f"  Saved best model to {model_path}")
        
        print(f"\nTraining complete! Best validation loss: {self.best_val_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
