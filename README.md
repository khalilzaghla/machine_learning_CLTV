# GPU-Accelerated CLTV Prediction

A production-grade Customer Lifetime Value (CLTV) prediction system using PyTorch with GPU acceleration and advanced neural architectures.

## Features

- **Scalable Data Processing**: Vectorized pandas operations for million-row datasets
- **GPU Acceleration**: CUDA support with automatic mixed precision (AMP)
- **Advanced Models**: Mixture-of-Experts with zero-inflation handling
- **Distribution-Aware**: Handles heavily skewed revenue distributions
- **Ranking Metrics**: Gini coefficient, Top-K revenue capture, Lift scores
- **Production-Ready**: Modular architecture, reproducible experiments

## Project Structure

```
CLTV/
├── data/                          # Data files
│   ├── online_retail_II.xlsx      # Raw data
│   ├── train_processed.csv        # Processed train set
│   └── test_processed.csv         # Processed test set
├── src/                           # Source code
│   ├── __init__.py
│   ├── data_preprocessing.py      # Data cleaning & feature engineering
│   ├── dataset.py                 # PyTorch Dataset & DataLoader
│   ├── models.py                  # Neural architectures
│   ├── training.py                # Training loops with mixed precision
│   ├── evaluation.py              # Metrics and evaluation
│   └── utils.py                   # Utility functions
├── experiments/                   # Training outputs
├── notebooks/                     # Jupyter notebooks (optional)
├── main.py                        # Main entry point
└── README.md                      # This file
```

## Installation

Install required packages:

```bash
pip install pandas numpy scikit-learn torch torchvision matplotlib seaborn openpyxl
```

For GPU support, ensure you have CUDA-compatible PyTorch installed:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Quick Start

Run the full pipeline:

```bash
python main.py
```

This will:
1. Load and preprocess the Online Retail II dataset
2. Engineer RFM-style features (Recency, Frequency, Monetary)
3. Create time-based train/test split (no data leakage)
4. Train baseline neural network
5. Train mixture-of-experts model with zero-inflation
6. Evaluate both models with comprehensive metrics
7. Generate visualizations (training curves, predictions, Lorenz curves)
8. Save all outputs to timestamped `experiments/` directory

### Custom Configuration

Modify hyperparameters in `main.py`:

```python
# Data preprocessing
preprocessor = DataPreprocessor(
    observation_days=365,  # Historical window
    prediction_days=90     # Prediction horizon
)

# DataLoader settings
batch_size = 256
num_workers = 4

# Model architecture
num_experts = 4
expert_hidden_dim = 128
gate_hidden_dim = 64

# Training
num_epochs = 50
learning_rate = 1e-3
weight_decay = 1e-5
```

### Using Individual Modules

```python
from src import DataPreprocessor, create_dataloaders, create_model, Trainer

# Preprocess data
preprocessor = DataPreprocessor(observation_days=365, prediction_days=90)
train_df, test_df = preprocessor.process('data/online_retail_II.xlsx')

# Create DataLoaders
train_loader, test_loader, scaler, features = create_dataloaders(
    train_df, test_df, batch_size=256
)

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model('moe', input_dim=len(features), device=device)

# Train
trainer = Trainer(model, train_loader, test_loader, device)
history = trainer.train(num_epochs=50)
```

## Model Architecture

### Baseline Neural Network
- 3-layer feedforward network (256 → 128 → 64)
- Batch normalization and dropout (0.2)
- ReLU activation for positive predictions
- Simple but effective baseline

### Mixture-of-Experts (MoE)
- **Multiple Expert Networks**: 4 specialized networks (default)
- **Gating Network**: Routes inputs to appropriate experts via softmax
- **Zero-Inflation Component**: Handles customers with zero future revenue
- **Custom Loss**: Combines MSE regression + binary classification for zeros
- **Architecture**: Each expert has 2 hidden layers with batch norm + dropout

## Evaluation Metrics

### Accuracy Metrics
- **RMSE** (Root Mean Squared Error): Overall prediction accuracy
- **MAE** (Mean Absolute Error): Average absolute prediction error

### Ranking Metrics
- **Gini Coefficient**: Ranking quality (0-1, higher = better)
- **Top-K Revenue Capture**: % of total revenue in top K% of predicted customers
- **Lift @ K**: How much better than random targeting (>1 = better)

These ranking metrics are crucial for business applications where you want to target high-value customers.

## Key Features

### 1. Scalable Data Preprocessing
- Vectorized pandas operations (no Python loops)
- Handles millions of transactions efficiently
- Removes cancellations, invalid data, outliers
- Creates RFM features: recency, frequency, monetary, and derived metrics

### 2. GPU Optimization
- Automatic CUDA detection
- Mixed precision training (FP16/FP32) for 2-3x speedup
- Pinned memory for faster CPU→GPU transfer
- Multi-worker data loading
- Gradient scaling for numerical stability

### 3. Zero-Inflated Modeling
- Many customers have zero future revenue
- MoE includes dedicated zero-inflation gate
- Custom loss balances regression and classification
- Better handles skewed distributions

### 4. Time-Based Split
- Observation period for feature engineering
- Separate prediction period for target
- Prevents data leakage
- Mimics real-world deployment scenario

## Outputs

Each experiment generates:
- **Model Checkpoints**: Best model weights (`.pt` files)
- **Training Plots**: Loss curves over epochs
- **Prediction Plots**: 
  - Scatter plot (predicted vs actual)
  - Residual plot
  - Distribution comparison
  - Lorenz curve (cumulative revenue capture)
- **Metrics CSV**: Comparison table for all models
- **Scaler & Features**: For deployment/inference

## Dataset

The pipeline expects Online Retail II format with columns:
- `Invoice`: Transaction ID
- `Customer ID`: Unique customer identifier
- `InvoiceDate`: Transaction timestamp
- `Quantity`: Items purchased
- `Price`: Unit price

Place your dataset as `data/online_retail_II.xlsx` or modify the path in `main.py`.

## Requirements

- Python 3.8+
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- PyTorch >= 1.12.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- openpyxl >= 3.0.0 (for Excel files)

## GPU Support

The pipeline automatically detects and uses CUDA if available:
- Mixed precision training with `torch.cuda.amp`
- Automatic fallback to CPU if no GPU found
- For CPU-only: set `num_workers=0` in DataLoaders

To check GPU availability:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

## Performance

On typical hardware:
- **Dataset**: ~1M transactions, ~5K customers
- **Preprocessing**: ~30 seconds (CPU)
- **Training**: ~2-3 min/epoch (GPU), ~10-15 min/epoch (CPU)
- **Total Runtime**: ~10-15 minutes (GPU), ~1-2 hours (CPU)

Mixed precision provides 2-3x speedup with no accuracy loss.

## Troubleshooting

### Out of Memory
Reduce batch size in `main.py`:
```python
batch_size = 128  # or 64
```

### CPU Only
Set workers to 0:
```python
num_workers = 0
pin_memory = False
```

### Import Errors
Ensure you're running from the CLTV directory:
```bash
cd CLTV
python main.py
```

## Citation

If you use this code in your research, please cite:

```
@software{cltv_prediction,
  title = {GPU-Accelerated CLTV Prediction with Mixture-of-Experts},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/cltv}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Contact

For questions or issues, please open a GitHub issue or contact [your email].
