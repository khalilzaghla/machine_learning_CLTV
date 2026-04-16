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

---

## 📚 References

This project draws from the following key academic works in Customer Lifetime Value (CLTV) prediction and related fields:

### Core CLTV Prediction Papers

| # | Paper | Authors | Venue | Year | Link |
|---|---|---|---|---|---|
| [1] | **OptDist: Learning Optimal Distribution for Customer Lifetime Value Prediction** | Yunpeng Weng et al. | CIKM | 2024 | [arXiv:2408.08585](https://arxiv.org/abs/2408.08585) |
| [2] | **A Deep Probabilistic Model for Customer Lifetime Value Prediction (ZILN)** | Xiaojing Wang, Tianqi Liu, Jingang Miao | arXiv | 2019 | [arXiv:1912.07753](https://arxiv.org/abs/1912.07753) |
| [3] | **Billion-user Customer Lifetime Value Prediction: An Industrial-scale Solution from Kuaishou (MDME)** | Kunpeng Li et al. | CIKM | 2022 | [ACM DL](https://dl.acm.org/doi/10.1145/3511808.3557152) |
| [4] | **MDAN: Multi-distribution Adaptive Networks for LTV Prediction** | Wenshuang Liu et al. | PAKDD | 2024 | [Springer](https://link.springer.com/chapter/10.1007/978-981-97-2266-2_32) |
| [5] | **Feature Missing-aware Routing-and-Fusion Network for CLTV Prediction (MarfNet)** | Xuejiao Yang et al. | WSDM | 2023 | [ACM DL](https://dl.acm.org/doi/10.1145/3539597.3570402) |
| [6] | **Out of the Box Thinking: Improving CLTV Modelling via Expert Routing and Game Whale Detection (ExpLTV)** | Shijie Zhang et al. | CIKM | 2023 | [ACM DL](https://dl.acm.org/doi/10.1145/3583780.3615205) |
| [7] | **perCLTV: A General System for Personalized Customer Lifetime Value Prediction in Online Games** | Shiwei Zhao et al. | ACM TOIS | 2023 | [ACM DL](https://dl.acm.org/doi/10.1145/3548456) |

### Classical & Statistical CLTV Methods

| # | Paper | Authors | Venue | Year |
|---|---|---|---|---|
| [8] | **RFM and CLV: Using Iso-value Curves for Customer Base Analysis** | Fader, Hardie, Lee | J. Marketing Research | 2005 |
| [9] | **Counting Your Customers: The Easy Way — Pareto/NBD Model** | Fader, Hardie, Lee | Marketing Science | 2005 |
| [10] | **Customer Lifetime Value Prediction Using Embeddings** | Chamberlain et al. | KDD | 2017 |
| [11] | **Improved Customer Lifetime Value Prediction with Sequence-to-Sequence Learning** | Bauer & Jannach | ACM TKDD | 2021 |

### Foundational Deep Learning Components Used in This Project

| # | Component | Paper | Authors | Year |
|---|---|---|---|---|
| [12] | **Mixture of Experts** | Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer | Shazeer et al. | 2017 |
| [13] | **Gumbel-Softmax** | Categorical Reparameterization with Gumbel-Softmax | Jang, Gu, Poole | ICLR 2017 |
| [14] | **Focal Loss** | Focal Loss for Dense Object Detection | Lin et al. | ICCV 2017 |
| [15] | **Adam Optimizer** | Adam: A Method for Stochastic Optimization | Kingma & Ba | ICLR 2015 |

### Dataset

| # | Dataset | Source | Description |
|---|---|---|---|
| [16] | **Online Retail II** | [UCI ML Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) | Transactional data from a UK-based online retailer (2009–2011), ~1M rows |

---

> **Note:** This project is inspired by the methodology of **OptDist** [1] and **ZILN** [2], adapting the distribution-aware and zero-inflation modeling paradigms into a GPU-accelerated PyTorch pipeline using the publicly available Online Retail II dataset.
