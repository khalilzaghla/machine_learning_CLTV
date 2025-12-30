"""
Scalable data preprocessing and feature engineering for CLTV prediction.
Handles large-scale transactional data with vectorized pandas operations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Efficient data preprocessing for transactional CLTV data."""
    
    def __init__(self, observation_days: int = 365, prediction_days: int = 90):
        """
        Args:
            observation_days: Days of historical data to use for features
            prediction_days: Days ahead to predict CLTV
        """
        self.observation_days = observation_days
        self.prediction_days = prediction_days
        self.split_date = None
        self.observation_end = None
        
    def load_and_clean(self, filepath: str) -> pd.DataFrame:
        """
        Load and clean transactional data efficiently.
        
        Args:
            filepath: Path to CSV or Excel file
            
        Returns:
            Cleaned DataFrame
        """
        print("Loading data...")
        # Handle both CSV and Excel
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath, encoding='latin1', parse_dates=['InvoiceDate'])
        else:
            # Use read_only mode for faster Excel reading
            df = pd.read_excel(filepath, engine='openpyxl')
            # Parse dates after loading
            if 'InvoiceDate' in df.columns:
                df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        print(f"Initial shape: {df.shape}")
        
        # Clean data vectorized
        print("Cleaning data...")
        
        # Remove missing CustomerID
        df = df[df['Customer ID'].notna()].copy()
        
        # Convert CustomerID to string for consistency
        df['CustomerID'] = df['Customer ID'].astype(str)
        
        # Remove cancellations (invoices starting with 'C')
        df = df[~df['Invoice'].astype(str).str.startswith('C')]
        
        # Remove invalid quantities and prices
        df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
        
        # Create revenue column
        df['Revenue'] = df['Quantity'] * df['Price']
        
        # Remove outliers (optional - top 0.1% of revenue)
        revenue_99_9 = df['Revenue'].quantile(0.999)
        df = df[df['Revenue'] <= revenue_99_9]
        
        print(f"Cleaned shape: {df.shape}")
        print(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
        print(f"Unique customers: {df['CustomerID'].nunique()}")
        
        return df[['CustomerID', 'InvoiceDate', 'Invoice', 'Quantity', 'Price', 'Revenue']]
    
    def create_time_split(self, df: pd.DataFrame) -> Tuple[datetime, datetime]:
        """
        Create time-based train/test split to avoid data leakage.
        
        Args:
            df: Cleaned transaction DataFrame
            
        Returns:
            Tuple of (observation_end_date, prediction_end_date)
        """
        max_date = df['InvoiceDate'].max()
        
        # Observation period ends prediction_days before max_date
        self.observation_end = max_date - timedelta(days=self.prediction_days)
        
        # Split date is observation_days before observation_end
        self.split_date = self.observation_end - timedelta(days=self.observation_days)
        
        print(f"\nTime split:")
        print(f"  Training period: {self.split_date} to {self.observation_end}")
        print(f"  Prediction period: {self.observation_end} to {max_date}")
        print(f"  Duration: {self.observation_days} days observation, {self.prediction_days} days prediction")
        
        return self.observation_end, max_date
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create customer-level RFM features efficiently using vectorized operations.
        
        Args:
            df: Cleaned transaction DataFrame
            
        Returns:
            Customer-level feature DataFrame
        """
        if self.observation_end is None:
            raise ValueError("Must call create_time_split() first")
        
        print("\nEngineering features...")
        
        # Filter to observation period
        obs_df = df[df['InvoiceDate'] <= self.observation_end].copy()
        
        # Calculate reference date for recency
        reference_date = self.observation_end
        
        # Group by customer and compute aggregations efficiently
        customer_features = obs_df.groupby('CustomerID').agg({
            'InvoiceDate': ['min', 'max', 'count'],  # First purchase, last purchase, transaction count
            'Invoice': 'nunique',  # Number of unique invoices (frequency)
            'Revenue': ['sum', 'mean', 'std', 'min', 'max'],  # Monetary features
            'Quantity': ['sum', 'mean']  # Quantity features
        }).reset_index()
        
        # Flatten column names
        customer_features.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                                     for col in customer_features.columns.values]
        
        # Rename for clarity
        customer_features.rename(columns={
            'InvoiceDate_min': 'first_purchase_date',
            'InvoiceDate_max': 'last_purchase_date',
            'InvoiceDate_count': 'transaction_count',
            'Invoice_nunique': 'frequency',
            'Revenue_sum': 'monetary',
            'Revenue_mean': 'avg_order_value',
            'Revenue_std': 'revenue_std',
            'Revenue_min': 'min_revenue',
            'Revenue_max': 'max_revenue',
            'Quantity_sum': 'total_quantity',
            'Quantity_mean': 'avg_quantity'
        }, inplace=True)
        
        # Calculate recency (days since last purchase)
        customer_features['recency'] = (
            reference_date - customer_features['last_purchase_date']
        ).dt.days
        
        # Calculate customer lifetime (days from first to last purchase)
        customer_features['customer_lifetime'] = (
            customer_features['last_purchase_date'] - customer_features['first_purchase_date']
        ).dt.days + 1  # Add 1 to avoid zero
        
        # Fill std with 0 for single-purchase customers
        customer_features['revenue_std'].fillna(0, inplace=True)
        
        # Derived features
        customer_features['avg_days_between_purchases'] = (
            customer_features['customer_lifetime'] / customer_features['frequency']
        )
        
        # Purchase rate (purchases per day)
        customer_features['purchase_rate'] = (
            customer_features['frequency'] / customer_features['customer_lifetime']
        )
        
        # Coefficient of variation in revenue
        customer_features['revenue_cv'] = (
            customer_features['revenue_std'] / (customer_features['avg_order_value'] + 1e-6)
        )
        
        # Drop date columns
        customer_features.drop(['first_purchase_date', 'last_purchase_date'], axis=1, inplace=True)
        
        print(f"Feature matrix shape: {customer_features.shape}")
        print(f"Features: {list(customer_features.columns)}")
        
        return customer_features
    
    def create_target(self, df: pd.DataFrame, customer_features: pd.DataFrame) -> pd.DataFrame:
        """
        Create CLTV target as future revenue in prediction period.
        
        Args:
            df: Cleaned transaction DataFrame
            customer_features: Customer-level features
            
        Returns:
            DataFrame with features and target
        """
        if self.observation_end is None:
            raise ValueError("Must call create_time_split() first")
        
        print("\nCreating target variable...")
        
        # Filter to prediction period
        future_df = df[df['InvoiceDate'] > self.observation_end].copy()
        
        # Calculate future revenue per customer
        future_revenue = future_df.groupby('CustomerID')['Revenue'].sum().reset_index()
        future_revenue.columns = ['CustomerID', 'future_revenue']
        
        # Merge with features
        result = customer_features.merge(future_revenue, on='CustomerID', how='left')
        
        # Fill missing future revenue with 0 (customers who didn't purchase)
        result['future_revenue'].fillna(0, inplace=True)
        
        print(f"Target statistics:")
        print(f"  Mean: ${result['future_revenue'].mean():.2f}")
        print(f"  Median: ${result['future_revenue'].median():.2f}")
        print(f"  Std: ${result['future_revenue'].std():.2f}")
        print(f"  Zero values: {(result['future_revenue'] == 0).sum()} ({(result['future_revenue'] == 0).mean()*100:.1f}%)")
        print(f"  Max: ${result['future_revenue'].max():.2f}")
        
        return result
    
    def process(self, filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Full preprocessing pipeline.
        
        Args:
            filepath: Path to raw data file
            
        Returns:
            Tuple of (train_df, test_df) with features and target
        """
        # Load and clean
        df = self.load_and_clean(filepath)
        
        # Create time split
        self.create_time_split(df)
        
        # Engineer features
        customer_features = self.engineer_features(df)
        
        # Create target
        data_with_target = self.create_target(df, customer_features)
        
        # Split into train/test (80/20 split by customers)
        np.random.seed(42)
        n_customers = len(data_with_target)
        n_train = int(0.8 * n_customers)
        
        shuffled_indices = np.random.permutation(n_customers)
        train_indices = shuffled_indices[:n_train]
        test_indices = shuffled_indices[n_train:]
        
        train_df = data_with_target.iloc[train_indices].reset_index(drop=True)
        test_df = data_with_target.iloc[test_indices].reset_index(drop=True)
        
        print(f"\nTrain set: {len(train_df)} customers")
        print(f"Test set: {len(test_df)} customers")
        
        return train_df, test_df


def save_processed_data(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str = 'data'):
    """Save processed data to CSV."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train_processed.csv')
    test_path = os.path.join(output_dir, 'test_processed.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nSaved processed data:")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")
