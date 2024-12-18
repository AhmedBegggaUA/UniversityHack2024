import pandas as pd 
from typing import Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from sklearn.preprocessing import StandardScaler
DATA_PATH = Path('../script exploración/processed_data')
TMP_PATH = Path('./tmp')
LOGS_PATH = Path('./logs')
class DataProcessor:
    """Handles all data loading and preprocessing operations."""
    
    @staticmethod
    def load_training_data() -> Tuple[pd.DataFrame, pd.Series]:
        """Load and preprocess training data."""
        data = pd.read_csv(DATA_PATH / 'train/train_data_clean.csv')
        data = data.drop('Producto 2', axis=1)
        
        # Remove columns with NaN correlation with target
        corr = data.corr()['Producto 1']
        columns_to_drop = corr[corr.isna()].index
        
        # Save columns to drop for later use
        with open(TMP_PATH / 'columns_to_drop.txt', 'w') as f:
            f.write('\n'.join(columns_to_drop))
        
        data = data.drop(columns_to_drop, axis=1)
        X = data.drop('Producto 1', axis=1)
        X.to_csv(TMP_PATH / 'X.csv', index=False)
        
        # Handle missing values and scale features
        for column in X.columns:
            X[column] = X[column].fillna(X[column].mean())
            
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns
        )
        
        return X_scaled, data['Producto 1']
    
    @staticmethod
    def load_test_data() -> pd.DataFrame:
        """Load and preprocess test data."""
        data = pd.read_csv(DATA_PATH / 'test/test_data_clean.csv')
        train_columns = pd.read_csv(TMP_PATH / 'X.csv').columns
        
        X = data[train_columns].copy()
        for column in X.columns:
            X[column] = X[column].fillna(X[column].mean())
            
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns
        )
        
        return X_scaled