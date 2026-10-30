"""
data_preprocessing.py
---------------------
Robust preprocessing pipeline for the IBM HR Analytics Employee Attrition dataset.
Handles data loading, cleaning, stratified train-test splitting, and feature transformation.
"""

import os
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Constant columns with 0 variance or identifier roles
COLUMNS_TO_DROP = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
TARGET_COLUMN = 'Attrition'

DEFAULT_DATA_PATHS = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'WA_Fn-UseC_-HR-Employee-Attrition.csv'),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'WA_Fn-UseC_-HR-Employee-Attrition.csv'),
    'data/WA_Fn-UseC_-HR-Employee-Attrition.csv',
    'WA_Fn-UseC_-HR-Employee-Attrition.csv',
]


def find_dataset_path(custom_path: Optional[str] = None) -> str:
    """Locates the dataset path from provided path or default locations."""
    if custom_path and os.path.exists(custom_path):
        return custom_path
    for path in DEFAULT_DATA_PATHS:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Could not locate 'WA_Fn-UseC_-HR-Employee-Attrition.csv'. "
        "Please ensure the dataset is placed inside the 'data/' directory."
    )


def load_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Loads the HR Attrition dataset into a pandas DataFrame."""
    path = find_dataset_path(filepath)
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw DataFrame:
    - Drops duplicates if any
    - Drops zero-variance / identifier columns
    """
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates()
    
    # Drop constant and identifier columns if present
    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df_clean.columns]
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)
        
    return df_clean


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identifies numerical and categorical feature columns (excluding target)."""
    feature_df = df.drop(columns=[TARGET_COLUMN], errors='ignore')
    
    # Also ignore columns to drop if not already removed
    feature_df = feature_df.drop(columns=[c for c in COLUMNS_TO_DROP if c in feature_df.columns], errors='ignore')
    
    num_cols = feature_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = feature_df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    
    return num_cols, cat_cols


def build_preprocessor(numerical_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that:
    - StandardScales numerical features
    - OneHotEncodes categorical features with handle_unknown='ignore'
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ],
        remainder='drop'
    )
    return preprocessor


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Separates independent features and target, and splits with stratification
    to handle class imbalance properly.
    """
    df_clean = clean_data(df)
    
    X = df_clean.drop(columns=[TARGET_COLUMN])
    # Encode target to binary (Yes: 1, No: 0)
    y = df_clean[TARGET_COLUMN].apply(lambda v: 1 if str(v).strip().lower() == 'yes' else 0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


def get_default_feature_values(df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Returns median/mode default values for all features to use for
    single-prediction inference when certain fields are omitted.
    """
    if df is None:
        df = load_data()
    df_clean = clean_data(df).drop(columns=[TARGET_COLUMN], errors='ignore')
    
    defaults = {}
    for col in df_clean.columns:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            defaults[col] = int(df_clean[col].median()) if df_clean[col].dtype == 'int64' else float(df_clean[col].median())
        else:
            defaults[col] = df_clean[col].mode()[0]
    return defaults


if __name__ == '__main__':
    raw_df = load_data()
    print(f"Loaded raw dataset with shape: {raw_df.shape}")
    cleaned_df = clean_data(raw_df)
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
    num_cols, cat_cols = get_feature_columns(cleaned_df)
    print(f"Numerical features ({len(num_cols)}): {num_cols}")
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
    X_tr, X_te, y_tr, y_te = split_data(cleaned_df)
    print(f"Train set: {X_tr.shape}, Test set: {X_te.shape}")
    print(f"Train attrition rate: {y_tr.mean():.2%}, Test attrition rate: {y_te.mean():.2%}")
