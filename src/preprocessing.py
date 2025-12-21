"""
Preprocessing Module

This module provides functions for data preprocessing:
- Train-test split
- Feature scaling
- Categorical encoding
- Class imbalance handling (SMOTE, undersampling)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def train_test_split_data(df, target_col='class', test_size=0.2, random_state=42, stratify=True):
    """
    Split data into training and testing sets with stratification.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of target column
    test_size : float
        Proportion of test set (default 0.2)
    random_state : int
        Random seed for reproducibility
    stratify : bool
        Whether to stratify split based on target
        
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify_param
    )
    
    print(f"✓ Train-test split complete:")
    print(f"  Train: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
    print(f"  Test: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(df)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test, scaler_type='standard'):
    """
    Scale features using StandardScaler or MinMaxScaler.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    X_test : pd.DataFrame or np.ndarray
        Test features
    scaler_type : str
        Type of scaler ('standard' or 'minmax')
        
    Returns:
    --------
    tuple
        X_train_scaled, X_test_scaled, scaler
    """
    # Convert to DataFrame if numpy array
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test)
    
    # Select numeric columns only
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    if scaler_type == 'standard':
        scaler = StandardScaler()
        print("  Using StandardScaler (mean=0, std=1)")
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
        print("  Using MinMaxScaler (range 0-1)")
    else:
        raise ValueError("scaler_type must be 'standard' or 'minmax'")
    
    # Fit on training data only (prevent data leakage)
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    if numeric_cols:
        X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
        
        print(f"  ✓ Scaled {len(numeric_cols)} numeric features")
    else:
        print("  ⚠ Warning: No numeric features found to scale")
    
    return X_train_scaled, X_test_scaled, scaler


def encode_categorical_features(df, categorical_cols, drop_first=True):
    """
    One-hot encode categorical features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    categorical_cols : list
        List of categorical column names
    drop_first : bool
        Whether to drop first category (avoid multicollinearity)
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with encoded features
    """
    df_encoded = df.copy()
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            # One-hot encode
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=drop_first)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded = df_encoded.drop(columns=[col])
    
    print(f"✓ Encoded {len(categorical_cols)} categorical features")
    print(f"  Original columns: {len(df.columns)}")
    print(f"  After encoding: {len(df_encoded.columns)}")
    
    return df_encoded


def handle_class_imbalance(X_train, y_train, method='smote', random_state=42, **kwargs):
    """
    Handle class imbalance using SMOTE or undersampling.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    y_train : pd.Series or np.ndarray
        Training labels
    method : str
        Method to use ('smote' or 'undersample')
    random_state : int
        Random seed for reproducibility
    **kwargs
        Additional parameters for resampling methods
        
    Returns:
    --------
    tuple
        X_resampled, y_resampled
    """
    # Convert to numpy if DataFrame/Series
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(y_train, pd.Series):
        y_train = y_train.values
    
    original_dist = pd.Series(y_train).value_counts().sort_index()
    print(f"\n📊 Original class distribution:")
    for cls, count in original_dist.items():
        print(f"  Class {cls}: {count:,} ({(count/len(y_train)*100):.2f}%)")
    
    if method == 'smote':
        print("\n🔄 Applying SMOTE (Synthetic Minority Over-sampling Technique)...")
        smote = SMOTE(random_state=random_state, **kwargs)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        print("  ✓ SMOTE applied - Creates synthetic samples for minority class")
        
    elif method == 'undersample':
        print("\n🔄 Applying Random Under-sampling...")
        rus = RandomUnderSampler(random_state=random_state, **kwargs)
        X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
        print("  ⚠ Under-sampling applied - Reduces majority class samples")
        print(f"  ⚠ Dataset reduced from {len(y_train):,} to {len(y_resampled):,} samples")
    else:
        raise ValueError("method must be 'smote' or 'undersample'")
    
    resampled_dist = pd.Series(y_resampled).value_counts().sort_index()
    print(f"\n📊 Resampled class distribution:")
    for cls, count in resampled_dist.items():
        print(f"  Class {cls}: {count:,} ({(count/len(y_resampled)*100):.2f}%)")
    
    return X_resampled, y_resampled

