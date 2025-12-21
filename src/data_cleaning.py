"""
Data Cleaning Module

This module provides functions for cleaning datasets:
- Handling missing values
- Removing duplicates
- Correcting data types
- Filtering invalid entries
"""

import pandas as pd
import numpy as np
from datetime import datetime


def standardize_datatypes(df, dataset_type='fraud_data'):
    """
    Standardize data types based on dataset type.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    dataset_type : str
        Type of dataset ('fraud_data' or 'creditcard')
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with standardized data types
    """
    df = df.copy()
    
    if dataset_type == 'fraud_data':
        # Convert datetime columns
        if 'signup_time' in df.columns:
            df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce')
        if 'purchase_time' in df.columns:
            df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce')
        
        # Ensure numeric types
        numeric_cols = ['user_id', 'purchase_value', 'age', 'ip_address', 'class']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
    elif dataset_type == 'creditcard':
        # All columns should be numeric except potentially class
        for col in df.columns:
            if col != 'Class' and col != 'class':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure class is integer
        if 'Class' in df.columns:
            df['Class'] = df['Class'].astype(int)
        if 'class' in df.columns:
            df['class'] = df['class'].astype(int)
    
    return df


def clean_data(df, dataset_name, critical_cols=None, age_col=None, value_col=None):
    """
    Comprehensive data cleaning function.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    dataset_name : str
        Name of the dataset (for logging)
    critical_cols : list, optional
        Columns that cannot have missing values
    age_col : str, optional
        Age column name (to filter invalid ages)
    value_col : str, optional
        Value column name (to filter invalid values)
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    df = df.copy()
    initial_shape = df.shape[0]
    
    print(f"\n{'='*60}")
    print(f"CLEANING {dataset_name}")
    print(f"{'='*60}")
    
    # 1. Remove duplicates
    duplicates_count = df.duplicated().sum()
    if duplicates_count > 0:
        df = df.drop_duplicates()
        print(f"✓ Removed {duplicates_count:,} duplicate rows")
    
    # 2. Handle missing values in critical columns
    if critical_cols:
        for col in critical_cols:
            if col in df.columns:
                missing = df[col].isnull().sum()
                if missing > 0:
                    print(f"⚠ Warning: {missing:,} missing values in critical column '{col}'")
                    df = df.dropna(subset=[col])
                    print(f"  → Dropped {missing:,} rows with missing {col}")
    
    # 3. Filter invalid ages (typically 0-120)
    if age_col and age_col in df.columns:
        invalid_ages = ((df[age_col] < 0) | (df[age_col] > 120)).sum()
        if invalid_ages > 0:
            df = df[(df[age_col] >= 0) & (df[age_col] <= 120)]
            print(f"✓ Removed {invalid_ages:,} rows with invalid age values")
    
    # 4. Filter invalid purchase/transaction values (non-negative)
    if value_col and value_col in df.columns:
        invalid_values = (df[value_col] < 0).sum()
        if invalid_values > 0:
            df = df[df[value_col] >= 0]
            print(f"✓ Removed {invalid_values:,} rows with negative {value_col}")
    
    # 5. Handle remaining missing values (impute or drop based on percentage)
    missing_summary = df.isnull().sum()
    missing_cols = missing_summary[missing_summary > 0]
    
    if len(missing_cols) > 0:
        print(f"\n📊 Missing Values Summary:")
        for col, count in missing_cols.items():
            pct = (count / len(df)) * 100
            print(f"  {col}: {count:,} ({pct:.2f}%)")
            
            # Strategy: Drop if >50% missing, otherwise impute
            if pct > 50:
                print(f"  → Dropping column '{col}' (>50% missing)")
                df = df.drop(columns=[col])
            elif col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(df[col].median())
                    print(f"  → Imputed '{col}' with median")
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
                    print(f"  → Imputed '{col}' with mode")
    
    final_shape = df.shape[0]
    removed = initial_shape - final_shape
    
    print(f"\n✓ Cleaning complete:")
    print(f"  Initial rows: {initial_shape:,}")
    print(f"  Final rows: {final_shape:,}")
    print(f"  Rows removed: {removed:,} ({(removed/initial_shape*100):.2f}%)")
    
    return df

