"""
EDA Utilities Module

This module provides utility functions for Exploratory Data Analysis:
- Class distribution analysis
- Missing values analysis
- Duplicate analysis
- Data summary
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_data_summary(df, dataset_name):
    """
    Get comprehensive data summary.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    dataset_name : str
        Name of the dataset
        
    Returns:
    --------
    dict
        Summary statistics
    """
    summary = {
        'dataset': dataset_name,
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'numeric_cols': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_cols': len(df.select_dtypes(include=['object']).columns),
        'datetime_cols': len(df.select_dtypes(include=['datetime64']).columns)
    }
    
    print(f"\n📋 {dataset_name} Summary:")
    print(f"  Shape: {summary['shape'][0]:,} rows × {summary['shape'][1]} columns")
    print(f"  Memory: {summary['memory_usage_mb']:.2f} MB")
    print(f"  Numeric columns: {summary['numeric_cols']}")
    print(f"  Categorical columns: {summary['categorical_cols']}")
    print(f"  Datetime columns: {summary['datetime_cols']}")
    
    return summary


def analyze_missing_values(df, dataset_name):
    """
    Analyze missing values in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    dataset_name : str
        Name of the dataset
        
    Returns:
    --------
    pd.Series
        Missing values count and percentage
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Count': missing,
        'Percentage': missing_pct
    })
    missing_df = missing_df[missing_df['Count'] > 0].sort_values('Count', ascending=False)
    
    if len(missing_df) > 0:
        print(f"\n⚠ {dataset_name} - Missing Values:")
        print(missing_df.to_string())
    else:
        print(f"\n✓ {dataset_name} - No missing values found")
    
    return missing


def analyze_duplicates(df, dataset_name):
    """
    Analyze duplicate rows in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    dataset_name : str
        Name of the dataset
        
    Returns:
    --------
    int
        Number of duplicate rows
    """
    duplicates = df.duplicated().sum()
    duplicate_pct = (duplicates / len(df)) * 100 if len(df) > 0 else 0
    
    if duplicates > 0:
        print(f"\n⚠ {dataset_name} - Found {duplicates:,} duplicate rows ({duplicate_pct:.2f}%)")
    else:
        print(f"\n✓ {dataset_name} - No duplicates found")
    
    return duplicates


def analyze_class_distribution(df, target_col, dataset_name, visualize=True):
    """
    Analyze class distribution for the target variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of the target column
    dataset_name : str
        Name of the dataset
    visualize : bool
        Whether to create visualization
        
    Returns:
    --------
    pd.Series
        Class distribution
    """
    if target_col not in df.columns:
        print(f"⚠ Warning: Target column '{target_col}' not found in {dataset_name}")
        return None
    
    class_dist = df[target_col].value_counts().sort_index()
    class_pct = (class_dist / len(df)) * 100
    
    print(f"\n{'='*60}")
    print(f"CLASS DISTRIBUTION - {dataset_name}")
    print(f"{'='*60}")
    
    dist_df = pd.DataFrame({
        'Count': class_dist,
        'Percentage': class_pct
    })
    print(dist_df.to_string())
    
    # Calculate imbalance ratio
    if len(class_dist) == 2:
        imbalance_ratio = class_dist.max() / class_dist.min()
        print(f"\n📊 Imbalance Ratio: {imbalance_ratio:.2f}:1 (Majority:Minority)")
        
        if imbalance_ratio > 10:
            print("  ⚠ SEVERE IMBALANCE - Resampling required!")
        elif imbalance_ratio > 3:
            print("  ⚠ Moderate imbalance - Consider resampling")
        else:
            print("  ✓ Relatively balanced")
    
    # Visualization
    if visualize:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart
        class_dist.plot(kind='bar', ax=axes[0], color=['green', 'red'] if len(class_dist) == 2 else None)
        axes[0].set_title(f'Class Distribution - {dataset_name}', fontweight='bold')
        axes[0].set_xlabel('Class (0=Legitimate, 1=Fraud)')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=0)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Pie chart
        class_dist.plot(kind='pie', ax=axes[1], autopct='%1.2f%%', 
                       colors=['green', 'red'] if len(class_dist) == 2 else None)
        axes[1].set_title(f'Class Distribution (Percentage)', fontweight='bold')
        axes[1].set_ylabel('')
        
        plt.tight_layout()
        plt.show()
    
    print(f"\n💡 Why imbalance is problematic:")
    print("  → Models tend to predict majority class (high accuracy but poor recall)")
    print("  → Fraud cases (minority) are often missed")
    print("  → Evaluation metrics like accuracy become misleading")
    print("  → Solution: Use SMOTE, undersampling, or class weights")
    
    return class_dist

