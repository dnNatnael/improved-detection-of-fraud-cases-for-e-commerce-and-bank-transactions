"""
Feature Engineering Module

This module provides functions for creating meaningful features:
- Time-based features (hour, day, weekend, night, time since signup)
- Transaction behavior features (frequency, velocity)
- Statistical aggregations
- Geolocation integration
"""

import pandas as pd
import numpy as np


def convert_ip_to_int(ip_address):
    """
    Convert IP address to integer format.
    
    Parameters:
    -----------
    ip_address : float or int
        IP address in numeric format (already converted)
        
    Returns:
    --------
    int
        Integer representation of IP address
    """
    return int(ip_address) if pd.notna(ip_address) else None


def merge_geolocation(fraud_df, ip_df):
    """
    Merge geolocation data using range-based join.
    
    Parameters:
    -----------
    fraud_df : pd.DataFrame
        Fraud data with ip_address column
    ip_df : pd.DataFrame
        IP address to country mapping with lower_bound and upper_bound
        
    Returns:
    --------
    pd.DataFrame
        Fraud data with country column added
    """
    fraud_df = fraud_df.copy()
    
    print("\n🌍 Merging Geolocation Data...")
    
    # Convert IP to integer if not already
    if 'ip_address' in fraud_df.columns:
        fraud_df['ip_address_int'] = fraud_df['ip_address'].apply(convert_ip_to_int)
        
        # Sort IP ranges for efficient lookup
        ip_df_sorted = ip_df.sort_values('lower_bound_ip_address').reset_index(drop=True)
        
        print(f"  Processing {len(fraud_df):,} transactions...")
        
        # Use merge_asof for efficient range-based merge
        # This is faster than row-by-row lookup
        def find_country(ip_int):
            if pd.isna(ip_int):
                return None
            # Binary search approach - find the range that contains the IP
            matching = ip_df_sorted[
                (ip_df_sorted['lower_bound_ip_address'] <= ip_int) &
                (ip_df_sorted['upper_bound_ip_address'] >= ip_int)
            ]
            if not matching.empty:
                return matching.iloc[0]['country']
            return None
        
        # Apply function to find countries
        fraud_df['country'] = fraud_df['ip_address_int'].apply(find_country)
        
        # Drop temporary column
        fraud_df = fraud_df.drop(columns=['ip_address_int'])
        
        # Fill missing countries (IPs not in mapping)
        missing_countries = fraud_df['country'].isnull().sum()
        if missing_countries > 0:
            fraud_df['country'] = fraud_df['country'].fillna('Unknown')
            print(f"  ⚠ {missing_countries:,} IPs not found in mapping (marked as 'Unknown')")
        
        matched = len(fraud_df) - missing_countries
        print(f"  ✓ Matched {matched:,} transactions to countries")
        print(f"  ✓ Found {fraud_df['country'].nunique()} unique countries")
    else:
        print("  ⚠ Warning: 'ip_address' column not found")
    
    return fraud_df


def engineer_features(df, dataset_type='fraud_data'):
    """
    Engineer features based on dataset type.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    dataset_type : str
        Type of dataset ('fraud_data' or 'creditcard')
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with engineered features added
    """
    df = df.copy()
    
    print(f"\n{'='*60}")
    print(f"FEATURE ENGINEERING - {dataset_type.upper()}")
    print(f"{'='*60}")
    
    if dataset_type == 'fraud_data':
        df = _engineer_fraud_features(df)
    elif dataset_type == 'creditcard':
        df = _engineer_creditcard_features(df)
    
    return df


def _engineer_fraud_features(df):
    """
    Engineer features for fraud_data dataset.
    
    Features created:
    - Time-based: hour_of_day, day_of_week, is_weekend, is_night
    - Behavioral: time_since_signup, quick_purchase, rapid_transactions
    """
    print("\n⏰ Creating Time-Based Features...")
    
    # Time-based features
    if 'purchase_time' in df.columns:
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        
        # Hour of day (0-23)
        df['hour_of_day'] = df['purchase_time'].dt.hour
        print("  ✓ hour_of_day: Purchase hour (0-23)")
        
        # Day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
        print("  ✓ day_of_week: Purchase day (0=Mon, 6=Sun)")
        
        # Weekend flag (Saturday=5, Sunday=6)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        print("  ✓ is_weekend: 1 if Saturday/Sunday, 0 otherwise")
        print("    → Useful: Fraud often occurs on weekends when monitoring is lower")
        
        # Night flag (22:00-06:00)
        df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] < 6)).astype(int)
        print("  ✓ is_night: 1 if 10 PM - 6 AM, 0 otherwise")
        print("    → Useful: Fraudsters often operate during off-hours")
    
    # Time since signup
    if 'signup_time' in df.columns and 'purchase_time' in df.columns:
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600  # hours
        print("\n⏱️ Creating Behavioral Features...")
        print("  ✓ time_since_signup: Hours between signup and purchase")
        print("    → Useful: Quick purchases after signup indicate fraud")
        
        # Quick purchase flag (< 5 minutes = 0.083 hours)
        df['quick_purchase'] = (df['time_since_signup'] < 0.083).astype(int)
        print("  ✓ quick_purchase: 1 if purchase within 5 minutes of signup")
        print("    → Useful: Fraudsters often act quickly")
    
    # Transaction frequency per user
    if 'user_id' in df.columns:
        user_transaction_count = df.groupby('user_id').size().reset_index(name='transaction_count')
        df = df.merge(user_transaction_count, on='user_id', how='left')
        
        # Rapid transactions flag (multiple transactions in short time)
        # For each user, calculate time between transactions
        if 'purchase_time' in df.columns:
            # Store original index to restore order later
            df['_original_index'] = range(len(df))
            
            # Sort for calculation
            df_sorted = df.sort_values(['user_id', 'purchase_time']).copy()
            df_sorted['prev_purchase_time'] = df_sorted.groupby('user_id')['purchase_time'].shift(1)
            df_sorted['time_between_transactions'] = (
                df_sorted['purchase_time'] - df_sorted['prev_purchase_time']
            ).dt.total_seconds() / 3600  # hours
            
            # Mark rapid transactions (within 1 hour)
            df_sorted['rapid_transactions'] = (df_sorted['time_between_transactions'] < 1).astype(int)
            df_sorted['rapid_transactions'] = df_sorted['rapid_transactions'].fillna(0)
            
            # Restore original order
            df = df_sorted.sort_values('_original_index').drop(columns=['_original_index', 'prev_purchase_time', 'time_between_transactions'])
            
            print("  ✓ transaction_count: Number of transactions per user")
            print("  ✓ rapid_transactions: 1 if transaction within 1 hour of previous")
            print("    → Useful: Multiple rapid transactions indicate suspicious behavior")
    
    print(f"\n✓ Feature engineering complete. Added {len([c for c in df.columns if c in ['hour_of_day', 'day_of_week', 'is_weekend', 'is_night', 'time_since_signup', 'quick_purchase', 'rapid_transactions', 'transaction_count']])} new features")
    
    return df


def _engineer_creditcard_features(df):
    """
    Engineer features for creditcard dataset.
    
    Features created:
    - Statistical: features_mean, features_std, features_min, features_max
    - Transformations: log_amount
    """
    print("\n📊 Creating Statistical Features...")
    
    # Get V-features (V1-V28)
    v_features = [col for col in df.columns if col.startswith('V')]
    
    if v_features:
        # Statistical aggregations of V-features
        df['features_mean'] = df[v_features].mean(axis=1)
        df['features_std'] = df[v_features].std(axis=1)
        df['features_min'] = df[v_features].min(axis=1)
        df['features_max'] = df[v_features].max(axis=1)
        print(f"  ✓ features_mean: Mean of V-features (captures overall transaction profile)")
        print(f"  ✓ features_std: Std deviation of V-features (captures variability)")
        print(f"  ✓ features_min: Min of V-features")
        print(f"  ✓ features_max: Max of V-features")
        print("    → Useful: Aggregated statistics capture overall transaction behavior")
    
    # Log transform amount (handles skewness)
    if 'Amount' in df.columns:
        df['log_amount'] = np.log1p(df['Amount'])  # log1p handles zeros
        print("  ✓ log_amount: Log-transformed transaction amount")
        print("    → Useful: Reduces skewness, improves model performance")
    
    # Time-based features (convert Time to hours)
    if 'Time' in df.columns:
        df['hour'] = (df['Time'] / 3600) % 24
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        print("  ✓ hour_sin/cos: Cyclical encoding of transaction hour")
        print("    → Useful: Captures periodic patterns (24-hour cycle)")
    
    print(f"\n✓ Feature engineering complete. Added statistical and transformation features")
    
    return df

