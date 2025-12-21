"""
Data Loading Module

This module provides functions for loading datasets with comprehensive error handling.
"""

import pandas as pd
import os


def load_data(file_path):
    """
    Load a dataset from a CSV file with error handling.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file to load
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
        
    Raises:
    -------
    FileNotFoundError
        If the file does not exist
    ValueError
        If the file cannot be read as CSV
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        print(f"✓ Successfully loaded: {os.path.basename(file_path)}")
        print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        return df
        
    except pd.errors.EmptyDataError:
        raise ValueError(f"File is empty: {file_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV file {file_path}: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error loading {file_path}: {str(e)}")

