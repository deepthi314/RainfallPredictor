import pandas as pd
import numpy as np
import os

def load_manila_data(file_path):
    """
    Load the Manila NASA POWER CSV file, skipping the 20-line header.
    
    Args:
        file_path (str): Path to the Manila.csv file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
    
    print(f"Loading data from {file_path} (skipping 20 lines)...")
    df = pd.read_csv(file_path, skiprows=20)
    
    # NASA POWER files sometimes have a trailing record that is not data
    # or extra empty lines. We should ensure we only keep valid rows.
    # Typically, the date columns are YEAR, DOY.
    df = df.dropna(subset=['YEAR', 'DOY'])
    
    return df

def save_processed_data(df, file_path):
    """
    Save the processed dataframe to CSV.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"Saved data to {file_path}")
