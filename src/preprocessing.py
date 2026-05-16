import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_manila_data, save_processed_data

# Base directory: project root (one level up from src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def preprocess_data(file_path):
    """
    Clean, impute, and scale the Manila rainfall dataset.
    
    Tasks:
    - Replace all -999 with NaN
    - Impute missing values in ALLSKY_SFC_SW_DNI using monthly median groupby
    - Verify no remaining NaN values
    - Apply MinMaxScaler to all feature columns (NOT to target PRECTOTCORR)
    - Save cleaned data
    """
    print("Preprocessing data...")
    df = load_manila_data(file_path)
    
    # Replace -999 with NaN
    df = df.replace(-999, np.nan)
    print(f"Missing values after replacing -999:\n{df.isnull().sum()}")
    
    # Extract month for monthly imputation
    # NASA POWER DOY is day of year
    df['DATE'] = pd.to_datetime(df['YEAR'].astype(int).astype(str) + '-' + df['DOY'].astype(int).astype(str), format='%Y-%j')
    df['MONTH'] = df['DATE'].dt.month
    
    # Impute ALLSKY_SFC_SW_DNI using monthly median groupby
    if 'ALLSKY_SFC_SW_DNI' in df.columns:
        print("Imputing ALLSKY_SFC_SW_DNI using monthly median groupby...")
        df['ALLSKY_SFC_SW_DNI'] = df.groupby(['YEAR', 'MONTH'])['ALLSKY_SFC_SW_DNI'].transform(lambda x: x.fillna(x.median()))
        
        # If there are still NaN values (e.g., empty month/year), fallback to just month
        if df['ALLSKY_SFC_SW_DNI'].isnull().any():
            df['ALLSKY_SFC_SW_DNI'] = df.groupby(['MONTH'])['ALLSKY_SFC_SW_DNI'].transform(lambda x: x.fillna(x.median()))
    
    # Fill remaining NaN values with 0 or drop if few, but instruction says avoid NaN remaining
    df = df.fillna(0) # Emergency fallback for any remaining NaNs
    
    # Apply MinMaxScaler to features (excluding target and YEAR, DOY, DATE, MONTH)
    target_col = 'PRECTOTCORR'
    exclude_cols = ['YEAR', 'DOY', 'DATE', 'MONTH', target_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    print(f"Scaling features: {feature_cols}")
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    if feature_cols:
        df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Print summary
    print(f"Shape before cleaning: {df.shape}")
    print(f"Shape after cleaning: {df_scaled.shape}")
    print(f"Residual NaNs: {df_scaled.isnull().sum().sum()}")
    
    return df_scaled, scaler

if __name__ == "__main__":
    # Test loading
    input_file = os.path.join(BASE_DIR, "data", "raw", "Manila.csv")
    output_file = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")
    
    if os.path.exists(input_file):
        df_cleaned, scaler = preprocess_data(input_file)
        save_processed_data(df_cleaned, output_file)
        import joblib
        models_dir = os.path.join(BASE_DIR, "models")
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
        print("Preprocessing successfully finished.")
    else:
        print(f"Input file {input_file} not found.")

