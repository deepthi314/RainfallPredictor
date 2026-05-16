import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import save_processed_data

# Base directory: project root (one level up from src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def engineer_features(df):
    """
    Extract date features, lags, and rolling means.
    
    Tasks:
    - Convert YEAR + DOY to datatime
    - Extract: MONTH, WEEK_OF_YEAR, SEASON
    - Create lags from PRECTOTCORR: LAG_1, LAG_7, LAG_30
    - Create rolling means: ROLLING_7_MEAN, ROLLING_30_MEAN
    - Drop NaNs introduced by lags
    """
    print("Engineering features...")
    df = df.copy()
    
    # DATE might already be present if from preprocessing.py
    if 'DATE' not in df.columns:
        df['DATE'] = pd.to_datetime(df['YEAR'].astype(int).astype(str) + '-' + df['DOY'].astype(int).astype(str), format='%Y-%j')
    else:
        df['DATE'] = pd.to_datetime(df['DATE'])
        
    df['MONTH'] = df['DATE'].dt.month
    df['WEEK_OF_YEAR'] = df['DATE'].dt.isocalendar().week.astype(int)
    
    # Season: 1=Dec-Feb, 2=Mar-May, 3=Jun-Sep monsoon, 4=Oct-Nov
    def get_season(month):
        if month in [12, 1, 2]: return 1
        elif month in [3, 4, 5]: return 2
        elif month in [6, 7, 8, 9]: return 3
        else: return 4
    
    df['SEASON'] = df['MONTH'].apply(get_season)
    
    # Lags from PRECTOTCORR
    df['LAG_1'] = df['PRECTOTCORR'].shift(1)
    df['LAG_7'] = df['PRECTOTCORR'].shift(7)
    df['LAG_30'] = df['PRECTOTCORR'].shift(30)
    
    # Rolling Means
    df['ROLLING_7_MEAN'] = df['PRECTOTCORR'].rolling(window=7).mean()
    df['ROLLING_30_MEAN'] = df['PRECTOTCORR'].rolling(window=30).mean()
    
    # Drop NaNs introduced by lags/rolling
    df = df.dropna()
    print(f"Shape after feature engineering and dropping NaNs: {df.shape}")
    
    return df

if __name__ == "__main__":
    cleaned_file = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")
    output_file = os.path.join(BASE_DIR, "data", "processed", "feature_engineered_data.csv")
    
    if os.path.exists(cleaned_file):
        df = pd.read_csv(cleaned_file)
        df_feat = engineer_features(df)
        save_processed_data(df_feat, output_file)
    else:
        print(f"Cleaned file {cleaned_file} not found.")
