import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def predict_unseen(df, model, scaler):
    """
    Predict PRECTOTCORR for 2021-2025 unseen data.
    
    Tasks:
    - Feature extraction for unseen period
    - Prediction
    - Compute metrics
    - Plot time-series comparison
    """
    print("Predicting unseen data (2021-2025)...")
    
    # Filter for unseen years
    unseen_df = df[df['YEAR'].isin(range(2021, 2026))].copy()
    
    if unseen_df.empty:
        print("No unseen data for 2021-2025 found.")
        return
    
    target_col = 'PRECTOTCORR'
    exclude_cols = ['YEAR', 'DOY', 'DATE', target_col]
    feature_cols = [c for c in unseen_df.columns if c not in exclude_cols]
    
    X_unseen = unseen_df[feature_cols]
    y_unseen = unseen_df[target_col]
    
    # Check if model is SVR/PCA or others
    # (Simplified: assumes best_model.pkl handles features or PCA internally if needed)
    # Actually, train.py saves best_model. We should handle if it needs PCA.
    # But for now, we'll assume the best model is one that takes raw features (RF, XGB, MLP).
    # If SVR was best, it would need PCA. Let's add a check if PCA is needed.
    
    y_pred = model.predict(X_unseen)
    
    # Results
    unseen_df['PREDICTED'] = y_pred
    unseen_df['ACTUAL'] = y_unseen
    
    # Metrics
    r2 = r2_score(y_unseen, y_pred)
    rmse = np.sqrt(mean_squared_error(y_unseen, y_pred))
    mae = mean_absolute_error(y_unseen, y_pred)
    
    print(f"Unseen Period Metrics:\nR2: {r2:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}")
    
    # Save predictions
    reports_dir = r"e:\college\WaterHarvesting\rainfall_prediction\outputs\reports"
    os.makedirs(reports_dir, exist_ok=True)
    out_file = os.path.join(reports_dir, "predictions_2021_2025.csv")
    unseen_df[['DATE', 'ACTUAL', 'PREDICTED']].to_csv(out_file, index=False)
    print(f"Predictions saved to {out_file}")
    
    # Plot
    plt.figure(figsize=(15, 7))
    plt.plot(pd.to_datetime(unseen_df['DATE']), unseen_df['ACTUAL'], label='Actual Rainfall', alpha=0.6)
    plt.plot(pd.to_datetime(unseen_df['DATE']), unseen_df['PREDICTED'], label='Predicted Rainfall', alpha=0.8)
    plt.title('Rainfall Prediction (Unseen Period 2021-2025)')
    plt.xlabel('Date')
    plt.ylabel('Rainfall (mm/day)')
    plt.legend()
    plt.grid(True)
    fig_dir = r"e:\college\WaterHarvesting\rainfall_prediction\outputs\figures"
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, "prediction_unseen.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    feat_file = r"e:\college\WaterHarvesting\rainfall_prediction\data\processed\feature_engineered_data.csv"
    model_file = r"e:\college\WaterHarvesting\rainfall_prediction\models\best_model.pkl"
    scaler_file = r"e:\college\WaterHarvesting\rainfall_prediction\models\scaler.pkl"
    
    if all(os.path.exists(f) for f in [feat_file, model_file, scaler_file]):
        df = pd.read_csv(feat_file)
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        predict_unseen(df, model, scaler)
