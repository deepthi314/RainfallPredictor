import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Base directory: project root (one level up from src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def predict_unseen(df, model, scaler, pca=None, model_name="Best Model"):
    """
    Predict PRECTOTCORR for 2021-2025 unseen data.
    - Computes metrics (R², RMSE, MAE, MSE)
    - Saves predictions CSV
    - Plots time-series comparison
    """
    print("Predicting unseen data (2021-2025)...")

    unseen_df = df[df['YEAR'].isin(range(2021, 2026))].copy()
    unseen_df['DATE'] = pd.to_datetime(unseen_df['DATE'])
    unseen_df = unseen_df.sort_values('DATE')

    if unseen_df.empty:
        print("No unseen data for 2021-2025 found in the dataset.")
        return

    target_col   = 'PRECTOTCORR'
    exclude_cols = ['YEAR', 'DOY', 'DATE', target_col]
    feature_cols = [c for c in unseen_df.columns if c not in exclude_cols]

    X_unseen = unseen_df[feature_cols]
    y_unseen = unseen_df[target_col]

    use_pca = (model_name == "SVR (PCA)") and (pca is not None)
    X_input = pca.transform(X_unseen) if use_pca else X_unseen

    y_pred = np.maximum(model.predict(X_input), 0)

    # ── Metrics ───────────────────────────────────────────────────────────────
    r2   = r2_score(y_unseen, y_pred)
    rmse = np.sqrt(mean_squared_error(y_unseen, y_pred))
    mae  = mean_absolute_error(y_unseen, y_pred)
    mse  = mean_squared_error(y_unseen, y_pred)

    print(f"\nUnseen Period (2021-2025) Metrics – {model_name}")
    print(f"  R²   : {r2:.4f}")
    print(f"  RMSE : {rmse:.4f} mm/day")
    print(f"  MAE  : {mae:.4f} mm/day")
    print(f"  MSE  : {mse:.4f}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    reports_dir = os.path.join(BASE_DIR, "outputs", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    out_df = unseen_df[['DATE', 'YEAR']].copy()
    out_df['Actual Rainfall (mm)']    = y_unseen.values
    out_df['Predicted Rainfall (mm)'] = y_pred
    out_df.to_csv(os.path.join(reports_dir, "predictions_2021_2025.csv"), index=False)
    print(f"Predictions saved to outputs/reports/predictions_2021_2025.csv")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig_dir = os.path.join(BASE_DIR, "outputs", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure(figsize=(16, 6))
    plt.plot(unseen_df['DATE'], y_unseen.values,
             label='Actual Rainfall',    color='royalblue',  linewidth=0.9, alpha=0.8)
    plt.plot(unseen_df['DATE'], y_pred,
             label='Predicted Rainfall', color='darkorange', linewidth=0.9, alpha=0.8)
    plt.title(f'Unseen Period (2021-2025) – Predicted vs Actual Rainfall\n'
              f'Model: {model_name}  |  R²={r2:.4f}  RMSE={rmse:.4f} mm/day',
              fontsize=12)
    plt.xlabel('Date')
    plt.ylabel('Rainfall (mm/day)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '07_unseen_pred_vs_actual.png'), dpi=150)
    plt.close()
    print("Unseen prediction plot saved.")

    return out_df


if __name__ == "__main__":
    feat_file  = os.path.join(BASE_DIR, "data", "processed", "feature_engineered_data.csv")
    model_file = os.path.join(BASE_DIR, "models", "best_model.pkl")
    scaler_file= os.path.join(BASE_DIR, "models", "scaler.pkl")
    pca_file   = os.path.join(BASE_DIR, "models", "pca.pkl")
    comp_file  = os.path.join(BASE_DIR, "outputs", "reports", "model_comparison.csv")

    missing = [f for f in [feat_file, model_file, scaler_file] if not os.path.exists(f)]
    if missing:
        print(f"Missing files: {missing}\nRun src/train.py first.")
    else:
        df     = pd.read_csv(feat_file)
        model  = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        pca    = joblib.load(pca_file) if os.path.exists(pca_file) else None

        best_name = "Best Model"
        if os.path.exists(comp_file):
            comp = pd.read_csv(comp_file)
            best_name = comp.sort_values("Test R2", ascending=False).iloc[0]["Model"]

        predict_unseen(df, model, scaler, pca, best_name)
