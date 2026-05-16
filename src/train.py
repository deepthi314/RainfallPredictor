import pandas as pd
import numpy as np
import os
import random
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA

# Base directory: project root (one level up from src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compute_metrics(y_true, y_pred, label):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    return {f"{label} R2": r2, f"{label} RMSE": rmse,
            f"{label} MAE": mae, f"{label} MSE": mse}


def train_models(df):
    """
    Train 5 models using year-wise split.

    Split Strategy:
    - Training & Testing: 1995-2020, 80:20 year-wise random split
    - Unseen: 2021-2025 (never used in training/testing)

    Models: Decision Tree, Random Forest, XGBoost, SVR (with PCA), MLP NN
    """
    random_state = 42
    np.random.seed(random_state)
    random.seed(random_state)

    # ── Year splits ──────────────────────────────────────────────────────────
    train_test_years = list(range(1995, 2021))
    unseen_years     = list(range(2021, 2026))

    random.shuffle(train_test_years)
    split_idx   = int(0.8 * len(train_test_years))
    train_years = sorted(train_test_years[:split_idx])
    test_years  = sorted(train_test_years[split_idx:])

    print(f"Training years  ({len(train_years)}): {train_years}")
    print(f"Testing years   ({len(test_years)}):  {test_years}")
    print(f"Unseen years    ({len(unseen_years)}): {unseen_years}")

    # ── Feature / target columns ─────────────────────────────────────────────
    target_col   = 'PRECTOTCORR'
    exclude_cols = ['YEAR', 'DOY', 'DATE', target_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # ── Data splits ───────────────────────────────────────────────────────────
    train_df  = df[df['YEAR'].isin(train_years)]
    test_df   = df[df['YEAR'].isin(test_years)]
    unseen_df = df[df['YEAR'].isin(unseen_years)]

    X_train,  y_train  = train_df[feature_cols],  train_df[target_col]
    X_test,   y_test   = test_df[feature_cols],   test_df[target_col]
    X_unseen, y_unseen = unseen_df[feature_cols], unseen_df[target_col]

    print(f"\nTrain samples:  {len(X_train)}")
    print(f"Test samples:   {len(X_test)}")
    print(f"Unseen samples: {len(X_unseen)}")

    # ── PCA for SVR (fit on train only) ───────────────────────────────────────
    pca = PCA(n_components=0.85, random_state=random_state)
    X_train_pca  = pca.fit_transform(X_train)
    X_test_pca   = pca.transform(X_test)
    X_unseen_pca = pca.transform(X_unseen)
    print(f"\nPCA components for 85% variance: {pca.n_components_}")

    # ── Model definitions ─────────────────────────────────────────────────────
    models = {
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=random_state),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=15,
                                               random_state=random_state, n_jobs=-1),
        "XGBoost":       XGBRegressor(n_estimators=300, learning_rate=0.05,
                                      max_depth=6, random_state=random_state,
                                      verbosity=0),
        "SVR (PCA)":     SVR(kernel='rbf', C=10, epsilon=0.1),
        "MLP NN":        MLPRegressor(hidden_layer_sizes=(128, 64, 32),
                                      max_iter=500, random_state=random_state),
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        use_pca = (name == "SVR (PCA)")

        Xtr = X_train_pca  if use_pca else X_train
        Xte = X_test_pca   if use_pca else X_test
        Xun = X_unseen_pca if use_pca else X_unseen

        model.fit(Xtr, y_train)

        ytr_pred = model.predict(Xtr)
        yte_pred = model.predict(Xte)
        yun_pred = model.predict(Xun)

        row = {"Model": name}
        row.update(compute_metrics(y_train,  ytr_pred, "Train"))
        row.update(compute_metrics(y_test,   yte_pred, "Test"))
        row.update(compute_metrics(y_unseen, yun_pred, "Unseen"))
        results.append(row)

        print(f"  Train  R²={row['Train R2']:.4f}  RMSE={row['Train RMSE']:.4f}")
        print(f"  Test   R²={row['Test R2']:.4f}  RMSE={row['Test RMSE']:.4f}")
        print(f"  Unseen R²={row['Unseen R2']:.4f}  RMSE={row['Unseen RMSE']:.4f}")

    results_df = pd.DataFrame(results)
    print("\n── Model Comparison ──────────────────────────────────────────────")
    print(results_df[["Model", "Train R2", "Test R2", "Unseen R2",
                       "Train RMSE", "Test RMSE", "Unseen RMSE"]].to_string(index=False))

    # ── Best model (by Test R²) ───────────────────────────────────────────────
    best_row       = results_df.sort_values("Test R2", ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    best_model      = models[best_model_name]
    print(f"\nBest model: {best_model_name}")

    # ── Save artefacts ────────────────────────────────────────────────────────
    models_dir  = os.path.join(BASE_DIR, "models")
    reports_dir = os.path.join(BASE_DIR, "outputs", "reports")
    os.makedirs(models_dir,  exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    joblib.dump(best_model, os.path.join(models_dir, "best_model.pkl"))
    joblib.dump(pca,        os.path.join(models_dir, "pca.pkl"))
    results_df.to_csv(os.path.join(reports_dir, "model_comparison.csv"), index=False)

    # Save unseen predictions for the best model
    use_pca  = (best_model_name == "SVR (PCA)")
    Xun_best = X_unseen_pca if use_pca else X_unseen
    yun_pred = best_model.predict(Xun_best)

    unseen_out = unseen_df[['DATE', 'YEAR']].copy()
    unseen_out['Actual Rainfall (mm)']    = y_unseen.values
    unseen_out['Predicted Rainfall (mm)'] = np.maximum(yun_pred, 0)
    unseen_out.to_csv(os.path.join(reports_dir, "predictions_2021_2025.csv"), index=False)
    print(f"Unseen predictions saved.")

    # Save train/test predictions too
    use_pca_best = (best_model_name == "SVR (PCA)")
    ytr_pred_best = best_model.predict(X_train_pca if use_pca_best else X_train)
    yte_pred_best = best_model.predict(X_test_pca  if use_pca_best else X_test)

    train_out = train_df[['DATE', 'YEAR']].copy()
    train_out['Actual Rainfall (mm)']    = y_train.values
    train_out['Predicted Rainfall (mm)'] = np.maximum(ytr_pred_best, 0)
    train_out.to_csv(os.path.join(reports_dir, "predictions_train.csv"), index=False)

    test_out = test_df[['DATE', 'YEAR']].copy()
    test_out['Actual Rainfall (mm)']    = y_test.values
    test_out['Predicted Rainfall (mm)'] = np.maximum(yte_pred_best, 0)
    test_out.to_csv(os.path.join(reports_dir, "predictions_test.csv"), index=False)

    print("Train/Test predictions saved.")

    return results_df, best_model_name, models, pca, \
           (X_train, y_train, train_df), \
           (X_test,  y_test,  test_df), \
           (X_unseen, y_unseen, unseen_df), \
           feature_cols


if __name__ == "__main__":
    feat_file = os.path.join(BASE_DIR, "data", "processed", "feature_engineered_data.csv")
    if os.path.exists(feat_file):
        df = pd.read_csv(feat_file)
        train_models(df)
    else:
        print(f"Feature file not found: {feat_file}")
