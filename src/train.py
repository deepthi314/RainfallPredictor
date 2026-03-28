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

def train_models(df):
    """
    Train 5 models using year-wise split.
    
    Split Strategy:
    - 80:20 split on years 1995-2020
    - Unseen data: 2021-2025
    """
    # Fix random seed
    random_state = 42
    np.random.seed(random_state)
    random.seed(random_state)
    
    # Filter years
    train_test_years = list(range(1995, 2021))
    unseen_years = list(range(2021, 2026))
    
    # Shuffle and split
    random.shuffle(train_test_years)
    split_idx = int(0.8 * len(train_test_years))
    train_years = sorted(train_test_years[:split_idx])
    test_years = sorted(train_test_years[split_idx:])
    
    print(f"Training years: {train_years}")
    print(f"Testing years: {test_years}")
    print(f"Unseen years: {unseen_years}")
    
    # Feature identification
    target_col = 'PRECTOTCORR'
    exclude_cols = ['YEAR', 'DOY', 'DATE', target_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Data splitting
    train_df = df[df['YEAR'].isin(train_years)]
    test_df = df[df['YEAR'].isin(test_years)]
    unseen_df = df[df['YEAR'].isin(unseen_years)]
    
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]
    X_unseen, y_unseen = unseen_df[feature_cols], unseen_df[target_col]
    
    models = {
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=random_state),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=15, random_state=random_state),
        "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=random_state),
        "SVR (PCA)": SVR(kernel='rbf', C=10, epsilon=0.1),
        "MLP NN": MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=random_state)
    }
    
    results = []
    
    # PCA for SVR
    # Determine components for 85% variance from training data
    pca = PCA(n_components=0.85)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    for name, model in models.items():
        print(f"Training {name}...")
        if name == "SVR (PCA)":
            model.fit(X_train_pca, y_train)
            y_train_pred = model.predict(X_train_pca)
            y_test_pred = model.predict(X_test_pca)
        else:
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
        metrics = {
            "Model": name,
            "Train R2": r2_score(y_train, y_train_pred),
            "Train RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            "Train MAE": mean_absolute_error(y_train, y_train_pred),
            "Train MSE": mean_squared_error(y_train, y_train_pred),
            "Test R2": r2_score(y_test, y_test_pred),
            "Test RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            "Test MAE": mean_absolute_error(y_test, y_test_pred),
            "Test MSE": mean_squared_error(y_test, y_test_pred)
        }
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    print("\nModel Comparison Table:")
    print(results_df.to_string(index=False))
    
    # Save the best model (based on Test R2)
    best_model_name = results_df.sort_values(by="Test R2", ascending=False).iloc[0]["Model"]
    print(f"Best model: {best_model_name}")
    
    best_model = models[best_model_name]
    models_dir = r"e:\college\WaterHarvesting\rainfall_prediction\models"
    reports_dir = r"e:\college\WaterHarvesting\rainfall_prediction\outputs\reports"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    joblib.dump(best_model, os.path.join(models_dir, "best_model.pkl"))
    results_df.to_csv(os.path.join(reports_dir, "model_comparison.csv"), index=False)
    
    return results_df, best_model_name

if __name__ == "__main__":
    feat_file = r"e:\college\WaterHarvesting\rainfall_prediction\data\processed\feature_engineered_data.csv"
    if os.path.exists(feat_file):
        df = pd.read_csv(feat_file)
        train_models(df)
    else:
        print(f"Feature file {feat_file} not found.")

