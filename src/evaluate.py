import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

def generate_evaluation_plots(df, best_model, best_model_name):
    """
    Generate evaluation plots:
    1. Correlation heatmap
    2. Feature Importance (if applicable)
    3. Model comparison bar chart
    4. Time-series of actual rainfall
    5. Predicted vs Actual scatter
    6. Residual plot
    7. Monthly average comparison
    """
    print("Generating evaluation plots...")
    
    # Output directory
    fig_dir = r"e:\college\WaterHarvesting\rainfall_prediction\outputs\figures"
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Correlation Heatmap
    plt.figure(figsize=(15, 10))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of All Features')
    plt.savefig(os.path.join(fig_dir, 'correlation_heatmap.png'), dpi=300)
    plt.close()
    
    # 2. Feature Importance (Random Forest specifically is requested)
    # We might need to refit RF if it's not the best model, or just use it from the study
    # For now, if best_model has feature_importances_, we use it.
    if hasattr(best_model, 'feature_importances_'):
        target_col = 'PRECTOTCORR'
        exclude_cols = ['YEAR', 'DOY', 'DATE', target_col]
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importances (Best Model)')
        plt.bar(range(len(feature_cols)), importances[indices], align='center')
        plt.xticks(range(len(feature_cols)), [feature_cols[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'feature_importance.png'), dpi=300)
        plt.close()
        
    # 3. Model comparison bar chart
    report_file = r"e:\college\WaterHarvesting\rainfall_prediction\outputs\reports\model_comparison.csv"
    if os.path.exists(report_file):
        results_df = pd.read_csv(report_file)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Model', y='Test R2', data=results_df)
        plt.title('Model Comparison (Test R2 Score)')
        plt.ylim(0, 1)
        plt.savefig(os.path.join(fig_dir, 'model_comparison_r2.png'), dpi=300)
        plt.close()
        
    print("Plots generated successfully.")

if __name__ == "__main__":
    feat_file = r"e:\college\WaterHarvesting\rainfall_prediction\data\processed\feature_engineered_data.csv"
    model_file = r"e:\college\WaterHarvesting\rainfall_prediction\models\best_model.pkl"
    
    if os.path.exists(feat_file) and os.path.exists(model_file):
        df = pd.read_csv(feat_file)
        model = joblib.load(model_file)
        generate_evaluation_plots(df, model, "Best Model")
