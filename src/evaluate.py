import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Base directory: project root (one level up from src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_evaluation_plots(df, best_model, best_model_name, pca=None):
    """
    Generate all evaluation plots required by the report:
    1.  Correlation heatmap
    2.  Feature importance (best model if tree-based)
    3.  Model comparison bar chart (Train / Test / Unseen R²)
    4.  Time-series: actual rainfall (full period)
    5.  Train – Predicted vs Actual (time-series)
    6.  Test  – Predicted vs Actual (time-series)
    7.  Unseen (2021-2025) – Predicted vs Actual (time-series)
    8.  Scatter: Predicted vs Actual (test set)
    9.  Residual plot (test set)
    10. Monthly average comparison (test set)
    11. PCA scree plot
    12. PCA biplot
    """
    print("Generating evaluation plots...")

    fig_dir     = os.path.join(BASE_DIR, "outputs", "figures")
    reports_dir = os.path.join(BASE_DIR, "outputs", "reports")
    os.makedirs(fig_dir,     exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    target_col   = 'PRECTOTCORR'
    exclude_cols = ['YEAR', 'DOY', 'DATE', target_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # ── Determine train / test / unseen splits (same seed as train.py) ────────
    import random
    random.seed(42)
    np.random.seed(42)
    train_test_years = list(range(1995, 2021))
    unseen_years     = list(range(2021, 2026))
    random.shuffle(train_test_years)
    split_idx   = int(0.8 * len(train_test_years))
    train_years = sorted(train_test_years[:split_idx])
    test_years  = sorted(train_test_years[split_idx:])

    train_df  = df[df['YEAR'].isin(train_years)].copy()
    test_df   = df[df['YEAR'].isin(test_years)].copy()
    unseen_df = df[df['YEAR'].isin(unseen_years)].copy()

    X_train  = train_df[feature_cols]
    X_test   = test_df[feature_cols]
    X_unseen = unseen_df[feature_cols]

    use_pca = (best_model_name == "SVR (PCA)") and (pca is not None)
    Xtr = pca.transform(X_train)  if use_pca else X_train
    Xte = pca.transform(X_test)   if use_pca else X_test
    Xun = pca.transform(X_unseen) if use_pca else X_unseen

    ytr_pred = np.maximum(best_model.predict(Xtr), 0)
    yte_pred = np.maximum(best_model.predict(Xte), 0)
    yun_pred = np.maximum(best_model.predict(Xun), 0)

    train_df['DATE'] = pd.to_datetime(train_df['DATE'])
    test_df['DATE']  = pd.to_datetime(test_df['DATE'])
    unseen_df['DATE']= pd.to_datetime(unseen_df['DATE'])

    # ── 1. Correlation Heatmap ────────────────────────────────────────────────
    print("  [1/12] Correlation heatmap")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    plt.figure(figsize=(16, 12))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm',
                fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap of All Features', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '01_correlation_heatmap.png'), dpi=150)
    plt.close()

    # ── 2. Feature Importance ─────────────────────────────────────────────────
    print("  [2/12] Feature importance")
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices     = np.argsort(importances)[::-1]
        plt.figure(figsize=(14, 6))
        plt.bar(range(len(feature_cols)),
                importances[indices], color='steelblue', align='center')
        plt.xticks(range(len(feature_cols)),
                   [feature_cols[i] for i in indices], rotation=45, ha='right')
        plt.title(f'Feature Importances – {best_model_name}', fontsize=13)
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, '02_feature_importance.png'), dpi=150)
        plt.close()
    else:
        print("    (skipped – model has no feature_importances_)")

    # ── 3. Model Comparison Bar Chart ─────────────────────────────────────────
    print("  [3/12] Model comparison bar chart")
    report_file = os.path.join(reports_dir, "model_comparison.csv")
    if os.path.exists(report_file):
        res = pd.read_csv(report_file)
        x   = np.arange(len(res))
        w   = 0.25
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - w, res['Train R2'],  w, label='Train R²',  color='steelblue')
        ax.bar(x,     res['Test R2'],   w, label='Test R²',   color='darkorange')
        ax.bar(x + w, res['Unseen R2'], w, label='Unseen R²', color='green')
        ax.set_xticks(x)
        ax.set_xticklabels(res['Model'], rotation=15, ha='right')
        ax.set_ylabel('R² Score')
        ax.set_title('Model Comparison – R² Score (Train / Test / Unseen)')
        ax.legend()
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, '03_model_comparison_r2.png'), dpi=150)
        plt.close()

    # ── 4. Full time-series of actual rainfall ────────────────────────────────
    print("  [4/12] Full actual rainfall time-series")
    full_df = df.copy()
    full_df['DATE'] = pd.to_datetime(full_df['DATE'])
    full_df = full_df.sort_values('DATE')
    plt.figure(figsize=(18, 5))
    plt.plot(full_df['DATE'], full_df[target_col],
             color='royalblue', linewidth=0.6, alpha=0.8)
    plt.title('Actual Daily Rainfall – Full Period (1995-2025)', fontsize=13)
    plt.xlabel('Date'); plt.ylabel('Rainfall (mm/day)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '04_actual_rainfall_full.png'), dpi=150)
    plt.close()

    # ── 5. Train: Predicted vs Actual ─────────────────────────────────────────
    print("  [5/12] Train predicted vs actual")
    _plot_pred_vs_actual(train_df['DATE'], train_df[target_col], ytr_pred,
                         'Training Set – Predicted vs Actual Rainfall',
                         os.path.join(fig_dir, '05_train_pred_vs_actual.png'))

    # ── 6. Test: Predicted vs Actual ──────────────────────────────────────────
    print("  [6/12] Test predicted vs actual")
    _plot_pred_vs_actual(test_df['DATE'], test_df[target_col], yte_pred,
                         'Test Set – Predicted vs Actual Rainfall',
                         os.path.join(fig_dir, '06_test_pred_vs_actual.png'))

    # ── 7. Unseen 2021-2025: Predicted vs Actual ──────────────────────────────
    print("  [7/12] Unseen 2021-2025 predicted vs actual")
    _plot_pred_vs_actual(unseen_df['DATE'], unseen_df[target_col], yun_pred,
                         'Unseen Period (2021-2025) – Predicted vs Actual Rainfall',
                         os.path.join(fig_dir, '07_unseen_pred_vs_actual.png'))

    # ── 8. Scatter: Predicted vs Actual (test) ────────────────────────────────
    print("  [8/12] Scatter plot (test set)")
    r2_te = r2_score(test_df[target_col], yte_pred)
    plt.figure(figsize=(7, 7))
    plt.scatter(test_df[target_col], yte_pred, alpha=0.4, s=10, color='darkorange')
    lim = max(test_df[target_col].max(), yte_pred.max()) * 1.05
    plt.plot([0, lim], [0, lim], 'r--', linewidth=1.5, label='Perfect fit')
    plt.xlabel('Actual Rainfall (mm/day)')
    plt.ylabel('Predicted Rainfall (mm/day)')
    plt.title(f'Predicted vs Actual – Test Set\nR² = {r2_te:.4f}')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '08_scatter_test.png'), dpi=150)
    plt.close()

    # ── 9. Residual plot (test) ───────────────────────────────────────────────
    print("  [9/12] Residual plot (test set)")
    residuals = test_df[target_col].values - yte_pred
    plt.figure(figsize=(10, 5))
    plt.scatter(yte_pred, residuals, alpha=0.4, s=10, color='purple')
    plt.axhline(0, color='red', linewidth=1.5, linestyle='--')
    plt.xlabel('Predicted Rainfall (mm/day)')
    plt.ylabel('Residual (Actual – Predicted)')
    plt.title('Residual Plot – Test Set')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '09_residual_plot_test.png'), dpi=150)
    plt.close()

    # ── 10. Monthly average comparison (test) ────────────────────────────────
    print("  [10/12] Monthly average comparison (test set)")
    test_df['Predicted'] = yte_pred
    monthly = test_df.groupby(test_df['DATE'].dt.month).agg(
        Actual=('PRECTOTCORR', 'mean'),
        Predicted=('Predicted', 'mean')
    ).reset_index()
    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    monthly['Month'] = monthly['DATE'].apply(lambda m: month_names[m-1])
    x = np.arange(12); w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w/2, monthly['Actual'],    w, label='Actual',    color='steelblue')
    ax.bar(x + w/2, monthly['Predicted'], w, label='Predicted', color='darkorange')
    ax.set_xticks(x); ax.set_xticklabels(month_names)
    ax.set_xlabel('Month'); ax.set_ylabel('Avg Rainfall (mm/day)')
    ax.set_title('Monthly Average Rainfall – Test Set (Actual vs Predicted)')
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '10_monthly_avg_test.png'), dpi=150)
    plt.close()

    # ── 11. PCA Scree Plot ────────────────────────────────────────────────────
    print("  [11/12] PCA scree plot")
    if pca is not None:
        _plot_pca_scree(pca, os.path.join(fig_dir, '11_pca_scree.png'))

    # ── 12. PCA Biplot ────────────────────────────────────────────────────────
    print("  [12/12] PCA biplot")
    if pca is not None:
        _plot_pca_biplot(X_train, feature_cols,
                         os.path.join(fig_dir, '12_pca_biplot.png'))

    print(f"\nAll plots saved to: {fig_dir}")


# ── Helper functions ──────────────────────────────────────────────────────────

def _plot_pred_vs_actual(dates, actual, predicted, title, save_path):
    """Time-series plot of actual vs predicted rainfall."""
    idx = np.argsort(dates.values)
    dates_s = dates.values[idx]
    actual_s    = np.array(actual)[idx]
    predicted_s = np.array(predicted)[idx]

    plt.figure(figsize=(16, 5))
    plt.plot(dates_s, actual_s,    label='Actual',    color='royalblue',
             linewidth=0.8, alpha=0.8)
    plt.plot(dates_s, predicted_s, label='Predicted', color='darkorange',
             linewidth=0.8, alpha=0.8)
    plt.title(title, fontsize=13)
    plt.xlabel('Date'); plt.ylabel('Rainfall (mm/day)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_pca_scree(pca, save_path):
    """Scree plot + cumulative explained variance."""
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    x   = range(1, len(evr) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(x, evr, alpha=0.6, color='steelblue', label='Individual')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio', color='steelblue')

    ax2 = ax1.twinx()
    ax2.plot(x, cum, 'ro-', linewidth=1.5, label='Cumulative')
    ax2.axhline(0.85, color='green', linestyle='--', linewidth=1,
                label='85% threshold')
    ax2.set_ylabel('Cumulative Explained Variance', color='red')
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    plt.title('PCA – Scree Plot & Cumulative Explained Variance')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_pca_biplot(X, feature_names, save_path):
    """PCA biplot of first two components with feature loading arrows."""
    from sklearn.decomposition import PCA as _PCA
    pca2 = _PCA(n_components=2)
    scores   = pca2.fit_transform(X)
    loadings = pca2.components_.T * np.sqrt(pca2.explained_variance_)

    plt.figure(figsize=(10, 8))
    plt.scatter(scores[:, 0], scores[:, 1], alpha=0.2, s=8, color='grey')
    scale = np.abs(scores).max() / np.abs(loadings).max() * 0.6
    for i, feat in enumerate(feature_names):
        plt.arrow(0, 0, loadings[i, 0] * scale, loadings[i, 1] * scale,
                  color='red', alpha=0.7, head_width=0.3, length_includes_head=True)
        plt.text(loadings[i, 0] * scale * 1.15,
                 loadings[i, 1] * scale * 1.15,
                 feat, color='darkred', fontsize=8, ha='center')
    plt.xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title('PCA Biplot – Feature Loadings (PC1 vs PC2)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    feat_file  = os.path.join(BASE_DIR, "data", "processed", "feature_engineered_data.csv")
    model_file = os.path.join(BASE_DIR, "models", "best_model.pkl")
    pca_file   = os.path.join(BASE_DIR, "models", "pca.pkl")

    if not os.path.exists(feat_file):
        print(f"Feature file not found: {feat_file}")
    elif not os.path.exists(model_file):
        print(f"Model file not found: {model_file}. Run src/train.py first.")
    else:
        df    = pd.read_csv(feat_file)
        model = joblib.load(model_file)
        pca   = joblib.load(pca_file) if os.path.exists(pca_file) else None

        # Read best model name from comparison CSV
        comp_file = os.path.join(BASE_DIR, "outputs", "reports", "model_comparison.csv")
        if os.path.exists(comp_file):
            comp = pd.read_csv(comp_file)
            best_name = comp.sort_values("Test R2", ascending=False).iloc[0]["Model"]
        else:
            best_name = "Best Model"

        generate_evaluation_plots(df, model, best_name, pca)
