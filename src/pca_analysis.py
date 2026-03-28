import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

def perform_pca_analysis(X_scaled, target_variance=0.85):
    """
    Perform PCA on the feature set and generate diagnostic plots.
    
    Tasks:
    - Apply PCA
    - Plot scree plot and cumulative explained variance
    - Determine # of components for target_variance
    - Plot biplot of first two components
    """
    print(f"Performing PCA analysis for target variance {target_variance*100}%...")
    
    pca = PCA()
    pca.fit(X_scaled)
    
    exp_var_ratio = pca.explained_variance_ratio_
    cum_exp_var = np.cumsum(exp_var_ratio)
    
    # Identify components needed for 85% variance
    n_components_optimal = np.argmax(cum_exp_var >= target_variance) + 1
    print(f"Number of components needed for {target_variance*100}% variance: {n_components_optimal}")
    
    # Scree plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(exp_var_ratio) + 1), exp_var_ratio, alpha=0.5, align='center', label='Individual Explained Variance')
    plt.step(range(1, len(exp_var_ratio) + 1), cum_exp_var, where='mid', label='Cumulative Explained Variance')
    plt.axhline(y=target_variance, color='r', linestyle='--', label=f'{target_variance*100}% Variance')
    plt.xlabel('Principal Component Index')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot & Cumulative Explained Variance')
    plt.legend(loc='best')
    plt.grid(True)
    fig_dir = r"e:\college\WaterHarvesting\rainfall_prediction\outputs\figures"
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, "pca_scree_plot.png"), dpi=300)
    plt.close()
    
    # Biplot (first 2 components)
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 10))
    # Plot feature loadings as arrows
    loadings = pca_2d.components_.T * np.sqrt(pca_2d.explained_variance_)
    for i, feature in enumerate(X_scaled.columns):
        plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='r', alpha=0.5, head_width=0.02)
        plt.text(loadings[i, 0]*1.2, loadings[i, 1]*1.2, feature, color='g', ha='center', va='center')
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Biplot (Feature Loadings)')
    plt.grid(True)
    plt.savefig(r"e:\college\WaterHarvesting\rainfall_prediction\outputs\figures\pca_biplot.png", dpi=300)
    plt.close()
    
    return n_components_optimal

if __name__ == "__main__":
    feat_file = r"e:\college\WaterHarvesting\rainfall_prediction\data\processed\feature_engineered_data.csv"
    if os.path.exists(feat_file):
        df = pd.read_csv(feat_file)
        # Drop columns not used in PCA
        cols_to_drop = ['DATE', 'YEAR', 'DOY', 'PRECTOTCORR']
        X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        
        n_comp = perform_pca_analysis(X)
        print(f"PCA Analysis complete. Optimal components: {n_comp}")
    else:
        print(f"Feature file {feat_file} not found.")

