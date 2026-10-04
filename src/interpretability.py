import shap
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython.display import Image, display
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def generate_shap_analysis(model, X_background, X_signal, feature_names):
    """
    Final fix: Extracts 2D SHAP values for the Signal class to prevent dimension errors.
    """
    print("Calculating SHAP values (optimized for speed)...")
    
    # 1. Use TreeExplainer
    explainer = shap.TreeExplainer(model)
    
    # 2. Sample 200 events for fast execution
    sample_background = X_background.sample(min(200, len(X_background)), random_state=42)
    
    print("Computing SHAP values for 200 events...")
    raw_explanation = explainer(sample_background)
    
    # 3. THE FIX: Extract the 2D array for the Positive Class (Signal = Index 1)
    # raw_explanation.values has shape (samples, features, 2 classes)
    shap_values_2d = raw_explanation.values[:, :, 1] 
    
    # Create a new, clean 2D Explanation object specifically for plotting
    explanation_2d = shap.Explanation(
        values=shap_values_2d,
        data=sample_background,
        feature_names=feature_names
    )
    
    os.makedirs('../outputs/figures', exist_ok=True)
    
    # 4. Summary Plot (Beeswarm plot) - Modern API
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(explanation_2d, max_display=10, show=False)
    plt.title("SHAP Summary: How Kinematic Features Drive Higgs Identification", fontsize=14)
    plt.tight_layout()
    
    plt.savefig('../outputs/figures/shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close() 
    print("✅ SHAP Summary (Beeswarm) plot saved.")
    display(Image(filename='../outputs/figures/shap_summary.png'))

    # 5. Bar Plot (Mean Absolute SHAP Value)
    plt.figure(figsize=(10, 6))
    shap.plots.bar(explanation_2d, max_display=10, show=False)
    plt.title("SHAP Feature Importance: Mean Absolute Impact on Model Output", fontsize=14)
    plt.tight_layout()
    
    plt.savefig('../outputs/figures/shap_bar.png', dpi=300, bbox_inches='tight')
    plt.close() 
    print("✅ SHAP Bar plot saved.")
    display(Image(filename='../outputs/figures/shap_bar.png'))


def generate_pca_visualization(X, y, feature_names):
    """
    Applies PCA to reduce 28D physics data to 2D for visual separation analysis.
    """
    print("Performing PCA dimensionality reduction...")
    
    # 1. Standardize the data (CRUCIAL for PCA)
    # PCA is sensitive to scale. Features like 'PRI_jet_all_pt' (hundreds of GeV) 
    # would dominate 'PRI_lep_eta' (small angles) without scaling.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Apply PCA to reduce to 2 components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Calculate how much variance is explained by these 2 components
    explained_variance = pca.explained_variance_ratio_
    print(f"✅ PCA Complete. PC1 explains {explained_variance[0]:.1%} and PC2 explains {explained_variance[1]:.1%} of total variance.")
    print(f"   (Total variance captured in 2D: {(explained_variance[0] + explained_variance[1]):.1%})")
    
    # 3. Sample the data for faster, cleaner plotting (e.g., 5000 events)
    # Plotting 250,000 points will crash the browser or look like a solid blob.
    sample_size = min(5000, len(X))
    indices = np.random.choice(len(X), sample_size, replace=False)
    
    X_pca_sample = X_pca[indices]
    y_sample = y.iloc[indices] # Use .iloc for pandas Series
    
    # 4. Plot the 2D PCA Projection
    plt.figure(figsize=(10, 8))
    
    # Plot Background (Z boson)
    plt.scatter(X_pca_sample[y_sample == 0, 0], X_pca_sample[y_sample == 0, 1], 
                c='blue', alpha=0.4, s=10, label='Background (Z boson)')
    
    # Plot Signal (Higgs boson)
    plt.scatter(X_pca_sample[y_sample == 1, 0], X_pca_sample[y_sample == 1, 1], 
                c='red', alpha=0.6, s=10, label='Signal (Higgs boson)')
    
    plt.title(f'PCA Projection of Higgs vs Z Boson Events\n(Total Variance Captured: {(explained_variance[0] + explained_variance[1]):.1%})', fontsize=14)
    plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.1%} variance)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save and display
    os.makedirs('../outputs/figures', exist_ok=True)
    plt.savefig('../outputs/figures/pca_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ PCA Visualization plot saved.")
    display(Image(filename='../outputs/figures/pca_visualization.png'))