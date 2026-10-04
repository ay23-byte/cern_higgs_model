import matplotlib.pyplot as plt
import numpy as np
import os

def advanced_evaluation(model, X, y):
    """
    Calculates Feature Importance and the Physics Significance Curve.
    """
    # 1. Feature Importance Analysis
    importances = model.feature_importances_
    feature_names = X.columns
    # Sort them from most important to least
    sorted_idx = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 8))
    plt.barh(feature_names[sorted_idx][:10], importances[sorted_idx][:10], color='teal') # Top 10
    plt.xlabel('Relative Importance')
    plt.title('Top 10 Kinematic Features Driving Higgs Identification')
    plt.gca().invert_yaxis() # Highest importance at the top
    plt.tight_layout()
    plt.savefig('../outputs/figures/feature_importance.png', dpi=300)
    print("✅ Feature Importance plot saved.")
    plt.show()
    
    # 2. The Physics Significance Curve (S / sqrt(S + B))
    # In HEP, we care about maximizing Signal (S) while minimizing Background (B)
    y_scores = model.predict_proba(X)[:, 1]
    
    # Sort events by predicted probability (from most likely signal to least)
    sorted_indices = np.argsort(y_scores)[::-1]
    sorted_y = y[sorted_indices]
    
    total_events = len(y)
    cumulative_signal = np.cumsum(sorted_y)
    cumulative_background = np.cumsum(1 - sorted_y)
    
    # Calculate Significance: S / sqrt(S + B)
    # Add a tiny epsilon to avoid division by zero
    significance = cumulative_signal / np.sqrt(cumulative_signal + cumulative_background + 1e-9)
    
    # Plot the Significance Curve
    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(0, 1, total_events), significance, color='crimson', lw=2)
    plt.xlabel('Fraction of Events Selected (Cut Threshold)')
    plt.ylabel('Approximate Significance ($S / \sqrt{S+B}$)')
    plt.title('Physics Significance vs Event Selection Cut')
    plt.grid(True, alpha=0.3)
    
    # Highlight the peak significance
    max_sig_idx = np.argmax(significance)
    max_sig = significance[max_sig_idx]
    plt.axvline(x=max_sig_idx/total_events, color='black', linestyle='--', label=f'Peak Significance: {max_sig:.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../outputs/figures/significance_curve.png', dpi=300)
    print(f"✅ Peak Physics Significance: {max_sig:.2f}")
    print("✅ Significance Curve saved.")
    plt.show()
