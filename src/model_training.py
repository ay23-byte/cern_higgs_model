import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from data_processing import load_and_clean_data
from sklearn.utils.class_weight import compute_class_weight

def train_higgs_model_advanced(file_path):
    """
    Advanced training pipeline using Stratified K-Fold Cross-Validation.
    """
    print("Loading and cleaning data...")
    X, y = load_and_clean_data(file_path)
    
    # 1. Initialize a more robust Random Forest
    # We increase n_estimators and remove max_depth to let the model learn complex physics boundaries
    model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=None, # Allow trees to grow fully
        min_samples_split=10, # Prevent overfitting on noise
        random_state=42, 
        n_jobs=-1
    )
    
    # 2. Stratified K-Fold Cross-Validation (5 Folds)
    # This ensures every fold has the same ratio of Signal/Background
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\nRunning 5-Fold Cross-Validation... (This may take a minute)")
    # We use 'roc_auc' as the scoring metric for each fold
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    print(f"Fold Scores: {cv_scores}")
    print(f"Mean CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # 3. Train the FINAL model on the ENTIRE dataset 
    # (Since we validated it with CV, it's safe to use all data for the final model)
    print("\nTraining final model on full dataset...")
    model.fit(X, y)
    
    return model, X, y

if __name__ == "__main__":
    model, X, y = train_higgs_model_advanced('../data/raw/training.csv')



def train_with_realistic_imbalance(X, y):
    """
    Simulates real LHC data imbalance and trains a model using class weights.
    """
    # 1. Artificially unbalance the data to mimic real LHC conditions
    # Keep all Signal (y=1), but only keep 10% of Background (y=0)
    X_signal = X[y == 1]
    X_background = X[y == 0]
    
    # Sample only 10% of background to create a 1:10 Signal/Background ratio
    # (In reality it's 1:1,000,000, but 1:10 is enough to prove the concept)
    X_background_sampled = X_background.sample(frac=0.1, random_state=42) 
    
    X_imbalanced = pd.concat([X_signal, X_background_sampled])
    y_imbalanced = pd.concat([y[y==1], y[y==0].sample(frac=0.1, random_state=42)])
    
    print(f"New Class Distribution: Signal={sum(y_imbalanced)}, Background={sum(1-y_imbalanced)}")
    
    # 2. Compute Class Weights
    # This tells the model: "Background is common, so penalize missing a Signal more heavily!"
    classes = np.unique(y_imbalanced)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_imbalanced)
    class_weight_dict = dict(zip(classes, weights))
    print(f"Computed Class Weights: {class_weight_dict}")
    
    # 3. Train the Random Forest with these weights
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=200, class_weight=class_weight_dict, random_state=42, n_jobs=-1)
    model.fit(X_imbalanced, y_imbalanced)
    
    return model