import joblib
import sys
import os

# Add src to path
sys.path.append('src')
from model_training import train_higgs_model_advanced

print("🚀 Training model for deployment (this will take ~1 minute)...")
model, X, y = train_higgs_model_advanced('data/raw/training.csv')

# Create output directory
os.makedirs('outputs/models', exist_ok=True)

# Save the trained model and the feature names
joblib.dump(model, 'outputs/models/higgs_rf_model.pkl')
joblib.dump(X.columns.tolist(), 'outputs/models/feature_names.pkl')

print("✅ Model saved successfully to outputs/models/!")