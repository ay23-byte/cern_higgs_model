import pandas as pd
import numpy as np

def load_and_clean_data(file_path):
    """
    Loads the CERN Higgs dataset and handles detector inefficiencies (-999.0).
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # 1. Identify Missing Values
    # In this specific dataset, -999.0 represents missing detector data.
    missing_mask = (df == -999.0).sum().sum()
    print(f"Total missing detector values found: {missing_mask}")
    
    # 2. Replace -999.0 with standard NaN
    df.replace(-999.0, np.nan, inplace=True)
    
    # 3. Separate Features (X) and Labels (y)
    # 'EventId' is just an index, 'Weight' is for advanced statistical analysis, 
    # 'Label' is our target (s = signal, b = background).
    y = df['Label'].apply(lambda x: 1 if x == 's' else 0) # Convert 's'/'b' to 1/0
    X = df.drop(['EventId', 'Label', 'Weight'], axis=1)
    
    # 4. Impute Missing Values
    # We use the median because physics distributions often have long tails (outliers).
    # Mean would be skewed by extreme high-energy events.
    X.fillna(X.median(), inplace=True)
    
    print(f"Data cleaning complete. Shape of X: {X.shape}, Shape of y: {y.shape}")
    return X, y

# Add this inside your data processing function
def add_physics_features(df):
    # Calculate angular separation between Tau lepton and leading Jet
    delta_eta = df['PRI_tau_eta'] - df['PRI_jet_leading_eta']
    delta_phi = df['PRI_tau_phi'] - df['PRI_jet_leading_phi']
    
    # Handle the periodic boundary condition for phi (angles wrap around at pi)
    delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi 
    
    df['DER_deltaR_tau_jet'] = np.sqrt(delta_eta**2 + delta_phi**2)
    return df
    
if __name__ == "__main__":
    # Test the function
    X, y = load_and_clean_data('../data/raw/training.csv')
    print(X.head())