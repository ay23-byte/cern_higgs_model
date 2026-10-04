import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np

class HiggsMLP(nn.Module):
    """A simple Multi-Layer Perceptron for Tabular Physics Data."""
    def __init__(self, input_dim):
        super(HiggsMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128), # Helps stabilize training
            nn.Dropout(0.3),     # Prevents overfitting
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid() # Output probability between 0 and 1
        )

    def forward(self, x):
        return self.network(x)

def train_pytorch_model(X, y, epochs=50, batch_size=256, learning_rate=0.001):
    """Trains a PyTorch MLP and returns the model and test AUC."""
    
    # Convert Pandas/NumPy to PyTorch Tensors
    X_tensor = torch.FloatTensor(X.values)
    y_tensor = torch.FloatTensor(y.values).unsqueeze(1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize Model, Loss, and Optimizer
    model = HiggsMLP(input_dim=X.shape[1])
    criterion = nn.BCELoss() # Binary Cross Entropy
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\nTraining PyTorch MLP for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}")
            
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test).numpy()
        auc = roc_auc_score(y_test.numpy(), test_preds)
        print(f"🎯 PyTorch MLP Test AUC: {auc:.4f}")
        
    return model, auc