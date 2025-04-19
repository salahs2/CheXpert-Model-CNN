import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

class TorchSVM(nn.Module):
    def __init__(self, input_dim):
        super(TorchSVM, self).__init__()
        # Linear layer for SVM
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        # Output raw scores (no activation)
        return self.fc(x)

def hinge_loss(outputs, labels):
    # Hinge loss: max(0, 1 - y * f(x))
    labels = 2 * labels - 1
    return torch.mean(torch.clamp(1 - outputs * labels, min=0))

# Initialize weights properly
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


def train_torch_svm(df, epochs=40, lr=1e-3, batch_size=32):
    
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # Define model
    model = TorchSVM(input_dim=X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.apply(init_weights)
    

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train).squeeze()
        loss = hinge_loss(outputs, y_train)
        
        loss.backward()
        optimizer.step()

        # Print loss for every epoch
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test).squeeze()
        predictions = (outputs > 0).float()  # Convert logits to binary predictions
        acc = accuracy_score(y_test.cpu(), predictions.cpu())
        print("\nAccuracy:", acc)
        print("\nClassification Report:\n", classification_report(y_test.cpu(), predictions.cpu()))

    return model
