import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Custom Dataset for PyTorch
class PneumoniaDataset(Dataset):
    def __init__(self, images, other_features, labels):
        self.images = torch.tensor(images, dtype=torch.float32)  # Convert images to tensors
        self.other_features = torch.tensor(other_features, dtype=torch.float32)  # Numerical features
        self.labels = torch.tensor(labels, dtype=torch.float32)  # Binary labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.other_features[idx], self.labels[idx]

# CNN Model
class PneumoniaCNN(nn.Module):
    def __init__(self, num_numerical_features):
        super(PneumoniaCNN, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Input channels=1 (grayscale)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 16 * 16 + num_numerical_features, 128),  # Adjust based on image size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Binary classification
        )

    def forward(self, image, numerical_features):
        x = self.cnn_layers(image)
        x = x.view(x.size(0), -1)  # Flatten
        combined = torch.cat((x, numerical_features), dim=1)
        out = self.fc_layers(combined)
        return out

# Training Function
def train_model(df, image_size=(128, 128), batch_size=32, epochs=10, lr=1e-4):
    # Preprocessing
    # Normalize images
    df['image_normalized'] = df['Image'].apply(lambda img: img / 255.0)

    # Encode gender
    gender_encoder = LabelEncoder()
    df['gender_encoded'] = gender_encoder.fit_transform(df['Sex'])

    # Prepare features and labels
    images = np.stack(df['image_normalized'].values)[:, np.newaxis, :, :]  # Add channel dimension (grayscale)
    other_features = df[['Age', 'gender_encoded']].values
    labels = df['Pneumonia'].values

    # Scale numerical features
    scaler = StandardScaler()
    other_features_scaled = scaler.fit_transform(other_features)

    # Train-test split
    X_train_img, X_test_img, X_train_other, X_test_other, y_train, y_test = train_test_split(
        images, other_features_scaled, labels, test_size=0.2, random_state=42
    )

    # Create DataLoader
    train_dataset = PneumoniaDataset(X_train_img, X_train_other, y_train)
    test_dataset = PneumoniaDataset(X_test_img, X_test_other, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PneumoniaCNN(num_numerical_features=X_train_other.shape[1]).to(device)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, features, labels in train_loader:
            images, features, labels = images.to(device), features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for images, features, labels in test_loader:
            images, features, labels = images.to(device), features.to(device), labels.to(device)
            outputs = model(images, features).squeeze()
            y_pred.extend((outputs > 0.5).cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    from sklearn.metrics import classification_report, accuracy_score
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

    return model
