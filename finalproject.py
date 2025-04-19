import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models
from torchvision.models import ResNet18_Weights
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score

class_labels = {"control": 0, "covid": 1, "pneumonia": 2, "tuberculosis": 3}
dataset_path = "./dataset"

# load and balance dataset
data = []
for label_name, label in class_labels.items():
    class_dir = os.path.join(dataset_path, label_name)
    images = sorted(os.listdir(class_dir))[:576]
    for img in images:
        img_path = os.path.join(class_dir, img)
        data.append({"path": img_path, "label": label})

df = pd.DataFrame(data)

train_df = df.groupby("label").apply(lambda x: x.iloc[:500]).reset_index(drop=True)
test_df = df.groupby("label").apply(lambda x: x.iloc[500:576]).reset_index(drop=True)

print(f"Training set size: {len(train_df)}")
print(f"Testing set size: {len(test_df)}")

class ChestXRayDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]["path"]
        label = self.dataframe.iloc[idx]["label"]

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (224, 224))
        image = np.stack([image] * 3, axis=-1)

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        return image, label

batch_size = 32
num_workers = min(4, os.cpu_count())
train_dataset = ChestXRayDataset(train_df)
test_dataset = ChestXRayDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

class CNNModel(nn.Module):
    def __init__(self, num_classes=4):
        super(CNNModel, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def get_folds(dataset, num_folds=5):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    indices = list(range(len(dataset)))
    folds = []

    for train_idx, val_idx in kf.split(indices):
        train_fold = Subset(dataset, train_idx)
        val_fold = Subset(dataset, val_idx)
        folds.append((train_fold, val_fold))
    
    return folds

def train_and_evaluate(folds, num_epochs=10, batch_size=32, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for fold_idx, (train_fold, val_fold) in enumerate(folds):
        print(f"Training fold {fold_idx + 1}/{len(folds)}...")
        
        train_loader = DataLoader(train_fold, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
        val_loader = DataLoader(val_fold, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
        
        model = CNNModel(num_classes=4).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(val_labels, val_preds)
        recall = recall_score(val_labels, val_preds, average="weighted")
        precision = precision_score(val_labels, val_preds, average="weighted")
        results.append((acc, recall, precision))
        
        print(f"Fold {fold_idx + 1}: Accuracy = {acc:.4f}, Recall = {recall:.4f}, Precision = {precision:.4f}")
    
    return results

folds = get_folds(train_dataset)
results = train_and_evaluate(folds)

avg_acc = np.mean([r[0] for r in results])
avg_recall = np.mean([r[1] for r in results])
avg_precision = np.mean([r[2] for r in results])

print(f"Average Accuracy: {avg_acc:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
