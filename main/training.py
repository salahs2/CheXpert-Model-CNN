import pandas as pd
import numpy as np
import pickle

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from torchvision.transforms import transforms

BATCH_SIZE = 64

'''SVM Model'''
class SVM:
    def __init__(self, kernel='linear', C=1.0):
        self.kernel = kernel
        self.C = C
        self.model = SVC(kernel=self.kernel, C=self.C, class_weight='balanced')

    def train(self, X, y):
        """Train the SVM model."""
        self.model.fit(X, y)
        print("Model trained successfully.")

    def save_model(self, filepath):
        """Save the trained model to a file."""
        with open(filepath, 'wb') as file:
            pickle.dump(self.model, file)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath):
        """Load a saved model from a file."""
        with open(filepath, 'rb') as file:
            return pickle.load(file)

def image_to_pixels(image_path, target_size=(128, 128)):
    """Convert an image to a resized numpy pixel array."""
    try:
        with Image.open(image_path) as img:
            img = img.convert('L')  # Convert to grayscale
            img = img.resize(target_size)  # Resize to the target size
            return np.array(img)  # Convert image to a NumPy array
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None  # Return None if there's an error


def batch_process_images(df, batch_size, target_size=(128, 128)):
    """Yield batches of features and labels."""
    num_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)

    for batch_num in range(num_batches):
        batch_df = df.iloc[batch_num * batch_size:(batch_num + 1) * batch_size]
        images = []
        for path in batch_df['Path']:
            image = image_to_pixels(path, target_size)
            if image is not None:
                images.append(image.flatten() / 255.0)  # Normalize pixels
        other_features = batch_df[['Age', 'gender_encoded']].values
        batch_features = np.hstack((images, other_features))
        batch_labels = batch_df['Lung Opacity'].values
        yield batch_features, batch_labels

'''Naive Bayes Model'''
class NB:
    def __init__(self):
        self.model = GaussianNB()
        self.scaler = None
    
    def save_model(self, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump(self.model, file)
        print(f"Model saved to {filepath}")

    def pre_processing(self, df):
        df['Image'] = df['Path'].apply(lambda x: image_to_pixels(x, target_size=(128, 128)))

        # encode f/m to 0/1
        gender_encoder = LabelEncoder()
        df['gender_encoded'] = gender_encoder.fit_transform(df['Sex'])

        # normalize and flatten
        df['image_normalized'] = df['Image'].apply(lambda img: img / 255.0)
        df['image_flattened'] = df['image_normalized'].apply(lambda img: img.flatten())

        # get features
        image_features = np.stack(df['image_flattened'].values)
        other_features = df[['Age', 'gender_encoded']].values
        self.scaler = StandardScaler()
        other_features_scaled = self.scaler.fit_transform(other_features)
        
        # combine image and non-image features
        combined_features = np.hstack((image_features, other_features_scaled))
        labels = df['Lung Opacity'].values

        return combined_features, labels

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, df):
        features, labels = self.pre_processing(df)

        # train-test split
        X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )

        # train/eval model
        self.model.fit(X_train_nb, y_train_nb)
        predictions_nb = self.model.predict(X_test_nb)
        nb_accuracy = accuracy_score(y_test_nb, predictions_nb)
        print(f"Naive Bayes Test Accuracy: {nb_accuracy:.4f}")
        print("\nClassification Report (NB):\n", classification_report(y_test_nb, predictions_nb))

        # save model
        self.save_model("nb_model.pkl")
        print("Naive Bayes Model saved as nb_model.pkl")

        return nb_accuracy, classification_report(y_test_nb, predictions_nb, output_dict=True)



'''NN Model'''
class Dataset(Dataset):
    def __init__(self, dataframe, target_size=(128, 128)):
        self.dataframe = dataframe
        self.target_size = target_size
        self.scaler = StandardScaler()
        self.dataframe['Age'] = self.scaler.fit_transform(self.dataframe[['Age']])
        self.gender_encoder = LabelEncoder()
        self.dataframe['gender_encoded'] = self.gender_encoder.fit_transform(self.dataframe['Sex'])
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row['Path']
        age = row['Age']
        gender_encoded = row['gender_encoded']
        label = row['Lung Opacity']


        try:
            with Image.open(image_path) as img:
                img = self.transform(img)
        except Exception as e:
            raise ValueError(f"Error processing image at {image_path}: {e}")
        features = torch.tensor([age, gender_encoded], dtype=torch.float32)
        return torch.tensor(img, dtype=torch.float32), features, torch.tensor(label, dtype=torch.float32)


# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 32 * 32 + 2, 128)  # 32x32 image after pooling, +6 for metadata
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image, metadata):
        x = self.pool(nn.ReLU()(self.conv1(image)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.cat((x, metadata), dim=1)  # Combine image features with metadata
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)

# Training Function
def train_cnn_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, metadata, labels in dataloader:
        images, metadata, labels = images.to(device), metadata.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images, metadata)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Evaluation Function
def evaluate_cnn_model(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for images, metadata, labels in dataloader:
            images, metadata, labels = images.to(device), metadata.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images, metadata)
            predictions = (outputs > 0.5).float()
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)
    return accuracy_score(all_labels, all_predictions)


if __name__ == "__main__":
    '''Load Data'''
    df = pd.read_csv('./main/data/train.csv')

    # select appropriate
    selected_features = [
        'Path', 'Sex', 'Age','Lung Opacity', 
    ]
    df = pd.read_csv('./main/data/train.csv')[selected_features].dropna()

    def get_last_dirs_and_file(path, num_dirs=3):
        parts = path.split('/')  # Split the path into components
        temp = '/'.join(parts[-(num_dirs + 1):])  # Join the last `num_dirs` + file name
        return './main/data/' + temp

    df['Path'] = df['Path'].apply(get_last_dirs_and_file)

    # make subsets for lung opacity balance
    df_0 = df[df['Lung Opacity'] == 0].head(500) 
    df_1 = df[df['Lung Opacity'] == 1].head(500)
    df = pd.concat([df_0, df_1]).reset_index(drop=True)

    '''NB Workflow'''
    nb = NB()
    df_nb = df.copy()
    nb_accuracy, nb_report = nb.evaluate(df_nb)

    '''Prepare SVM DataFrame'''
    # convert labels to -1 and 1 for SVM
    df_svm = df.copy()
    df_svm['Lung Opacity'] = df_svm['Lung Opacity'].apply(lambda x: 2 * x - 1)
    
    # encode gender for SVM
    gender_encoder = LabelEncoder()
    df_svm['gender_encoded'] = gender_encoder.fit_transform(df_svm['Sex'])

    # scale age for SVM
    scaler = StandardScaler()
    df_svm['Age'] = scaler.fit_transform(df_svm[['Age']])

    '''Prepare CNN DataFrame'''
    # convert labels to 0 and 1 for CNN
    df_cnn = df.copy()
    df_cnn['Lung Opacity'] = df_cnn['Lung Opacity'].astype(int)

    # encode gender for CNN
    df_cnn['gender_encoded'] = gender_encoder.fit_transform(df_cnn['Sex'])

    # scale age
    df_cnn['Age'] = scaler.transform(df_cnn[['Age']])

    '''Train-Test Split'''
    train_df_svm, test_df_svm = train_test_split(df_svm, test_size=0.2, random_state=42)
    train_df_cnn, test_df_cnn = train_test_split(df_cnn, test_size=0.2, random_state=42)

    '''SVM Workflow'''
    # train SVM using batch processing
    svm = SVM(kernel='rbf', C=1.0)

    for batch_features, batch_labels in batch_process_images(train_df_svm, batch_size=BATCH_SIZE, target_size=(128, 128)):
        if 'X_train_svm' not in locals():
            X_train_svm = batch_features
            y_train_svm = batch_labels
        else:
            X_train_svm = np.vstack((X_train_svm, batch_features))
            y_train_svm = np.hstack((y_train_svm, batch_labels))

    svm.train(X_train_svm, y_train_svm)
    svm.save_model("svm_model.pkl")

    # evaluate SVM
    all_test_features_svm = []
    all_test_labels_svm = []
    for batch_features, batch_labels in batch_process_images(test_df_svm, batch_size=BATCH_SIZE, target_size=(128, 128)):
        all_test_features_svm.append(batch_features)
        all_test_labels_svm.append(batch_labels)

    X_test_svm = np.vstack(all_test_features_svm)
    y_test_svm = np.hstack(all_test_labels_svm)
    predictions_svm = svm.model.predict(X_test_svm)
    svm_accuracy = accuracy_score(y_test_svm, predictions_svm)
    print(f"SVM Test Accuracy: {svm_accuracy:.4f}")

    '''CNN Workflow'''
    # create data loaders for CNN
    train_dataset_cnn = Dataset(train_df_cnn)
    test_dataset_cnn = Dataset(test_df_cnn)
    train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=BATCH_SIZE, shuffle=True)
    test_loader_cnn = DataLoader(test_dataset_cnn, batch_size=BATCH_SIZE, shuffle=False)

    # train CNN
    model = CNN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    for epoch in range(epochs):
        train_loss = train_cnn_model(model, train_loader_cnn, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}")

    # evaluate CNN
    cnn_accuracy = evaluate_cnn_model(model, test_loader_cnn, device)
    print(f"CNN Test Accuracy: {cnn_accuracy:.4f}")

    # save to pkl
    torch.save(model.state_dict(), "cnn_model.pkl")
    print("CNN Model saved as 'cnn_model.pkl'")
