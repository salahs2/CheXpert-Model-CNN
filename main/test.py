import numpy as np
import pandas as pd
from PIL import Image
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torchvision.transforms import transforms

# Function to calculate confusion matrix metrics
def calculate_metrics(y_true, y_pred):
    assert len(y_true) == len(y_pred), "Mismatch between true and predicted labels."
    print(f"Number of samples: {len(y_true)}")  # Debug print

    # Ensure binary labels (convert -1 to 0 if necessary)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = np.where(y_true == -1, 0, y_true)  # Convert -1 to 0
    y_pred = np.where(y_pred == -1, 0, y_pred)  # Convert -1 to 0

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    # Handle case where confusion matrix is not 2x2
    if cm.shape != (2, 2):
        raise ValueError("Confusion matrix shape is not 2x2. Check input labels.")

    tn, fp, fn, tp = cm.ravel()
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Negatives (FN): {fn}")

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity/Recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    print(f"True Positive Rate (Sensitivity/Recall): {tpr:.4f}")
    print(f"False Positive Rate: {fpr:.4f}")

    return tp, fp, tn, fn, tpr, fpr

# Function to analyze bias
def analyze_bias(df, y_true, y_pred, feature):
    bias_results = df.groupby(feature).apply(
        lambda group: calculate_metrics(
            y_true[group.index], y_pred[group.index]
        )
    )
    print(f"Bias analysis for {feature}:")
    print(bias_results)

# CNN Model Definition
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 32 * 32 + 2, 128)  # 32x32 image after pooling, +2 for metadata
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

# Model Tester Class
class ModelTester:
    def __init__(self, model=None, model_type="svm"):
        self.model = model
        self.model_type = model_type

    def load_model(self, filepath):
        """Load a model from a file."""
        if self.model_type == "svm":
            with open(filepath, 'rb') as file:
                self.model = pickle.load(file)
            print(f"SVM model loaded from {filepath}")
        elif self.model_type == "cnn":
            self.model = CNN()
            self.model.load_state_dict(torch.load(filepath, weights_only=True))
            self.model.eval()
            print(f"CNN model loaded from {filepath}")
        elif self.model_type == "nb":
            with open(filepath, 'rb') as file:
                self.model = pickle.load(file)
            print(f"NB model loaded from {filepath}")

    def test_model(self, X, y, df=None):
        """Test the model with the given data."""
        if self.model is None:
            raise ValueError("No model loaded or set. Please load a model first.")
        
        if self.model_type == "svm":
            predictions = self.model.predict(X)
            accuracy = accuracy_score(y, predictions)
            print(f"SVM Model Accuracy: {accuracy:.4f}")
            tp, fp, tn, fn, tpr, fpr = calculate_metrics(y, predictions)
            if df is not None:
                analyze_bias(df, y, predictions, "gender_encoded")
                analyze_bias(df, y, predictions, "Age")
            return accuracy, tp, fp, tn, fn, tpr, fpr
        elif self.model_type == "cnn":
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(device)

            # Prepare CNN inputs
            images, metadata, labels = X
            images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)
            
            with torch.no_grad():
                outputs = self.model(images, metadata)
                predictions = (outputs > 0.5).float().cpu().numpy()
                y_true = labels.cpu().numpy()
                accuracy = accuracy_score(y_true, predictions)
                print(f"CNN Model Accuracy: {accuracy:.4f}")
                tp, fp, tn, fn, tpr, fpr = calculate_metrics(y_true, predictions)
                if df is not None:
                    analyze_bias(df, y_true, predictions, "gender_encoded")
                    analyze_bias(df, y_true, predictions, "Age")
                return accuracy, tp, fp, tn, fn, tpr, fpr
        elif self.model_type == "nb":
            predictions = self.model.predict(X)
            accuracy = accuracy_score(y, predictions)
            print(f"NB Model Accuracy: {accuracy:.4f}")
            tp, fp, tn, fn, tpr, fpr = calculate_metrics(y, predictions)
            if df is not None:
                analyze_bias(df, y, predictions, "gender_encoded")
                analyze_bias(df, y, predictions, "Age")
            return accuracy, tp, fp, tn, fn, tpr, fpr

if __name__ == "__main__":
    # Read the csv for data
    df = pd.read_csv('./main/data/train.csv')

    # Selecting appropriate features
    df = df[['Path', 'Sex', 'Age', 'Lung Opacity']].dropna()

    def get_last_dirs_and_file(path, num_dirs=3):
        parts = path.split('/')  # Split the path into components
        temp = '/'.join(parts[-(num_dirs + 1):])  # Join the last `num_dirs` + file name
        return './main/data/' + temp

    df['Path'] = df['Path'].apply(get_last_dirs_and_file)
    df['Lung Opacity'] = df['Lung Opacity'].astype(int)

    # Subset data for balance 
    df_0 = df[df['Lung Opacity'] == 0].head(200) #30% in full dataset
    df_1 = df[df['Lung Opacity'] == 1].head(200) #70% in full dataset
    df = pd.concat([df_0, df_1]).reset_index(drop=True)

    # Encode labels
    gender_encoder = LabelEncoder()
    df['gender_encoded'] = gender_encoder.fit_transform(df['Sex'])

    # Train-test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    # Preprocessing for SVM
    def image_to_pixels(image_path, target_size=(128, 128)):
        try:
            with Image.open(image_path) as img:
                img = img.convert('L')  # Convert to grayscale
                img = img.resize(target_size)
                return np.array(img).flatten() / 255.0
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    test_df['Image'] = test_df['Path'].apply(lambda x: image_to_pixels(x))
    test_df = test_df[test_df['Image'].notnull()]
    image_features = np.vstack(test_df['Image'].values)
    other_features = test_df[['Age', 'gender_encoded']].values
    features = np.hstack((image_features, other_features))
    labels = test_df['Lung Opacity'].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Test NB
    tester_nb = ModelTester(model_type="nb")
    tester_nb.load_model("nb_model.pkl")
    tester_nb.test_model(features_scaled, labels)

    # Test SVM
    tester_svm = ModelTester(model_type="svm")
    tester_svm.load_model("svm_model.pkl")
    tester_svm.test_model(features_scaled, labels)



    # Preprocessing for CNN
    class PneumoniaDataset(torch.utils.data.Dataset):
        def __init__(self, dataframe, target_size=(128, 128)):
            self.dataframe = dataframe
            self.target_size = target_size
            self.scaler = StandardScaler()
            self.dataframe['Age'] = self.scaler.fit_transform(self.dataframe[['Age']])
            self.transform = transforms.Compose([ # test transform is different than train
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            row = self.dataframe.iloc[idx]
            image_path = row['Path']
            age = row['Age']
            gender_encoded = row['gender_encoded']
            label = row['Lung Opacity']

            with Image.open(image_path) as img:
                img = self.transform(img)

            metadata = torch.tensor([age, gender_encoded], dtype=torch.float32)
            return torch.tensor(img, dtype=torch.float32), metadata, torch.tensor(label, dtype=torch.float32)

    test_dataset = PneumoniaDataset(test_df)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Test CNN
    tester_cnn = ModelTester(model_type="cnn")
    tester_cnn.load_model("cnn_model.pkl")

    # Prepare CNN test inputs
    all_images, all_metadata, all_labels = [], [], []
    for images, metadata, labels in test_loader:
        all_images.append(images)
        all_metadata.append(metadata)
        all_labels.append(labels)
    X_cnn_test = (torch.cat(all_images), torch.cat(all_metadata), torch.cat(all_labels))

    tester_cnn.test_model(X_cnn_test, all_labels)
