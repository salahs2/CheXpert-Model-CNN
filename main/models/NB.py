import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Naive Gaussian Classifier
class NB:
    def __init__(self):
        self.model = GaussianNB()
        self.scaler = None

    def pre_processing(self, df):
        # encode f/m to 0/1
        gender_encoder = LabelEncoder()
        df['gender_encoded'] = gender_encoder.fit_transform(df['Sex'])

        # normalize image
        df['image_normalized'] = df['Image'].apply(lambda img: img / 255.0)

        # flatten
        df['image_flattened'] = df['image_normalized'].apply(lambda img: img.flatten())

        # get image feature
        image_features = np.stack(df['image_flattened'].values)
        
        # other features
        other_features = df[['Age', 'gender_encoded']].values
        self.scaler = StandardScaler()
        other_features_scaled = self.scaler.fit_transform(other_features)

        # combine image and non-image features
        combined_features = np.hstack((image_features, other_features_scaled))
        labels = df['Pneumonia'].values

        return combined_features, labels

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        return accuracy, report

    def run(self, df):
        features, labels = self.pre_processing(df)

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        self.train(X_train, y_train)
        accuracy, report = self.evaluate(X_test, y_test)

        print(f"Test Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)

        return self.model