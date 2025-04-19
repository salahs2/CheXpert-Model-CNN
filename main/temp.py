import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

# ==============================
# 1. Data Loading and Preprocessing
# ==============================

# Load Data
# Read the CSV for data
df = pd.read_csv('./main/data/train.csv')

# Selecting appropriate features
selected_features = [
    'Path', 'Sex', 'Age', 'Lung Opacity'
]
df = df[selected_features].dropna()

# ==============================
# 2. Count 'Lung Opacity' Values
# ==============================

# Print the count of 'Lung Opacity' being 0 and 1
lung_opacity_counts = df['Lung Opacity'].value_counts()
print("Counts of 'Lung Opacity':")
print(lung_opacity_counts)

