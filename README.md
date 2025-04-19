# 4AL3 Final Project Data Installation Guide

# Data Overview

Data is a subset of the CheXpert Dataset from the Stanford ML Group

[ChexPert Page](https://stanfordmlgroup.github.io/competitions/chexpert/)

[Subset Dataset Page](https://www.kaggle.com/datasets/ashery/chexpert/data?select=train.csv)

# Getting Data

### Warning, This is a large dataset (12gb)

Step 1: Create / Locate Data folder

Step 2: Install the dataset 

## Install dataset

### Method 1 (CLI)
`kaggle datasets download ashery/chexpert` note: kaggle package is needed for this step found below

### Method 2 (directly)

download dataset from [kaggle](https://www.kaggle.com/datasets/ashery/chexpert/data?select=train.csv)

create a data directory inside `\main` 

Step 3: extract the zip file to the data directory     
`note: Use Winrar or 7zip to extract dataset (takes way too long otherwise)`

## Packages Required

Latest Versions as of December, 2024

please install all required dependencies

`pip install kaggle, pandas, numpy, scikit-learn, torch, pickle`

kaggle

pandas v2.2.3

numpy v2.1.3

scikit-learn v1.5.2

torch v2.5.1

matplotlib v3.9.3

pickle