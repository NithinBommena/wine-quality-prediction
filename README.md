
# Wine Quality Prediction

This project is focused on predicting the quality of wine based on various chemical properties. The dataset used consists of 6,500 entries, and the goal is to classify wines as "Good Quality" or "Bad Quality" using various machine learning algorithms.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Libraries Used](#libraries-used)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling](#modeling)
- [Prediction System](#prediction-system)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction
Wine quality is an important factor that can be influenced by several chemical properties such as fixed acidity, volatile acidity, citric acid,residual sugar, pH, sulphates etc..

This project utilizes machine learning to predict whether a wine is of good or bad quality based on these properties.

## Dataset
The dataset contains 6,500 entries with various features such as acidity, sugar content, alcohol level, and more. The target variable is the quality of the wine, labeled as either "Good" or "Bad."

## Libraries Used
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
```

## Data Preprocessing
To handle missing data, the following approach was used:

```python
df.isnull().sum()

for col, value in df.items():
    if col != 'type':
        df[col] = df[col].fillna(df[col].mean())

df.isnull().sum()
```

This method fills missing values with the mean of their respective columns.

## Exploratory Data Analysis
A correlation matrix was defined to explore the relationships between features:

```python
corr = df.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr, annot=True, cmap='coolwarm')
```

## Modeling
The following machine learning models were applied to the dataset:

- **Logistic Regression**: Accuracy = 0.816
- **Support Vector Machine (SVM)**: Accuracy = 0.829
- **K-Nearest Neighbors (KNN)**: Accuracy = 0.832
- **Decision Tree**: Accuracy = 0.838
- **Random Forest**: Accuracy = 0.838

## Prediction System
A simple predictive system was built to classify the quality of wine based on user input:

```python
input_data=input().split()
if input_data[0]=='white':
    input_data[0]='1'
else:
    input_data[0]='0'

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model5.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==1):
  print('Good Quality Wine')
else:
  print('Bad Quality Wine')
```

## Results
The Random Forest model yielded the highest accuracy at **0.838**.


## Conclusion
This project demonstrates the application of machine learning techniques in predicting wine quality with a reasonably high accuracy. The Random Forest model performed the best, making it a strong candidate for future enhancements.
