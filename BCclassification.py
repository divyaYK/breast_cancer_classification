"""
    Breast Cancer Classification
"""

# TODO: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

# TODO: Import data
cancer = load_breast_cancer()
# printing the datset and its info
print(f"dataset: {cancer}")
print(f"\nKeys it contains: {cancer.keys()}")
print(f"\nDescription of the dataset: {cancer['DESCR']}")
print(f"\nTarget of the dataset: {cancer['target']}")
print(f"\nTarget names of the dataset: {cancer['target_names']}")
print(f"\nFeature names of the dataset: {cancer['feature_names']}")

dataframe_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns=np.append(cancer['feature_names'], ['target']))
print(f"\nFirst five rows: \n{dataframe_cancer.head()}")

# TODO: Visualize data
# using pairplot
sns.pairplot(dataframe_cancer, hue='target',vars=['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])
# plt.show()

# using countplot
sns.countplot(dataframe_cancer['target'])
# plt.show()

# using scatterplot
sns.scatterplot(x='mean area', y='mean smoothness', hue='target', data=dataframe_cancer)
# plt.show()

# using heatmap
plt.figure(figsize=(20,10))
sns.heatmap(dataframe_cancer.corr(), annot=True)
# plt.show()

# TODO: Train the Model
X = dataframe_cancer.drop(['target'], axis=1)
y = dataframe_cancer['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
svc_model = SVC()
svc_model.fit(X_train, y_train)

# TODO: Evaluate the model