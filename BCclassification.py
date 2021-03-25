"""
    Breast Cancer Classification
"""

# TODO: Import libraries
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
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

dataframe_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns=np.append(
    cancer['feature_names'], ['target']))
print(f"\nFirst five rows: \n{dataframe_cancer.head()}")

# TODO: Visualize data
# using pairplot
# sns.pairplot(dataframe_cancer, hue='target',vars=['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])
# plt.show()

# using countplot
# sns.countplot(dataframe_cancer['target'])
# plt.show()

# using scatterplot
# sns.scatterplot(x='mean area', y='mean smoothness', hue='target', data=dataframe_cancer)
# plt.show()

# using heatmap
# plt.figure(figsize=(20,10))
# sns.heatmap(dataframe_cancer.corr(), annot=True)
# plt.show()

# TODO: Train the Model
X = dataframe_cancer.drop(['target'], axis=1)
y = dataframe_cancer['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

svc_model = SVC()
svc_model.fit(X_train, y_train)

# TODO: Evaluate the model
y_predict = svc_model.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
print(f"\nConfusion Matrix: {cm}")

# sns.heatmap(cm)
# plt.show()

# TODO: Improve the model: Part 1
# data normalization (Feature Scaling)
min_train = X_train.min()
range_train = (X_train-min_train).max()
X_train_scaled = (X_train-min_train)/range_train
# sns.scatterplot(x=X_train_scaled['mean area'], y=X_train_scaled['mean smoothness'], hue=y_train)
# plt.show()

min_test = X_test.min()
range_test = (X_test-min_test).max()
X_test_scaled = (X_test-min_test)/range_train
svc_model.fit(X_train_scaled, y_train)

y_predict = svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_predict)
print(f"\nConfusion Matrix: {cm}")

# sns.heatmap(cm, annot=True)
# plt.show()

print(f"\nClassification report: {classification_report(y_test, y_predict)}")

# TODO: Improving the model: Part 2
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [
    1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=4)
grid.fit(X_train_scaled, y_train)
print(f"\nBest grid parameters: {grid.best_params_}")

grid_predictions = grid.predict(X_test_scaled)
cm = confusion_matrix(y_test, grid_predictions)
print(f"\nConfusion Matrix: {cm}")

sns.heatmap(cm, annot=True)
plt.show()

print(f"\nClassification report: {classification_report(y_test, y_predict)}")
