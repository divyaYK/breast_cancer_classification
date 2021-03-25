# Breast Cancer Classification

### PROBLEM STATEMENT

- Predicting if the cancer diagnosis is benign or malignant based on several observations/features
- 30 features are used, examples:<br>
<nbsp> - radius (mean of distances from center to points on the perimeter)<br>
<nbsp> - texture (standard deviation of gray scale values)<br>
<nbsp> - perimeter<br>
<nbsp> - area<br>
<nbsp> - smoothness (local variation in radius length)<br>
<nbsp> - compactness (perimeter^2/area -1.0)<br>
<nbsp> - concavity (severity of concave portions of the contour)<br>
<nbsp> - concave points (Number of concave portions on the contour)<br>
<nbsp> - symmetry<br>
<nbsp> - fractal dimension ("coastline approximation"-1)
- Datasets are linearly separable using all 30 input features
- Number of Instances: 569
- Class Distribution: 212 Malignant, 357 Benign
- Target Class: <br>
<nbsp> - Malignant<br>
<nbsp> - Benign<br>

### Dataset
Link: https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(Diagnostic) <br><br>
### Algorithm
Support Vector Machine<br>
Libraries used: numpy, pandas, matplotlib, seaborn, sklearn

### Data Visualization
- Using pair plot:
![Pair Plot graph](./pairplot.png)

- Using count plot:
![Count Plot graph](./countplot.png)

- Using scatter plot:
![Scatter Plot graph](./scatterplot.png)

- Using heatmap:
![Heatmap](./heatmap.png)

### Trained Model
Heatmap of confusion matrix:
![Confusion Matrix](./trainedheatmap.png)