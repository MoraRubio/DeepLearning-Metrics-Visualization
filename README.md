# Metrics visualizations for TensorFlow and Keras models

Implementation based on [Yellowbrick library](https://www.scikit-yb.org/en/latest/index.html) adapted to directly support trained Deep Learning models from TensorFlow and Keras.

Currently supports visualizations for Classification Report, Confusion Matrix, ROC-AUC and Class Prediction Error.

## How to use
```python
classes = ['Cat', 'Dog'] 
name = 'VGG19' 
path_to_save = './images' 
if not os.path.exists(path_to_save): 
    os.makedirs(path_to_save) 
vz = MetricsVisualizer(model, X_test, y_test, classes, name, path_to_save) 
vz.ClassificationReportViz() 
vz.ConfusionMatrixViz() 
vz.ROCAUCViz() 
vz.ClassPredictionErrorViz()
```
