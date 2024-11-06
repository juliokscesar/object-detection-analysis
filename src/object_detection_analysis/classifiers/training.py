import numpy as np

from object_detection_analysis.classifiers import BaseClassifier

def train_classifier(clf: BaseClassifier, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, **clf_kwargs):
    pass
