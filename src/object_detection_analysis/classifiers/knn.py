from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from object_detection_analysis.classifiers import BaseClassifier

class KNNClassifier(BaseClassifier):
    def __init__(self, n_neighbors: int, weights: str = "distance", enable_nca = False, state_file: str = None, preprocess=None, preprocess_kwargs=None):
        if state_file is not None:
            self.load_state(state_file)
            return
        if enable_nca:
            self._clf = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("nca", NeighborhoodComponentsAnalysis(n_components=256, random_state=42)),
                    ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)),
                ]
            )
        else:
            self._clf = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)),
                ]
            )
        self._n_neighbors = n_neighbors
        self._preprocess = preprocess
        self._prep_args = preprocess_kwargs if preprocess_kwargs is not None else {} 

    def fit(self, X_train, y_train):
        if self._preprocess is not None:
            X_train = self._preprocess(X_train, **self._prep_args)
        self._clf.fit(X_train, y_train)


    def predict(self, X):
        if self._preprocess is not None:
            X = self._preprocess(X, **self._prep_args)
        return self._clf.predict(X)


