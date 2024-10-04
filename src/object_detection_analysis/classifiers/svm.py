from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from object_detection_analysis.classifiers import BaseClassifier

class SVMClassifier(BaseClassifier):
    def __init__(self, kernel="rbf", state_file: str = None, preprocess = None):
        if state_file is not None:
            self.load_state(state_file)
            return
        self._clf = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("svc", svm.SVC(gamma="auto", kernel=kernel))
            ]
        )
        self._preprocess = preprocess


    def fit(self, X_train, y_train):
        if self._preprocess is not None:
            X_train = self._preprocess(X_train)
        self._clf.fit(X_train, y_train)


    def predict(self, X):
        if self._preprocess is not None:
            X = self._preprocess(X)
        return self._clf.predict(X)


    def evaluate(self, X_test, y_test, disp_labels=None):
        predictions = self.predict(X_test)
        cm = confusion_matrix(y_test, predictions, labels=self._clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)
        print(classification_report(y_test, predictions, labels=self._clf.classes_, target_names=disp_labels))
        disp.plot()
        plt.show()

