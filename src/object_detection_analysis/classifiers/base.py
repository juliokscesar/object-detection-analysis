from abc import ABC, abstractmethod
import dill
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

class BaseClassifier(ABC):
    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    def evaluate(self, X_test, y_test, show_confusion_matrix=False, disp_labels=None) -> dict:
        predictions = self.predict(X_test)
        if show_confusion_matrix:
            cm = confusion_matrix(y_test, predictions)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)
            print(classification_report(y_test, predictions, target_names=disp_labels))
            disp.plot()
            plt.show()
        return classification_report(y_test, predictions, output_dict=True)

    @staticmethod
    def from_state(state_file: str):
        with open(state_file, "rb") as f:
            obj = dill.load(f)
        return obj

    def save_state(self, file_name: str):
        with open(file_name, "wb") as f:
            dill.dump(self, f)

