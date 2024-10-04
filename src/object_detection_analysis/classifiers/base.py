from abc import ABC, abstractmethod
import dill


class BaseClassifier(ABC):
    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test, disp_labels=None):
        pass

    @staticmethod
    def from_state(state_file: str):
        with open(state_file, "rb") as f:
            obj = dill.load(f)
        return obj

    def save_state(self, file_name: str):
        with open(file_name, "wb") as f:
            dill.dump(self, f)


