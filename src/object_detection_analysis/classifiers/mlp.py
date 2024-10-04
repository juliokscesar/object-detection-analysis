import torch
import torch.nn as nn
import torch.optim as optim
import dill
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from object_detection_analysis.classifiers import BaseClassifier

class MLPClassifier(nn.Module, BaseClassifier):
    def __init__(self, n_features: int, n_classes: int, preprocess=None, state_file=None):
        if state_file is not None:
            self.load_state(state_file)
            return
        
        super().__init__()

        self.fc1 = nn.Linear(n_features, n_features//2)
        self.bn1 = nn.BatchNorm1d(n_features//2)
        self.ac1 = nn.ReLU()
        self.fc2 = nn.Linear(n_features//2, n_features//4)
        self.bn2 = nn.BatchNorm1d(n_features//4)
        self.ac2 = nn.ReLU()
        self.fc3 = nn.Linear(n_features//4, n_classes)

        self._preprocess = preprocess
        device = "cuda" if torch.cuda.is_available() else "cpu"
        nn.Module.to(self, device)
        self._device = device

    @staticmethod
    def from_state(state_file: str):
        with open(state_file, "rb") as f:
            obj = dill.load(f)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        obj._device = device
        nn.Module.to(obj, device)
        obj.eval()
        return obj

    def forward(self, X):
        X = self.fc1(X)
        X = self.bn1(X)
        X = self.ac1(X)
        X = self.fc2(X)
        X = self.bn2(X)
        X = self.ac2(X)
        X = self.fc3(X)
        return X

    def fit(self, X_train, y_train, epochs=20):
        if self._preprocess is not None:
            X_train = self._preprocess(X_train)

        X = torch.tensor(deepcopy(X_train)).to(self._device)
        labels = torch.tensor(deepcopy(y_train)).to(self._device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=1e-3)

        correct = 0
        total = 0

        hist_loss = []
        hist_acc = []
        self.train() # Set to training mode (from nn.Module)
        for epoch in range(epochs):
            optimizer.zero_grad() # Zero the gradients

            # Forward pass and compute loss
            outputs = self(X)
            loss = criterion(outputs, labels)

            # Backpropagate errors to adjust parameters
            loss.backward() # Backpropagate
            optimizer.step() # Update weights

            _, predicted =  torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100.0 * correct / total

            hist_loss.append(loss.item())
            hist_acc.append(accuracy)

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

        _, axs = plt.subplots(nrows=1, ncols=2, figsize=(14,8))
        axs[0].set(xlabel="Epoch", ylabel="Loss", title="Loss")
        axs[1].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
        ep = np.arange(1, epochs+1)
        axs[0].plot(ep, hist_loss)
        axs[1].plot(ep, hist_acc)
        plt.show()

    
    def predict(self, X):
        if self._preprocess is not None:
            X = self._preprocess(X)
        trans = torch.tensor(deepcopy(X)).to(self._device)

        with torch.no_grad():
            output = self(trans)

        _, predicted = torch.max(output, 1)
        return predicted.cpu().numpy(force=True)


    def evaluate(self, X_test, y_test, disp_labels=None):
        self.eval()

        predictions = self.predict(X_test)
        cm = confusion_matrix(y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)
        print(classification_report(y_test, predictions, target_names=disp_labels))
        disp.plot()
        plt.show()

