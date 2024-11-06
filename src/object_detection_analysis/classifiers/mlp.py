import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import dill
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from object_detection_analysis.classifiers import BaseClassifier

class MLPClassifier(nn.Module, BaseClassifier):
    def __init__(self, n_features: int, n_classes: int, preprocess=None, preprocess_kwargs: dict = None, state_file=None):
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
        self._prep_args = preprocess_kwargs if preprocess_kwargs is not None else {}

        self._device = "cpu"

    def to(self, device):
        self._device = device
        super().to(device)

    @staticmethod
    def from_state(state_file: str):
        with open(state_file, "rb") as f:
            obj = dill.load(f)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        obj.to(device)
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

    def fit(self, X_train, y_train, X_val, y_val, epochs=20, plot=False):
        if self._preprocess is not None:
            X_train = self._preprocess(X_train, **self._prep_args)

        X = torch.tensor(deepcopy(X_train)).to(self._device)
        labels = torch.tensor(deepcopy(y_train)).to(self._device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=1e-3)

        hist_loss = []
        hist_val_loss = []
        hist_val_acc = []
        for epoch in range(epochs):
            self.train() # Set to training mode (from nn.Module)
            optimizer.zero_grad() # Zero the gradients

            # Forward pass and compute loss
            outputs = self(X)
            loss = criterion(outputs, labels)

            # Backpropagate errors to adjust parameters
            loss.backward() # Backpropagate
            optimizer.step() # Update weights

            hist_loss.append(loss.item())

            # Validation phase
            _, val_loss, val_acc = self._evaluate_model(X_val, y_val, criterion, batch=4)
            hist_val_loss.append(val_loss)
            hist_val_acc.append(val_acc)

            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {hist_loss[-1]:.4f}, Val Loss: {hist_val_loss[-1]:.4f}, Val Accuracy: {hist_val_acc[-1]:.2f}%")

        if plot:
            _, axs = plt.subplots(nrows=1, ncols=3, figsize=(14,8))
            axs[0].set(xlabel="Epoch", ylabel="Train Loss", title="Train Loss")
            axs[1].set(xlabel="Epoch", ylabel="Valid Accuracy", title="Val Accuracy")
            axs[2].set(xlabel="Epoch", ylabel="Valid Loss", title="Val Loss")
            ep = np.arange(1, epochs+1)
            axs[0].plot(ep, hist_loss)
            axs[1].plot(ep, hist_val_acc)
            axs[2].plot(ep, hist_val_loss)
            plt.show()

        return hist_loss, hist_val_loss, hist_val_acc

    
    def predict(self, X):
        if self._preprocess is not None:
            X = self._preprocess(X, **self._prep_args)
        trans = torch.tensor(deepcopy(X)).to(self._device)

        with torch.no_grad():
            output = self(trans)

        _, predicted = torch.max(output, 1)
        return predicted.cpu().numpy(force=True)


    def evaluate(self, X_test, y_test, show_confusion_matrix=False, disp_labels=None):
        self.eval()
        return super().evaluate(X_test, y_test, show_confusion_matrix, disp_labels)

    def _evaluate_model(self, X_val, y_val, criterion, batch=4):
        if not isinstance(X_val, np.ndarray):
            X_val = np.array(X_val)
        if not isinstance(y_val, np.ndarray):
            y_val = np.array(y_val)
        if self._preprocess is not None:
            X_val = self._preprocess(X_val, **self._prep_args)
        X = torch.from_numpy(X_val).to(self._device)
        y = torch.from_numpy(y_val).to(self._device)
        
        val_dataset = TensorDataset(X, y)
        val_loader = DataLoader(val_dataset, batch_size=batch)

        self.eval()
        correct = total = 0
        loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                output = self(inputs)
                _, predicted = torch.max(output, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss += criterion(output, labels).item()
        acc = (100.0*(float(correct)/float(total)))

        return correct, loss, acc

