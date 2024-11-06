import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from object_detection_analysis.classifiers import BaseClassifier

class CNNFCClassifier(nn.Module, BaseClassifier):
    def __init__(self, n_classes: int, preprocess=None, preprocess_kwargs: dict = None):
        super().__init__()
        self._preprocess = preprocess
        self._prep_args = preprocess_kwargs if preprocess_kwargs is not None else {}
        self._device = "cpu"

        self.backbone = nn.Sequential(
            # GPT suggested
            # nn.Conv2d(3, 32, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            
            # nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            # nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2), 
            
            # Raisa Ater and Md Imran Hosen
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.20),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.35),
        )

        self.fc = nn.Sequential(
            # GPT suggested
            # nn.Linear(128*4*4, 128),
            # nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            # nn.Linear(64, n_classes),
            
            # Raisa Ater and Md Imran Hosen
            nn.Linear(128*4*4, 384),
            nn.ReLU(),
            nn.BatchNorm1d(384),
            nn.Linear(384, n_classes),
        )
    
    def to(self, device):
        self._device = device
        super().to(device)

    def forward(self, X):
        X = self.backbone(X)
        X = torch.flatten(X, 1)
        X = self.fc(X)
        return torch.nn.functional.softmax(X, dim=1)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, epochs=20, batch=4, plot=False, save_best=True):
        if not isinstance(X_train, np.ndarray):
            X_train = np.array(X_train)
        if not isinstance(y_train, np.ndarray):
            y_train = np.array(y_train)
        if self._preprocess is not None:
            X_train = self._preprocess(X_train, **self._prep_args)

        X = X_train.to(self._device)
        labels = torch.from_numpy(y_train).to(self._device)
        train_dataset = TensorDataset(X, labels)
        train_loader = DataLoader(train_dataset, batch_size=batch)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=1e-3)

        hist_loss = []
        hist_val_loss = []
        hist_val_acc = []
        best_acc = 0.0

        for epoch in range(epochs):
            self.train()
            hist_loss.append(0.0)
            for inputs, tgts in train_loader:
                optimizer.zero_grad()
                
                outputs = self(inputs)
                loss = criterion(outputs, tgts)
                loss.backward()
                optimizer.step()
                hist_loss[-1] += loss.item()
            hist_loss[-1] /= (len(train_loader.dataset) / batch)

            _, val_loss, val_acc = self._evaluate_model(X_val, y_val, criterion, batch=batch)
            hist_val_loss.append(val_loss)
            hist_val_acc.append(val_acc)
            if val_acc > best_acc:
                self.save_state("cnn_best.pt")
                best_acc = val_acc

            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {hist_loss[-1]:.4f}, Val Loss: {hist_val_loss[-1]:.4f}, Val Accuracy: {hist_val_acc[-1]:.2f}%")
        
        if plot:
            _, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,8))
            axs[0].set(xlabel="Epoch", ylabel="Training Loss", title="Training Loss")
            axs[1].set(xlabel="Epoch", ylabel="Valid Accuracy", title="Valid Accuracy")
            axs[2].set(xlabel="Epoch", ylabel="Valid Loss", title="Valid Loss")
            ep = np.arange(1, epochs+1)
            axs[0].plot(ep, hist_loss)
            axs[1].plot(ep, hist_val_acc)
            axs[2].plot(ep, hist_val_loss)
            plt.show()

        return hist_loss, hist_val_loss, hist_val_acc


    def predict(self, X):
        if self._preprocess is not None:
            X = self._preprocess(X, **self._prep_args)
        trans = X.to(self._device)
        with torch.no_grad():
            output = self(trans)
        _, predicted = torch.max(output, 1)
        return predicted.cpu().numpy(force=True)

    def _evaluate_model(self, X_val, y_val, criterion, batch=4): 
        if not isinstance(X_val, np.ndarray):
            X_val = np.array(X_val)
        if not isinstance(y_val, np.ndarray):
            y_val = np.array(y_val) 
        if self._preprocess is not None:
            X_val = self._preprocess(X_val, **self._prep_args)
        X = X_val.to(self._device)
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

    def evaluate(self, X_test, y_test, show_confusion_matrix=False, disp_labels=None):
        self.eval()
        return super().evaluate(X_test, y_test, show_confusion_matrix, disp_labels)

