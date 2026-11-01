# Hand Gesture Classification
# This module provides functions to classify hand gestures using different machine learning models.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, TensorDataset


# KNN-based Gesture Classification

# Train a KNN classifier on the provided features and labels.
def train_knn(features, labels, k=3):
    """
    Trains a KNN classifier on the given features and labels.

    Parameters:
        features (list): List of feature vectors (shape: N x D).
        labels (list): Corresponding labels for each sample.
        k (int): Number of neighbors to consider.

    Returns:
        model (KNeighborsClassifier): Trained KNN model.
    """
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(features, labels)
    return model

# Predict gesture classes using the trained KNN model.
def predict_knn(model, features):
    """
    Predicts gesture classes using a trained KNN model.

    Parameters:
        model (KNeighborsClassifier): Trained KNN model.
        features (list): List of feature vectors to classify.

    Returns:
        predictions (list): Predicted class labels.
    """
    return model.predict(features)

# SVM-based Gesture Classification

# Train an SVM classifier on the provided features and labels.
def train_svm(features, labels, kernel='rbf', C=1.0, gamma='scale'):
    """
    Trains an SVM classifier on the given features and labels.

    Parameters:
        features (list): List of feature vectors (shape: N x D).
        labels (list): Corresponding labels for each sample.
        kernel (str): Kernel type ('linear', 'poly', 'rbf', 'sigmoid').
        C (float): Regularization parameter.
        gamma (str or float): Kernel coefficient.

    Returns:
        model (SVC): Trained SVM model.
    """
    model = SVC(kernel=kernel, C=C, gamma=gamma)
    model.fit(features, labels)
    return model

# Predict gesture classes using the trained SVM model.
def predict_svm(model, features):
    """
    Predicts gesture classes using a trained SVM model.

    Parameters:
        model (SVC): Trained SVM model.
        features (list): List of feature vectors to classify.

    Returns:
        predictions (list): Predicted class labels.
    """
    return model.predict(features)

# CNN-based Gesture Classification using ResNet18
class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=20):
        super(ResNet18Classifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Helper function to create data loaders
def create_dataloader(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])
    dataset = ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return loader, dataset.class_to_idx

# Train a ResNet18 model from data directory
def train_resnet18(data_dir, num_classes=20, epochs=5, lr=0.001, batch_size=32, device='cpu'):
    train_loader, _ = create_dataloader(data_dir, batch_size)
    model = ResNet18Classifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    return model

# Predict gesture classes using the trained ResNet18 model
def predict_resnet18(model, data_dir, batch_size=32, device='cpu'):
    loader, _ = create_dataloader(data_dir, batch_size)
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.numpy())

    return predictions, true_labels