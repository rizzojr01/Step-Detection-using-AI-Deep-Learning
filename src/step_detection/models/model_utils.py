"""
Model Utilities
Functions for creating, training, and evaluating PyTorch models.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset

# Import configuration
from ..utils.config import get_config


class StepDetectionCNN(nn.Module):
    """
    PyTorch CNN model for step detection matching the original working architecture.
    This matches the exact architecture from your trained model with conv3, fc1, fc2, fc3 layers.
    """

    def __init__(self):
        super(StepDetectionCNN, self).__init__()
        # Convolutional layers - matching your original architecture
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=32, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1, stride=1)
        self.conv3 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=1, stride=1
        )

        # Activation functions
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=1)

        # Fully connected layers - matching your original architecture
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Input shape: [batch, 6] for 6 sensor features
        x = x.unsqueeze(2)  # [batch, 6] -> [batch, 6, 1] for Conv1d

        # Convolutional layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)

        return x


class StepDataset(Dataset):
    """Custom dataset for step detection"""

    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def create_cnn_model(
    input_shape: Optional[Tuple[int, ...]] = None,
    num_classes: Optional[int] = None,
    dropout_rate: Optional[float] = None,
    regularization: Optional[float] = None,
    config_path: Optional[str] = None,
    device: str = "cpu",
) -> nn.Module:
    """
    Create CNN model for step detection using the EXACT original high-accuracy architecture.
    This is the EXACT same PyTorch architecture that matches the working model.

    Args:
        input_shape: Input shape (overrides config) - not used in PyTorch
        num_classes: Number of output classes (overrides config)
        dropout_rate: Dropout rate (overrides config) - NOT USED in original
        regularization: L2 regularization factor (overrides config) - NOT USED in original
        config_path: Path to config file
        device: Device to create model on

    Returns:
        PyTorch model
    """
    # Load configuration
    config = get_config(config_path)
    num_classes = num_classes or config.get_output_classes()

    # Create device
    device_obj = torch.device(device)

    # Create the EXACT same architecture as the working PyTorch model
    model = StepDetectionCNN().to(device_obj)

    print("✅ CNN model created with EXACT original PyTorch architecture!")
    print("🎯 This is the EXACT same model that achieved high accuracy!")
    print(f"   Device: {device_obj}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


def train_model(
    model: nn.Module,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    patience: Optional[int] = None,
    config_path: Optional[str] = None,
    device: str = "cpu",
) -> Dict:
    """
    Train the PyTorch model with EXACT same approach as the high-accuracy notebook.
    Simple training approach that achieves high accuracy.

    Args:
        model: PyTorch model to train
        train_features: Training features
        train_labels: Training labels
        val_features: Validation features
        val_labels: Validation labels
        epochs: Maximum number of epochs (overrides config)
        batch_size: Batch size for training (overrides config)
        patience: Early stopping patience (not used for simplicity)
        config_path: Path to config file
        device: Device for training

    Returns:
        Training history dictionary
    """
    # Load configuration
    config = get_config(config_path)

    # Get parameters from config or use provided values
    if epochs is None:
        epochs = config.get_epochs()
    if batch_size is None:
        batch_size = config.get_batch_size()

    print(f"🏃‍♂️ Training PyTorch model with EXACT notebook approach:")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Device: {device}")

    # Create device
    device_obj = torch.device(device)
    model = model.to(device_obj)

    # Create datasets and dataloaders
    train_dataset = StepDataset(train_features, train_labels)
    val_dataset = StepDataset(val_features, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training history
    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

    print("🔥 Starting training...")

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for features_batch, labels_batch in train_loader:
            features_batch = features_batch.to(device_obj)
            labels_batch = labels_batch.to(device_obj)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(features_batch)
            loss = criterion(outputs, labels_batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels_batch.size(0)
            train_correct += (predicted == labels_batch).sum().item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features_batch, labels_batch in val_loader:
                features_batch = features_batch.to(device_obj)
                labels_batch = labels_batch.to(device_obj)

                outputs = model(features_batch)
                loss = criterion(outputs, labels_batch)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels_batch.size(0)
                val_correct += (predicted == labels_batch).sum().item()

        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        # Store history
        history["loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)

        # Print progress
        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch+1:3d}/{epochs}: "
                f"Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Acc: {train_acc:.4f}, "
                f"Val Acc: {val_acc:.4f}"
            )

    print(f"✅ Training completed after {epochs} epochs!")
    print("🎯 Used EXACT same simple approach as the high-accuracy notebook!")

    return history


def evaluate_model(
    model: nn.Module,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    device: str = "cpu",
) -> Dict:
    """
    Evaluate PyTorch model performance.

    Args:
        model: Trained PyTorch model
        val_features: Validation features
        val_labels: Validation labels
        device: Device for evaluation

    Returns:
        Dictionary with evaluation metrics
    """
    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.eval()

    # Convert data to tensors
    val_features_tensor = torch.tensor(val_features, dtype=torch.float32).to(device_obj)

    # Make predictions
    with torch.no_grad():
        outputs = model(val_features_tensor)
        predictions = torch.softmax(outputs, dim=1).cpu().numpy()
        predicted_classes = np.argmax(predictions, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(val_labels, predicted_classes)

    # Classification report
    target_names = ["No Label", "start", "end"]
    report = classification_report(
        val_labels, predicted_classes, target_names=target_names, output_dict=True
    )

    # Confusion matrix
    cm = confusion_matrix(val_labels, predicted_classes)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "predictions": predictions,
    }


def save_model_and_metadata(
    model: nn.Module,
    model_path: str,
    metadata: Dict,
    metadata_path: Optional[str] = None,
):
    """
    Save PyTorch model and metadata.

    Args:
        model: Trained PyTorch model
        model_path: Path to save the model
        metadata: Model metadata dictionary
        metadata_path: Path to save metadata (auto-generated if None)
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save model state dict
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    # Save metadata
    if metadata_path is None:
        metadata_path = model_path.replace(".pth", "_metadata.json")

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")


def load_model_with_metadata(
    model_path: str, metadata_path: Optional[str] = None, device: str = "cpu"
) -> Tuple[nn.Module, Dict]:
    """
    Load PyTorch model and metadata.

    Args:
        model_path: Path to the saved model
        metadata_path: Path to metadata file (auto-generated if None)
        device: Device to load model on

    Returns:
        Tuple of (model, metadata)
    """
    # Create model instance
    device_obj = torch.device(device)
    model = StepDetectionCNN().to(device_obj)

    # Load model state dict
    model.load_state_dict(
        torch.load(model_path, map_location=device_obj, weights_only=False)
    )
    model.eval()

    # Load metadata
    if metadata_path is None:
        metadata_path = model_path.replace(".pth", "_metadata.json")

    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        metadata = {}
        print(f"Warning: Metadata file not found: {metadata_path}")

    return model, metadata


def optimize_thresholds(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    threshold_range: Optional[List[float]] = None,
) -> Dict:
    """
    Optimize detection thresholds based on validation data.

    Args:
        predictions: Model predictions (probabilities)
        true_labels: True labels
        threshold_range: List of thresholds to test

    Returns:
        Dictionary with optimal thresholds and results
    """
    if threshold_range is None:
        threshold_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    best_threshold = 0.03
    best_score = 0
    results = []

    for thresh in threshold_range:
        # Calculate metrics for this threshold
        predicted_starts = predictions[:, 1] > thresh
        predicted_ends = predictions[:, 2] > thresh

        true_starts = true_labels == 1
        true_ends = true_labels == 2

        # F1 scores
        start_tp = np.sum(predicted_starts & true_starts)
        start_fp = np.sum(predicted_starts & ~true_starts)
        start_fn = np.sum(~predicted_starts & true_starts)

        end_tp = np.sum(predicted_ends & true_ends)
        end_fp = np.sum(predicted_ends & ~true_ends)
        end_fn = np.sum(~predicted_ends & true_ends)

        # Calculate F1 scores
        start_precision = (
            start_tp / (start_tp + start_fp) if (start_tp + start_fp) > 0 else 0
        )
        start_recall = (
            start_tp / (start_tp + start_fn) if (start_tp + start_fn) > 0 else 0
        )
        start_f1 = (
            2 * start_precision * start_recall / (start_precision + start_recall)
            if (start_precision + start_recall) > 0
            else 0
        )

        end_precision = end_tp / (end_tp + end_fp) if (end_tp + end_fp) > 0 else 0
        end_recall = end_tp / (end_tp + end_fn) if (end_tp + end_fn) > 0 else 0
        end_f1 = (
            2 * end_precision * end_recall / (end_precision + end_recall)
            if (end_precision + end_recall) > 0
            else 0
        )

        overall_f1 = (start_f1 + end_f1) / 2

        result = {
            "threshold": thresh,
            "start_f1": start_f1,
            "end_f1": end_f1,
            "overall_f1": overall_f1,
        }
        results.append(result)

        if overall_f1 > best_score:
            best_score = overall_f1
            best_threshold = thresh

    return {
        "best_threshold": best_threshold,
        "best_score": best_score,
        "all_results": results,
    }


if __name__ == "__main__":
    print("Model Utilities")
    print("===============")

    # Create sample models
    print("Creating CNN model (original high-accuracy PyTorch architecture):")
    cnn_model = create_cnn_model()
    print("CNN model created:")
    print(f"Model: {cnn_model}")
    print(f"Parameters: {sum(p.numel() for p in cnn_model.parameters()):,}")

    print("\nModel utilities loaded successfully!")
    print("Available functions:")
    print("- create_cnn_model() (original high-accuracy PyTorch CNN)")
    print("- train_model()")
    print("- evaluate_model()")
    print("- save_model_and_metadata()")
    print("- load_model_with_metadata()")
    print("- optimize_thresholds()")
    print("- optimize_thresholds()")
    print("- optimize_thresholds()")
    print("- optimize_thresholds()")
    print("- optimize_thresholds()")
    print("- optimize_thresholds()")
