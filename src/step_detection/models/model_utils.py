"""
Model Utilities
Functions for creating, training, and evaluating TensorFlow models.
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers


def create_cnn_model(
    input_shape: Tuple[int, ...] = (6,), num_classes: int = 3
) -> keras.Model:
    """
    Create CNN model for step detection.

    Args:
        input_shape: Input shape (number of sensor features)
        num_classes: Number of output classes

    Returns:
        Compiled Keras model
    """
    model = keras.Sequential(
        [
            layers.Reshape((1, input_shape[0]), input_shape=input_shape),
            layers.Conv1D(filters=32, kernel_size=1, strides=1, activation="relu"),
            layers.MaxPooling1D(pool_size=1),
            layers.Conv1D(filters=64, kernel_size=1, strides=1, activation="relu"),
            layers.Flatten(),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def train_model(
    model: keras.Model,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 10,
) -> keras.callbacks.History:
    """
    Train the model with early stopping and learning rate reduction.

    Args:
        model: Keras model to train
        train_features: Training features
        train_labels: Training labels
        val_features: Validation features
        val_labels: Validation labels
        epochs: Maximum number of epochs
        batch_size: Batch size for training
        patience: Early stopping patience

    Returns:
        Training history
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_features,
        train_labels,
        validation_data=(val_features, val_labels),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    return history


def evaluate_model(
    model: keras.Model, val_features: np.ndarray, val_labels: np.ndarray
) -> Dict:
    """
    Evaluate model performance.

    Args:
        model: Trained Keras model
        val_features: Validation features
        val_labels: Validation labels

    Returns:
        Dictionary with evaluation metrics
    """
    # Make predictions
    predictions = model.predict(val_features, verbose=0)
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
    model: keras.Model, model_path: str, metadata: Dict, metadata_path: str = None
):
    """
    Save model and metadata.

    Args:
        model: Trained Keras model
        model_path: Path to save the model
        metadata: Model metadata dictionary
        metadata_path: Path to save metadata (auto-generated if None)
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save model (using native Keras format to avoid warnings)
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    # Save metadata
    if metadata_path is None:
        metadata_path = model_path.replace(".keras", "_metadata.json")

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")


def load_model_with_metadata(
    model_path: str, metadata_path: str = None
) -> Tuple[keras.Model, Dict]:
    """
    Load model and metadata.

    Args:
        model_path: Path to the saved model
        metadata_path: Path to metadata file (auto-generated if None)

    Returns:
        Tuple of (model, metadata)
    """
    # Load model
    model = keras.models.load_model(model_path)

    # Load metadata
    if metadata_path is None:
        metadata_path = model_path.replace(".keras", "_metadata.json")

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
    threshold_range: List[float] = None,
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

    # Create a sample model
    model = create_cnn_model()
    print("Sample model created:")
    model.summary()

    print("\nModel utilities loaded successfully!")
    print("Available functions:")
    print("- create_cnn_model()")
    print("- train_model()")
    print("- evaluate_model()")
    print("- save_model_and_metadata()")
    print("- load_model_with_metadata()")
    print("- optimize_thresholds()")
