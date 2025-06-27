# Model Training Guide

## Overview

This guide covers how to train, evaluate, and optimize the step detection CNN model.

## Model Architecture

### CNN Design

The step detection model uses a 1D Convolutional Neural Network optimized for time-series sensor data:

```python
def create_cnn_model(input_shape=(6,), num_classes=3):
    model = keras.Sequential([
        layers.Reshape((1, input_shape[0]), input_shape=input_shape),
        layers.Conv1D(filters=32, kernel_size=1, strides=1, activation="relu"),
        layers.MaxPooling1D(pool_size=1),
        layers.Conv1D(filters=64, kernel_size=1, strides=1, activation="relu"),
        layers.Flatten(),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
```

### Architecture Details

| Layer     | Type         | Parameters           | Output Shape  |
| --------- | ------------ | -------------------- | ------------- |
| Input     | Input        | -                    | (None, 6)     |
| Reshape   | Reshape      | -                    | (None, 1, 6)  |
| Conv1D #1 | Conv1D       | 32 filters, kernel=1 | (None, 1, 32) |
| MaxPool1D | MaxPooling1D | pool_size=1          | (None, 1, 32) |
| Conv1D #2 | Conv1D       | 64 filters, kernel=1 | (None, 1, 64) |
| Flatten   | Flatten      | -                    | (None, 64)    |
| Dense     | Dense        | 3 units, softmax     | (None, 3)     |

**Total Parameters**: ~2,000-3,000 trainable parameters

## Data Preparation

### Data Format

Training data should be in CSV format with the following columns:

```csv
accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,Label
1.234,-0.567,9.801,0.123,0.234,-0.098,No Label
2.456,1.789,8.234,-0.345,0.789,0.456,start
0.345,-2.123,11.123,0.567,-0.234,-0.345,end
```

### Label Classes

| Class | Label    | Description                 | Expected Count |
| ----- | -------- | --------------------------- | -------------- |
| 0     | No Label | Normal movement, not a step | ~95% of data   |
| 1     | start    | Beginning of a step motion  | ~2.5% of data  |
| 2     | end      | End of a step motion        | ~2.5% of data  |

### Data Preprocessing

The `prepare_data_for_training` function handles:

1. **Feature Extraction**: First 6 columns as sensor features
2. **Label Encoding**: String labels → integers (0, 1, 2)
3. **Train/Validation Split**: 80/20 split with stratification
4. **Data Type Conversion**: float32 for features, int for labels

```python
def prepare_data_for_training(df, test_size=0.2, random_state=42):
    # Extract features and labels
    features = df.iloc[:, :6].values.astype(np.float32)
    labels = df.iloc[:, 6].values

    # Convert labels to numeric
    label_mapping = {"No Label": 0, "start": 1, "end": 2}
    numeric_labels = np.array([label_mapping[label] for label in labels])

    # Split data with stratification
    return train_test_split(
        features, numeric_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=numeric_labels
    )
```

## Training Process

### Step 1: Data Loading

```bash
python main.py
# Choose option 1: Train new model
```

Or programmatically:

```python
from src.step_detection import load_step_data, prepare_data_for_training

# Load data
df = load_step_data("data/raw")
train_X, val_X, train_y, val_y = prepare_data_for_training(df)

print(f"Training samples: {len(train_X)}")
print(f"Validation samples: {len(val_X)}")
print(f"Label distribution: {np.bincount(train_y)}")
```

### Step 2: Model Creation

```python
from src.step_detection import create_cnn_model

model = create_cnn_model(input_shape=(6,), num_classes=3)
model.summary()
```

### Step 3: Training Configuration

```python
from src.step_detection import train_model

# Training with callbacks
history = train_model(
    model=model,
    train_features=train_X,
    train_labels=train_y,
    val_features=val_X,
    val_labels=val_y,
    epochs=50,
    batch_size=32,
    patience=10
)
```

### Training Callbacks

The training includes several callbacks:

1. **EarlyStopping**: Stops training when validation loss stops improving

   - Monitor: `val_loss`
   - Patience: 10 epochs
   - Restore best weights: True

2. **ReduceLROnPlateau**: Reduces learning rate when loss plateaus
   - Monitor: `val_loss`
   - Factor: 0.5
   - Patience: 5 epochs
   - Min LR: 1e-7

### Step 4: Model Evaluation

```python
from src.step_detection import evaluate_model

results = evaluate_model(model, val_X, val_y)
print(f"Validation Accuracy: {results['accuracy']:.4f}")
print(f"Classification Report:\n{results['classification_report']}")
print(f"Confusion Matrix:\n{results['confusion_matrix']}")
```

### Step 5: Model Saving

```python
from src.step_detection import save_model_and_metadata

metadata = {
    "model_type": "CNN",
    "framework": "TensorFlow/Keras",
    "input_shape": [6],
    "output_classes": 3,
    "validation_accuracy": float(results["accuracy"]),
    "epochs_trained": len(history.history["loss"]),
    "training_date": datetime.now().isoformat()
}

save_model_and_metadata(
    model,
    "models/step_detection_model.keras",
    metadata,
    "models/model_metadata.json"
)
```

## Hyperparameter Tuning

### Key Hyperparameters

| Parameter           | Default | Range        | Impact             |
| ------------------- | ------- | ------------ | ------------------ |
| filters (Conv1D #1) | 32      | 16-64        | Model capacity     |
| filters (Conv1D #2) | 64      | 32-128       | Feature extraction |
| learning_rate       | 0.001   | 1e-4 to 1e-2 | Convergence speed  |
| batch_size          | 32      | 16-128       | Training stability |
| dropout             | 0.0     | 0.0-0.5      | Regularization     |

### Tuning Process

1. **Start with baseline**: Use default parameters
2. **Adjust capacity**: Increase/decrease filters if under/overfitting
3. **Tune learning rate**: Use learning rate finder
4. **Add regularization**: If overfitting occurs
5. **Optimize batch size**: For training speed/stability

### Example: Advanced Model

```python
def create_advanced_cnn_model(input_shape=(6,), num_classes=3, dropout=0.2):
    model = keras.Sequential([
        layers.Reshape((1, input_shape[0]), input_shape=input_shape),

        # First block
        layers.Conv1D(filters=64, kernel_size=1, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(dropout),

        # Second block
        layers.Conv1D(filters=128, kernel_size=1, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(dropout),

        # Third block
        layers.Conv1D(filters=64, kernel_size=1, activation="relu"),
        layers.GlobalMaxPooling1D(),
        layers.Dropout(dropout),

        # Output
        layers.Dense(32, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(num_classes, activation="softmax"),
    ])

    return model
```

## Threshold Optimization

After training, optimize detection thresholds:

```bash
python main.py
# Choose option 3: Optimize detection thresholds
```

### Optimization Process

1. **Get Model Predictions**: Run inference on validation set
2. **Test Multiple Thresholds**: From 0.01 to 0.3
3. **Calculate Scores**: Balance precision and recall
4. **Select Best Threshold**: Maximize F1 score or custom metric

```python
def optimize_thresholds(model, val_features, val_labels):
    predictions = model.predict(val_features)
    thresholds = [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    best_threshold = 0.03
    best_score = 0

    for thresh in thresholds:
        # Count predictions above threshold
        start_preds = (predictions[:, 1] > thresh).sum()
        end_preds = (predictions[:, 2] > thresh).sum()

        # Count actual labels
        actual_starts = (val_labels == 1).sum()
        actual_ends = (val_labels == 2).sum()

        # Calculate score
        start_score = min(start_preds, actual_starts) / max(start_preds, actual_starts, 1)
        end_score = min(end_preds, actual_ends) / max(end_preds, actual_ends, 1)
        overall_score = (start_score + end_score) / 2

        if overall_score > best_score:
            best_score = overall_score
            best_threshold = thresh

    return best_threshold, best_score
```

## Model Evaluation Metrics

### Classification Metrics

- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Step Detection Metrics

- **Step Count Accuracy**: Correct total steps / Actual total steps
- **Step Timing Precision**: Correctly timed steps / All detected steps
- **False Positive Rate**: Incorrect steps / Total detected steps

### Validation Process

```python
def comprehensive_evaluation(model, val_features, val_labels):
    # Standard classification metrics
    predictions = model.predict(val_features)
    pred_classes = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(val_labels, pred_classes)
    report = classification_report(val_labels, pred_classes)
    confusion = confusion_matrix(val_labels, pred_classes)

    # Step detection specific metrics
    detector = StepDetector(model, threshold=0.15)
    detected_steps = 0
    actual_steps = count_actual_steps(val_labels)

    for features in val_features:
        result = detector.process_reading(*features)
        if result.get('completed_step'):
            detected_steps += 1

    step_accuracy = detected_steps / actual_steps if actual_steps > 0 else 0

    return {
        'classification_accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': confusion,
        'step_count_accuracy': step_accuracy,
        'detected_steps': detected_steps,
        'actual_steps': actual_steps
    }
```

## Troubleshooting

### Common Issues

1. **Low Accuracy (<80%)**

   - Check data quality and labeling
   - Increase model capacity
   - Adjust learning rate
   - Add more training data

2. **Overfitting**

   - Add dropout layers
   - Reduce model capacity
   - Increase regularization
   - Use early stopping

3. **Underfitting**

   - Increase model capacity
   - Reduce regularization
   - Check for data leakage
   - Extend training time

4. **Poor Step Detection**
   - Optimize thresholds
   - Check data distribution
   - Validate sensor calibration
   - Review labeling criteria

### Performance Tips

1. **Data Quality**

   - Ensure consistent sensor calibration
   - Remove outliers and noise
   - Balance class distribution
   - Validate label quality

2. **Model Architecture**

   - Start simple, add complexity gradually
   - Use batch normalization for stability
   - Consider attention mechanisms
   - Experiment with different optimizers

3. **Training Process**
   - Use learning rate scheduling
   - Monitor validation metrics
   - Save model checkpoints
   - Use cross-validation

## Advanced Techniques

### Data Augmentation

```python
def augment_sensor_data(features, labels, noise_factor=0.1):
    """Add noise to sensor data for augmentation."""
    augmented_features = features + np.random.normal(0, noise_factor, features.shape)
    return np.vstack([features, augmented_features]), np.hstack([labels, labels])
```

### Transfer Learning

```python
def create_transfer_model(base_model_path, num_classes=3):
    """Create model using transfer learning."""
    base_model = tf.keras.models.load_model(base_model_path)
    base_model.trainable = False

    model = keras.Sequential([
        base_model.layers[:-1],  # Remove last layer
        layers.Dense(num_classes, activation='softmax')
    ])

    return model
```

### Ensemble Methods

```python
def create_ensemble(model_paths, weights=None):
    """Create ensemble of multiple models."""
    models = [tf.keras.models.load_model(path) for path in model_paths]

    def ensemble_predict(features):
        predictions = [model.predict(features) for model in models]
        if weights:
            weighted_preds = [pred * w for pred, w in zip(predictions, weights)]
            return np.mean(weighted_preds, axis=0)
        return np.mean(predictions, axis=0)

    return ensemble_predict
```
