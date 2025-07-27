"""
Train PyTorch Model for Step Detection
=====================================

This script trains the PyTorch CNN model using your real dataset and saves
the trained weights for production use.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset

# Add src directory to path
sys.path.append("src")
from initialize_model import StepDetectionCNN


class StepDataset(Dataset):
    """Custom dataset for step detection"""

    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_real_data():
    """Load your actual step detection dataset"""
    data_folder = "data/raw"
    step_data_frames = []

    print("🔍 Loading real step detection data...")

    if not os.path.exists(data_folder):
        print("❌ Real data folder not found!")
        return None, None

    # Load real data
    files_processed = 0
    for root, dirs, files in os.walk(data_folder):
        for filename in files:
            if filename.endswith(".csv") and "Clipped" in filename:
                csv_path = os.path.join(root, filename)
                step_mixed_path = os.path.join(
                    root, filename.replace("Clipped", "") + ".stepMixed"
                )

                if os.path.exists(step_mixed_path):
                    try:
                        # Load sensor data
                        step_data = pd.read_csv(csv_path, usecols=[1, 2, 3, 4, 5, 6])
                        step_data = step_data.dropna()

                        # Load step labels
                        col_names = ["start_index", "end_index"]
                        step_indices = pd.read_csv(step_mixed_path, names=col_names)
                        step_indices = step_indices.dropna()
                        step_indices = step_indices.loc[
                            step_indices.end_index < step_data.shape[0]
                        ]

                        # Create labels
                        step_data["Label"] = "No Label"
                        for index, row in step_indices.iterrows():
                            step_data.loc[int(row["start_index"]), "Label"] = "start"
                            step_data.loc[int(row["end_index"]), "Label"] = "end"

                        step_data_frames.append(step_data)
                        files_processed += 1
                        print(f"   ✅ Loaded {filename} ({len(step_data)} samples)")

                    except Exception as e:
                        print(f"   ❌ Error loading {csv_path}: {e}")

    if step_data_frames:
        print(f"📊 Combining data from {files_processed} files...")
        combined_df = pd.concat(step_data_frames, ignore_index=True)

        # Extract features and labels
        features = combined_df.iloc[:, :6].values.astype(np.float32)
        labels = combined_df.iloc[:, 6].values

        # Convert labels to numbers
        label_mapping = {"No Label": 0, "start": 1, "end": 2}
        labels = np.array([label_mapping[label] for label in labels])

        print(
            f"✅ Dataset loaded: {features.shape[0]} samples, {features.shape[1]} features"
        )
        print(f"   Class distribution: {np.bincount(labels)} (no-step, start, end)")

        return features, labels
    else:
        print("❌ No data files found!")
        return None, None


def train_model():
    """Train the PyTorch step detection model"""

    print("🚀 Starting PyTorch Model Training")
    print("=" * 50)

    # Load data
    features, labels = load_real_data()
    if features is None or labels is None:
        print("❌ Failed to load data. Training aborted.")
        return None, None

    # Split data
    train_size = int(0.8 * len(features))
    X_train = features[:train_size]
    y_train = labels[:train_size]
    X_val = features[train_size:]
    y_val = labels[train_size:]

    print(f"📊 Data split: Train={len(X_train)}, Val={len(X_val)}")

    # Create datasets and dataloaders
    train_dataset = StepDataset(X_train, y_train)
    val_dataset = StepDataset(X_val, y_val)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StepDetectionCNN().to(device)

    print(f"🔧 Model initialized on {device}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Class weights to handle imbalanced data
    class_counts = np.bincount(y_train)
    class_weights = len(y_train) / (len(class_counts) * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print(f"⚖️  Class weights: {class_weights.cpu().numpy()}")

    # Training loop
    epochs = 50
    best_val_acc = 0.0
    best_model_path = "models/best_pytorch_model.pth"

    # Create models directory
    os.makedirs("models", exist_ok=True)

    print(f"🔥 Training for {epochs} epochs...")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for features_batch, labels_batch in train_loader:
            features_batch = features_batch.to(device)
            labels_batch = labels_batch.to(device)

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
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for features_batch, labels_batch in val_loader:
                features_batch = features_batch.to(device)
                labels_batch = labels_batch.to(device)

                outputs = model(features_batch)
                loss = criterion(outputs, labels_batch)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels_batch.size(0)
                val_correct += (predicted == labels_batch).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())

        # Calculate accuracies
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 New best model saved! Val accuracy: {val_acc:.4f}")

        # Print progress
        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch+1:2d}/{epochs}: "
                f"Train Loss: {train_loss/len(train_loader):.4f}, "
                f"Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss/len(val_loader):.4f}, "
                f"Val Acc: {val_acc:.4f}"
            )

    print(f"\n✅ Training completed!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
    print(f"💾 Best model saved to: {best_model_path}")

    # Load best model for final evaluation
    model.load_state_dict(torch.load(best_model_path, weights_only=False))
    model.eval()

    # Final evaluation
    all_predictions = []
    all_labels = []
    print("\n📊 Final Model Performance:")
    print("=" * 40)

    # Get step confidence statistics
    step_indices = np.where((np.array(all_labels) == 1) | (np.array(all_labels) == 2))[
        0
    ]

    if len(step_indices) > 0:
        # Calculate step confidence for validation set
        step_confidences = []

        with torch.no_grad():
            for features_batch, labels_batch in val_loader:
                features_batch = features_batch.to(device)
                outputs = model(features_batch)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

                for i, (probs, label) in enumerate(zip(probabilities, labels_batch)):
                    if label in [1, 2]:  # Step labels
                        step_conf = max(probs[1], probs[2])  # Max step confidence
                        step_confidences.append(step_conf)

        avg_step_conf = np.mean(step_confidences)
        high_conf_steps = np.sum(np.array(step_confidences) > 0.3)

        print(f"🚶‍♂️ Step Detection Performance:")
        print(
            f"   Average step confidence: {avg_step_conf:.3f} ({avg_step_conf*100:.1f}%)"
        )
        print(
            f"   High confidence steps (>30%): {high_conf_steps}/{len(step_confidences)} ({high_conf_steps/len(step_confidences)*100:.1f}%)"
        )

        if avg_step_conf > 0.25:
            print("✅ Good step confidence achieved!")
        else:
            print("⚠️  Step confidence could be improved")

    # Print classification report
    print(f"\n📈 Classification Report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=["No Step", "Step Start", "Step End"],
        )
    )

    return model, best_model_path


def test_trained_model(model_path="models/best_pytorch_model.pth"):
    """Test the trained model with sample data"""

    print("\n🧪 Testing Trained Model")
    print("=" * 30)

    try:
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = StepDetectionCNN().to(device)
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=False)
        )
        model.eval()

        print(f"✅ Model loaded from {model_path}")

        # Test with sample sensor data
        test_data = [
            [0.5, -9.8, 0.2, 0.1, 0.0, 0.0],  # Normal standing
            [2.0, -7.5, 1.5, 0.8, 0.2, 0.3],  # Step start
            [1.0, -11.2, 0.8, 0.4, 0.1, 0.2],  # Step end
        ]

        print("\n🔍 Sample Predictions:")
        for i, data in enumerate(test_data):
            sensor_tensor = torch.tensor([data], dtype=torch.float32).to(device)

            with torch.no_grad():
                outputs = model(sensor_tensor)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

            no_step, start, end = probabilities
            step_conf = max(start, end)
            predicted_class = np.argmax(probabilities)

            class_names = ["No Step", "Step Start", "Step End"]
            print(
                f"   Sample {i+1}: {class_names[predicted_class]} "
                f"(step confidence: {step_conf:.3f})"
            )
            print(
                f"      Probabilities: no_step={no_step:.3f}, start={start:.3f}, end={end:.3f}"
            )

        return True

    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False


if __name__ == "__main__":
    # Train the model
    result = train_model()

    if result is not None and len(result) == 2:
        model, model_path = result

        if model is not None and model_path is not None:
            # Test the trained model
            test_trained_model(model_path)

            print(f"\n🎯 Next Steps:")
            print(f"1. The trained model is saved at: {model_path}")
            print(f"2. Update your API to use this trained model")
            print(f"3. Test real-time step detection with your app")
            print(f"4. The model should now have much better step confidence!")

            # Copy to production location
            production_path = "models/trained_step_detection_model.pth"
            import shutil

            shutil.copy2(model_path, production_path)
            print(f"💾 Model also copied to: {production_path}")
        else:
            print("\n❌ Model or path is None.")
    else:
        print("\n❌ Training failed. Please check the error messages above.")
        print("\n❌ Training failed. Please check the error messages above.")
