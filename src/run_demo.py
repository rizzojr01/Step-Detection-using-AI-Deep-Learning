"""Main script to demonstrate the step detection system."""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from step_detection.core.detector import StepDetector
from step_detection.models.model_utils import (
    create_cnn_model,
    evaluate_model,
    train_model,
)
from step_detection.utils.data_processor import (
    load_step_data,
    prepare_data_for_training,
)


def main():
    """Main function to run the step detection pipeline."""
    print("🚀 Step Detection System - Main Script")
    print("=" * 50)

    # Load data
    print("📊 Loading data...")
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    combined_df = load_step_data(str(data_dir))

    # Prepare data for training
    print("🔧 Preparing data for training...")
    train_features, val_features, train_labels, val_labels = prepare_data_for_training(
        combined_df
    )

    # Create and train model
    print("🏗️ Creating and training model...")
    model = create_cnn_model()
    history = train_model(model, train_features, train_labels, val_features, val_labels)

    # Evaluate model
    print("📈 Evaluating model...")
    accuracy = evaluate_model(model, val_features, val_labels)
    print(f"✅ Model accuracy: {accuracy:.4f}")

    # Initialize step detector
    print("🚶‍♂️ Initializing step detector...")
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    # Save model
    model_path = models_dir / "step_detection_model.keras"
    model.save(str(model_path))
    print(f"💾 Model saved to: {model_path}")

    # Demo step detection
    print("🎬 Running step detection demo...")
    detector = StepDetector(str(model_path))

    # Process some sample data
    sample_count = 0
    detected_steps = 0

    for i in range(min(100, len(val_features))):
        reading = val_features[i]
        result = detector.process_reading(
            reading[0],
            reading[1],
            reading[2],  # accelerometer
            reading[3],
            reading[4],
            reading[5],  # gyroscope
        )

        # Check if a step was completed in this reading
        if "completed_step" in result:
            detected_steps += 1
        sample_count += 1

        if (i + 1) % 25 == 0:
            print(f"  Processed {i+1} samples, detected {detected_steps} steps")

    print(f"\n🎯 Demo Results:")
    print(f"  Processed {sample_count} samples")
    print(f"  Detected {detected_steps} steps")
    print(f"  Total steps in detector: {detector.get_step_count()}")

    print("\n✅ Step Detection System Demo Complete!")


if __name__ == "__main__":
    main()
