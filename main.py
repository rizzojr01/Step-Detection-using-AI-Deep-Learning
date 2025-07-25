#!/usr/bin/env python3
"""
Main script for step detection project.
Demonstrates how to use the modular components.
"""

import os
import sys

import numpy as np

# Import from the step detection package
from src.step_detection import (
    SimpleStepCounter,
    StepDetector,
    create_cnn_model,
    evaluate_model,
    load_step_data,
    prepare_data_for_training,
    save_model_and_metadata,
    train_model,
)

# Import the balanced model retraining function
from src.step_detection.models.retrain_balanced_model import retrain_with_class_balance


def optimize_thresholds():
    """Find optimal thresholds for step detection."""
    print("\n🔧 Optimizing Step Detection Thresholds")
    print("=" * 40)

    model_path = "models/step_detection_model.keras"

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please train a model first using the training option.")
        return False

    try:
        # Load model and data
        import tensorflow as tf

        model = tf.keras.models.load_model(model_path)

        print("Loading validation data...")
        df = load_step_data("data/raw")
        train_features, val_features, train_labels, val_labels = (
            prepare_data_for_training(df)
        )

        # Get predictions on validation data
        print("Getting model predictions...")
        predictions = model.predict(val_features, verbose=0)

        # Handle labels - they should already be integers (0, 1, 2) for sparse_categorical_crossentropy
        if len(val_labels.shape) > 1 and val_labels.shape[1] > 1:
            # If one-hot encoded, convert to integers
            val_true_classes = np.argmax(val_labels, axis=1)
        else:
            # Already integers
            val_true_classes = val_labels

        # Test different thresholds
        thresholds = [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        best_threshold = 0.03
        best_score = 0

        print("\nTesting different thresholds:")
        for thresh in thresholds:
            # Count predictions above threshold
            start_predictions = (predictions[:, 1] > thresh).sum()
            end_predictions = (predictions[:, 2] > thresh).sum()

            # Count actual labels
            actual_starts = (val_true_classes == 1).sum()
            actual_ends = (val_true_classes == 2).sum()

            # Simple scoring based on how close predictions are to actual counts
            start_score = min(start_predictions, actual_starts) / max(
                start_predictions, actual_starts, 1
            )
            end_score = min(end_predictions, actual_ends) / max(
                end_predictions, actual_ends, 1
            )
            overall_score = (start_score + end_score) / 2

            print(
                f"  Threshold {thresh:.3f}: Start preds={start_predictions:4d} (actual={actual_starts}), "
                f"End preds={end_predictions:4d} (actual={actual_ends}), Score={overall_score:.3f}"
            )

            if overall_score > best_score:
                best_score = overall_score
                best_threshold = thresh

        print(
            f"\n🎯 Recommended threshold: {best_threshold:.3f} (score: {best_score:.3f})"
        )
        print(f"This threshold balances detection sensitivity with accuracy.")

        return True

    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        return False


def train_new_model():
    """Train a new step detection model."""
    print("🚀 Training New Step Detection Model")
    print("=" * 40)

    try:
        # Load and prepare data
        print("Loading data...")
        df = load_step_data("data/raw")
        train_X, val_X, train_y, val_y = prepare_data_for_training(df)

        # Create model
        print("Creating model...")
        model = create_cnn_model()

        # Train model
        print("Training model...")
        history = train_model(model, train_X, train_y, val_X, val_y, epochs=50)

        # Evaluate model
        print("Evaluating model...")
        results = evaluate_model(model, val_X, val_y)
        print(f"Final accuracy: {results['accuracy']:.4f}")

        # Save model
        print("Saving model...")
        metadata = {
            "model_type": "CNN",
            "framework": "TensorFlow/Keras",
            "input_shape": [6],
            "output_classes": 3,
            "validation_accuracy": float(results["accuracy"]),
            "epochs_trained": len(history.history["loss"]),
        }

        save_model_and_metadata(
            model,
            "models/step_detection_model.keras",
            metadata,
            "models/model_metadata.json",
        )

        print("✅ Model training completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error during training: {e}")
        return False


def retrain_balanced_model():
    """Retrain the model with class balancing to fix prediction issues."""
    print("⚖️ Retraining Model with Class Balancing")
    print("=" * 40)
    print("This will address the issue where the model predicts 99%+ 'No Step'")
    print("by balancing the training data and using class weights.")

    try:
        # Check if original model exists
        model_path = "models/step_detection_model.keras"
        if not os.path.exists(model_path):
            print(
                "⚠️ No existing model found. Training a new balanced model from scratch."
            )
        else:
            print("📊 Existing model found. Will retrain with balanced approach.")

        print("\n🔄 Starting balanced retraining process...")

        # Call the balanced retraining function
        model = retrain_with_class_balance()

        if model is not None:
            print("✅ Balanced model retraining completed successfully!")
            print("\n🎯 Key improvements made:")
            print("   • Applied class weights to balance training")
            print("   • Used data augmentation for minority classes")
            print("   • Adjusted model architecture for better sensitivity")
            print("   • Optimized thresholds for real-world usage")

            # Test the retrained model quickly
            print("\n🧪 Quick test of retrained model...")
            test_quick_predictions(model)

            return True
        else:
            print("❌ Balanced retraining failed. Check the logs for details.")
            return False

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required dependencies are installed.")
        return False
    except Exception as e:
        print(f"❌ Error during balanced retraining: {e}")
        return False


def test_quick_predictions(model):
    """Quick test to verify the retrained model produces better predictions."""
    print("Testing model predictions on sample data...")

    # Test with various input patterns
    test_cases = [
        ([0.1, 0.2, 9.8, 0.01, 0.02, 0.01], "Small movement (should be No Step)"),
        ([2.5, -1.2, 9.8, 0.3, 0.1, -0.2], "Normal walking step"),
        ([4.0, -2.0, 8.5, 0.8, 0.3, -0.4], "Strong walking step"),
    ]

    print(f"{'Test Case':<30} {'No Step':<10} {'Start':<10} {'End':<10}")
    print("-" * 60)

    for data, description in test_cases:
        input_data = np.array([data], dtype=np.float32)
        predictions = model.predict(input_data, verbose=0)[0]

        print(
            f"{description:<30} {predictions[0]:<10.3f} {predictions[1]:<10.3f} {predictions[2]:<10.3f}"
        )

    print("\n💡 Look for more balanced predictions (not 99%+ No Step)")


def test_real_time_detection():
    """Test real-time detection with saved model."""
    print("\n🔬 Testing Real-time Detection")
    print("=" * 40)

    model_path = "models/step_detection_model.keras"

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please train a model first using the training option.")
        return False

    try:
        # Initialize detectors with optimized thresholds
        # Based on threshold optimization results, using 0.15 for better balance
        detector = StepDetector(model_path)
        counter = SimpleStepCounter(model_path)

        print("✅ Detectors initialized successfully!")
        print(f"Using optimized thresholds: start=0.15, end=0.15")

        # Test with sample data
        print("Testing with sample sensor readings...")

        # Use more realistic sensor readings that might trigger step detection
        test_readings = [
            (1.2, -0.5, 9.8, 0.1, 0.2, -0.1),  # Normal standing
            (2.5, 1.8, 8.2, -0.3, 0.8, 0.4),  # Step start motion
            (0.3, -2.1, 11.1, 0.5, -0.2, -0.3),  # Step impact
            (1.8, 0.4, 9.2, -0.1, 0.3, 0.2),  # Step transition
            (0.7, -1.2, 10.3, 0.3, -0.4, 0.1),  # Step end motion
        ]

        for i, reading in enumerate(test_readings):
            result = detector.process_reading(*reading)
            step_detected = counter.process_reading(*reading)

            print(
                f"Reading {i+1}: Start prob={result['predictions']['start_prob']:.4f}, "
                f"End prob={result['predictions']['end_prob']:.4f}, "
                f"Step detected={step_detected}"
            )

        print(f"\nSession summary:")
        summary = detector.get_session_summary()
        print(f"Total readings: {summary['total_readings']}")
        print(f"Total steps: {summary['total_steps']}")
        print(f"Simple counter: {counter.step_count} steps")

        # Also test with some real validation data if available
        print("\n📊 Testing with real validation data...")
        try:
            # Load some real data for testing
            df = load_step_data("data/raw")
            if len(df) > 0:
                train_features, val_features, train_labels, val_labels = (
                    prepare_data_for_training(df)
                )

                # Test with first 10 samples from validation set
                real_detected_steps = 0
                for i in range(min(10, len(val_features))):
                    reading = val_features[i]
                    result = detector.process_reading(
                        reading[0],
                        reading[1],
                        reading[2],  # accelerometer
                        reading[3],
                        reading[4],
                        reading[5],  # gyroscope
                    )

                    if "completed_step" in result:
                        real_detected_steps += 1

                    if (
                        result["predictions"]["start_prob"] > 0.05
                        or result["predictions"]["end_prob"] > 0.05
                    ):
                        print(
                            f"  Sample {i+1}: Start={result['predictions']['start_prob']:.4f}, End={result['predictions']['end_prob']:.4f}"
                        )

                print(
                    f"Real data test: {real_detected_steps} steps detected from {min(10, len(val_features))} samples"
                )
            else:
                print("No real data available for testing")
        except Exception as e:
            print(f"Could not test with real data: {e}")

        return True

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False


def start_api_server():
    """Start the FastAPI server."""
    print("\n🌐 Starting API Server")
    print("=" * 40)

    try:
        import uvicorn

        print("Starting FastAPI server...")
        print("API documentation will be available at: http://localhost:8000/docs")

        # Use import string to enable reload functionality
        uvicorn.run(
            "src.step_detection.api.api:app", host="0.0.0.0", port=8000, reload=True
        )

    except ImportError:
        print(
            "❌ FastAPI/Uvicorn not installed. Install with: pip install -r requirements.txt"
        )
    except Exception as e:
        print(f"❌ Error starting server: {e}")


def main():
    """Main function with menu."""
    print("🚶‍♂️ Step Detection Project")
    print("=" * 50)
    print("Choose an option:")
    print("1. Train new model")
    print("2. Test real-time detection")
    print("3. Optimize detection thresholds")
    print("4. Retrain model with class balancing")
    print("5. Start API server")
    print("6. Exit")

    while True:
        try:
            choice = input("\nEnter your choice (1-6): ").strip()

            if choice == "1":
                train_new_model()
            elif choice == "2":
                test_real_time_detection()
            elif choice == "3":
                optimize_thresholds()
            elif choice == "4":
                retrain_balanced_model()
            elif choice == "5":
                start_api_server()
                break
            elif choice == "6":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-6.")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
