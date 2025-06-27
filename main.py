#!/usr/bin/env python3
"""
Main script for step detection project.
Demonstrates how to use the modular components.
"""

import os
import sys

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
        # Initialize detectors
        detector = StepDetector(model_path)
        counter = SimpleStepCounter(model_path)

        print("✅ Detectors initialized successfully!")

        # Test with sample data
        print("Testing with sample sensor readings...")

        # Simulate some sensor readings
        test_readings = [
            (1.2, -0.5, 9.8, 0.1, 0.2, -0.1),
            (2.1, 1.2, 8.9, -0.2, 0.5, 0.3),
            (0.8, -1.1, 10.1, 0.3, -0.1, -0.2),
            (1.5, 0.2, 9.5, 0.0, 0.1, 0.1),
            (0.9, -0.8, 9.9, 0.2, -0.3, 0.0),
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
        print(f"Simple counter: {counter.get_count()} steps")

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

        # Import the app directly to avoid circular imports
        from src.step_detection.api.api import app

        print("Starting FastAPI server...")
        print("API documentation will be available at: http://localhost:8000/docs")

        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

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
    print("3. Start API server")
    print("4. Exit")

    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()

            if choice == "1":
                train_new_model()
            elif choice == "2":
                test_real_time_detection()
            elif choice == "3":
                start_api_server()
                break
            elif choice == "4":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-4.")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
