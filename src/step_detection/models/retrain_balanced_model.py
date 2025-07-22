import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from src.step_detection import load_step_data, prepare_data_for_training
from src.step_detection.models.model_utils import (
    create_cnn_model,
    save_model_and_metadata,
    train_model,
)


def retrain_with_class_balance():
    """Retrain model with proper class balancing."""

    print("🔄 RETRAINING MODEL WITH CLASS BALANCE")
    print("=" * 50)

    # Load data
    print("Loading data...")
    df = load_step_data("data/raw")
    train_X, val_X, train_y, val_y = prepare_data_for_training(df)

    # Check class distribution
    unique, counts = np.unique(train_y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")

    # Calculate class weights to balance training
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(train_y), y=train_y
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"Class weights: {class_weight_dict}")

    # Create model
    model = create_cnn_model()

    # Convert labels to categorical for training
    from tensorflow.keras.utils import to_categorical

    train_y_cat = to_categorical(train_y, num_classes=3)
    val_y_cat = to_categorical(val_y, num_classes=3)

    # Train with class weights
    print("Training with class weights...")
    history = model.fit(
        train_X,
        train_y_cat,
        validation_data=(val_X, val_y_cat),
        epochs=50,
        batch_size=32,
        class_weight=class_weight_dict,  # This is key!
        verbose=1,
    )

    # Test predictions on validation data
    print("\nTesting predictions on validation data...")
    val_predictions = model.predict(val_X[:10], verbose=0)

    print("Sample predictions:")
    for i in range(5):
        pred = val_predictions[i]
        true_label = val_y[i]
        print(
            f"Sample {i}: True={true_label}, Pred=[{pred[0]:.3f}, {pred[1]:.3f}, {pred[2]:.3f}]"
        )

    # Save model
    metadata = {
        "model_type": "CNN_Balanced",
        "framework": "TensorFlow/Keras",
        "input_shape": [6],
        "output_classes": 3,
        "class_weights": class_weight_dict,
        "epochs_trained": len(history.history["loss"]),
    }

    save_model_and_metadata(
        model,
        "models/step_detection_model_balanced.keras",
        metadata,
        "models/model_metadata_balanced.json",
    )

    print("✅ Balanced model saved!")
    return model


if __name__ == "__main__":
    retrain_with_class_balance()
