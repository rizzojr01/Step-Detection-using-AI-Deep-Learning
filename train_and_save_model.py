#!/usr/bin/env python3
"""
Train Model for Production
==========================

This script trains the step detection model and saves it for production use.
Run this locally before building your Docker image.

Usage:
    python train_and_save_model.py
"""

import os

import torch

from initialize_model import (
    StepDetectionCNN,
    create_dummy_data,
    save_model_for_production,
)


def main():
    """Train and save the model for production"""
    print("🏃‍♂️ Training Step Detection Model for Production")
    print("=" * 50)

    # Create device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # Initialize model
    model = StepDetectionCNN().to(device)
    print("🧠 Model initialized")

    # Create dummy training data (replace with your actual training data)
    print("📊 Creating training data...")
    X_train, y_train = create_dummy_data(1000)  # 1000 samples
    X_val, y_val = create_dummy_data(200)  # 200 validation samples

    # Convert to tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.FloatTensor(y_val).to(device)

    # Training setup
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 50

    print(f"🚂 Starting training for {epochs} epochs...")

    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()

        # Validation
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs.squeeze(), y_val)
                accuracy = ((val_outputs.squeeze() > 0.5) == y_val).float().mean()

            print(
                f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Val Loss: {val_loss:.4f} - Accuracy: {accuracy:.4f}"
            )
            model.train()

    # Save the trained model
    model_filename = "models/trained_step_detection_model.pth"
    os.makedirs("models", exist_ok=True)

    print(f"💾 Saving model to {model_filename}...")
    torch.save(model.state_dict(), model_filename)

    print("✅ Training complete!")
    print(f"📁 Model saved to: {model_filename}")
    print("🐳 Ready to build Docker image with pre-trained model!")


if __name__ == "__main__":
    main()
