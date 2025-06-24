# Optimized Production Deployment Guide

This guide shows how to deploy with a pre-trained model instead of training data.

## 🚀 Quick Start

### Step 1: Train the Model Locally

```bash
# Train and save the model
python train_and_save_model.py
```

This will create a `models/trained_step_detection_model.pth` file.

### Step 2: Build & Push Docker Image

```bash
# Build and push to Docker Hub
./build.sh
```

### Step 3: Deploy on Server

```bash
# Pull and run the container
./deploy.sh
```

## 📊 Optimization Benefits

### Before (with sample data):

- **Image size**: ~2-3 GB
- **Build time**: 5-10 minutes
- **Contains**: All training data, notebooks, dependencies

### After (with pre-trained model):

- **Image size**: ~800 MB - 1.2 GB
- **Build time**: 2-3 minutes
- **Contains**: Only the trained model and API code

## 🔧 What Changed

1. **Dockerfile**: Copies `models/` instead of `sample_data/`
2. **API**: Loads pre-trained model on startup
3. **Workflow**: Train locally → Build → Deploy

## 📁 File Structure

```
├── models/
│   └── trained_step_detection_model.pth  # Pre-trained model
├── train_and_save_model.py               # Training script
├── build.sh                              # Build & push
├── deploy.sh                             # Deploy script
└── step_detection_api.py                 # API (loads model)
```

## 🛠️ Advanced Usage

### Custom Model Training

Edit `train_and_save_model.py` to use your actual training data:

```python
# Replace dummy data with your real data
X_train, y_train = load_your_training_data()
```

### Different Model Architectures

You can train different models (CNN, LSTM, Transformer) and save them:

```python
# Train CNN
python train_and_save_model.py

# Or train LSTM (modify script)
python train_lstm_model.py
```

## 🔍 Troubleshooting

- **Model not found**: Make sure to run `train_and_save_model.py` first
- **Build fails**: Check if `models/` directory exists
- **API errors**: Check model loading in container logs
