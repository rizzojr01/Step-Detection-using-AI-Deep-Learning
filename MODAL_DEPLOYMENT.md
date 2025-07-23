# 🚀 Modal Deployment Guide

This guide provides step-by-step instructions for deploying the Step Detection AI application using Modal, a serverless cloud platform optimized for machine learning workloads.

## 📋 Prerequisites

- Python 3.10+ installed on your local machine
- Git installed
- A Modal account (sign up at [modal.com](https://modal.com))
- Modal CLI configured with your credentials

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/rizzojr01/Step-Detection-using-AI-Deep-Learning.git
cd Step-Detection-using-AI-Deep-Learning
```

### 2. Create and Activate Virtual Environment

**On macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Upgrade pip to latest version
python -m pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# Install Modal CLI
pip install modal==1.0.0
```

### 4. Configure Modal Authentication

#### Option A: Interactive Login

```bash
modal token new
```


### 5. Verify Modal Setup

```bash
modal --help
```

## 🚢 Deployment Process

### 1. Deploy the Application

```bash
modal deploy modal_app.py
```

This command will:

- Build the container image with all dependencies
- Upload your code to Modal's cloud
- Deploy the FastAPI application as a serverless function
- Return a public URL for your API



## 🔄 CI/CD with GitHub Actions

The project includes automated deployment via GitHub Actions:

### Setup GitHub Secrets

1. Go to your GitHub repository
2. Navigate to Settings → Secrets and variables → Actions
3. Add the following secrets:
   - `MODAL_TOKEN_ID`: Your Modal token ID
   - `MODAL_TOKEN_SECRET`: Your Modal token secret

### Automatic Deployment

Pushes to the `main` branch will automatically:

1. Install dependencies
2. Deploy to Modal
3. Update the live application

## 📊 Monitoring and Logs

### View Application Logs

```bash
modal logs your-app-name
```



### Common Issues

#### 1. Authentication Errors

```bash
# Re-authenticate with Modal
modal token new
```

## 🔄 Development Workflow



### Staging Deployment

```bash
# Deploy to staging environment
modal deploy modal_app.py --name step-detection-staging
```

### Production Deployment

```bash
# Deploy to production
modal deploy modal_app.py --name step-detection-prod
```

