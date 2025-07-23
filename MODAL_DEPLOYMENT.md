# 🚀 Modal Deployment Guide

This guide provides step-by-step instructions for deploying the Step Detection AI application using Modal, a serverless cloud platform optimized for machine learning workloads.

## 📋 Table of Contents

- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)

## 📋 Prerequisites

- **Python 3.10+** installed on your local machine
- **Modal account** (sign up at [modal.com](https://modal.com))


## ⚡ Quick Start

For experienced developers who want to get up and running quickly:

```bash
# 1. Clone and setup
git clone https://github.com/rizzojr01/Step-Detection-using-AI-Deep-Learning.git
cd Step-Detection-using-AI-Deep-Learning
python3 -m venv venv && source venv/bin/activate

# 2. Install Modal
pip install modal==1.0.0

# 3. Authenticate with Modal
modal token new

# 4. Deploy
modal deploy modal_app.py
```

