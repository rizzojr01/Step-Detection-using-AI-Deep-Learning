"""API server script."""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

import uvicorn

from step_detection.api.api import app

if __name__ == "__main__":
    print("🚀 Starting Step Detection API Server...")
    uvicorn.run("step_detection.api.api:app", host="0.0.0.0", port=8000, reload=True)
