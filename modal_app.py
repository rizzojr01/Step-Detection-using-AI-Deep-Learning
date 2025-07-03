import modal

app = modal.App("step-detection-app")

# Create a volume for persistent model storage (optional - for production)
model_volume = modal.Volume.from_name("step-detection-models", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    # Add source code
    .add_local_dir("src", remote_path="/root/src")
    # Add configuration files
    .add_local_dir("config", remote_path="/root/config")
    # Add models directory
    .add_local_dir("models", remote_path="/root/models")
    # Add data directory (25MB is reasonable for image mounting)
    .add_local_dir(
        "data",
        remote_path="/root/data",
        ignore=["*.tmp", "*.log", "__pycache__", ".DS_Store"],
    )
    # Add specific files if needed
    .add_local_file("requirements.txt", remote_path="/root/requirements.txt")
    .add_local_file("pyproject.toml", remote_path="/root/pyproject.toml")
)


@app.function(
    image=image,
    # Optional: Mount persistent volume for models (uncomment if needed)
    # volumes={"/root/persistent_models": model_volume}
)
@modal.asgi_app()
def fastapi_app():
    from src.step_detection.api.api import app as fastapi_app

    return fastapi_app
