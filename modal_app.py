import modal

app = modal.App("step-detection-app")

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", remote_path="/root")
)

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    from src.step_detection.api.api import app as fastapi_app
    return fastapi_app
