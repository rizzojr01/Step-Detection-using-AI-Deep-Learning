import modal

app = modal.App("step-detection-app")

model_volume = modal.Volume.from_name("step-detection-models", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", remote_path="/root")
)


@app.function(
    image=image,
    scaledown_window=600,
    max_containers=3,
    concurrency_limit=3,
)
@modal.asgi_app()
def fastapi_app():
    from src.step_detection.api.api import app as fastapi_app

    return fastapi_app
