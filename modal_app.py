import modal

stub = modal.Stub("step-detection-app")

image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")

@stub.function(image=image, keep_warm=1)
@modal.asgi_app()
def fastapi_app():
    from src.step_detection.api.api import app
    return app
