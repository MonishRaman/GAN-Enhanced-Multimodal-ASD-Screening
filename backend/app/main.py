"""
ASD Multimodal Inference API

Application entry point.

The actual model-loading and inference services will be connected
in the next development stage.
"""

from fastapi import FastAPI

app = FastAPI(
    title="ASD Multimodal Inference API",
    version="1.0.0",
    description="Inference API for the verified ASD multimodal project.",
)


@app.get("/")
def root():
    return {
        "project": "ASD Multimodal Project",
        "status": "ready",
        "message": "ASD multimodal inference API",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }
