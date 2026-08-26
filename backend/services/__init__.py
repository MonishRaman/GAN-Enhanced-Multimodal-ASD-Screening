"""
ASD Multimodal Analysis — Backend Services

This package contains the inference-service layer for the
frozen ASD multimodal machine-learning pipeline.

Services:
    - genetic_service
    - behavioral_service
    - environmental_service
    - fusion_service

The service modules are responsible only for inference and
data transformation. They must not retrain or modify the
frozen machine-learning artifacts.
"""

__all__ = [
    "genetic_service",
    "behavioral_service",
    "environmental_service",
    "fusion_service",
]