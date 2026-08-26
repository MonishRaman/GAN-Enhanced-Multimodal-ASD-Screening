"""
ASD Multimodal Analysis Backend — Schema Package.

This package contains the data schemas used to validate
patient inputs and inference results.

The schemas define the interface between:
    Frontend → Backend → Frozen Multimodal Models → Backend → Frontend

No model loading, training, preprocessing, or fusion calculations
are performed in this package.
"""