"""
Pydantic schemas for inference requests and responses.

Detailed schemas will be added during backend implementation.
"""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
