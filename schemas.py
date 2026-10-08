from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class PrimaryScreening(BaseModel):
    detected_pathology: str = Field(
        ..., description="Highest probability pathology identified"
    )
    confidence_percent: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Temperature-scaled confidence percentage"
    )

class ReferenceCase(BaseModel):
    case_id: str
    similarity: float = Field(
        ..., description="Cosine similarity score against query emberdding"
    )
    pathology: str
    impression: str
    findings: str

class DiagnosticRespone(BaseModel):
    filename: str
    primary_screening: PrimaryScreening
    differential_distribution: Dict[str, float]
    retrieved_reference_cohort: List[ReferenceCase]
    processing_time_ms: float