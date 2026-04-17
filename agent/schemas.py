"""
Shared Pydantic v2 models used across the agent, API, and tests.

These models are imported by rca_agent.py, report_generator.py, and api/main.py.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RCAStructuredOutput(BaseModel):
    """Intermediate structured output produced by the RCA LLM agent.

    Contains the analysis content without session metadata (timing, language).
    This is the schema passed to `create_agent(response_format=...)` so the
    LLM produces a validated JSON object at the end of its reasoning loop.
    """

    anomaly_summary: str = Field(
        description="One-paragraph plain-language description of the detected anomaly."
    )
    root_cause: str = Field(
        description=(
            "The most probable root cause of the anomaly based on historical data "
            "and incident reports. Be specific — name components, failure modes, "
            "and contributing factors."
        )
    )
    similar_incidents: list[str] = Field(
        description="Top 3 similar past incidents, each as a one-sentence summary.",
        max_length=3,
    )
    recommended_actions: list[str] = Field(
        description=(
            "Ordered list of corrective actions to resolve the anomaly. "
            "Action 1 is the most urgent."
        )
    )
    escalate: bool = Field(
        description=(
            "True if the anomaly warrants immediate escalation to a senior engineer "
            "or requires emergency shutdown. False for routine corrective maintenance."
        )
    )
    sources: list[str] = Field(
        description="Names of the tools/sources consulted during analysis."
    )


class RCAReport(BaseModel):
    """Complete RCA report including session metadata and language tag.

    Returned by RCAAgent.analyze() and serialized by report_generator.
    """

    sensor_id: str
    severity: float = Field(ge=0.0, le=1.0)
    anomaly_summary: str
    root_cause: str
    similar_incidents: list[str]
    recommended_actions: list[str]
    escalate: bool
    sources: list[str]
    generation_time_seconds: float = Field(ge=0.0)
    language: str = Field(default="en", pattern="^(en|de)$")


class HistoricalAnomalyRecord(BaseModel):
    """One row returned from the historical_anomalies PostgreSQL table."""

    id: str
    sensor_id: str
    anomaly_type: str
    detected_at: str
    duration_minutes: int
    severity: float
    root_cause: str
    resolution: str
    resolved_by: str
    resolution_time_hours: float


class IncidentSearchResult(BaseModel):
    """One result returned from the ChromaDB incident report collection."""

    text: str
    source_file: str
    anomaly_type: str
    similarity_score: float


class ActionRecommendation(BaseModel):
    """One result returned from the ChromaDB maintenance logs collection."""

    text: str
    source_file: str
    maintenance_type: str
    similarity_score: float
