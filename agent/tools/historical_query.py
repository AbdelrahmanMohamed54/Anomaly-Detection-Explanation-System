"""
LangChain tool: query PostgreSQL for historical anomaly records similar to a
detected event.

Exposes:
    query_similar_anomalies()  — raw function (used in tests / directly)
    historical_anomaly_tool    — LangChain @tool decorated callable

PostgreSQL connection is read from the POSTGRES_URL environment variable.
Connection errors are caught and return an empty list with a log warning so
the agent can continue without a live database.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError

load_dotenv(Path(__file__).parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

POSTGRES_URL: str = os.getenv(
    "POSTGRES_URL", "postgresql://postgres:password@localhost:5432/anomaly_db"
)

_QUERY_SQL = text(
    """
    SELECT
        id::text, sensor_id, anomaly_type,
        detected_at::text, duration_minutes, severity,
        root_cause, resolution, resolved_by, resolution_time_hours
    FROM historical_anomalies
    WHERE sensor_id = :sensor_id
       OR anomaly_type = :anomaly_type
    ORDER BY detected_at DESC
    LIMIT :limit
    """
)


def query_similar_anomalies(
    sensor_id: str,
    anomaly_type: str,
    limit: int = 5,
) -> list[dict]:
    """Query historical_anomalies for records matching this sensor or anomaly type.

    Args:
        sensor_id:    Sensor identifier (e.g. 'PUMP_01').
        anomaly_type: Anomaly category ('bearing_wear', 'pressure_drop', 'overload').
        limit:        Maximum number of records to return (default 5).

    Returns:
        List of row dicts with all table columns. Empty list on any DB error.
    """
    try:
        engine = create_engine(POSTGRES_URL, pool_pre_ping=True)
        with engine.connect() as conn:
            rows = conn.execute(
                _QUERY_SQL,
                {"sensor_id": sensor_id, "anomaly_type": anomaly_type, "limit": limit},
            ).fetchall()
        results = [dict(row._mapping) for row in rows]
        logger.info(
            "historical_query: %d records for sensor=%s type=%s",
            len(results), sensor_id, anomaly_type,
        )
        return results

    except OperationalError:
        logger.warning(
            "historical_query: PostgreSQL unavailable (sensor=%s, type=%s). "
            "Returning empty list.",
            sensor_id,
            anomaly_type,
        )
        return []

    except SQLAlchemyError as exc:
        logger.warning("historical_query: DB error — %s. Returning empty list.", exc)
        return []


# ── LangChain tool ─────────────────────────────────────────────────────────────


@tool
def historical_anomaly_tool(sensor_id: str, anomaly_type: str, limit: int = 5) -> str:
    """Query past anomaly records from the database for similar incidents.

    Use this tool FIRST when analysing any detected anomaly. Returns structured
    records of similar past events including root causes and resolutions.

    Args:
        sensor_id:    The sensor ID of the anomalous equipment (e.g. 'PUMP_01').
        anomaly_type: The classified anomaly type (e.g. 'bearing_wear').
        limit:        How many historical records to retrieve (default 5).

    Returns:
        JSON string of matching historical anomaly records.
    """
    records = query_similar_anomalies(sensor_id, anomaly_type, limit)
    if not records:
        return (
            f"No historical records found for sensor='{sensor_id}' "
            f"or anomaly_type='{anomaly_type}'. Database may be unavailable."
        )
    return json.dumps(records, default=str, indent=2)
