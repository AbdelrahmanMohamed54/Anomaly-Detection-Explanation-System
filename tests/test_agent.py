"""
Integration tests for the RCAAgent.

All three tools and the LLM are mocked so tests run without live services
(no PostgreSQL, ChromaDB, or Gemini API required).

Coverage:
    test_rca_report_has_all_fields_populated  — RCAReport returned with every field set
    test_escalate_true_when_severity_above_threshold — severity >= 0.8 forces escalate=True
    test_escalate_false_for_low_severity      — severity < 0.8 with LLM escalate=False
    test_generate_report_called               — report_generator.generate_report is invoked
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.rca_agent import ESCALATION_SEVERITY_THRESHOLD, RCAAgent
from agent.schemas import RCAReport, RCAStructuredOutput


# ── Test fixtures ─────────────────────────────────────────────────────────────


def _make_anomaly(
    sensor_id: str = "PUMP_01",
    anomaly_type: str = "bearing_wear",
    severity: float = 0.72,
    reconstruction_error: float = 5.5,
) -> object:
    """Build a minimal AnomalyEvent-compatible object for tests."""
    # Import the dataclass from the detector
    from model.anomaly_detector import AnomalyEvent

    return AnomalyEvent(
        sensor_id=sensor_id,
        anomaly_type=anomaly_type,
        severity=severity,
        timestamp="2024-03-12T02:47:00",
        detected_values={
            "temperature": 96.0,
            "vibration": 3.8,
            "pressure": 5.0,
            "rpm": 1495.0,
            "current_draw": 13.5,
        },
        reconstruction_error=reconstruction_error,
    )


def _make_structured_output(escalate: bool = False) -> RCAStructuredOutput:
    """Build a canned RCAStructuredOutput for LLM mock returns."""
    return RCAStructuredOutput(
        anomaly_summary=(
            "PUMP_01 drive-end bearing showing elevated vibration (3.8 mm/s) and "
            "temperature (96°C), consistent with early-stage outer race spalling."
        ),
        root_cause=(
            "Lubrication degradation: re-lubrication interval in CMMS set to 4,000 hours "
            "vs. OEM-specified 2,000 hours. Overdue greasing led to oxidised lubricant and "
            "progressive bearing fatigue."
        ),
        similar_incidents=[
            "INC-2024-0312: PUMP_01 outer race spalling from missed re-lubrication, resolved by bearing replacement.",
            "INC-2024-0418: MOTOR_01 inner race fatigue from 103% rated torque, bearings replaced.",
            "INC-2024-0815: PUMP_01 vibration trending leading to planned bearing replacement.",
        ],
        recommended_actions=[
            "Isolate PUMP_01 under PTW and remove drive-end bearing for inspection.",
            "Replace bearing with SKF 6311-2RS1 and repack with 15 g Mobilith SHC 460.",
            "Correct CMMS re-lubrication interval to 2,000 hours.",
            "Verify shaft alignment post-replacement with laser alignment tool.",
        ],
        escalate=escalate,
        sources=[
            "historical_anomaly_tool",
            "incident_search_tool",
            "corrective_action_tool",
        ],
    )


# ── Helper: patch the agent's _run_agent coroutine ───────────────────────────


def _patch_run_agent(structured_output: RCAStructuredOutput):
    """Return a context manager that replaces RCAAgent._run_agent with an AsyncMock."""
    return patch.object(
        RCAAgent,
        "_run_agent",
        new=AsyncMock(return_value=structured_output),
    )


def _patch_build_llm():
    """Return a context manager that replaces RCAAgent._build_llm with a MagicMock."""
    return patch.object(
        RCAAgent,
        "_build_llm",
        return_value=MagicMock(),
    )


def _patch_create_agent():
    """Return a context manager that stubs create_agent to return a MagicMock graph."""
    return patch("agent.rca_agent.create_agent", return_value=MagicMock())


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestRCAAgent:
    """Integration tests for RCAAgent.analyze()."""

    def test_rca_report_has_all_fields_populated(self) -> None:
        """analyze() must return an RCAReport with every required field non-empty."""
        structured = _make_structured_output(escalate=False)
        anomaly = _make_anomaly(severity=0.72)

        with _patch_build_llm(), _patch_create_agent(), _patch_run_agent(structured):
            agent = RCAAgent()
            report: RCAReport = asyncio.run(agent.analyze(anomaly))

        assert isinstance(report, RCAReport), "analyze() must return an RCAReport"
        assert report.sensor_id == "PUMP_01"
        assert 0.0 <= report.severity <= 1.0
        assert report.anomaly_summary, "anomaly_summary must not be empty"
        assert report.root_cause, "root_cause must not be empty"
        assert len(report.similar_incidents) >= 1, "similar_incidents must have at least 1 entry"
        assert len(report.recommended_actions) >= 1, "recommended_actions must have at least 1 entry"
        assert isinstance(report.escalate, bool)
        assert len(report.sources) >= 1, "sources must reference at least one tool"
        assert report.generation_time_seconds >= 0.0
        assert report.language in ("en", "de")

    def test_escalate_true_when_severity_above_threshold(self) -> None:
        """severity >= 0.8 must force escalate=True regardless of LLM output."""
        # LLM says escalate=False, but severity is above the threshold
        structured = _make_structured_output(escalate=False)
        anomaly = _make_anomaly(severity=ESCALATION_SEVERITY_THRESHOLD)  # exactly at threshold

        with _patch_build_llm(), _patch_create_agent(), _patch_run_agent(structured):
            agent = RCAAgent()
            report: RCAReport = asyncio.run(agent.analyze(anomaly))

        assert report.escalate is True, (
            f"escalate must be True when severity={ESCALATION_SEVERITY_THRESHOLD} "
            f">= threshold={ESCALATION_SEVERITY_THRESHOLD}"
        )

    def test_escalate_true_when_severity_well_above_threshold(self) -> None:
        """severity=0.95 must always produce escalate=True."""
        structured = _make_structured_output(escalate=False)
        anomaly = _make_anomaly(severity=0.95)

        with _patch_build_llm(), _patch_create_agent(), _patch_run_agent(structured):
            agent = RCAAgent()
            report: RCAReport = asyncio.run(agent.analyze(anomaly))

        assert report.escalate is True

    def test_escalate_false_for_low_severity(self) -> None:
        """severity < 0.8 with LLM escalate=False must keep escalate=False."""
        structured = _make_structured_output(escalate=False)
        anomaly = _make_anomaly(severity=0.60)

        with _patch_build_llm(), _patch_create_agent(), _patch_run_agent(structured):
            agent = RCAAgent()
            report: RCAReport = asyncio.run(agent.analyze(anomaly))

        assert report.escalate is False, (
            "escalate must remain False when severity < threshold and LLM says False"
        )

    def test_llm_escalate_true_overrides_low_severity(self) -> None:
        """LLM escalate=True must be honoured even when severity < threshold."""
        structured = _make_structured_output(escalate=True)
        anomaly = _make_anomaly(severity=0.50)

        with _patch_build_llm(), _patch_create_agent(), _patch_run_agent(structured):
            agent = RCAAgent()
            report: RCAReport = asyncio.run(agent.analyze(anomaly))

        assert report.escalate is True, (
            "If LLM explicitly sets escalate=True, it must be honoured"
        )

    def test_generate_report_called(self) -> None:
        """generate_report from report_generator must be called during analyze()."""
        structured = _make_structured_output()
        anomaly = _make_anomaly()

        with (
            _patch_build_llm(),
            _patch_create_agent(),
            _patch_run_agent(structured),
            patch("agent.rca_agent.generate_report", wraps=__import__(
                "agent.report_generator", fromlist=["generate_report"]
            ).generate_report) as mock_gen,
        ):
            agent = RCAAgent()
            asyncio.run(agent.analyze(anomaly))

        mock_gen.assert_called_once()

    def test_similar_incidents_capped_at_three(self) -> None:
        """RCAReport.similar_incidents must never exceed 3 entries."""
        structured = _make_structured_output()
        # Add extra incidents to the structured output
        structured.similar_incidents = [f"Incident {i}" for i in range(10)]
        anomaly = _make_anomaly()

        with _patch_build_llm(), _patch_create_agent(), _patch_run_agent(structured):
            agent = RCAAgent()
            report: RCAReport = asyncio.run(agent.analyze(anomaly))

        assert len(report.similar_incidents) <= 3, (
            "similar_incidents in RCAReport must be capped at 3"
        )
