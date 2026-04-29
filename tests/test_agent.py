"""
Integration tests for the RCAAgent and ReportGenerator.

All LLM calls and tools are mocked so tests run without live services
(no PostgreSQL, ChromaDB, or Gemini API required).

Coverage:
    TestRCAAgent:
        test_rca_report_has_all_fields_populated
        test_escalate_true_when_severity_above_threshold
        test_escalate_false_for_low_severity
        test_generate_report_called

    TestReportGenerator:
        test_en_report_has_all_required_fields
        test_de_report_contains_german_text
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.rca_agent import ESCALATION_SEVERITY_THRESHOLD, RCAAgent
from agent.report_generator import ReportGenerator, _TranslatedFields
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


# ─────────────────────────────────────────────────────────────────────────────
# TestReportGenerator — bilingual formatter (Session 4)
# ─────────────────────────────────────────────────────────────────────────────


def _make_en_report(severity: float = 0.72) -> RCAReport:
    """Return a fully populated English RCAReport for use in generator tests."""
    return RCAReport(
        sensor_id="PUMP_01",
        severity=severity,
        anomaly_summary=(
            "PUMP_01 drive-end bearing exhibits elevated vibration (3.8 mm/s) and "
            "temperature (96°C), consistent with early-stage outer race spalling."
        ),
        root_cause=(
            "Lubrication degradation: re-lubrication interval set to 4,000 hours "
            "in CMMS vs. OEM-specified 2,000 hours. Oxidised grease caused bearing fatigue."
        ),
        similar_incidents=[
            "INC-2024-0312: outer race spalling resolved by bearing replacement.",
            "INC-2024-0418: inner race fatigue from overloading, bearings replaced.",
            "INC-2024-0815: planned bearing replacement based on vibration trending.",
        ],
        recommended_actions=[
            "Isolate PUMP_01 and remove drive-end bearing for inspection.",
            "Replace bearing with SKF 6311-2RS1, repack with 15 g Mobilith SHC 460.",
            "Correct CMMS re-lubrication interval to 2,000 hours.",
        ],
        escalate=False,
        sources=["historical_anomaly_tool", "incident_search_tool", "corrective_action_tool"],
        generation_time_seconds=4.2,
        language="en",
    )


class TestReportGenerator:
    """Tests for ReportGenerator.generate() and generate_bilingual()."""

    def test_en_report_has_all_required_fields(self) -> None:
        """generate(language='EN') must return an RCAReport with all fields populated."""
        en_report = _make_en_report()
        rca_result = en_report.model_dump()

        with patch.object(ReportGenerator, "_build_llm", return_value=MagicMock()):
            gen = ReportGenerator()
            result: RCAReport = asyncio.run(gen.generate(rca_result, language="EN"))

        assert isinstance(result, RCAReport), "generate() must return an RCAReport"
        assert result.sensor_id == "PUMP_01"
        assert result.language == "en"
        assert 0.0 <= result.severity <= 1.0
        assert result.anomaly_summary, "anomaly_summary must not be empty"
        assert result.root_cause, "root_cause must not be empty"
        assert len(result.similar_incidents) >= 1
        assert len(result.recommended_actions) >= 1
        assert isinstance(result.escalate, bool)
        assert result.generation_time_seconds >= 0.0

    def test_bilingual_reports_have_different_text(self) -> None:
        """EN and DE reports must have different anomaly_summary text."""
        en_report = _make_en_report()
        rca_result = en_report.model_dump()

        german_fields = _TranslatedFields(
            anomaly_summary="Das Lager zeigt erhoehte Schwingungen.",
            root_cause="Schmierstoffversagen durch verpasstes Nachschmierintervall.",
            similar_incidents=["INC-2024-0312: Lagerschaden behoben."],
            recommended_actions=["Lager unter Erlaubnisschein isolieren."],
        )

        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=german_fields)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_chain

        with patch.object(ReportGenerator, "_build_llm", return_value=mock_llm):
            gen = ReportGenerator()
            de_report: RCAReport = asyncio.run(gen.generate(rca_result, language="DE"))

        assert en_report.anomaly_summary != de_report.anomaly_summary, (
            "EN and DE anomaly_summary must differ"
        )
        assert de_report.language == "de"
        assert en_report.language == "en"

    def test_de_report_contains_german_text(self) -> None:
        """generate(language='DE') must return a report with German-language content."""
        en_report = _make_en_report()
        rca_result = en_report.model_dump()

        # Mock the LLM translation to return plausible German text
        german_fields = _TranslatedFields(
            anomaly_summary=(
                "Das Lager der Antriebsseite von PUMP_01 zeigt erhoehte Schwingungen "
                "(3,8 mm/s) und Temperatur (96 Grad C), was fruehzeitigem Aussenring-Pittingschaden entspricht."
            ),
            root_cause=(
                "Schmierstoffversagen: Nachschmierintervall im CMMS auf 4.000 Stunden eingestellt "
                "statt der vom Hersteller vorgeschriebenen 2.000 Stunden. "
                "Oxidiertes Fett verursachte Lagerermuedung."
            ),
            similar_incidents=[
                "INC-2024-0312: Aussenring-Pittingschaden durch Lagertausch behoben.",
                "INC-2024-0418: Innenring-Ermuedung durch Ueberlastung, Lager ersetzt.",
                "INC-2024-0815: Geplanter Lagertausch aufgrund von Schwingungsanalyse.",
            ],
            recommended_actions=[
                "PUMP_01 unter Erlaubnisschein isolieren und Antriebslager ausbauen.",
                "Lager durch SKF 6311-2RS1 ersetzen, mit 15 g Mobilith SHC 460 nachschmieren.",
                "Nachschmierintervall im CMMS auf 2.000 Stunden korrigieren.",
            ],
        )

        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=german_fields)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_chain

        with patch.object(ReportGenerator, "_build_llm", return_value=mock_llm):
            gen = ReportGenerator()
            de_report: RCAReport = asyncio.run(gen.generate(rca_result, language="DE"))

        assert de_report.language == "de", "language field must be 'de'"

        # Verify German words appear in the translated fields
        german_indicators = ["Lager", "Schwingungen", "Schmierstoff", "Lagertausch", "isolieren"]
        all_de_text = (
            de_report.anomaly_summary
            + de_report.root_cause
            + " ".join(de_report.similar_incidents)
            + " ".join(de_report.recommended_actions)
        )
        found = [word for word in german_indicators if word in all_de_text]
        assert found, (
            f"DE report must contain German text. "
            f"Checked for: {german_indicators}. Text sample: {all_de_text[:200]}"
        )

        # Sensor metadata must be unchanged
        assert de_report.sensor_id == en_report.sensor_id
        assert de_report.severity == en_report.severity
        assert de_report.escalate == en_report.escalate


# ─────────────────────────────────────────────────────────────────────────────
# TestSeverityLabel — severity_label() mapping
# ─────────────────────────────────────────────────────────────────────────────


class TestSeverityLabel:
    """Tests for the severity_label() helper in report_generator."""

    def test_high_severity_mapping(self) -> None:
        """Float 0.9 must map to 'HIGH' in English."""
        from agent.report_generator import severity_label
        assert severity_label(0.9) == "HIGH"

    def test_medium_severity_mapping(self) -> None:
        """Float 0.5 must map to 'MEDIUM' in English."""
        from agent.report_generator import severity_label
        assert severity_label(0.5) == "MEDIUM"

    def test_low_severity_mapping(self) -> None:
        """Float 0.2 must map to 'LOW' in English."""
        from agent.report_generator import severity_label
        assert severity_label(0.2) == "LOW"

    def test_boundary_high_threshold(self) -> None:
        """Exactly 0.8 must be 'HIGH'; just below 0.8 must be 'MEDIUM'."""
        from agent.report_generator import severity_label
        assert severity_label(0.8) == "HIGH"
        assert severity_label(0.799) == "MEDIUM"

    def test_boundary_medium_threshold(self) -> None:
        """Exactly 0.5 must be 'MEDIUM'; just below 0.5 must be 'LOW'."""
        from agent.report_generator import severity_label
        assert severity_label(0.5) == "MEDIUM"
        assert severity_label(0.499) == "LOW"

    def test_german_high_label(self) -> None:
        """severity_label(0.9, 'de') must return 'HOCH'."""
        from agent.report_generator import severity_label
        assert severity_label(0.9, "de") == "HOCH"

    def test_german_medium_label(self) -> None:
        """severity_label(0.6, 'de') must return 'MITTEL'."""
        from agent.report_generator import severity_label
        assert severity_label(0.6, "de") == "MITTEL"

    def test_german_low_label(self) -> None:
        """severity_label(0.1, 'de') must return 'NIEDRIG'."""
        from agent.report_generator import severity_label
        assert severity_label(0.1, "de") == "NIEDRIG"


# ─────────────────────────────────────────────────────────────────────────────
# TestFullRCAAgentPipeline — all 3 tools mocked, valid RCAReport returned
# ─────────────────────────────────────────────────────────────────────────────


class TestFullRCAAgentPipeline:
    """Verify the full analyze() path with all three tools mocked."""

    def test_full_rca_agent_returns_valid_report(self) -> None:
        """Full RCAAgent.analyze() with all tools mocked must return RCAReport."""
        structured = _make_structured_output(escalate=False)
        anomaly = _make_anomaly(sensor_id="PUMP_01", severity=0.72)

        with _patch_build_llm(), _patch_create_agent(), _patch_run_agent(structured):
            agent = RCAAgent()
            report: RCAReport = asyncio.run(agent.analyze(anomaly))

        assert isinstance(report, RCAReport)
        assert report.sensor_id == "PUMP_01"
        assert report.anomaly_summary, "anomaly_summary must not be empty"
        assert report.root_cause, "root_cause must not be empty"
        assert len(report.similar_incidents) >= 1
        assert len(report.recommended_actions) >= 1
        assert report.sources, "sources must reference at least one tool"
        assert report.generation_time_seconds >= 0.0

    def test_full_rca_agent_all_tools_referenced_in_sources(self) -> None:
        """RCAReport.sources must include all three expected tool names."""
        structured = _make_structured_output(escalate=False)
        anomaly = _make_anomaly(sensor_id="MOTOR_01", severity=0.65)

        with _patch_build_llm(), _patch_create_agent(), _patch_run_agent(structured):
            agent = RCAAgent()
            report: RCAReport = asyncio.run(agent.analyze(anomaly))

        expected_tools = {
            "historical_anomaly_tool",
            "incident_search_tool",
            "corrective_action_tool",
        }
        assert expected_tools.issubset(set(report.sources)), (
            f"Expected all 3 tool names in sources. Got: {report.sources}"
        )
