"""
FastAPI endpoint tests using TestClient.

All external services (TensorFlow model, RCA agent, PostgreSQL, ChromaDB)
are mocked so tests run without any live infrastructure.

Coverage:
    test_health_returns_200_with_model_loaded
    test_simulate_returns_valid_rca_report_structure
    test_history_returns_list
    test_health_degraded_when_model_missing
    test_simulate_both_languages_returns_en_and_de
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from agent.schemas import RCAReport


# ── Shared fixtures ────────────────────────────────────────────────────────────


def _make_rca_report(
    sensor_id: str = "PUMP_01",
    anomaly_type: str = "bearing_wear",
    severity: float = 0.85,
    language: str = "en",
) -> RCAReport:
    """Return a fully-populated RCAReport for mocking the RCA agent."""
    return RCAReport(
        sensor_id=sensor_id,
        severity=severity,
        anomaly_summary=(
            f"{sensor_id} drive-end bearing shows elevated vibration and temperature, "
            "consistent with outer race spalling."
        ),
        root_cause=(
            "Lubrication degradation due to missed re-lubrication interval. "
            "Oxidised grease caused progressive bearing fatigue."
        ),
        similar_incidents=[
            "INC-2024-0312: outer race spalling, resolved by bearing replacement.",
            "INC-2024-0418: inner race fatigue, bearings replaced.",
            "INC-2024-0815: planned bearing replacement from vibration trending.",
        ],
        recommended_actions=[
            "Isolate equipment under PTW and remove drive-end bearing.",
            "Replace bearing with OEM part, repack grease.",
            "Correct CMMS re-lubrication interval to OEM specification.",
        ],
        escalate=severity >= 0.8,
        sources=["historical_anomaly_tool", "incident_search_tool", "corrective_action_tool"],
        generation_time_seconds=3.7,
        language=language,
    )


def _make_de_report(en_report: RCAReport) -> RCAReport:
    """Return a German translation of the given EN report."""
    return en_report.model_copy(
        update={
            "anomaly_summary": "Das Lager zeigt erhoehte Schwingungen und Temperatur.",
            "root_cause": "Schmierstoffversagen durch verpasstes Nachschmierintervall.",
            "similar_incidents": [
                "INC-2024-0312: Aussenring-Pittingschaden behoben.",
                "INC-2024-0418: Innenring-Ermuedung, Lager ersetzt.",
                "INC-2024-0815: Geplanter Lagertausch.",
            ],
            "recommended_actions": [
                "Anlage unter Erlaubnisschein isolieren und Lager ausbauen.",
                "Lager tauschen und Schmierstoff erneuern.",
                "Nachschmierintervall im CMMS korrigieren.",
            ],
            "language": "de",
        }
    )


def _make_test_client() -> TestClient:
    """Create a TestClient with all heavy dependencies mocked."""
    en_report = _make_rca_report()
    de_report = _make_de_report(en_report)

    mock_detector = MagicMock()
    mock_detector.threshold = 0.994
    mock_detector.detect.return_value = None   # no anomaly from detector itself

    mock_agent = MagicMock()
    mock_agent.analyze = AsyncMock(return_value=en_report)

    mock_report_gen = MagicMock()
    mock_report_gen.generate = AsyncMock(return_value=de_report)
    mock_report_gen.generate_bilingual = AsyncMock(return_value=(en_report, de_report))

    import api.main as main_module

    with (
        patch.object(main_module, "_detector", mock_detector),
        patch.object(main_module, "_rca_agent", mock_agent),
        patch.object(main_module, "_report_gen", mock_report_gen),
        patch.object(main_module, "_analysis_history", []),
        # Prevent real startup loading
        patch("api.main.AnomalyDetector", return_value=mock_detector),
        patch("api.main.RCAAgent", return_value=mock_agent),
        patch("api.main.ReportGenerator", return_value=mock_report_gen),
        patch("api.main._check_postgres", return_value=True),
        patch("api.main._check_chromadb", return_value=True),
    ):
        from api.main import app
        client = TestClient(app, raise_server_exceptions=True)
        # Store mocks on client so tests can inspect them
        client._mock_agent = mock_agent
        client._mock_report_gen = mock_report_gen
        client._en_report = en_report
        client._de_report = de_report
        return client


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200_with_model_loaded(self) -> None:
        """GET /health must return 200 and model_loaded=True when detector is available."""
        import api.main as main_module

        mock_detector = MagicMock()
        mock_detector.threshold = 0.994

        with (
            patch.object(main_module, "_detector", mock_detector),
            patch("api.main._check_postgres", return_value=True),
            patch("api.main._check_chromadb", return_value=True),
        ):
            from api.main import app
            client = TestClient(app)
            response = client.get("/health")

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert data["model_loaded"] is True, "model_loaded must be True"
        assert data["status"] == "ok"
        assert "timestamp" in data

    def test_health_degraded_when_model_not_loaded(self) -> None:
        """GET /health returns status='degraded' and model_loaded=False when no detector."""
        import api.main as main_module

        with (
            patch.object(main_module, "_detector", None),
            patch("api.main._check_postgres", return_value=False),
            patch("api.main._check_chromadb", return_value=True),
        ):
            from api.main import app
            client = TestClient(app)
            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is False
        assert data["status"] == "degraded"
        assert data["db_connected"] is False

    def test_health_response_has_required_fields(self) -> None:
        """Health response must contain all HealthResponse fields."""
        import api.main as main_module

        with (
            patch.object(main_module, "_detector", MagicMock()),
            patch("api.main._check_postgres", return_value=True),
            patch("api.main._check_chromadb", return_value=True),
        ):
            from api.main import app
            client = TestClient(app)
            response = client.get("/health")

        data = response.json()
        for field in ("status", "model_loaded", "db_connected", "chromadb_accessible", "timestamp"):
            assert field in data, f"Missing field: {field}"


class TestSimulateEndpoint:
    """Tests for POST /simulate."""

    def test_simulate_returns_valid_rca_report_structure(self) -> None:
        """POST /simulate must return a dict with en_report containing RCAReport fields."""
        import api.main as main_module

        en_report = _make_rca_report(sensor_id="PUMP_01", severity=0.85)
        de_report = _make_de_report(en_report)

        mock_agent = MagicMock()
        mock_agent.analyze = AsyncMock(return_value=en_report)
        mock_report_gen = MagicMock()
        mock_report_gen.generate_bilingual = AsyncMock(return_value=(en_report, de_report))
        mock_report_gen.generate = AsyncMock(return_value=de_report)

        with (
            patch.object(main_module, "_detector", MagicMock()),
            patch.object(main_module, "_rca_agent", mock_agent),
            patch.object(main_module, "_report_gen", mock_report_gen),
            patch.object(main_module, "_analysis_history", []),
        ):
            from api.main import app
            client = TestClient(app)
            response = client.post(
                "/simulate",
                json={
                    "sensor_id": "PUMP_01",
                    "anomaly_type": "bearing_wear",
                    "severity": 0.85,
                    "language": "en",
                },
            )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()

        assert "en_report" in data, "Response must have 'en_report' key"
        en = data["en_report"]
        for field in ("sensor_id", "severity", "anomaly_summary", "root_cause",
                      "similar_incidents", "recommended_actions", "escalate", "sources",
                      "generation_time_seconds", "language"):
            assert field in en, f"en_report missing field: {field}"

        assert en["sensor_id"] == "PUMP_01"
        assert 0.0 <= en["severity"] <= 1.0
        assert isinstance(en["escalate"], bool)
        assert isinstance(en["similar_incidents"], list)
        assert isinstance(en["recommended_actions"], list)

    def test_simulate_both_languages_returns_en_and_de(self) -> None:
        """POST /simulate with language='both' must return both en_report and de_report."""
        import api.main as main_module

        en_report = _make_rca_report(sensor_id="MOTOR_01", severity=0.70)
        de_report = _make_de_report(en_report)

        mock_agent = MagicMock()
        mock_agent.analyze = AsyncMock(return_value=en_report)
        mock_report_gen = MagicMock()
        mock_report_gen.generate_bilingual = AsyncMock(return_value=(en_report, de_report))

        with (
            patch.object(main_module, "_detector", MagicMock()),
            patch.object(main_module, "_rca_agent", mock_agent),
            patch.object(main_module, "_report_gen", mock_report_gen),
            patch.object(main_module, "_analysis_history", []),
        ):
            from api.main import app
            client = TestClient(app)
            response = client.post(
                "/simulate",
                json={
                    "sensor_id": "MOTOR_01",
                    "anomaly_type": "overload",
                    "severity": 0.70,
                    "language": "both",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "en_report" in data, "Must have en_report for language='both'"
        assert "de_report" in data, "Must have de_report for language='both'"
        assert data["en_report"]["language"] == "en"
        assert data["de_report"]["language"] == "de"

    def test_simulate_escalate_true_for_high_severity(self) -> None:
        """Simulate with severity=0.9 must produce escalate=True in the report."""
        import api.main as main_module

        en_report = _make_rca_report(sensor_id="PUMP_02", severity=0.90)
        de_report = _make_de_report(en_report)

        mock_agent = MagicMock()
        mock_agent.analyze = AsyncMock(return_value=en_report)
        mock_report_gen = MagicMock()
        mock_report_gen.generate_bilingual = AsyncMock(return_value=(en_report, de_report))
        mock_report_gen.generate = AsyncMock(return_value=de_report)

        with (
            patch.object(main_module, "_detector", MagicMock()),
            patch.object(main_module, "_rca_agent", mock_agent),
            patch.object(main_module, "_report_gen", mock_report_gen),
            patch.object(main_module, "_analysis_history", []),
        ):
            from api.main import app
            client = TestClient(app)
            response = client.post(
                "/simulate",
                json={
                    "sensor_id": "PUMP_02",
                    "anomaly_type": "overload",
                    "severity": 0.90,
                    "language": "en",
                },
            )

        assert response.status_code == 200
        assert response.json()["en_report"]["escalate"] is True


class TestHistoryEndpoint:
    """Tests for GET /history."""

    def test_history_returns_list(self) -> None:
        """GET /history must return a dict with 'analyses' key containing a list."""
        import api.main as main_module

        with patch.object(main_module, "_analysis_history", []):
            from api.main import app
            client = TestClient(app)
            response = client.get("/history")

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "analyses" in data, "Response must have 'analyses' key"
        assert isinstance(data["analyses"], list), "'analyses' must be a list"
        assert "count" in data, "Response must have 'count' key"

    def test_history_accumulates_after_simulate(self) -> None:
        """After a /simulate call the /history endpoint must reflect it."""
        import api.main as main_module

        history_store: list = []
        en_report = _make_rca_report()
        de_report = _make_de_report(en_report)

        mock_agent = MagicMock()
        mock_agent.analyze = AsyncMock(return_value=en_report)
        mock_report_gen = MagicMock()
        mock_report_gen.generate = AsyncMock(return_value=de_report)
        mock_report_gen.generate_bilingual = AsyncMock(return_value=(en_report, de_report))

        with (
            patch.object(main_module, "_detector", MagicMock()),
            patch.object(main_module, "_rca_agent", mock_agent),
            patch.object(main_module, "_report_gen", mock_report_gen),
            patch.object(main_module, "_analysis_history", history_store),
        ):
            from api.main import app
            client = TestClient(app)

            # Simulate one call
            client.post(
                "/simulate",
                json={"sensor_id": "PUMP_01", "anomaly_type": "bearing_wear",
                      "severity": 0.7, "language": "en"},
            )

            # Now check history
            response = client.get("/history")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] >= 1, "History must have at least 1 entry after simulate"

    def test_history_reverse_chronological_order(self) -> None:
        """GET /history must return analyses newest-first."""
        import api.main as main_module

        history_store: list = []
        en_report_1 = _make_rca_report(sensor_id="PUMP_01")
        en_report_2 = _make_rca_report(sensor_id="MOTOR_01")
        de_report_1 = _make_de_report(en_report_1)
        de_report_2 = _make_de_report(en_report_2)

        mock_agent = MagicMock()
        mock_report_gen = MagicMock()
        mock_report_gen.generate = AsyncMock(side_effect=[de_report_1, de_report_2])
        mock_report_gen.generate_bilingual = AsyncMock(
            side_effect=[(en_report_1, de_report_1), (en_report_2, de_report_2)]
        )

        with (
            patch.object(main_module, "_detector", MagicMock()),
            patch.object(main_module, "_rca_agent", mock_agent),
            patch.object(main_module, "_report_gen", mock_report_gen),
            patch.object(main_module, "_analysis_history", history_store),
        ):
            from api.main import app
            client = TestClient(app)

            mock_agent.analyze = AsyncMock(return_value=en_report_1)
            client.post("/simulate", json={
                "sensor_id": "PUMP_01", "anomaly_type": "bearing_wear",
                "severity": 0.7, "language": "en"
            })
            mock_agent.analyze = AsyncMock(return_value=en_report_2)
            client.post("/simulate", json={
                "sensor_id": "MOTOR_01", "anomaly_type": "overload",
                "severity": 0.6, "language": "en"
            })

            response = client.get("/history")

        assert response.status_code == 200
        analyses = response.json()["analyses"]
        assert len(analyses) >= 2
        # Most recent simulate (MOTOR_01) must appear first
        assert analyses[0]["sensor_id"] == "MOTOR_01", (
            "Newest entry (MOTOR_01) must be first in reverse chronological order"
        )


class TestAnalyzeEndpoint:
    """Tests for POST /analyze."""

    def test_analyze_with_sensor_id_returns_200(self) -> None:
        """POST /analyze with valid sensor_id must return 200 and en_report."""
        import api.main as main_module

        en_report = _make_rca_report(sensor_id="PUMP_03", severity=0.75)
        de_report = _make_de_report(en_report)

        mock_agent = MagicMock()
        mock_agent.analyze = AsyncMock(return_value=en_report)
        mock_report_gen = MagicMock()
        mock_report_gen.generate = AsyncMock(return_value=de_report)
        mock_report_gen.generate_bilingual = AsyncMock(return_value=(en_report, de_report))

        with (
            patch.object(main_module, "_detector", MagicMock()),
            patch.object(main_module, "_rca_agent", mock_agent),
            patch.object(main_module, "_report_gen", mock_report_gen),
            patch.object(main_module, "_analysis_history", []),
        ):
            from api.main import app
            client = TestClient(app)
            response = client.post(
                "/analyze",
                json={
                    "sensor_id": "PUMP_03",
                    "anomaly_type": "bearing_wear",
                    "severity": 0.75,
                    "language": "en",
                },
            )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        assert "en_report" in data, "Response must have en_report"
        assert data["en_report"]["sensor_id"] == "PUMP_03"

    def test_analyze_response_includes_sensor_metadata(self) -> None:
        """POST /analyze response must echo back sensor_id, anomaly_type, timestamp."""
        import api.main as main_module

        en_report = _make_rca_report(sensor_id="PUMP_01")
        de_report = _make_de_report(en_report)

        mock_agent = MagicMock()
        mock_agent.analyze = AsyncMock(return_value=en_report)
        mock_report_gen = MagicMock()
        mock_report_gen.generate_bilingual = AsyncMock(return_value=(en_report, de_report))

        with (
            patch.object(main_module, "_detector", MagicMock()),
            patch.object(main_module, "_rca_agent", mock_agent),
            patch.object(main_module, "_report_gen", mock_report_gen),
            patch.object(main_module, "_analysis_history", []),
        ):
            from api.main import app
            client = TestClient(app)
            response = client.post(
                "/analyze",
                json={
                    "sensor_id": "PUMP_01",
                    "anomaly_type": "bearing_wear",
                    "severity": 0.7,
                    "language": "both",
                },
            )

        data = response.json()
        assert data["sensor_id"] == "PUMP_01"
        assert data["anomaly_type"] == "bearing_wear"
        assert "timestamp" in data
        assert "analysis_time_seconds" in data


class TestSimulateAllAnomalyTypes:
    """Test /simulate endpoint with all supported anomaly types."""

    def _run_simulate(self, anomaly_type: str) -> dict:
        import api.main as main_module

        severity = 0.75
        en_report = _make_rca_report(sensor_id="PUMP_01", anomaly_type=anomaly_type, severity=severity)
        de_report = _make_de_report(en_report)

        mock_agent = MagicMock()
        mock_agent.analyze = AsyncMock(return_value=en_report)
        mock_report_gen = MagicMock()
        mock_report_gen.generate = AsyncMock(return_value=de_report)
        mock_report_gen.generate_bilingual = AsyncMock(return_value=(en_report, de_report))

        with (
            patch.object(main_module, "_detector", MagicMock()),
            patch.object(main_module, "_rca_agent", mock_agent),
            patch.object(main_module, "_report_gen", mock_report_gen),
            patch.object(main_module, "_analysis_history", []),
        ):
            from api.main import app
            client = TestClient(app)
            response = client.post(
                "/simulate",
                json={"sensor_id": "PUMP_01", "anomaly_type": anomaly_type, "severity": severity, "language": "en"},
            )
        assert response.status_code == 200, f"Failed for anomaly_type={anomaly_type}: {response.text}"
        return response.json()

    def test_simulate_bearing_wear(self) -> None:
        """POST /simulate with anomaly_type='bearing_wear' must return 200."""
        data = self._run_simulate("bearing_wear")
        assert "en_report" in data

    def test_simulate_pressure_drop(self) -> None:
        """POST /simulate with anomaly_type='pressure_drop' must return 200."""
        data = self._run_simulate("pressure_drop")
        assert "en_report" in data

    def test_simulate_overload(self) -> None:
        """POST /simulate with anomaly_type='overload' must return 200."""
        data = self._run_simulate("overload")
        assert "en_report" in data
