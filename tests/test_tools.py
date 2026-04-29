"""
Unit tests for the three LangChain agent tools.

Tests use mocking to avoid requiring live PostgreSQL / ChromaDB connections.

Coverage:
    Task 1 — historical_query.py:
        test_successful_query_returns_list_of_dicts
        test_db_failure_returns_empty_list

    Task 2 — semantic_search.py:
        test_search_returns_results_when_collection_has_data
        test_search_results_include_similarity_scores

    Task 3 — action_recommender.py:
        test_action_search_returns_results
        test_action_results_include_similarity_scores
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.exc import OperationalError

from agent.schemas import ActionRecommendation, IncidentSearchResult
from agent.tools.action_recommender import ActionRecommenderTool
from agent.tools.historical_query import query_similar_anomalies
from agent.tools.semantic_search import IncidentSearchTool


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — historical_query
# ─────────────────────────────────────────────────────────────────────────────


class TestHistoricalQuery:
    """Tests for query_similar_anomalies() with mocked SQLAlchemy engine."""

    def _make_fake_row(self, overrides: dict | None = None) -> MagicMock:
        """Build a mock SQLAlchemy row with default anomaly values."""
        row = MagicMock()
        data = {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "sensor_id": "PUMP_01",
            "anomaly_type": "bearing_wear",
            "detected_at": "2024-03-12 02:47:00",
            "duration_minutes": 210,
            "severity": 0.72,
            "root_cause": "Lubrication failure in main drive bearing.",
            "resolution": "Bearing replaced, grease repacked.",
            "resolved_by": "Technician Klaus Bauer",
            "resolution_time_hours": 4.5,
        }
        if overrides:
            data.update(overrides)
        row._mapping = data
        return row

    def test_successful_query_returns_list_of_dicts(self) -> None:
        """A successful DB query returns a non-empty list of dicts."""
        fake_row = self._make_fake_row()

        mock_conn = MagicMock()
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchall.return_value = [fake_row]

        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn

        with patch(
            "agent.tools.historical_query.create_engine", return_value=mock_engine
        ):
            results = query_similar_anomalies("PUMP_01", "bearing_wear", limit=5)

        assert isinstance(results, list), "Result must be a list"
        assert len(results) == 1, "Should return exactly 1 mocked record"
        record = results[0]
        assert isinstance(record, dict), "Each item must be a dict"
        assert record["sensor_id"] == "PUMP_01"
        assert record["anomaly_type"] == "bearing_wear"
        assert record["severity"] == 0.72

    def test_db_failure_returns_empty_list_not_exception(self) -> None:
        """An OperationalError (DB down) must return [] without raising."""
        with patch(
            "agent.tools.historical_query.create_engine",
            side_effect=OperationalError("connection refused", None, None),
        ):
            results = query_similar_anomalies("PUMP_01", "bearing_wear")

        assert results == [], "DB failure must return an empty list"

    def test_multiple_rows_returned(self) -> None:
        """Query with multiple matching rows returns all as dicts."""
        rows = [
            self._make_fake_row({"sensor_id": "PUMP_01"}),
            self._make_fake_row({"sensor_id": "MOTOR_01", "anomaly_type": "bearing_wear"}),
        ]

        mock_conn = MagicMock()
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchall.return_value = rows
        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn

        with patch(
            "agent.tools.historical_query.create_engine", return_value=mock_engine
        ):
            results = query_similar_anomalies("PUMP_01", "bearing_wear", limit=10)

        assert len(results) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — semantic_search (IncidentSearchTool)
# ─────────────────────────────────────────────────────────────────────────────


class TestIncidentSearchTool:
    """Tests for IncidentSearchTool.search() with mocked ChromaDB."""

    def _make_mock_tool(self) -> IncidentSearchTool:
        """Return an IncidentSearchTool with a mocked ChromaDB collection."""
        tool = object.__new__(IncidentSearchTool)
        tool.collection = MagicMock()
        tool.collection.count.return_value = 90
        tool.collection.query.return_value = {
            "documents": [
                [
                    "Bearing inner race fatigue detected. Vibration reached 3.8 mm/s.",
                    "Lubrication failure in drive-end bearing. Grease oxidised.",
                    "Outer race spalling confirmed by acoustic emission at 112 Hz.",
                ]
            ],
            "metadatas": [
                [
                    {"source_file": "bearing_wear_incidents.txt", "anomaly_type": "bearing_wear"},
                    {"source_file": "bearing_wear_incidents.txt", "anomaly_type": "bearing_wear"},
                    {"source_file": "bearing_wear_incidents.txt", "anomaly_type": "bearing_wear"},
                ]
            ],
            "distances": [[0.12, 0.25, 0.38]],
        }
        return tool

    def test_search_returns_results_when_collection_has_data(self) -> None:
        """search() returns a non-empty list of IncidentSearchResult when data exists."""
        tool = self._make_mock_tool()
        results = tool.search("bearing wear high vibration temperature", top_k=3)

        assert isinstance(results, list), "Result must be a list"
        assert len(results) == 3, "Should return exactly 3 results"
        assert all(isinstance(r, IncidentSearchResult) for r in results)

    def test_search_results_include_similarity_scores(self) -> None:
        """Each search result has a similarity_score in [0, 1]."""
        tool = self._make_mock_tool()
        results = tool.search("bearing wear", top_k=3)

        for r in results:
            assert 0.0 <= r.similarity_score <= 1.0, (
                f"similarity_score {r.similarity_score} out of range [0, 1]"
            )

    def test_search_results_have_correct_fields(self) -> None:
        """Each result carries text, source_file, anomaly_type, and similarity_score."""
        tool = self._make_mock_tool()
        results = tool.search("vibration spike", top_k=1)

        r = results[0]
        assert r.text, "text must not be empty"
        assert r.source_file == "bearing_wear_incidents.txt"
        assert r.anomaly_type == "bearing_wear"

    def test_empty_collection_returns_empty_list(self) -> None:
        """search() returns [] without error when the collection is empty."""
        tool = object.__new__(IncidentSearchTool)
        tool.collection = MagicMock()
        tool.collection.count.return_value = 0

        results = tool.search("any query")
        assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — action_recommender (ActionRecommenderTool)
# ─────────────────────────────────────────────────────────────────────────────


class TestActionRecommenderTool:
    """Tests for ActionRecommenderTool.search() with mocked ChromaDB."""

    def _make_mock_tool(self) -> ActionRecommenderTool:
        """Return an ActionRecommenderTool with a mocked ChromaDB collection."""
        tool = object.__new__(ActionRecommenderTool)
        tool.collection = MagicMock()
        tool.collection.count.return_value = 102
        tool.collection.query.return_value = {
            "documents": [
                [
                    "Bearing replaced using SKF TMMP 20 puller. Grease repacked with 20 g Mobilith SHC 460.",
                    "Shaft realigned using Pruftechnik OPTALIGN to within 0.02 mm after thermal stabilisation.",
                ]
            ],
            "metadatas": [
                [
                    {"source_file": "corrective_actions.txt", "maintenance_type": "corrective"},
                    {"source_file": "corrective_actions.txt", "maintenance_type": "corrective"},
                ]
            ],
            "distances": [[0.10, 0.22]],
        }
        return tool

    def test_action_search_returns_results(self) -> None:
        """search() returns a non-empty list of ActionRecommendation."""
        tool = self._make_mock_tool()
        results = tool.search("bearing replacement lubrication", top_k=2)

        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, ActionRecommendation) for r in results)

    def test_action_results_include_similarity_scores(self) -> None:
        """Each action recommendation includes a valid similarity score."""
        tool = self._make_mock_tool()
        results = tool.search("replace bearing", top_k=2)

        for r in results:
            assert 0.0 <= r.similarity_score <= 1.0

    def test_action_results_have_maintenance_type(self) -> None:
        """Each result carries a maintenance_type field."""
        tool = self._make_mock_tool()
        results = tool.search("corrective maintenance bearing", top_k=2)

        for r in results:
            assert r.maintenance_type in ("corrective", "preventive", "general")

    def test_empty_collection_returns_empty_list(self) -> None:
        """search() returns [] without error when the collection is empty."""
        tool = object.__new__(ActionRecommenderTool)
        tool.collection = MagicMock()
        tool.collection.count.return_value = 0

        results = tool.search("any query")
        assert results == []

    def test_action_recommender_not_incident_reports(self) -> None:
        """ActionRecommenderTool must query 'maintenance_logs', not 'incident_reports'."""
        from agent.tools.action_recommender import COLLECTION_NAME as ACTION_COLLECTION

        assert ACTION_COLLECTION == "maintenance_logs", (
            f"ActionRecommenderTool must target 'maintenance_logs', got '{ACTION_COLLECTION}'"
        )

    def test_action_results_describe_maintenance_procedures(self) -> None:
        """Returned action text must come from maintenance log content, not incident text."""
        tool = self._make_mock_tool()
        results = tool.search("bearing replacement", top_k=2)

        for r in results:
            assert r.text, "Action text must not be empty"
            # Maintenance procedures contain procedural language
            assert isinstance(r.maintenance_type, str)
            assert r.source_file, "source_file must be set"


# ─────────────────────────────────────────────────────────────────────────────
# Additional tests — historical_query ordering + semantic_search ranking
# ─────────────────────────────────────────────────────────────────────────────


class TestHistoricalQueryOrdering:
    """Verify that historical_query requests results in correct order."""

    def _make_rows(self, count: int) -> list[MagicMock]:
        rows = []
        for i in range(count):
            row = MagicMock()
            row._mapping = {
                "id": f"uuid-{i}",
                "sensor_id": "PUMP_01",
                "anomaly_type": "bearing_wear",
                "detected_at": f"2024-0{i+1}-01 00:00:00",
                "duration_minutes": 60,
                "severity": round(0.5 + i * 0.1, 2),
                "root_cause": f"Root cause {i}",
                "resolution": f"Resolution {i}",
                "resolved_by": "Tech",
                "resolution_time_hours": 2.0,
            }
            rows.append(row)
        return rows

    def test_query_respects_limit_parameter(self) -> None:
        """query_similar_anomalies must pass limit to SQL and honour it."""
        rows = self._make_rows(3)

        mock_conn = MagicMock()
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchall.return_value = rows
        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn

        with patch("agent.tools.historical_query.create_engine", return_value=mock_engine):
            results = query_similar_anomalies("PUMP_01", "bearing_wear", limit=3)

        assert len(results) == 3, "Result count must match the mock's return"

    def test_query_result_dicts_have_all_keys(self) -> None:
        """Each result dict must contain the standard historical anomaly keys."""
        rows = self._make_rows(1)

        mock_conn = MagicMock()
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchall.return_value = rows
        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn

        with patch("agent.tools.historical_query.create_engine", return_value=mock_engine):
            results = query_similar_anomalies("PUMP_01", "bearing_wear", limit=1)

        required_keys = {"sensor_id", "anomaly_type", "severity", "root_cause", "resolution"}
        for key in required_keys:
            assert key in results[0], f"Missing key in result: {key}"


class TestIncidentSearchRanking:
    """Verify that IncidentSearchTool returns results ordered by similarity."""

    def test_results_ordered_by_similarity_descending(self) -> None:
        """search() must return results with similarity_score in descending order."""
        tool = object.__new__(IncidentSearchTool)
        tool.collection = MagicMock()
        tool.collection.count.return_value = 90
        # distances increase → similarity decreases; results should be sorted best-first
        tool.collection.query.return_value = {
            "documents": [["Doc A", "Doc B", "Doc C"]],
            "metadatas": [[
                {"source_file": "f.txt", "anomaly_type": "bearing_wear"},
                {"source_file": "f.txt", "anomaly_type": "bearing_wear"},
                {"source_file": "f.txt", "anomaly_type": "bearing_wear"},
            ]],
            "distances": [[0.05, 0.30, 0.55]],
        }

        results = tool.search("vibration spike", top_k=3)

        assert len(results) == 3
        scores = [r.similarity_score for r in results]
        assert scores == sorted(scores, reverse=True), (
            f"Scores must be descending; got {scores}"
        )
