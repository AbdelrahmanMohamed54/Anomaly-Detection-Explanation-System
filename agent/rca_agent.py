"""
LangChain RCA (Root Cause Analysis) Agent — core of the anomaly explanation system.

The agent orchestrates three tools in sequence:
    1. historical_anomaly_tool   — PostgreSQL: similar past incidents
    2. incident_search_tool      — ChromaDB:   semantically similar incident reports
    3. corrective_action_tool    — ChromaDB:   recommended maintenance procedures

Primary LLM:  Google Gemini 2.5 Flash (via GOOGLE_API_KEY)
Fallback LLM: Groq Llama 3.3 70B   (via GROQ_API_KEY, if Gemini fails)

Usage:
    from agent.rca_agent import RCAAgent
    from model.anomaly_detector import AnomalyEvent

    agent = RCAAgent()
    report = await agent.analyze(anomaly_event)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from agent.report_generator import generate_report
from agent.schemas import RCAReport, RCAStructuredOutput
from agent.tools.action_recommender import corrective_action_tool
from agent.tools.historical_query import historical_anomaly_tool
from agent.tools.semantic_search import incident_search_tool
from model.anomaly_detector import AnomalyEvent

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

ESCALATION_SEVERITY_THRESHOLD: float = 0.8

SYSTEM_PROMPT: str = """You are an expert manufacturing engineer specialising in root cause \
analysis (RCA) for rotating equipment and process systems. You are precise, methodical, \
and base all conclusions on evidence gathered from your tools.

When an anomaly is reported to you, you MUST use all three tools in this order before \
drawing any conclusions:
  1. historical_anomaly_tool  — query the structured database for similar past events
  2. incident_search_tool     — search incident reports for matching symptom patterns
  3. corrective_action_tool   — retrieve maintenance procedures relevant to the root cause

Your final response MUST be structured and include:
  - anomaly_summary: a plain-language paragraph describing the detected anomaly
  - root_cause: the most probable root cause with specific component-level detail
  - similar_incidents: up to 3 one-sentence summaries of the most relevant past incidents
  - recommended_actions: an ordered list of corrective actions (most urgent first)
  - escalate: true only if the anomaly poses an immediate safety or production risk
  - sources: list of tools consulted

Always cite which tool result informed each part of your analysis. Be concise and \
actionable — your report will be read by maintenance technicians on the shop floor."""


# ── RCA Agent ─────────────────────────────────────────────────────────────────


class RCAAgent:
    """LangChain agent that performs root cause analysis for detected anomalies.

    Attributes:
        _llm:         The primary language model (Gemini 2.5 Flash).
        _tools:       The three LangChain tools available to the agent.
        _agent:       The compiled LangGraph agent graph.
    """

    def __init__(self) -> None:
        """Initialise LLM, tools, and the agent graph.

        Raises:
            EnvironmentError: If neither GOOGLE_API_KEY nor GROQ_API_KEY is set.
        """
        self._llm = self._build_llm()
        self._tools = [
            historical_anomaly_tool,
            incident_search_tool,
            corrective_action_tool,
        ]
        self._agent = create_agent(
            model=self._llm,
            tools=self._tools,
            system_prompt=SYSTEM_PROMPT,
            response_format=RCAStructuredOutput,
        )
        logger.info("RCAAgent initialised with %d tools.", len(self._tools))

    # ── Public API ────────────────────────────────────────────────────────────

    async def analyze(self, anomaly: AnomalyEvent) -> RCAReport:
        """Run the full RCA workflow for a detected anomaly.

        Workflow:
            1. Build a structured prompt from the AnomalyEvent
            2. Invoke the agent — it autonomously calls all 3 tools
            3. Extract the structured RCAStructuredOutput from the agent result
            4. Apply severity-based escalation override
            5. Call report_generator to produce the English-language RCAReport

        Args:
            anomaly: AnomalyEvent from AnomalyDetector.detect()

        Returns:
            RCAReport with all fields populated.
        """
        start_time = time.monotonic()
        prompt = self._build_prompt(anomaly)

        logger.info(
            "RCAAgent.analyze: starting for sensor=%s type=%s severity=%.3f",
            anomaly.sensor_id,
            anomaly.anomaly_type,
            anomaly.severity,
        )

        structured_output: RCAStructuredOutput = await self._run_agent(prompt)

        # Override escalation if severity exceeds hard threshold
        escalate = structured_output.escalate or (
            anomaly.severity >= ESCALATION_SEVERITY_THRESHOLD
        )

        elapsed = round(time.monotonic() - start_time, 3)

        rca_report = RCAReport(
            sensor_id=anomaly.sensor_id,
            severity=anomaly.severity,
            anomaly_summary=structured_output.anomaly_summary,
            root_cause=structured_output.root_cause,
            similar_incidents=structured_output.similar_incidents[:3],
            recommended_actions=structured_output.recommended_actions,
            escalate=escalate,
            sources=structured_output.sources,
            generation_time_seconds=elapsed,
            language="en",
        )

        logger.info(
            "RCAAgent.analyze: completed in %.2fs — escalate=%s",
            elapsed,
            escalate,
        )

        return generate_report(rca_report, language="en")

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _run_agent(self, prompt: str) -> RCAStructuredOutput:
        """Invoke the LangGraph agent and extract the structured output.

        Args:
            prompt: Human-readable analysis request built from the AnomalyEvent.

        Returns:
            RCAStructuredOutput with fields populated by the LLM.

        Raises:
            RuntimeError: If the agent produces no structured response.
        """
        result = await self._agent.ainvoke(
            {"messages": [HumanMessage(content=prompt)]}
        )

        # LangChain 1.x create_agent with response_format stores the structured
        # response in result['structured_response'].
        structured = result.get("structured_response")
        if isinstance(structured, RCAStructuredOutput):
            return structured

        # Fallback: if structured output is missing, synthesise a minimal response
        # from the last assistant message to avoid crashing downstream.
        logger.warning(
            "_run_agent: no structured_response in agent result — using fallback."
        )
        messages = result.get("messages", [])
        last_content = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content:
                last_content = str(msg.content)
                break

        return RCAStructuredOutput(
            anomaly_summary=last_content[:500] or "Anomaly detected — see raw output.",
            root_cause="Root cause could not be structured automatically.",
            similar_incidents=[],
            recommended_actions=["Inspect equipment manually."],
            escalate=True,
            sources=["historical_anomaly_tool", "incident_search_tool",
                     "corrective_action_tool"],
        )

    @staticmethod
    def _build_prompt(anomaly: AnomalyEvent) -> str:
        """Format an AnomalyEvent into a structured analysis prompt.

        Args:
            anomaly: The detected anomaly to analyse.

        Returns:
            Multi-line string prompt for the agent.
        """
        values_str = "\n".join(
            f"    {k}: {v:.3f}" for k, v in anomaly.detected_values.items()
        )
        return (
            f"ANOMALY DETECTED — PLEASE PERFORM ROOT CAUSE ANALYSIS\n\n"
            f"Sensor ID:       {anomaly.sensor_id}\n"
            f"Anomaly Type:    {anomaly.anomaly_type}\n"
            f"Severity Score:  {anomaly.severity:.3f} (1.0 = critical)\n"
            f"Timestamp:       {anomaly.timestamp}\n"
            f"Reconstruction Error: {anomaly.reconstruction_error:.4f}\n\n"
            f"Detected Sensor Values:\n{values_str}\n\n"
            "Please investigate this anomaly using all three tools in order:\n"
            "1. Query historical anomalies for this sensor and anomaly type\n"
            "2. Search incident reports for similar symptom patterns\n"
            "3. Retrieve corrective actions based on identified root cause\n\n"
            "Then provide your structured RCA report."
        )

    @staticmethod
    def _build_llm() -> ChatGoogleGenerativeAI | ChatGroq:
        """Initialise the LLM — Gemini primary, Groq fallback.

        Returns:
            A LangChain chat model instance.

        Raises:
            EnvironmentError: If neither API key is configured.
        """
        google_key = os.getenv("GOOGLE_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")

        if google_key:
            logger.info("RCAAgent: using Gemini 2.5 Flash as primary LLM.")
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=google_key,
                temperature=0.1,
            )
        if groq_key:
            logger.warning("RCAAgent: GOOGLE_API_KEY not set — falling back to Groq.")
            return ChatGroq(
                model="llama-3.3-70b-versatile",
                groq_api_key=groq_key,
                temperature=0.1,
            )
        raise EnvironmentError(
            "No LLM API key found. Set GOOGLE_API_KEY or GROQ_API_KEY in .env"
        )


# ── Convenience sync wrapper (for non-async callers) ─────────────────────────


def analyze_sync(anomaly: AnomalyEvent) -> RCAReport:
    """Synchronous wrapper around RCAAgent.analyze() for use in scripts/tests.

    Args:
        anomaly: The detected anomaly to analyse.

    Returns:
        RCAReport with all fields populated.
    """
    agent = RCAAgent()
    return asyncio.run(agent.analyze(anomaly))
