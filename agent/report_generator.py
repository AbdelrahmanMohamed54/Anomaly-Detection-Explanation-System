"""
Bilingual EN/DE RCA report generator.

Provides:
    ReportGenerator          — async class used by the API layer
    generate_report()        — sync wrapper kept for rca_agent.py compatibility

Flow:
    1. rca_agent.analyze()  produces an RCAReport in English
    2. ReportGenerator.generate_bilingual(report.model_dump())
       translates the text fields to German via a second LLM call
    3. Both EN and DE RCAReport objects are returned to the API
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(Path(__file__).parent.parent / ".env")

from agent.schemas import RCAReport

logger = logging.getLogger(__name__)

# ── German severity labels ────────────────────────────────────────────────────

_SEVERITY_LABELS: dict[str, dict[str, str]] = {
    "en": {"HIGH": "HIGH", "MEDIUM": "MEDIUM", "LOW": "LOW"},
    "de": {"HIGH": "HOCH", "MEDIUM": "MITTEL", "LOW": "NIEDRIG"},
}


def severity_label(score: float, language: str = "en") -> str:
    """Convert a 0-1 severity float to a human-readable label.

    Args:
        score:    Severity float in [0, 1].
        language: Output language code ('en' or 'de').

    Returns:
        'HIGH' / 'MEDIUM' / 'LOW' (or German equivalents).
    """
    lang = language.lower()
    labels = _SEVERITY_LABELS.get(lang, _SEVERITY_LABELS["en"])
    if score >= 0.8:
        return labels["HIGH"]
    if score >= 0.5:
        return labels["MEDIUM"]
    return labels["LOW"]


# ── Internal translation schema ───────────────────────────────────────────────


class _TranslatedFields(BaseModel):
    """Structured output schema for the German translation LLM call."""

    anomaly_summary: str
    root_cause: str
    similar_incidents: list[str]
    recommended_actions: list[str]


# ── ReportGenerator class ─────────────────────────────────────────────────────


class ReportGenerator:
    """Formats and optionally translates RCAReport objects.

    Uses the same LLM stack as RCAAgent (Gemini primary, Groq fallback).
    The German translation call uses structured output so field boundaries
    are preserved exactly.
    """

    def __init__(self) -> None:
        """Initialise the LLM for translation."""
        self._llm = self._build_llm()

    async def generate(
        self, rca_result: dict, language: str = "EN"
    ) -> RCAReport:
        """Format a raw RCA result dict into a validated RCAReport.

        For language='EN': validates the dict and returns it unchanged.
        For language='DE': translates all narrative text fields via LLM.

        Args:
            rca_result: Dict representation of an RCAReport (e.g. report.model_dump()).
            language:   Target language — 'EN' or 'DE' (case-insensitive).

        Returns:
            RCAReport with the correct language field set.
        """
        lang = language.upper()
        en_report = RCAReport(**{**rca_result, "language": "en"})

        if lang == "EN":
            return en_report
        if lang == "DE":
            return await self._translate_to_german(en_report)

        logger.warning("generate: unknown language '%s', defaulting to EN.", language)
        return en_report

    async def generate_bilingual(
        self, rca_result: dict
    ) -> tuple[RCAReport, RCAReport]:
        """Produce both English and German versions of an RCA report.

        Args:
            rca_result: Dict representation of an English-language RCAReport.

        Returns:
            Tuple of (en_report, de_report).
        """
        en_report = await self.generate(rca_result, language="EN")
        de_report = await self._translate_to_german(en_report)
        return en_report, de_report

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _translate_to_german(self, report: RCAReport) -> RCAReport:
        """Translate the narrative fields of an EN RCAReport into German.

        Technical identifiers (sensor_id, sources), numeric values (severity,
        generation_time_seconds), and boolean fields (escalate) are kept as-is.

        Args:
            report: English-language RCAReport.

        Returns:
            RCAReport with narrative fields translated to German.
        """
        translate_llm = self._llm.with_structured_output(_TranslatedFields)

        prompt = (
            "Translate the following technical RCA (Root Cause Analysis) report fields "
            "from English to German. Maintain all technical terms in their standard "
            "German engineering equivalents. Preserve the same number of list items. "
            "Return only the translated fields — do not add explanations.\n\n"
            f"anomaly_summary:\n{report.anomaly_summary}\n\n"
            f"root_cause:\n{report.root_cause}\n\n"
            f"similar_incidents (translate each item):\n"
            + "\n".join(f"- {s}" for s in report.similar_incidents)
            + "\n\nrecommended_actions (translate each item):\n"
            + "\n".join(f"- {a}" for a in report.recommended_actions)
        )

        try:
            translated: _TranslatedFields = await translate_llm.ainvoke(prompt)
            logger.info(
                "ReportGenerator: translated report to DE for sensor %s", report.sensor_id
            )
            return report.model_copy(
                update={
                    "anomaly_summary": translated.anomaly_summary,
                    "root_cause": translated.root_cause,
                    "similar_incidents": translated.similar_incidents,
                    "recommended_actions": translated.recommended_actions,
                    "language": "de",
                }
            )

        except Exception as exc:
            logger.error(
                "ReportGenerator: German translation failed — %s. "
                "Returning EN report with language tag overridden to 'de'.",
                exc,
            )
            return report.model_copy(update={"language": "de"})

    @staticmethod
    def _build_llm():
        """Initialise translation LLM — Gemini primary, Groq fallback."""
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_groq import ChatGroq

        google_key = os.getenv("GOOGLE_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")

        if google_key:
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=google_key,
                temperature=0.1,
            )
        if groq_key:
            return ChatGroq(
                model="llama-3.3-70b-versatile",
                groq_api_key=groq_key,
                temperature=0.1,
            )
        raise EnvironmentError(
            "No LLM API key found. Set GOOGLE_API_KEY or GROQ_API_KEY in .env"
        )


# ── Module-level convenience function (backward compat with rca_agent.py) ────


def generate_report(report: RCAReport, language: str = "en") -> RCAReport:
    """Synchronous wrapper — sets the language tag on an existing RCAReport.

    For English: returns the report with language='en' (no-op if already set).
    For German:  schedules an async translation and blocks until complete.
                 Note: will fail if called from within a running event loop;
                 in that case use ReportGenerator.generate() directly.

    Args:
        report:   English-language RCAReport from RCAAgent.analyze().
        language: Target language code — 'en' or 'de'.

    Returns:
        RCAReport with language field set.
    """
    lang = language.lower()
    if lang not in ("en", "de"):
        logger.warning("generate_report: unknown language '%s', defaulting to 'en'.", language)
        lang = "en"

    if lang == "en":
        result = report.model_copy(update={"language": "en"})
        logger.info(
            "generate_report: EN report ready for sensor %s", report.sensor_id
        )
        return result

    # DE — run async translation synchronously
    gen = ReportGenerator()
    try:
        result = asyncio.run(gen._translate_to_german(report))
    except RuntimeError:
        # Already inside an event loop (e.g. called from async context)
        logger.warning(
            "generate_report: cannot call asyncio.run inside an active event loop. "
            "Returning EN report with 'de' language tag."
        )
        result = report.model_copy(update={"language": "de"})

    return result
