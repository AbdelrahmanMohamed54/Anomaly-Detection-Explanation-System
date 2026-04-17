"""
Bilingual EN/DE RCA report generator.

Session 3 stub — returns the RCAReport as-is with the requested language tag.
Full bilingual LLM-based implementation is delivered in Session 4.

Interface contract (do not change):
    generate_report(report: RCAReport, language: str) -> RCAReport
"""

from __future__ import annotations

import logging

from agent.schemas import RCAReport

logger = logging.getLogger(__name__)


def generate_report(report: RCAReport, language: str = "en") -> RCAReport:
    """Format an RCAReport for the requested language.

    Session 3 stub: copies the report and sets the language tag.
    Session 4 will replace this with LLM-based bilingual translation.

    Args:
        report:   The English-language RCAReport produced by RCAAgent.analyze().
        language: Target language code — 'en' (English) or 'de' (German).

    Returns:
        RCAReport with the language field set to *language*.
    """
    if language not in ("en", "de"):
        logger.warning("generate_report: unknown language '%s', defaulting to 'en'.", language)
        language = "en"

    localised = report.model_copy(update={"language": language})
    logger.info("generate_report: produced %s report for sensor %s", language, report.sensor_id)
    return localised
