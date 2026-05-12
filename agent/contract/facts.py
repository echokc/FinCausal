from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agent.schemas import Intent


@dataclass(frozen=True)
class IntentFacts:
    """Structured facts extracted from an Intent — no policy, just detection."""

    time_column: str | None
    is_temporal: bool
    output_variable_name: str
    output_kind: str
    semantic: str
    domain_hints: list[str] = field(default_factory=list)


def infer_intent_facts(intent: Intent) -> IntentFacts:
    """Extract structured facts from a user Intent.

    Pure extraction — no side effects, no knowledge retrieval, no policy decisions.
    Later this function can be swapped for an LLM-based extractor producing the
    same IntentFacts shape.
    """
    time_column = _infer_time_column_from_intent(intent)
    output_kind = _infer_output_kind(intent)
    return IntentFacts(
        time_column=time_column,
        is_temporal=_is_temporal(intent, time_column),
        output_variable_name=_infer_output_name(intent),
        output_kind=output_kind,
        semantic=intent.requested_output,
        domain_hints=_infer_domain_hints(intent, time_column),
    )


def _infer_time_column_from_intent(intent: Intent) -> str | None:
    metadata = intent.metadata or {}
    for key in ("time_column", "time_col", "timestamp_col", "date_col"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value

    text = _intent_text(intent)
    candidates = [
        ("timestamp", ("timestamp", "minute", "intraday", "event time")),
        ("date", ("date", "daily", "day")),
        ("time", ("time", "temporal", "time-indexed", "time series")),
    ]
    for column, needles in candidates:
        if any(needle in text for needle in needles):
            return column
    return None


def _infer_output_name(intent: Intent) -> str:
    metadata = intent.metadata or {}
    for key in ("output_variable_name", "output_name", "variable_name"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value

    text = intent.requested_output.lower()
    if "dataframe" in text or "data frame" in text:
        return "result_df"
    if "series" in text or "signal" in text:
        return "signal"
    return "result"


def _infer_output_kind(intent: Intent) -> str:
    metadata = intent.metadata or {}
    value = metadata.get("output_kind")
    if isinstance(value, str) and value:
        return value

    text = intent.requested_output.lower()
    if "dataframe" in text or "data frame" in text:
        return "dataframe"
    if "series" in text or "signal" in text:
        return "series"
    if "dict" in text:
        return "dict"
    if "vector" in text or "weights" in text:
        return "vector"
    return "scalar"


def _is_temporal(intent: Intent, time_column: str | None) -> bool:
    if time_column:
        return True
    return _intent_mentions(
        intent,
        (
            "time",
            "timestamp",
            "date",
            "temporal",
            "rolling",
            "online",
            "point-in-time",
            "look-ahead",
            "lookahead",
            "future",
        ),
    )


def _infer_domain_hints(intent: Intent, time_column: str | None) -> list[str]:
    """Detect keywords that hint at specific hazards or domains."""
    hints: list[str] = []
    if _intent_mentions(intent, ("regime", "structural break", "state label")):
        hints.append("regime")
    if _intent_mentions(intent, ("risk", "tail", "drawdown", "var", "cvar", "leverage")):
        hints.append("risk")
    if time_column or _is_temporal(intent, time_column):
        hints.append("temporal")
    return hints


def _intent_mentions(intent: Intent, needles: tuple[str, ...]) -> bool:
    text = _intent_text(intent)
    for needle in needles:
        if " " in needle or "-" in needle:
            if needle in text:
                return True
            continue
        if re.search(rf"\b{re.escape(needle)}\b", text):
            return True
    return False


def _intent_text(intent: Intent) -> str:
    values = [
        intent.task_type,
        intent.domain,
        intent.input_data_description,
        intent.requested_output,
        " ".join(intent.explicit_user_constraints),
        " ".join(f"{key} {value}" for key, value in intent.metadata.items()),
    ]
    return " ".join(str(value) for value in values).lower()
