from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from agent.schemas import Intent, ToolResult


KNOWLEDGE_ROOT = Path(__file__).with_name("knowledge")


def retrieve_causal_principles(
    query: str,
    *,
    domain: str | None = None,
    causal_pillars: list[str] | None = None,
    top_k: int = 5,
) -> ToolResult:
    items = _load_json("causal_principles.json")
    filtered = _filter_items(items, domain=domain, pillars=causal_pillars)
    ranked = _rank_items(filtered, query)[:top_k]
    return ToolResult.success(
        {"items": ranked},
        metadata={"tool_name": "retrieve_causal_principles", "top_k": top_k},
    )


def retrieve_hazard_definitions(
    query: str = "",
    *,
    pillar: str | None = None,
    severity: list[str] | None = None,
    hazard_family: str | None = None,
    top_k: int = 5,
) -> ToolResult:
    items = _load_json("hazard_definitions.json")
    if pillar:
        items = [item for item in items if _normalize(item.get("pillar")) == _normalize(pillar)]
    if severity:
        allowed = {_normalize(value) for value in severity}
        items = [item for item in items if _normalize(item.get("severity")) in allowed]
    if hazard_family:
        query = f"{query} {hazard_family}".strip()
    ranked = _rank_items(items, query)[:top_k]
    return ToolResult.success(
        {"items": ranked},
        metadata={"tool_name": "retrieve_hazard_definitions", "top_k": top_k},
    )


def retrieve_invariant_definitions(query: str = "", *, top_k: int = 5) -> ToolResult:
    ranked = _rank_items(_load_json("invariant_definitions.json"), query)[:top_k]
    return ToolResult.success(
        {"items": ranked},
        metadata={"tool_name": "retrieve_invariant_definitions", "top_k": top_k},
    )


def retrieve_similar_contracts(
    intent: Intent,
    *,
    strategy_type: str | None = None,
    asset_class: str | None = None,
    data_frequency: str | None = None,
    top_k: int = 3,
) -> ToolResult:
    items = _load_json("similar_contracts.json")
    if strategy_type:
        items = [item for item in items if _normalize(item.get("strategy_type")) == _normalize(strategy_type)]
    if asset_class:
        items = [item for item in items if _normalize(item.get("asset_class")) == _normalize(asset_class)]
    if data_frequency:
        items = [item for item in items if _normalize(item.get("data_frequency")) == _normalize(data_frequency)]

    query = " ".join(
        [
            intent.domain,
            intent.input_data_description,
            intent.requested_output,
            " ".join(intent.explicit_user_constraints),
        ]
    )
    ranked = _rank_items(items, query)[:top_k]
    return ToolResult.success(
        {"items": ranked},
        metadata={"tool_name": "retrieve_similar_contracts", "top_k": top_k},
    )


def retrieve_hazard_ids_for_intent(intent: Intent, *, severity: list[str] | None = None, top_k: int = 8) -> list[str]:
    query = " ".join(
        [
            intent.domain,
            intent.input_data_description,
            intent.requested_output,
            " ".join(intent.explicit_user_constraints),
            " ".join(f"{key} {value}" for key, value in intent.metadata.items()),
        ]
    )
    result = retrieve_hazard_definitions(query, severity=severity, top_k=top_k)
    if not result.ok:
        return []
    return [item["id"] for item in result.result["items"]]


@lru_cache(maxsize=None)
def _load_json(filename: str) -> list[dict[str, Any]]:
    path = KNOWLEDGE_ROOT / filename
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _filter_items(
    items: list[dict[str, Any]],
    *,
    domain: str | None = None,
    pillars: list[str] | None = None,
) -> list[dict[str, Any]]:
    if domain:
        items = [item for item in items if _normalize(item.get("domain")) == _normalize(domain)]
    if pillars:
        allowed = {_normalize(item) for item in pillars}
        items = [item for item in items if _normalize(item.get("pillar")) in allowed]
    return items


def _rank_items(items: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    query_tokens = set(_tokens(query))

    def score(item: dict[str, Any]) -> tuple[int, str]:
        item_text = " ".join(
            str(value)
            for key, value in item.items()
            if key not in {"known_hazards", "required_invariants"}
        )
        item_tokens = set(_tokens(item_text))
        keyword_tokens = set(_tokens(" ".join(item.get("keywords", []))))
        overlap = len(query_tokens & item_tokens) + (2 * len(query_tokens & keyword_tokens))
        return (-overlap, str(item.get("id", "")))

    return sorted(items, key=score)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower()
