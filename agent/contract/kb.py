from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.contract.facts import IntentFacts
    from agent.schemas import Intent


@dataclass(frozen=True)
class ContractKnowledge:
    """Retrieved knowledge — IDs from the local knowledge base."""

    hazard_ids: list[str] = field(default_factory=list)


def retrieve_contract_knowledge(
    intent: Intent,
    facts: IntentFacts,  # noqa: ARG001
    *,
    use_local_knowledge: bool = True,
) -> ContractKnowledge:
    """Retrieve hazard IDs from the local knowledge base.

    Always includes the base execution-hazard set. When use_local_knowledge=True,
    also queries the local KB for task-specific hazards.

    This is a thin wrapper. Later the KB backend can be replaced (vector DB, LLM)
    without changing the ContractKnowledge return type.
    """
    hazard_ids: set[str] = {"hardcoded_absolute_path", "forbidden_optional_import"}

    if use_local_knowledge:
        from agent.retrieval import retrieve_hazard_ids_for_intent

        kb_ids = retrieve_hazard_ids_for_intent(intent, severity=["high", "critical"])
        hazard_ids.update(kb_ids)

    return ContractKnowledge(hazard_ids=sorted(hazard_ids))
