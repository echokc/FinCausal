from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class UniverseFixture:
    name: str
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
