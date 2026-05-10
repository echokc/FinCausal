from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class PromptFileSpec:
    variable: str
    filename: str
    columns: List[str]


@dataclass(frozen=True)
class PromptSpec:
    framing: str
    data_path: str
    files: List[PromptFileSpec]
    load_snippet: str
    task_steps: List[str]
    critical_requirements: List[str]
    required_output_columns: List[str]
    output_dataframe_name: Optional[str] = None
