from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class ExecutionResult:
    success: bool
    output_df: Optional[pd.DataFrame]
    error: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    data_path: str = ""
    bindings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InputBinding:
    """
    Explicit contract between prompt variables and sandbox paths.

    kind="file": variable is a CSV file path, suitable for pd.read_csv(DATA_PATH).
    kind="directory": variable is a directory path, suitable for os.path.join(DATA_PATH, "clean.csv").
    kind="dataframe": variable is a loaded pandas DataFrame object.
    """

    name: str
    kind: str
    path: Optional[str] = None
    files: Optional[Dict[str, pd.DataFrame]] = None
    data: Optional[pd.DataFrame] = None
    filename: str = "data.csv"

    def validate(self) -> None:
        if self.kind not in {"file", "directory", "dataframe"}:
            raise ValueError(f"Unsupported binding kind for {self.name}: {self.kind}")
        if self.kind in {"file", "directory"} and not self.path and not self.files and self.data is None:
            raise ValueError(f"Binding {self.name} requires path, files, or data")
        if self.kind == "dataframe" and self.data is None and not self.path:
            raise ValueError(f"DataFrame binding {self.name} requires data or path")


def directory_binding(name: str, path: str) -> InputBinding:
    return InputBinding(name=name, kind="directory", path=path)


def file_binding(name: str, path: str) -> InputBinding:
    return InputBinding(name=name, kind="file", path=path)


def dataframe_binding(name: str, data: pd.DataFrame, filename: str = "data.csv") -> InputBinding:
    return InputBinding(name=name, kind="dataframe", data=data, filename=filename)


def execution_to_dict(result: ExecutionResult) -> Dict[str, Any]:
    return {
        "success": result.success,
        "error": result.error,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "data_path": result.data_path,
        "bindings": result.bindings,
        "rows": None if result.output_df is None else len(result.output_df),
        "columns": [] if result.output_df is None else list(result.output_df.columns),
    }
