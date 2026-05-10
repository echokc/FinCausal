import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd

from eval.execution.code_extraction import sanitize_llm_code
from eval.execution.execution_models import ExecutionResult, InputBinding, directory_binding


RESULT_KEYS = ("result_df", "merged_df", "output_df", "regime_df")


class LocalSubprocessSandboxExecutor:
    """
    Minimal phase-1 sandbox: executes model code in a separate process with a
    timeout and controlled DATA_PATH. Docker can replace this behind the same
    interface once the image is available in every environment.
    """

    def __init__(self, timeout_seconds: int = 10):
        self.timeout_seconds = timeout_seconds

    def run(self, code: str, data_path: str) -> ExecutionResult:
        return self.run_with_bindings(code, [directory_binding("DATA_PATH", data_path)])

    def run_with_bindings(self, code: str, bindings: Iterable[InputBinding]) -> ExecutionResult:
        clean_code = sanitize_llm_code(code)
        bindings = list(bindings)
        for binding in bindings:
            binding.validate()

        with tempfile.TemporaryDirectory() as tmp_dir:
            prepared = self._prepare_bindings(bindings, Path(tmp_dir))
            runner_path = Path(tmp_dir) / "runner.py"
            output_path = Path(tmp_dir) / "output.csv"
            error_path = Path(tmp_dir) / "error.txt"

            runner_code = self._build_runner(clean_code, prepared, str(output_path), str(error_path))
            runner_path.write_text(runner_code, encoding="utf-8")

            try:
                proc = subprocess.run(
                    [sys.executable, str(runner_path)],
                    cwd=tmp_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    env={**os.environ, **{name: spec["value"] for name, spec in prepared.items()}},
                )
            except subprocess.TimeoutExpired as exc:
                return ExecutionResult(
                    success=False,
                    output_df=None,
                    error=f"Timeout after {self.timeout_seconds}s",
                    stdout=exc.stdout or "",
                    stderr=exc.stderr or "",
                    data_path=prepared.get("DATA_PATH", {}).get("value", ""),
                    bindings=prepared,
                )

            if output_path.exists():
                try:
                    return ExecutionResult(
                        success=True,
                        output_df=pd.read_csv(output_path),
                        stdout=proc.stdout,
                        stderr=proc.stderr,
                        data_path=prepared.get("DATA_PATH", {}).get("value", ""),
                        bindings=prepared,
                    )
                except Exception as exc:
                    return ExecutionResult(
                        success=False,
                        output_df=None,
                        error=f"Could not read output.csv: {type(exc).__name__}: {exc}",
                        stdout=proc.stdout,
                        stderr=proc.stderr,
                        data_path=prepared.get("DATA_PATH", {}).get("value", ""),
                        bindings=prepared,
                    )

            error = error_path.read_text(encoding="utf-8") if error_path.exists() else proc.stderr
            return ExecutionResult(
                success=False,
                output_df=None,
                error=error or "No output DataFrame produced",
                stdout=proc.stdout,
                stderr=proc.stderr,
                data_path=prepared.get("DATA_PATH", {}).get("value", ""),
                bindings=prepared,
            )

    def _prepare_bindings(self, bindings: Iterable[InputBinding], tmp_root: Path) -> Dict[str, Dict[str, Any]]:
        prepared: Dict[str, Dict[str, Any]] = {}
        inputs_root = tmp_root / "inputs"
        inputs_root.mkdir(parents=True, exist_ok=True)

        for binding in bindings:
            binding_root = inputs_root / binding.name
            binding_root.mkdir(parents=True, exist_ok=True)

            if binding.kind == "directory":
                if binding.files:
                    for filename, df in binding.files.items():
                        target = binding_root / filename
                        target.parent.mkdir(parents=True, exist_ok=True)
                        df.to_csv(target, index=False)
                elif binding.path:
                    source = Path(binding.path)
                    if source.is_dir():
                        for child in source.iterdir():
                            target = binding_root / child.name
                            if child.is_dir():
                                shutil.copytree(child, target)
                            else:
                                shutil.copy2(child, target)
                    elif source.is_file():
                        shutil.copy2(source, binding_root / source.name)
                    else:
                        raise FileNotFoundError(f"Directory binding path does not exist: {binding.path}")
                elif binding.data is not None:
                    binding.data.to_csv(binding_root / binding.filename, index=False)
                value = str(binding_root)

            elif binding.kind == "file":
                if binding.data is not None:
                    file_path = binding_root / binding.filename
                    binding.data.to_csv(file_path, index=False)
                elif binding.path:
                    source = Path(binding.path)
                    if source.is_dir():
                        candidates = [p for p in source.iterdir() if p.suffix == ".csv"]
                        if len(candidates) != 1:
                            raise ValueError(
                                f"File binding {binding.name} expected exactly one CSV in {binding.path}; found {len(candidates)}"
                            )
                        source = candidates[0]
                    if not source.exists():
                        raise FileNotFoundError(f"File binding path does not exist: {binding.path}")
                    file_path = binding_root / (binding.filename or source.name)
                    shutil.copy2(source, file_path)
                elif binding.files:
                    if len(binding.files) != 1:
                        raise ValueError(f"File binding {binding.name} requires exactly one file")
                    filename, df = next(iter(binding.files.items()))
                    file_path = binding_root / filename
                    df.to_csv(file_path, index=False)
                value = str(file_path)

            else:
                if binding.data is not None:
                    file_path = binding_root / binding.filename
                    binding.data.to_csv(file_path, index=False)
                else:
                    source = Path(binding.path)
                    if source.is_dir():
                        candidates = [p for p in source.iterdir() if p.suffix == ".csv"]
                        if len(candidates) != 1:
                            raise ValueError(
                                f"DataFrame binding {binding.name} expected exactly one CSV in {binding.path}; found {len(candidates)}"
                            )
                        source = candidates[0]
                    file_path = binding_root / (binding.filename or source.name)
                    shutil.copy2(source, file_path)
                value = str(file_path)

            prepared[binding.name] = {
                "kind": binding.kind,
                "value": value,
                "root": str(binding_root),
            }

        return prepared

    def _build_runner(self, clean_code: str, bindings: Dict[str, Dict[str, Any]], output_path: str, error_path: str) -> str:
        binding_lines = []
        globals_lines = []
        for name, spec in bindings.items():
            if spec["kind"] == "dataframe":
                binding_lines.append(f"{name} = pd.read_csv({spec['value']!r})")
            else:
                binding_lines.append(f"{name} = {spec['value']!r}")
            binding_lines.append(f"os.environ[{name!r}] = {name} if isinstance({name}, str) else {spec['value']!r}")
            globals_lines.append(f"{name!r}: {name}")

        return "\n".join(
            [
                "import os",
                "import traceback",
                "import numpy as np",
                "import pandas as pd",
                "",
                *binding_lines,
                "",
                "try:",
                "    g = {",
                '        "os": os,',
                '        "np": np,',
                '        "pd": pd,',
                f"        {','.join(globals_lines)},",
                "    }",
                f"    exec({clean_code!r}, g)",
                "",
                '    if "run_analysis" in g and callable(g["run_analysis"]):',
                '        candidate = g["run_analysis"](g.get("DATA_PATH"))',
                "        if isinstance(candidate, pd.DataFrame):",
                '            g["output_df"] = candidate',
                "",
                "    target_df = None",
                f"    for key in {list(RESULT_KEYS)!r}:",
                "        if key in g and isinstance(g[key], pd.DataFrame):",
                "            target_df = g[key]",
                "            break",
                "",
                "    if target_df is None:",
                "        for key, value in g.items():",
                "            if isinstance(value, pd.DataFrame):",
                "                target_df = value",
                "                break",
                "",
                "    if target_df is None:",
                '        raise ValueError("No pandas DataFrame output found")',
                "",
                f"    target_df.to_csv({output_path!r}, index=False)",
                "",
                "except Exception:",
                f"    with open({error_path!r}, \"w\", encoding=\"utf-8\") as f:",
                "        f.write(traceback.format_exc())",
                "",
            ]
        )
