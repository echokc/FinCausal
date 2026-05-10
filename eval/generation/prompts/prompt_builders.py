from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, Mapping


@dataclass(frozen=True)
class RecipePromptOptions:
    data_variable: str = "DATA_PATH"
    schema_variant: str | None = None
    prompt_variant: str | None = None
    include_known_traps: bool = True
    include_universe_descriptions: bool = False
    require_markdown_code_block: bool = True


def build_prompt_from_recipe(recipe: Any, options: RecipePromptOptions | None = None) -> str:
    """Render a candidate-facing prompt from a recipe-like object.

    The builder intentionally consumes only the common recipe surface:
    schema variants, prompt variants, output, task steps, load snippet, and
    known traps. This keeps prompt generation generic for new scenarios.
    """

    options = options or RecipePromptOptions()
    schema_key = options.schema_variant or recipe.default_schema_variant
    prompt_key = options.prompt_variant or recipe.default_prompt_variant
    schema = recipe.schema_variants[schema_key]
    prompt_variant = recipe.prompt_variants[prompt_key]

    output = _recipe_output(recipe)
    sections = [
        _render_header(prompt_variant),
        _render_data_section(options.data_variable, schema_key, schema, recipe),
        _render_task_steps(recipe.task_steps),
        _render_output_contract(output),
    ]

    if options.include_known_traps:
        sections.append(_render_known_traps(getattr(recipe, "known_traps", [])))
    if options.include_universe_descriptions:
        sections.append(_render_universe_context(getattr(recipe, "universes", [])))

    sections.append(_render_response_contract(options.require_markdown_code_block))
    return "\n\n".join(section for section in sections if section).strip() + "\n"


def build_repair_prompt(
    *,
    original_prompt: str,
    candidate_code: str,
    recipe: Any,
    error_summary: str,
    schema_variant: str | None = None,
) -> str:
    """Build a contract-only repair prompt.

    The repair prompt intentionally excludes evaluation universe names, probe
    names, expected thresholds, and hidden behavioral outcomes. It tells the
    model only what a normal execution/contract validator could reveal.
    """

    schema_key = schema_variant or recipe.default_schema_variant
    schema = recipe.schema_variants[schema_key]
    output = _recipe_output(recipe)
    return f"""Your previous Python code did not satisfy the executable interface contract.

Rewrite the solution as a complete replacement. Preserve the original task semantics, but fix only code, schema, dependency, output, or unit-contract issues exposed below.

Do not infer or mention any hidden evaluation cases. Do not hardcode values for a specific input file. Do not add example usage.

## Original Task Prompt
{original_prompt.strip()}

## Candidate Code That Failed
```python
{candidate_code.strip()}
```

## Public Contract Failure
{_sanitize_repair_error(error_summary)}

## Schema Reminder
{_render_schema_contract(schema_key, schema)}

## Output Reminder
{_render_output_contract(output)}

## Repair Response Format
Return executable Python code in exactly one ```python code block.
The code must load from the provided data variable, assign one accepted output variable, and use only standard packages already available in the execution environment: pandas, numpy, os, math, statistics, and Python standard library.
"""


def prompt_quality_checks(recipe: Any, prompt: str) -> Dict[str, bool]:
    output = _recipe_output(recipe)
    accepted_names = _accepted_names(output)
    normalized_prompt = _normalize_for_check(prompt)
    return {
        "mentions_data_path": "DATA_PATH" in prompt,
        "mentions_primary_output": output.variable_name in prompt,
        "mentions_any_accepted_output": any(name in prompt for name in accepted_names),
        "mentions_code_block_contract": "```python" in prompt,
        "mentions_task_steps": all(_normalize_for_check(step)[:24] in normalized_prompt for step in recipe.task_steps[:2]),
        "mentions_real_column_contract": "Do not use schema metadata keys as dataframe column names" in prompt,
        "mentions_no_function_only": "Do not only define functions" in prompt,
    }


def _render_header(prompt_variant: Any) -> str:
    lines = [prompt_variant.framing]
    if getattr(prompt_variant, "causal_requirement", None):
        lines.extend(["", "Critical requirement:", prompt_variant.causal_requirement])
    return "\n".join(lines)


def _render_data_section(data_variable: str, schema_key: str, schema: Mapping[str, Any], recipe: Any) -> str:
    load_snippet = getattr(recipe, "load_snippet", f"df = pd.read_csv({data_variable})")
    load_snippet = load_snippet.replace("DATA_PATH", data_variable)
    schema_contract = _render_schema_contract(schema_key, schema)
    return f"""## Data
The input data is available through the Python variable `{data_variable}`.

Use this loading pattern unless the task steps explicitly require a different variable name:

```python
{load_snippet}
```

{schema_contract}"""


def _render_task_steps(task_steps: Iterable[str]) -> str:
    rows = "\n".join(f"{idx}. {step}" for idx, step in enumerate(task_steps, start=1))
    return f"""## Task
{rows}"""


def _render_output_contract(output: Any) -> str:
    accepted = ", ".join(f"`{name}`" for name in _accepted_names(output))
    lines = [
        "## Output Contract",
        f"- Output kind: `{getattr(output, 'kind', 'scalar')}`",
        f"- Primary output variable: `{output.variable_name}`",
        f"- Accepted output variable names: {accepted}",
        f"- Output semantic: `{output.semantic}`",
    ]
    if getattr(output, "valid_range", None) is not None:
        lines.append(f"- Valid range: `{output.valid_range}`")
    if getattr(output, "shape", None) is not None:
        lines.append(f"- Expected shape: `{output.shape}`")
    lines.extend(
        [
            "",
            "Your final code must assign the required result to one of the accepted output variable names.",
            "Do not only define functions; call your function or otherwise compute the result at top level.",
            "Do not place the required assignment inside `if __name__ == \"__main__\"`.",
        ]
    )
    lines.extend(_output_kind_requirements(output))
    return "\n".join(lines)


def _render_known_traps(known_traps: Iterable[Mapping[str, Any]]) -> str:
    traps = list(known_traps)
    if not traps:
        return """## Common Failure Modes
- Avoid shortcuts that violate the causal or robustness requirement."""
    rows = "\n".join(f"- {trap.get('description', trap.get('type', 'Unspecified trap'))}" for trap in traps[:5])
    return f"""## Common Failure Modes
{rows}"""


def _render_universe_context(universes: Iterable[Any]) -> str:
    universe_rows = []
    for universe in universes:
        universe_rows.append(f"- `{universe.name}` ({universe.role}): {universe.description}")
    if not universe_rows:
        return ""
    return "## Evaluation Universes\n" + "\n".join(universe_rows)


def _render_response_contract(require_markdown_code_block: bool) -> str:
    if require_markdown_code_block:
        return """## Response Format
Return executable Python code in exactly one ```python code block.
Do not hardcode absolute file paths.
Do not overwrite the provided data path variable.
Do not include example usage that calls external files.
Use only packages that are normally available in this sandbox: pandas, numpy, os, math, statistics, and Python standard library.
Do not import optional third-party packages such as sklearn, scipy, cvxpy, statsmodels, or talib."""
    return """## Response Format
Return executable Python code.
Do not hardcode absolute file paths.
Do not overwrite the provided data path variable.
Use only packages that are normally available in this sandbox: pandas, numpy, os, math, statistics, and Python standard library."""


def _render_schema_contract(schema_key: str, schema: Mapping[str, Any]) -> str:
    if not schema:
        return f"""Schema variant `{schema_key}`:
- No structured schema metadata provided."""

    exact_columns = _exact_dataframe_columns(schema)
    lines = [f"Schema variant `{schema_key}`:", _render_mapping(schema)]
    lines.extend(
        [
            "",
            "Column-name contract:",
            "- Do not use schema metadata keys as dataframe column names.",
        ]
    )
    if exact_columns:
        columns = ", ".join(f"`{col}`" for col in exact_columns)
        lines.append(f"- When loading a tabular CSV, use these real dataframe column names: {columns}.")
    if "asset_prefix" in schema:
        prefix = schema["asset_prefix"]
        lines.append(f"- The wide panel contains asset columns whose names start with `{prefix}`; select those real columns from the loaded dataframe.")
    if "layout" in schema:
        lines.append(f"- Layout: `{schema['layout']}`.")
    lines.append("- Metadata keys such as `time_col`, `price_col`, `side_col`, or `return_col` describe which real column to use; they are not literal dataframe columns unless explicitly listed as real columns.")
    return "\n".join(lines)


def _render_mapping(mapping: Mapping[str, Any]) -> str:
    if not mapping:
        return "- No structured schema metadata provided."
    return "\n".join(f"- `{key}`: `{value}`" for key, value in mapping.items())


def _exact_dataframe_columns(schema: Mapping[str, Any]) -> list[str]:
    columns: list[str] = []
    for key, value in schema.items():
        if not isinstance(value, str):
            continue
        if key.endswith("_col") or key.endswith("_column"):
            columns.append(value)
    return list(dict.fromkeys(columns))


def _output_kind_requirements(output: Any) -> list[str]:
    kind = getattr(output, "kind", "scalar")
    variable = output.variable_name
    if kind == "scalar":
        return [f"For scalar output, `{variable}` must be a plain int, float, bool, or numpy scalar, not just a printed value."]
    if kind == "dict":
        return [f"For dict output, `{variable}` must be assigned to a Python dict whose keys are the requested output fields."]
    if kind == "series":
        return [
            f"For series output, `{variable}` must be a pandas Series aligned to the input rows.",
            "If the series is boolean flags, `True` must mean the row triggers the requested event, signal, or protection condition; `False` must mean no trigger on that row.",
            "Do not return the inverse safe-row mask where normal rows are `True` and event rows are `False`.",
        ]
    if kind == "dataframe":
        return [f"For dataframe output, `{variable}` must be a pandas DataFrame with the requested shape and columns."]
    return []


def _sanitize_repair_error(error_summary: str) -> str:
    text = error_summary.strip() or "The code failed validation without a detailed error."
    text = re.sub(r"/var/folders/[^\s'\"]+", "<tmp_path>", text)
    text = re.sub(r"/private/var/folders/[^\s'\"]+", "<tmp_path>", text)
    text = re.sub(r"/tmp/[^\s'\"]+", "<tmp_path>", text)
    return text[:4000]


def _recipe_output(recipe: Any) -> Any:
    return recipe.output


def _accepted_names(output: Any) -> list[str]:
    names = list(getattr(output, "accepted_names", []) or [])
    if output.variable_name not in names:
        names.insert(0, output.variable_name)
    return names


def _normalize_for_check(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("`", "")).strip().lower()
