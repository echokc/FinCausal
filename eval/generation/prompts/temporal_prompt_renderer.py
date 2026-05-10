from eval.generation.prompts.prompt_models import PromptSpec


def render_temporal_prompt(spec: PromptSpec) -> str:
    if len(spec.files) == 1:
        file_lines = "\n".join(f"- `{file.filename}`" for file in spec.files)
        data_section = (
            "The directory contains exactly one CSV file for this task:\n"
            f"{file_lines}"
        )
    else:
        rows = "\n".join(
            f"| `{file.variable}` | `{file.filename}` | "
            f"{', '.join(f'`{column}`' for column in file.columns)} |"
            for file in spec.files
        )
        data_section = (
            "Two CSV files are available inside the `DATA_PATH` directory. Always use the\n"
            "`data_path` function argument and never hardcode absolute paths.\n\n"
            "| Variable | File | Columns |\n"
            "|----------|------|---------|\n"
            f"{rows}"
        )

    task_lines = "\n".join(f"{idx}. {step}" for idx, step in enumerate(spec.task_steps, start=1))
    critical_lines = "\n".join(
        f"{idx}. {requirement}" for idx, requirement in enumerate(spec.critical_requirements, start=1)
    )
    output_columns = "\n".join(f"- `{column}`" for column in spec.required_output_columns)
    output_name = (
        f"\n- Final DataFrame must be named `{spec.output_dataframe_name}`"
        if spec.output_dataframe_name
        else ""
    )

    return f"""{spec.framing}

## Data
`DATA_PATH` is a directory path:

DATA_PATH = '{spec.data_path}'

{data_section}

Load the data inside `run_analysis` with:
```python
{spec.load_snippet}
```

## Task
{task_lines}

## Critical Requirements
{critical_lines}

Required output columns:
{output_columns}{output_name}

## Output Format
Encapsulate all logic in a single function:

```python
def run_analysis(data_path: str) -> pd.DataFrame:
    ...
```

Output requirements:
- Output complete executable Python code only.
- Define `run_analysis(data_path: str) -> pd.DataFrame`.
- Do not call `run_analysis`.
- Do not assign or overwrite `DATA_PATH`.
- Do not hardcode any absolute file paths.
- Do not include example usage, print statements, markdown, or explanation."""
