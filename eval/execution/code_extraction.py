import re


def extract_candidate_code(raw_response: str) -> str:
    """Extract executable Python from common markdown/code-only LLM responses."""
    if not raw_response:
        return ""

    fenced = re.search(r"```(?:python)?\s*\n(.*?)\n```", raw_response, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    # Some models include prose before a top-level import/def. Keep the likely
    # executable suffix instead of failing the candidate at generation time.
    lines = raw_response.strip().splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("import ", "from ", "def ", "DATA_PATH", "result_df", "merged_df")):
            return "\n".join(lines[idx:]).strip()

    return raw_response.strip()


def sanitize_llm_code(llm_code: str) -> str:
    clean_code = extract_candidate_code(llm_code)
    clean_code = re.sub(
        r"""^(DATA_DIR|DATA_PATH|data_dir|data_path)\s*=\s*['"][^'"]+['"]""",
        r"\1 = DATA_PATH",
        clean_code,
        flags=re.MULTILINE,
    )
    return clean_code
