import os
from typing import Any, Dict, List, Optional


def schema_manifest(files: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"files": files}


def schema_file(
    *,
    filename: str,
    role: str,
    primary_time_column: str,
    entity_column: Optional[str],
    columns: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "filename": filename,
        "role": role,
        "primary_time_column": primary_time_column,
        "entity_column": entity_column,
        "columns": columns,
    }


def column_metadata(semantic_type: str, availability: str, **extra: Any) -> Dict[str, Any]:
    payload = {
        "semantic_type": semantic_type,
        "availability": availability,
    }
    payload.update(extra)
    return payload


def feature_columns(names: List[str], availability: str = "current_or_past") -> Dict[str, Dict[str, Any]]:
    return {name: column_metadata("feature", availability) for name in names}


def future_target_column(**extra: Any) -> Dict[str, Any]:
    return column_metadata(
        "target",
        "future_only",
        forbidden_for_decision=True,
        **extra,
    )


def temporal_case_metadata(
    *,
    variant_a: Dict[str, Any],
    variant_b: Dict[str, Any],
    behavior_name: str,
    diversity: Dict[str, Any],
    task_columns: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "legacy_variant_a_id": variant_a["id"],
        "legacy_variant_b_id": variant_b["id"],
        "behavior_name": behavior_name,
        "diversity": diversity,
        "task_columns": task_columns,
    }


def all_files_exist(base_path: str, filenames: List[str]) -> bool:
    return all(os.path.exists(os.path.join(base_path, filename)) for filename in filenames)
