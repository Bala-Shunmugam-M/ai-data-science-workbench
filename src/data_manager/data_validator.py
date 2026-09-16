"""
src.data_manager.data_validator
===============================

PURPOSE
-------
Validate observed data against the human-approved semantic schema
(:mod:`src.domain.schema`): required/unexpected columns, storage type,
nullability, numeric bounds, whole-number requirement, allowed categories, and
duplicate rows. Produces a validation report and an overall status.

PIPELINE POSITION
-----------------
    ingestion -> [validation] -> preprocessing

Preprocessing is gated on this stage's status (PASSED / PASSED WITH WARNINGS).

Adapted from the professor's ``sematic_schema_validation.py`` (typo'd filename
in the starter code; clean module name here). The rule set, severities, and the
FAILED/PASSED WITH WARNINGS/PASSED status logic are preserved exactly. Outputs
are written under ``results/validation/`` (clean name) and the file-driven CLI
was refactored into ``validate_data`` / ``run_validation`` functions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config.paths import RESULTS_DIR, ensure_dir
from src.domain.schema import SEMANTIC_SCHEMA, schema_columns
from src.utils.file_utils import save_csv, save_text
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

VALIDATION_DIR: Path = RESULTS_DIR / "validation"

# Statuses that permit preprocessing to proceed (professor's gate policy).
ACCEPTED_STATUSES = {"PASSED", "PASSED WITH WARNINGS"}


def _add_issue(
    issues: list[dict[str, object]],
    severity: str,
    rule: str,
    column: str | None,
    count: int,
    message: str,
) -> None:
    """Append one issue record to the running list."""

    issues.append(
        {
            "severity": severity,
            "rule": rule,
            "column": column or "",
            "affected_count": int(count),
            "message": message,
        }
    )


def validate_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate ``data`` against the semantic schema.

    Returns ``(issues_df, flagged_observations_df)``. ``issues_df`` has columns
    severity / rule / column / affected_count / message. Flagged observations
    capture rows violating a minimum bound (as in the professor's script).
    """

    issues: list[dict[str, object]] = []
    observations: list[pd.DataFrame] = []

    expected = schema_columns()
    observed = list(data.columns)

    missing = [c for c in expected if c not in observed]
    unexpected = [c for c in observed if c not in expected]

    if missing:
        _add_issue(
            issues, "ERROR", "required_columns", None, len(missing),
            "Missing columns: " + ", ".join(missing),
        )
    if unexpected:
        _add_issue(
            issues, "WARNING", "unexpected_columns", None, len(unexpected),
            "Unexpected columns: " + ", ".join(unexpected),
        )

    for column, rules in SEMANTIC_SCHEMA.items():
        if column not in data.columns:
            continue

        series = data[column]
        expected_type = rules.get("expected_storage_type")

        if expected_type == "numeric":
            if not pd.api.types.is_numeric_dtype(series.dtype):
                _add_issue(
                    issues, "ERROR", "storage_type", column, len(series),
                    f"Expected numeric; observed {series.dtype}.",
                )
        elif expected_type == "string":
            is_string_like = (
                pd.api.types.is_object_dtype(series.dtype)
                or pd.api.types.is_string_dtype(series.dtype)
                or isinstance(series.dtype, pd.CategoricalDtype)
            )
            if not is_string_like:
                _add_issue(
                    issues, "ERROR", "storage_type", column, len(series),
                    f"Expected string; observed {series.dtype}.",
                )

        missing_count = int(series.isna().sum())
        if missing_count:
            # Nullable columns downgrade a missing-value finding to WARNING.
            severity = "WARNING" if rules.get("nullable", False) else "ERROR"
            _add_issue(
                issues, severity, "missing_values", column, missing_count,
                (
                    "Missing values allowed with documented treatment."
                    if severity == "WARNING"
                    else "Missing values found in non-nullable column."
                ),
            )

        if pd.api.types.is_numeric_dtype(series.dtype):
            numeric = pd.to_numeric(series, errors="coerce")

            minimum = rules.get("minimum")
            if minimum is not None:
                mask = numeric.notna() & (numeric < minimum)
                if int(mask.sum()):
                    _add_issue(
                        issues, "ERROR", "minimum_bound", column, int(mask.sum()),
                        f"Values below minimum {minimum}.",
                    )
                    flagged = data.loc[mask].copy()
                    flagged.insert(0, "validation_rule", "minimum_bound")
                    flagged.insert(1, "validation_column", column)
                    observations.append(flagged)

            maximum = rules.get("maximum")
            if maximum is not None:
                mask = numeric.notna() & (numeric > maximum)
                if int(mask.sum()):
                    _add_issue(
                        issues, "ERROR", "maximum_bound", column, int(mask.sum()),
                        f"Values above maximum {maximum}.",
                    )

            if rules.get("whole_number") is True:
                mask = numeric.notna() & ~np.isclose(numeric, np.round(numeric))
                if int(mask.sum()):
                    _add_issue(
                        issues, "ERROR", "whole_number_requirement", column,
                        int(mask.sum()), "Non-whole-number values detected.",
                    )

        allowed = rules.get("allowed_values")
        if allowed:
            mask = series.notna() & ~series.astype(str).isin(allowed)
            if int(mask.sum()):
                invalid = sorted(series.loc[mask].astype(str).unique())
                _add_issue(
                    issues, "ERROR", "allowed_values", column, int(mask.sum()),
                    "Unexpected values: " + ", ".join(invalid),
                )

    duplicates = int(data.duplicated().sum())
    if duplicates:
        _add_issue(
            issues, "WARNING", "duplicate_rows", None, duplicates,
            "Duplicate rows detected.",
        )

    issues_df = pd.DataFrame(
        issues,
        columns=["severity", "rule", "column", "affected_count", "message"],
    )

    if observations:
        observations_df = pd.concat(observations, ignore_index=False)
    else:
        observations_df = pd.DataFrame(
            columns=["validation_rule", "validation_column", *data.columns]
        )

    return issues_df, observations_df


def status_from_issues(issues: pd.DataFrame) -> tuple[str, int, int]:
    """Return (status, error_count, warning_count) from an issues frame."""

    errors = int((issues["severity"] == "ERROR").sum()) if not issues.empty else 0
    warnings = (
        int((issues["severity"] == "WARNING").sum()) if not issues.empty else 0
    )
    status = (
        "FAILED"
        if errors
        else "PASSED WITH WARNINGS"
        if warnings
        else "PASSED"
    )
    return status, errors, warnings


def run_validation(
    data: pd.DataFrame,
    output_dir: Path = VALIDATION_DIR,
    raise_on_failure: bool = False,
) -> dict[str, object]:
    """
    Validate ``data``, persist the report under ``results/validation/``, and
    return a small result dict. When ``raise_on_failure`` is True a FAILED
    status raises ``RuntimeError`` (hard gate).
    """

    ensure_dir(output_dir)
    issues, observations = validate_data(data)
    status, errors, warnings = status_from_issues(issues)

    summary = pd.DataFrame(
        [
            {
                "observations": len(data),
                "columns": len(data.columns),
                "validation_errors": errors,
                "validation_warnings": warnings,
                "duplicate_rows": int(data.duplicated().sum()),
                "total_missing_values": int(data.isna().sum().sum()),
                "status": status,
            }
        ]
    )

    save_csv(summary, output_dir / "validation_summary.csv")
    save_csv(issues, output_dir / "validation_issues.csv")
    save_csv(observations, output_dir / "validation_observations.csv", index=True)
    save_text(f"Validation status: {status}\n", output_dir / "validation_status.txt")

    logger.info(
        "Validation status=%s (errors=%d, warnings=%d)", status, errors, warnings
    )

    if status == "FAILED" and raise_on_failure:
        raise RuntimeError(
            "Validation failed. Review results/validation/validation_issues.csv."
        )

    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
        "issues": issues,
        "output_dir": output_dir,
    }
