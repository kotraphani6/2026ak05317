"""Shared column definitions and cleaning for the Online Shoppers dataset.

Used by both train_models.py and app.py so that training and inference
see exactly the same feature layout.
"""
import pandas as pd

TARGET = "Revenue"

NUMERIC_COLS = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
]
# Integer-coded or text columns that carry no numeric order -> one-hot encode
CATEGORICAL_COLS = [
    "Month", "OperatingSystems", "Browser", "Region",
    "TrafficType", "VisitorType", "Weekend",
]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

_TRUTHY = {"true": 1, "1": 1, "yes": 1, "false": 0, "0": 0, "no": 0}


def to_binary(series: pd.Series) -> pd.Series:
    """Map TRUE/FALSE (bool or text) to 1/0."""
    if series.dtype == bool:
        return series.astype(int)
    return series.astype(str).str.strip().str.lower().map(_TRUTHY)


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with consistent dtypes for the model pipeline."""
    out = df.copy()
    out["Weekend"] = to_binary(out["Weekend"])
    for col in ["OperatingSystems", "Browser", "Region", "TrafficType"]:
        out[col] = out[col].astype(str)
    out["Month"] = out["Month"].astype(str).str.strip()
    out["VisitorType"] = out["VisitorType"].astype(str).str.strip()
    if TARGET in out.columns:
        out[TARGET] = to_binary(out[TARGET])
    return out


def missing_columns(df: pd.DataFrame) -> list:
    return [c for c in FEATURE_COLS + [TARGET] if c not in df.columns]
