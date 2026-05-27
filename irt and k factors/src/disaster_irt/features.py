from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data import LONG
from .utils import parse_maybe_dict


def flatten_case_metadata(case_metadata: pd.DataFrame) -> pd.DataFrame:
    """Flatten household_attributes / feature_key into analysis-ready columns.

    Ground-truth columns are preserved, but the returned feature columns can be used with
    exclude_label=True in make_feature_matrix to avoid leakage.
    """
    records = []
    for _, row in case_metadata.iterrows():
        d = parse_maybe_dict(row.get("feature_key"))
        if not d:
            d = parse_maybe_dict(row.get("household_attributes"))
        soc = d.get("SocioEconomicParameters", {}) if isinstance(d, dict) else {}
        rec = {
            "item_id": row.get("item_id"),
            "case_id": row.get("case_id", row.get("item_id")),
            "ground_truth": row.get("ground_truth", d.get("DisplacementDuration")),
            "ground_truth_mapped": row.get("ground_truth_mapped"),
            "State": soc.get("State", ""),
            "MSA": soc.get("MSA", ""),
            "Tenure": soc.get("Tenure", ""),
            "Income": soc.get("Income", ""),
            "Occupants": soc.get("Occupants", np.nan),
            "Kids5years": soc.get("Kids5years", np.nan),
            "Kids5_11years": soc.get("Kids5-11years", np.nan),
            "Kids12_17years": soc.get("Kids12-17years", np.nan),
            "EmploymentStatus": soc.get("EmploymentStatus", ""),
            "BuildingDamage": d.get("BuildingDamage", ""),
            "DisasterType": d.get("DisasterType", ""),
            "WaterAccess": d.get("WaterAccess", ""),
            "PowerAccess": d.get("PowerAccess", ""),
            "FoodAccess": d.get("FoodAccess", ""),
            "UnsanitaryConditions": d.get("UnsanitaryConditions", ""),
        }
        records.append(rec)
    out = pd.DataFrame(records)
    out["true_long"] = (out["ground_truth_mapped"] == LONG).astype(int)
    out = add_severity_scores(out)
    return out


def _map_contains(value, rules):
    s = str(value).lower()
    for key, val in rules:
        if key in s:
            return val
    return np.nan


def add_severity_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple ordinal severity scores for disaster/recovery cues."""
    out = df.copy()
    damage_map = {
        "No Damage": 0,
        "Some Damage": 1,
        "Moderate Damage": 2,
        "A lot of Damage": 3,
    }
    out["damage_score"] = out.get("BuildingDamage", pd.Series(index=out.index)).map(damage_map).astype(float)

    shortage_rules = [("no ", 0), ("never", 0), ("some", 1), ("a lot", 2), ("lot", 2), ("always", 0)]
    out["water_score"] = out.get("WaterAccess", pd.Series(index=out.index)).apply(lambda x: _map_contains(x, shortage_rules)).astype(float)
    out["power_score"] = out.get("PowerAccess", pd.Series(index=out.index)).apply(lambda x: _map_contains(x, shortage_rules)).astype(float)
    out["food_score"] = out.get("FoodAccess", pd.Series(index=out.index)).apply(lambda x: _map_contains(x, shortage_rules)).astype(float)
    out["unsanitary_score"] = out.get("UnsanitaryConditions", pd.Series(index=out.index)).apply(lambda x: _map_contains(x, shortage_rules)).astype(float)

    score_cols = ["damage_score", "water_score", "power_score", "food_score", "unsanitary_score"]
    for c in score_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    out["total_severity_score"] = out[score_cols].sum(axis=1)

    # Cue-label conflict variables. These are deliberately simple and interpretable.
    out["severe_cues"] = ((out["damage_score"] >= 2) | (out["total_severity_score"] >= 5)).astype(int)
    out["weak_cues"] = ((out["damage_score"] <= 0) & (out["total_severity_score"] <= 1)).astype(int)
    if "true_long" in out.columns:
        out["severe_cues_short"] = ((out["severe_cues"] == 1) & (out["true_long"] == 0)).astype(int)
        out["weak_cues_long"] = ((out["weak_cues"] == 1) & (out["true_long"] == 1)).astype(int)
        out["simple_cue_label_conflict"] = ((out["severe_cues_short"] == 1) | (out["weak_cues_long"] == 1)).astype(int)
    return out


DEFAULT_NUMERIC_COLS = ["Occupants", "Kids5years", "Kids5_11years", "Kids12_17years", "damage_score", "water_score", "power_score", "food_score", "unsanitary_score", "total_severity_score"]
DEFAULT_CATEGORICAL_COLS = ["State", "MSA", "Tenure", "Income", "EmploymentStatus", "BuildingDamage", "DisasterType", "WaterAccess", "PowerAccess", "FoodAccess", "UnsanitaryConditions"]


def make_feature_frame(flat_metadata: pd.DataFrame, include_scores: bool = True) -> pd.DataFrame:
    """Return a leakage-safe feature frame indexed by item_id.

    It excludes ground_truth, ground_truth_mapped, model prediction columns, and true_long.
    """
    cols = DEFAULT_CATEGORICAL_COLS + [c for c in DEFAULT_NUMERIC_COLS if include_scores]
    available = [c for c in cols if c in flat_metadata.columns]
    Xdf = flat_metadata[["item_id"] + available].copy().set_index("item_id")
    for c in DEFAULT_NUMERIC_COLS:
        if c in Xdf.columns:
            Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")
            Xdf[c] = Xdf[c].fillna(Xdf[c].median())
    return Xdf


def one_hot_feature_matrix(flat_metadata: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Create a simple one-hot encoded feature matrix used by K-factor cold-start."""
    Xdf = make_feature_frame(flat_metadata)
    cat_cols = [c for c in DEFAULT_CATEGORICAL_COLS if c in Xdf.columns]
    Xoh = pd.get_dummies(Xdf, columns=cat_cols, dummy_na=True)
    for c in Xoh.columns:
        Xoh[c] = pd.to_numeric(Xoh[c], errors="coerce").fillna(0.0)
    return Xoh, Xoh.to_numpy(dtype=float)
