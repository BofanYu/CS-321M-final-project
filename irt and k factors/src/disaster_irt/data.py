from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import json

import numpy as np
import pandas as pd

from .paths import find_first
from .utils import normalize_model_name_from_json_path

LONG = "LeaveHomeForMoreThanAWeek"
SHORT = "LeaveHomeForLessThanAWeek"


def load_core_tables(irt_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load the standard outputs under results/irt_baseline_ruleset.

    Expected filenames:
    - response_matrix.csv: model x item correctness matrix, values 0/1/NaN.
    - case_metadata.csv: item metadata and raw model predictions.
    - irt_1pl_item_parameters.csv: item difficulty output.
    - irt_1pl_subject_ability.csv: model ability output.
    - missing_response_summary.csv: optional logged missing/error responses.
    """
    irt_dir = Path(irt_dir)
    response_path = find_first(irt_dir, ["response_matrix.csv", "response_matrix(3).csv"])
    meta_path = find_first(irt_dir, ["case_metadata.csv", "case_metadata(3).csv"])
    item_path = find_first(irt_dir, ["irt_1pl_item_parameters.csv", "irt_1pl_item_parameters(3).csv"])
    subject_path = find_first(irt_dir, ["irt_1pl_subject_ability.csv", "irt_1pl_subject_ability(3).csv"])
    missing_path = find_first(irt_dir, ["missing_response_summary.csv"], required=False)

    response_matrix = pd.read_csv(response_path)
    if "model" not in response_matrix.columns:
        # Some older exports used the index column as model.
        response_matrix = pd.read_csv(response_path, index_col=0).reset_index().rename(columns={"index": "model"})
    case_metadata = pd.read_csv(meta_path)
    item_params = pd.read_csv(item_path)
    subject_ability = pd.read_csv(subject_path)
    missing = pd.read_csv(missing_path) if missing_path else pd.DataFrame()
    return {
        "response_matrix": response_matrix,
        "case_metadata": case_metadata,
        "item_params": item_params,
        "subject_ability": subject_ability,
        "missing_summary": missing,
        "paths": pd.DataFrame({
            "table": ["response_matrix", "case_metadata", "item_params", "subject_ability", "missing_summary"],
            "path": [str(response_path), str(meta_path), str(item_path), str(subject_path), str(missing_path) if missing_path else ""],
        })
    }


def response_matrix_to_array(response_matrix: pd.DataFrame) -> Tuple[List[str], List[str], np.ndarray]:
    """Return model names, item ids, and M x N correctness array."""
    if "model" not in response_matrix.columns:
        raise ValueError("response_matrix must have a 'model' column")
    models = response_matrix["model"].astype(str).tolist()
    item_cols = [c for c in response_matrix.columns if c != "model"]
    Y = response_matrix[item_cols].to_numpy(dtype=float)
    return models, item_cols, Y


def align_metadata_to_matrix(case_metadata: pd.DataFrame, item_ids: Iterable[str]) -> pd.DataFrame:
    """Return metadata ordered by response matrix item_ids."""
    md = case_metadata.copy()
    if "item_id" not in md.columns:
        raise ValueError("case_metadata must contain item_id")
    md = md.drop_duplicates("item_id")
    missing = sorted(set(item_ids) - set(md["item_id"]))
    if missing:
        raise ValueError(f"case_metadata is missing {len(missing)} item ids, e.g. {missing[:5]}")
    return md.set_index("item_id").loc[list(item_ids)].reset_index()


def prediction_columns(case_metadata: pd.DataFrame) -> List[str]:
    return [c for c in case_metadata.columns if c.endswith("_prediction")]


def build_decision_matrix_from_metadata(case_metadata: pd.DataFrame, model_names: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Build a model x item matrix of raw LLM decisions.

    Cell is 1 if the model predicted LeaveHomeForMoreThanAWeek, 0 if LessThanAWeek, NaN otherwise.
    This uses columns like phi4_baseline_prediction in case_metadata.csv.
    """
    md = case_metadata.copy()
    if "item_id" not in md.columns:
        raise ValueError("case_metadata must contain item_id")
    pred_cols = prediction_columns(md)
    rows = []
    wanted = set(model_names) if model_names is not None else None
    for col in pred_cols:
        model = col[:-len("_prediction")]
        if wanted is not None and model not in wanted:
            continue
        vals = md[col].map({LONG: 1.0, SHORT: 0.0}).astype(float).to_numpy()
        rows.append(pd.Series(vals, index=md["item_id"].astype(str), name=model))
    if not rows:
        raise ValueError("No *_prediction columns found for decision matrix")
    out = pd.DataFrame(rows).reset_index().rename(columns={"index": "model"})
    return out


def load_agent_decisions_from_json_folders(results_dir: Path, model_folders: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Optional loader for raw JSON results under results/<model>/results_<model>_combined.json.

    Returns a long table with model, row_number, and AgentDecision.
    Use this only if you need to rebuild prediction columns from raw JSON.
    """
    results_dir = Path(results_dir)
    folders = [results_dir / m for m in model_folders] if model_folders else [p for p in results_dir.iterdir() if p.is_dir()]
    rows = []
    for folder in folders:
        json_files = list(folder.glob("results_*_combined.json")) + list(folder.glob("*_combined.json"))
        if not json_files:
            continue
        p = json_files[0]
        model = normalize_model_name_from_json_path(p)
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        decisions = obj.get("AgentDecisions", []) if isinstance(obj, dict) else []
        for idx, dec in enumerate(decisions):
            rows.append({"model": model, "row_number": idx, "AgentDecision": dec})
    return pd.DataFrame(rows)


def save_tables(tables: Dict[str, pd.DataFrame], output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(output_dir / f"{name}.csv", index=False)
