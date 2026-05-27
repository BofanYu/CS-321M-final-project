from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .data import response_matrix_to_array
from .utils import sigmoid


def ability_table(subject_ability: pd.DataFrame, performance: pd.DataFrame = None) -> pd.DataFrame:
    """Merge IRT ability with optional performance metrics."""
    out = subject_ability.copy()
    if performance is not None:
        perf_cols = [c for c in ["model", "accuracy", "coverage", "balanced_accuracy", "long_recall", "short_specificity", "predicted_long_rate"] if c in performance.columns]
        out = out.drop(columns=[c for c in perf_cols if c != "model" and c in out.columns], errors="ignore")
        out = out.merge(performance[perf_cols], on="model", how="left", suffixes=("", "_perf"))
    return out.sort_values("ability", ascending=False).reset_index(drop=True)


def item_difficulty_table(item_params: pd.DataFrame, flat_metadata: pd.DataFrame = None) -> pd.DataFrame:
    """Return item-level IRT difficulty table merged with flattened metadata."""
    out = item_params.copy()
    if flat_metadata is not None:
        keep = [c for c in flat_metadata.columns if c not in out.columns or c == "item_id"]
        out = out.merge(flat_metadata[keep], on="item_id", how="left")
    return out


def irt_probability_by_difficulty(subject_ability: pd.DataFrame, difficulty_values: Iterable[float]) -> pd.DataFrame:
    """Compute Rasch P(correct)=sigmoid(theta-b) for selected difficulty values."""
    rows = []
    for _, s in subject_ability.iterrows():
        theta = float(s["ability"])
        for b in difficulty_values:
            rows.append({"model": s["model"], "ability": theta, "difficulty": b, "p_correct": float(sigmoid(theta - b))})
    return pd.DataFrame(rows)


def difficulty_decile_summary(item_table: pd.DataFrame, difficulty_col: str = "difficulty") -> pd.DataFrame:
    """Summarize item properties by IRT difficulty decile."""
    df = item_table.copy()
    df = df.dropna(subset=[difficulty_col])
    df["difficulty_decile"] = pd.qcut(df[difficulty_col], q=10, labels=False, duplicates="drop") + 1
    agg = {
        "item_id": ("item_id", "size"),
        difficulty_col: (difficulty_col, "mean"),
    }
    for c in [
        "empirical_accuracy", "empirical_difficulty", "simple_cue_label_conflict",
        "feature_label_surprisal", "feature_ambiguity", "severe_cues_short", "weak_cues_long",
        "damage_score", "total_severity_score"
    ]:
        if c in df.columns:
            agg[c] = (c, "mean")
    out = df.groupby("difficulty_decile").agg(**agg).reset_index()
    out = out.rename(columns={"item_id": "n_items", difficulty_col: "mean_irt_difficulty"})
    return out


def damage_label_difficulty(item_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize difficulty by BuildingDamage and binary label."""
    df = item_table.copy()
    if "true_long" not in df.columns:
        raise ValueError("item_table must contain true_long. Use flatten_case_metadata first.")
    df["binary_label"] = np.where(df["true_long"] == 1, "MoreThanAWeek", "LessThanAWeek")
    group_cols = ["BuildingDamage", "binary_label"]
    out = df.groupby(group_cols, dropna=False).agg(
        n=("item_id", "size"),
        mean_irt_difficulty=("difficulty", "mean"),
        empirical_accuracy=("empirical_accuracy", "mean"),
        true_long_rate=("true_long", "mean"),
        conflict_rate=("simple_cue_label_conflict", "mean") if "simple_cue_label_conflict" in df.columns else ("true_long", "mean"),
    ).reset_index()
    return out.sort_values("mean_irt_difficulty", ascending=False).reset_index(drop=True)


def residual_factor_analysis(response_matrix: pd.DataFrame, subject_ability: pd.DataFrame, item_params: pd.DataFrame, n_factors: int = 5) -> Dict[str, pd.DataFrame]:
    """SVD of 1PL standardized residuals.

    This is an interpretable diagnostic for multidimensional behavior beyond scalar ability.
    """
    models, item_ids, Y = response_matrix_to_array(response_matrix)
    theta_map = subject_ability.set_index("model")["ability"].to_dict()
    b_map = item_params.drop_duplicates("item_id").set_index("item_id")["difficulty"].to_dict()
    theta = np.array([theta_map.get(m, np.nan) for m in models], dtype=float)
    b = np.array([b_map.get(i, np.nan) for i in item_ids], dtype=float)
    eta = theta[:, None] - b[None, :]
    p = sigmoid(eta)
    obs = ~np.isnan(Y) & ~np.isnan(p)
    denom = np.sqrt(np.clip(p * (1 - p), 1e-4, None))
    R = np.zeros_like(Y, dtype=float)
    R[obs] = (Y[obs] - p[obs]) / denom[obs]
    U, s, Vt = np.linalg.svd(R, full_matrices=False)
    K = min(n_factors, len(s))
    subj = pd.DataFrame({"model": models})
    item = pd.DataFrame({"item_id": item_ids})
    for k in range(K):
        subj[f"factor{k+1}"] = U[:, k] * np.sqrt(s[k])
        item[f"factor{k+1}"] = Vt[k, :] * np.sqrt(s[k])
    var = pd.DataFrame({
        "factor": np.arange(1, K + 1),
        "singular_value": s[:K],
        "residual_variance_share": (s[:K] ** 2) / np.sum(s ** 2),
    })
    return {"subject_factors": subj, "item_factors": item, "variance": var}
