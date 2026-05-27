from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from .features import DEFAULT_CATEGORICAL_COLS, DEFAULT_NUMERIC_COLS, make_feature_frame


def feature_label_model_scores(flat_metadata: pd.DataFrame, n_splits: int = 5, random_state: int = 2026) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Cross-fitted feature-only label model to quantify feature surprisal and ambiguity.

    This model predicts the true binary displacement label from item metadata only. Its predicted
    probability is used to define:
    - feature_label_surprisal = -log P(true label | features)
    - feature_ambiguity = 1 - |2p - 1|, high near p=0.5.
    """
    Xdf = make_feature_frame(flat_metadata)
    y = flat_metadata.set_index("item_id").loc[Xdf.index, "true_long"].astype(int).to_numpy()
    cat_cols = [c for c in DEFAULT_CATEGORICAL_COLS if c in Xdf.columns]
    num_cols = [c for c in DEFAULT_NUMERIC_COLS if c in Xdf.columns]
    pre = ColumnTransformer([
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
    ])
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight=None, solver="lbfgs")
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    prob = cross_val_predict(pipe, Xdf, y, cv=cv, method="predict_proba")[:, 1]
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    p_true = np.where(y == 1, prob, 1 - prob)
    scored = flat_metadata.copy()
    scored = scored.set_index("item_id").loc[Xdf.index].reset_index()
    scored["feature_p_long"] = prob
    scored["feature_label_surprisal"] = -np.log(p_true)
    scored["feature_ambiguity"] = 1 - np.abs(2 * prob - 1)
    metrics = {
        "feature_label_auc": float(roc_auc_score(y, prob)),
        "feature_label_accuracy_at_0p5": float(accuracy_score(y, (prob >= 0.5).astype(int))),
        "n_items": int(len(y)),
    }
    return scored, metrics


def difficulty_validity_correlations(item_table: pd.DataFrame, difficulty_col: str = "difficulty") -> pd.DataFrame:
    """Spearman correlations between IRT difficulty and construct-validity variables."""
    variables = [
        ("empirical difficulty", "empirical_difficulty"),
        ("feature label surprisal", "feature_label_surprisal"),
        ("simple cue-label conflict", "simple_cue_label_conflict"),
        ("feature ambiguity", "feature_ambiguity"),
        ("total severity score", "total_severity_score"),
        ("damage score", "damage_score"),
    ]
    rows = []
    for label, col in variables:
        if col not in item_table.columns:
            continue
        df = item_table[[difficulty_col, col]].dropna()
        if len(df) < 3:
            rho, p = np.nan, np.nan
        else:
            rho, p = spearmanr(df[difficulty_col], df[col])
        rows.append({"construct_validity_variable": label, "column": col, "spearman_corr_with_irt_difficulty": rho, "p_value": p, "n": len(df)})
    return pd.DataFrame(rows)
