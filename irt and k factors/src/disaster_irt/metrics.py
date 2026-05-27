from itertools import combinations
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, accuracy_score

from .data import response_matrix_to_array, LONG, SHORT


def safe_binary_metrics(y_true, y_pred_or_prob, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred_or_prob).astype(float)
    yhat = (p >= threshold).astype(int)
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, yhat)),
        "brier": float(brier_score_loss(y_true, np.clip(p, 1e-6, 1 - 1e-6))),
        "logloss": float(log_loss(y_true, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
    }
    if len(np.unique(y_true)) == 2:
        out["auc"] = float(roc_auc_score(y_true, p))
    else:
        out["auc"] = np.nan
    return out


def model_performance(response_matrix: pd.DataFrame, case_metadata: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy, coverage, recall/specificity, and predicted-long rate by model.

    response_matrix is correctness, so class-specific metrics are read from prediction columns
    in case_metadata when available.
    """
    models, item_ids, Y = response_matrix_to_array(response_matrix)
    md = case_metadata.drop_duplicates("item_id").set_index("item_id").loc[item_ids].reset_index()
    true_long = (md["ground_truth_mapped"] == LONG).astype(int).to_numpy()
    rows = []
    for mi, model in enumerate(models):
        correct = Y[mi, :]
        obs = ~np.isnan(correct)
        row = {
            "model": model,
            "observed_total": int(obs.sum()),
            "missing_total": int((~obs).sum()),
            "coverage": float(obs.mean()),
            "accuracy": float(np.nanmean(correct)) if obs.any() else np.nan,
        }
        pred_col = f"{model}_prediction"
        if pred_col in md.columns:
            pred = md[pred_col].map({LONG: 1.0, SHORT: 0.0}).to_numpy(dtype=float)
            valid = obs & ~np.isnan(pred)
            if valid.any():
                yt = true_long[valid]
                yp = pred[valid].astype(int)
                tp = ((yt == 1) & (yp == 1)).sum()
                tn = ((yt == 0) & (yp == 0)).sum()
                fp = ((yt == 0) & (yp == 1)).sum()
                fn = ((yt == 1) & (yp == 0)).sum()
                long_recall = tp / (tp + fn) if (tp + fn) else np.nan
                short_specificity = tn / (tn + fp) if (tn + fp) else np.nan
                bal = np.nanmean([long_recall, short_specificity])
                row.update({
                    "balanced_accuracy": float(bal),
                    "long_recall": float(long_recall),
                    "short_specificity": float(short_specificity),
                    "predicted_long_rate": float(np.mean(yp == 1)),
                    "true_long_rate_observed": float(np.mean(yt == 1)),
                    "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
                })
        rows.append(row)
    out = pd.DataFrame(rows)
    if "accuracy" in out.columns:
        out["accuracy_rank"] = out["accuracy"].rank(ascending=False, method="min").astype(int)
    return out.sort_values(["accuracy", "coverage"], ascending=False).reset_index(drop=True)


def exact_mcnemar_from_correctness(a_correct, b_correct) -> Dict[str, float]:
    """Exact McNemar test for paired correctness vectors.

    b10 = A correct, B wrong. b01 = A wrong, B correct.
    The two-sided exact p-value is a binomial test with p=0.5 on discordant pairs.
    """
    a = np.asarray(a_correct).astype(int)
    b = np.asarray(b_correct).astype(int)
    b10 = int(((a == 1) & (b == 0)).sum())
    b01 = int(((a == 0) & (b == 1)).sum())
    n_disc = b10 + b01
    p = binomtest(min(b10, b01), n_disc, 0.5, alternative="two-sided").pvalue if n_disc > 0 else 1.0
    return {"a_correct_b_wrong": b10, "a_wrong_b_correct": b01, "discordant": n_disc, "mcnemar_exact_p": float(p)}


def paired_comparisons(response_matrix: pd.DataFrame, pairs: Optional[List[tuple]] = None) -> pd.DataFrame:
    """Paired accuracy deltas and McNemar tests among models."""
    models, item_ids, Y = response_matrix_to_array(response_matrix)
    model_to_i = {m: i for i, m in enumerate(models)}
    if pairs is None:
        # Useful default: common baselines/ruleset comparisons when present.
        candidate_pairs = [
            ("baseline", "ruleset"), ("baseline", "5mini"), ("ruleset", "5mini"),
            ("deepseek_r1_14b_baseline", "deepseek_r1_14b_ruleset"),
            ("llama31_8b_baseline", "llama31_8b_ruleset"),
            ("phi4_baseline", "phi4_ruleset"),
            ("qwen253b_baseline", "qwen253b_ruleset"),
            ("qwen314b_baseline", "qwen314b_ruleset"),
        ]
        pairs = [(a, b) for a, b in candidate_pairs if a in model_to_i and b in model_to_i]
    rows = []
    for a, b in pairs:
        ya = Y[model_to_i[a], :]
        yb = Y[model_to_i[b], :]
        obs = ~np.isnan(ya) & ~np.isnan(yb)
        if obs.sum() == 0:
            continue
        aa = ya[obs].astype(int)
        bb = yb[obs].astype(int)
        diff_vec = bb - aa
        mean_delta = float(diff_vec.mean())
        # Normal approximation CI for paired mean difference.
        se = float(diff_vec.std(ddof=1) / np.sqrt(len(diff_vec))) if len(diff_vec) > 1 else np.nan
        ci_low = mean_delta - 1.96 * se if not np.isnan(se) else np.nan
        ci_high = mean_delta + 1.96 * se if not np.isnan(se) else np.nan
        row = {
            "model_a": a,
            "model_b": b,
            "n_overlap": int(obs.sum()),
            "accuracy_a": float(aa.mean()),
            "accuracy_b": float(bb.mean()),
            "delta_b_minus_a": mean_delta,
            "ci95_low": float(ci_low),
            "ci95_high": float(ci_high),
        }
        row.update(exact_mcnemar_from_correctness(aa, bb))
        rows.append(row)
    return pd.DataFrame(rows)


def vote_fraction_analysis(decision_matrix: pd.DataFrame, case_metadata: pd.DataFrame) -> pd.DataFrame:
    """Analyze cross-model long-vote fraction as a risk score."""
    models, item_ids, D = response_matrix_to_array(decision_matrix)
    md = case_metadata.drop_duplicates("item_id").set_index("item_id").loc[item_ids].reset_index()
    true_long = (md["ground_truth_mapped"] == LONG).astype(int).to_numpy()
    vote_frac = np.nanmean(D, axis=0)
    obs_count = np.sum(~np.isnan(D), axis=0)
    out = pd.DataFrame({"item_id": item_ids, "long_vote_fraction": vote_frac, "n_observed_models": obs_count, "true_long": true_long})
    bins = np.linspace(0, 1, 11)
    out["vote_bin"] = pd.cut(out["long_vote_fraction"], bins=bins, include_lowest=True)
    summary = out.groupby("vote_bin", observed=True).agg(
        n_cases=("item_id", "size"),
        mean_long_vote_fraction=("long_vote_fraction", "mean"),
        empirical_true_long_rate=("true_long", "mean"),
        mean_observed_models=("n_observed_models", "mean"),
    ).reset_index()
    summary["vote_bin"] = summary["vote_bin"].astype(str)
    return summary
