from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler

from .data import response_matrix_to_array
from .features import one_hot_feature_matrix
from .utils import safe_logit


@dataclass
class KFactorFit:
    theta: np.ndarray
    b: np.ndarray
    U: np.ndarray
    V: np.ndarray
    singular_values: np.ndarray
    train_item_idx: np.ndarray


def fit_residual_svd_kfactor(Y: np.ndarray, train_item_idx: np.ndarray, kmax: int = 8) -> KFactorFit:
    """Fast K-factor approximation around a 1PL model.

    This is not a full maximum-likelihood multidimensional IRT fit. It is a clean,
    fast proof-of-concept:
        1. estimate model ability from model means;
        2. estimate item difficulty from item means;
        3. compute standardized residuals;
        4. factorize residuals with SVD.
    """
    Ytr = Y[:, train_item_idx].astype(float)
    obs = ~np.isnan(Ytr)
    model_mean = np.nanmean(Ytr, axis=1)
    item_mean = np.nanmean(Ytr, axis=0)
    theta = safe_logit(model_mean)
    theta = theta - np.nanmean(theta)
    b = -safe_logit(item_mean)
    b = b - np.nanmean(b)
    eta0 = theta[:, None] - b[None, :]
    p0 = expit(eta0)
    denom = np.sqrt(np.clip(p0 * (1 - p0), 1e-4, None))
    R = np.zeros_like(Ytr, dtype=float)
    R[obs] = (Ytr[obs] - p0[obs]) / denom[obs]
    U_s, s, Vt = np.linalg.svd(R, full_matrices=False)
    k = min(kmax, len(s))
    U = U_s[:, :k] * np.sqrt(s[:k])
    V = Vt[:k, :].T * np.sqrt(s[:k])
    return KFactorFit(theta=theta, b=b, U=U, V=V, singular_values=s[:k], train_item_idx=np.asarray(train_item_idx))


def _metrics(y, p) -> Dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=int)
    out = {
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "acc_at_0p5": float(accuracy_score(y, (p >= 0.5).astype(int))),
    }
    return out


def cold_start_one_split(
    response_matrix: pd.DataFrame,
    flat_metadata: pd.DataFrame,
    seed: int = 2026,
    test_size: int = 500,
    K: int = 4,
    alpha: float = 10.0,
    min_observed_models: int = 10,
    return_predictions: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Hold out entire items, learn metadata -> item factors, test on held-out items.

    response_matrix can be either:
    - correctness matrix: Y_mi=1 if model m is correct;
    - raw decision matrix: Y_mi=1 if model m predicts Long.

    The code is identical; only the interpretation changes.
    """
    models, item_ids, Y = response_matrix_to_array(response_matrix)
    N = len(item_ids)
    md = flat_metadata.drop_duplicates("item_id").set_index("item_id").loc[item_ids].reset_index()
    Xdf, X_all_raw = one_hot_feature_matrix(md)
    # Standardize all columns. This uses the full feature design only for column definition; targets are train-only.
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_all = scaler.fit_transform(X_all_raw)

    rng = np.random.default_rng(seed)
    eligible = np.where(np.sum(~np.isnan(Y), axis=0) >= min_observed_models)[0]
    if len(eligible) < test_size:
        raise ValueError(f"Only {len(eligible)} eligible items, fewer than test_size={test_size}")
    test_idx = np.sort(rng.choice(eligible, size=test_size, replace=False))
    train_idx = np.setdiff1d(np.arange(N), test_idx)

    kfit = fit_residual_svd_kfactor(Y, train_idx, kmax=max(8, K))
    if K == 0:
        T = kfit.b[:, None]
    else:
        T = np.column_stack([kfit.b, kfit.V[:, :K]])
    tmean = T.mean(axis=0)
    tstd = T.std(axis=0)
    tstd[tstd == 0] = 1
    Tz = (T - tmean) / tstd

    ridge = Ridge(alpha=alpha)
    ridge.fit(X_all[train_idx], Tz)
    Tpred_train = ridge.predict(X_all[train_idx]) * tstd + tmean
    Tpred_test = ridge.predict(X_all[test_idx]) * tstd + tmean
    if Tpred_train.ndim == 1:
        Tpred_train = Tpred_train[:, None]
        Tpred_test = Tpred_test[:, None]
    bpred_train = Tpred_train[:, 0]
    bpred_test = Tpred_test[:, 0]

    Yte = Y[:, test_idx]
    obste = ~np.isnan(Yte)
    mte, ite = np.where(obste)
    yte = Yte[mte, ite].astype(int)
    p_model_mean = np.nanmean(Y[:, train_idx], axis=1)[mte]
    eta_test = kfit.theta[mte] - bpred_test[ite]
    p_1pl = expit(eta_test)

    if K > 0:
        Vpred_train = Tpred_train[:, 1:]
        Vpred_test = Tpred_test[:, 1:]
        Ytr = Y[:, train_idx]
        obstr = ~np.isnan(Ytr)
        mtr, itr = np.where(obstr)
        ytr = Ytr[mtr, itr].astype(int)
        eta_train = kfit.theta[mtr] - bpred_train[itr]
        f_train = np.sum(kfit.U[mtr, :K] * Vpred_train[itr], axis=1)
        # Calibrate a scalar factor weight on training responses.
        cs = np.linspace(-2, 2, 161)
        losses = [log_loss(ytr, expit(eta_train + c * f_train), labels=[0, 1]) for c in cs]
        c_best = float(cs[int(np.argmin(losses))])
        f_test = np.sum(kfit.U[mte, :K] * Vpred_test[ite], axis=1)
        p_k = expit(eta_test + c_best * f_test)
    else:
        c_best = np.nan
        p_k = p_1pl

    item_mean_test = np.nanmean(Yte, axis=0)
    p_oracle_item_mean = item_mean_test[ite]

    method_name = "feature_1pl_b_only_K0" if K == 0 else f"feature_K{K}_b_plus_loading"
    rows = []
    for name, p in [
        ("model_mean_only", p_model_mean),
        (method_name, p_k),
        ("oracle_item_mean_not_coldstart", p_oracle_item_mean),
    ]:
        row = _metrics(yte, p)
        row.update({"method": name, "K": K, "seed": seed, "test_items": test_size, "n_test_responses": int(len(yte)), "alpha": alpha, "factor_scale_c": c_best if name == method_name else np.nan})
        rows.append(row)
    metrics = pd.DataFrame(rows)

    b_emp = -safe_logit(item_mean_test)
    diag = pd.DataFrame([{
        "seed": seed,
        "K": K,
        "test_items": test_size,
        "n_test_responses": int(len(yte)),
        "b_empirical_corr": float(np.corrcoef(b_emp, bpred_test)[0, 1]) if len(b_emp) > 1 else np.nan,
        "b_empirical_r2": float(r2_score(b_emp, bpred_test)) if len(b_emp) > 1 else np.nan,
        "factor_scale_c": c_best,
    }])

    pred_df = None
    if return_predictions:
        rows = []
        for m, j, yval, pk, p1, pm in zip(mte, ite, yte, p_k, p_1pl, p_model_mean):
            global_item_idx = test_idx[j]
            rows.append({
                "item_id": item_ids[global_item_idx],
                "model": models[m],
                "observed_response": int(yval),
                "p_K_coldstart": float(pk),
                "p_feature_1pl": float(p1),
                "p_model_mean": float(pm),
            })
        pred_df = pd.DataFrame(rows)
    return metrics, diag, pred_df


def run_cold_start_experiment(
    response_matrix: pd.DataFrame,
    flat_metadata: pd.DataFrame,
    seeds: Iterable[int] = (2026, 2027, 2028, 2029, 2030),
    Ks: Iterable[int] = (0, 1, 2, 4, 8),
    test_size: int = 500,
    alpha: float = 10.0,
    min_observed_models: int = 10,
) -> Dict[str, pd.DataFrame]:
    """Run repeated new-item cold-start experiment."""
    all_metrics = []
    all_diag = []
    for seed in seeds:
        for K in Ks:
            metrics, diag, _ = cold_start_one_split(
                response_matrix=response_matrix,
                flat_metadata=flat_metadata,
                seed=seed,
                test_size=test_size,
                K=K,
                alpha=alpha,
                min_observed_models=min_observed_models,
                return_predictions=False,
            )
            all_metrics.append(metrics)
            all_diag.append(diag)
    metrics = pd.concat(all_metrics, ignore_index=True)
    diag = pd.concat(all_diag, ignore_index=True)
    summary = metrics.groupby(["method", "K"], dropna=False).agg(
        splits=("seed", "nunique"),
        mean_auc=("auc", "mean"), sd_auc=("auc", "std"),
        mean_logloss=("logloss", "mean"), sd_logloss=("logloss", "std"),
        mean_brier=("brier", "mean"), sd_brier=("brier", "std"),
        mean_acc_at_0p5=("acc_at_0p5", "mean"), sd_acc_at_0p5=("acc_at_0p5", "std"),
        mean_test_responses=("n_test_responses", "mean"),
    ).reset_index().sort_values(["method", "K"])
    diag_summary = diag.groupby("K").agg(
        mean_b_empirical_corr=("b_empirical_corr", "mean"),
        sd_b_empirical_corr=("b_empirical_corr", "std"),
        mean_b_empirical_r2=("b_empirical_r2", "mean"),
        sd_b_empirical_r2=("b_empirical_r2", "std"),
        mean_factor_scale_c=("factor_scale_c", "mean"),
    ).reset_index()
    return {"metrics_by_split": metrics, "diagnostics_by_split": diag, "summary": summary, "diagnostics_summary": diag_summary}
