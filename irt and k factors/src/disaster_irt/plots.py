from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _save(fig, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ability(ability: pd.DataFrame, path: Path, top_n: Optional[int] = None):
    df = ability.sort_values("ability", ascending=True).copy()
    if top_n is not None:
        df = df.tail(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(df))))
    ax.barh(df["model"], df["ability"])
    ax.set_xlabel("IRT ability θ")
    ax.set_ylabel("Model / prompt condition")
    ax.set_title("1PL/Rasch model ability")
    _save(fig, path)


def plot_accuracy_vs_ability(ability: pd.DataFrame, path: Path):
    df = ability.dropna(subset=["ability", "accuracy"]).copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["accuracy"], df["ability"])
    for _, r in df.iterrows():
        ax.annotate(r["model"], (r["accuracy"], r["ability"]), fontsize=7, alpha=0.8)
    ax.set_xlabel("Observed accuracy")
    ax.set_ylabel("IRT ability θ")
    ax.set_title("Accuracy vs. IRT ability")
    _save(fig, path)


def plot_coverage(performance: pd.DataFrame, path: Path):
    df = performance.sort_values("coverage", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(df))))
    ax.barh(df["model"], df["coverage"])
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Model / prompt condition")
    ax.set_xlim(0, 1.05)
    ax.set_title("Observed response coverage")
    _save(fig, path)


def plot_validity_correlations(corr: pd.DataFrame, path: Path):
    df = corr.sort_values("spearman_corr_with_irt_difficulty", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(7, max(3.5, 0.45 * len(df))))
    ax.barh(df["construct_validity_variable"], df["spearman_corr_with_irt_difficulty"])
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("Spearman correlation with IRT difficulty")
    ax.set_title("Construct-validity checks for item difficulty")
    _save(fig, path)


def plot_damage_label_difficulty(damage_label: pd.DataFrame, path: Path):
    df = damage_label.copy()
    df["group"] = df["BuildingDamage"].astype(str) + " | " + df["binary_label"].astype(str)
    df = df.sort_values("mean_irt_difficulty", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(df))))
    ax.barh(df["group"], df["mean_irt_difficulty"])
    ax.set_xlabel("Mean IRT difficulty")
    ax.set_title("Item difficulty by damage cue and true label")
    _save(fig, path)


def plot_difficulty_deciles(deciles: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(deciles["difficulty_decile"], deciles["empirical_accuracy"], marker="o", label="Empirical accuracy")
    if "simple_cue_label_conflict" in deciles.columns:
        ax.plot(deciles["difficulty_decile"], deciles["simple_cue_label_conflict"], marker="o", label="Cue-label conflict rate")
    if "feature_ambiguity" in deciles.columns:
        ax.plot(deciles["difficulty_decile"], deciles["feature_ambiguity"], marker="o", label="Feature ambiguity")
    ax.set_xlabel("IRT difficulty decile (1=easiest, 10=hardest)")
    ax.set_ylabel("Mean value")
    ax.set_title("Behavioral complexity across IRT difficulty deciles")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_vote_fraction(vote_summary: pd.DataFrame, path: Path):
    df = vote_summary.copy()
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, df["empirical_true_long_rate"], marker="o")
    ax.set_xticks(x)
    ax.set_xticklabels(df["vote_bin"], rotation=45, ha="right")
    ax.set_xlabel("Long-vote fraction bin")
    ax.set_ylabel("Empirical true long rate")
    ax.set_title("Cross-model vote fraction as risk score")
    _save(fig, path)


def plot_residual_factor_variance(variance: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(variance["factor"].astype(str), variance["residual_variance_share"])
    ax.set_xlabel("Residual factor")
    ax.set_ylabel("Residual variance share")
    ax.set_title("Low-rank residual structure beyond 1PL")
    _save(fig, path)


def plot_subject_factor(subject_factors: pd.DataFrame, path: Path, factor: str = "factor1"):
    df = subject_factors.sort_values(factor, ascending=True).copy()
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(df))))
    ax.barh(df["model"], df[factor])
    ax.set_xlabel(f"Subject loading: {factor}")
    ax.set_ylabel("Model")
    ax.set_title(f"Model loadings on residual {factor}")
    _save(fig, path)


def plot_coldstart_auc(summary: pd.DataFrame, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in summary["method"].unique():
        df = summary[summary["method"] == method].sort_values("K")
        ax.plot(df["K"], df["mean_auc"], marker="o", label=method)
    ax.set_xlabel("Number of item factors K")
    ax.set_ylabel("Held-out AUC")
    ax.set_title(title)
    ax.legend(fontsize=7)
    _save(fig, path)


def plot_coldstart_brier(summary: pd.DataFrame, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in summary["method"].unique():
        df = summary[summary["method"] == method].sort_values("K")
        ax.plot(df["K"], df["mean_brier"], marker="o", label=method)
    ax.set_xlabel("Number of item factors K")
    ax.set_ylabel("Held-out Brier score")
    ax.set_title(title)
    ax.legend(fontsize=7)
    _save(fig, path)
