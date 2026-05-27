"""Optional non-notebook runner.

Run from the pyrecode E6 root:
    python scripts/run_all_analysis.py
"""
from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from disaster_irt.paths import default_paths, ensure_dir
from disaster_irt.data import load_core_tables, build_decision_matrix_from_metadata, save_tables
from disaster_irt.features import flatten_case_metadata
from disaster_irt.metrics import model_performance, paired_comparisons, vote_fraction_analysis
from disaster_irt.irt import ability_table, item_difficulty_table, difficulty_decile_summary, damage_label_difficulty, residual_factor_analysis
from disaster_irt.validity import feature_label_model_scores, difficulty_validity_correlations
from disaster_irt.kfactor import run_cold_start_experiment
from disaster_irt import plots


def main():
    paths = default_paths(CODE_ROOT)
    out = ensure_dir(paths["output_dir"])
    figs = ensure_dir(out / "figures")
    tables = ensure_dir(out / "tables")

    data = load_core_tables(paths["irt_dir"])
    response = data["response_matrix"]
    meta = data["case_metadata"]
    item_params = data["item_params"]
    subject = data["subject_ability"]

    flat = flatten_case_metadata(meta)
    perf = model_performance(response, meta)
    ability = ability_table(subject, perf)
    paired = paired_comparisons(response)

    scored_meta, feature_metrics = feature_label_model_scores(flat)
    item_table = item_difficulty_table(item_params, scored_meta)
    corr = difficulty_validity_correlations(item_table)
    deciles = difficulty_decile_summary(item_table)
    damage_label = damage_label_difficulty(item_table)

    decision = build_decision_matrix_from_metadata(meta, model_names=response["model"])
    vote_summary = vote_fraction_analysis(decision, meta)
    residual = residual_factor_analysis(response, subject, item_params, n_factors=5)

    correctness_cold = run_cold_start_experiment(response, flat, test_size=500, Ks=(0, 1, 2, 4, 8))
    decision_cold = run_cold_start_experiment(decision, flat, test_size=500, Ks=(0, 1, 2, 4, 8))

    save_tables({
        "flattened_case_metadata": flat,
        "model_performance": perf,
        "irt_ability_with_performance": ability,
        "paired_comparisons": paired,
        "feature_label_model_metrics": __import__('pandas').DataFrame([feature_metrics]),
        "difficulty_validity_correlations": corr,
        "difficulty_decile_summary": deciles,
        "damage_label_difficulty": damage_label,
        "vote_fraction_summary": vote_summary,
        "residual_factor_subject_loadings": residual["subject_factors"],
        "residual_factor_item_loadings": residual["item_factors"],
        "residual_factor_variance": residual["variance"],
        "kfactor_correctness_coldstart_summary": correctness_cold["summary"],
        "kfactor_correctness_coldstart_by_split": correctness_cold["metrics_by_split"],
        "kfactor_decision_coldstart_summary": decision_cold["summary"],
        "kfactor_decision_coldstart_by_split": decision_cold["metrics_by_split"],
    }, tables)

    plots.plot_ability(ability, figs / "irt_ability.png")
    plots.plot_accuracy_vs_ability(ability, figs / "accuracy_vs_ability.png")
    plots.plot_coverage(perf, figs / "coverage.png")
    plots.plot_validity_correlations(corr, figs / "difficulty_validity_correlations.png")
    plots.plot_damage_label_difficulty(damage_label, figs / "damage_label_difficulty.png")
    plots.plot_difficulty_deciles(deciles, figs / "difficulty_deciles.png")
    plots.plot_vote_fraction(vote_summary, figs / "vote_fraction_calibration.png")
    plots.plot_residual_factor_variance(residual["variance"], figs / "residual_factor_variance.png")
    plots.plot_subject_factor(residual["subject_factors"], figs / "residual_factor1_subject_loadings.png", "factor1")
    plots.plot_coldstart_auc(correctness_cold["summary"], figs / "kfactor_correctness_coldstart_auc.png", "Correctness cold-start: metadata → item loading")
    plots.plot_coldstart_auc(decision_cold["summary"], figs / "kfactor_decision_coldstart_auc.png", "Raw decision cold-start: metadata → item loading")
    print(f"Done. Outputs saved to {out}")


if __name__ == "__main__":
    main()
