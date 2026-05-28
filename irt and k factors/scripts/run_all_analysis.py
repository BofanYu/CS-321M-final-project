"""Run the full IRT/K-factor analysis pipeline from prepared CSV inputs.

The script reads a prepared IRT folder containing ``response_matrix.csv``,
``case_metadata.csv``, and fitted 1PL item/subject tables, then regenerates the
analysis tables and figures used in the paper.
"""
import argparse
from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

import pandas as pd

from disaster_irt.paths import default_paths, ensure_dir
from disaster_irt.data import load_core_tables, build_decision_matrix_from_metadata, save_tables
from disaster_irt.features import flatten_case_metadata
from disaster_irt.metrics import model_performance, paired_comparisons, vote_fraction_analysis
from disaster_irt.irt import ability_table, item_difficulty_table, difficulty_decile_summary, damage_label_difficulty, residual_factor_analysis
from disaster_irt.validity import feature_label_model_scores, difficulty_validity_correlations
from disaster_irt.kfactor import run_cold_start_experiment
from disaster_irt import plots


def _parse_int_list(text: str) -> tuple[int, ...]:
    """Parse comma-separated integers from a CLI option."""
    try:
        return tuple(int(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected comma-separated integers, got {text!r}") from exc


def parse_args(argv=None):
    """Return command-line options for the reproducible analysis runner."""
    defaults = default_paths(CODE_ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--irt-dir",
        type=Path,
        default=defaults["irt_dir"],
        help="Folder containing response_matrix.csv, case_metadata.csv, and 1PL IRT outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=defaults["output_dir"],
        help="Folder where regenerated tables and figures will be written.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=500,
        help="Number of held-out items per cold-start split.",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_int_list,
        default=(2026, 2027, 2028, 2029, 2030),
        help="Comma-separated random seeds for cold-start splits.",
    )
    parser.add_argument(
        "--ks",
        type=_parse_int_list,
        default=(0, 1, 2, 4, 8),
        help="Comma-separated K-factor dimensions to evaluate.",
    )
    parser.add_argument(
        "--residual-factors",
        type=int,
        default=5,
        help="Number of residual factors to save for exploratory SVD analysis.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Regenerate all analysis outputs from committed prepared inputs."""
    args = parse_args(argv)
    if args.test_size <= 0:
        raise ValueError(f"--test-size must be positive, got {args.test_size}")
    if not args.seeds:
        raise ValueError("--seeds must contain at least one integer")
    if not args.ks:
        raise ValueError("--ks must contain at least one integer")

    paths = default_paths(CODE_ROOT)
    paths["irt_dir"] = args.irt_dir
    paths["output_dir"] = args.output_dir

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
    residual = residual_factor_analysis(response, subject, item_params, n_factors=args.residual_factors)

    correctness_cold = run_cold_start_experiment(
        response,
        flat,
        seeds=args.seeds,
        test_size=args.test_size,
        Ks=args.ks,
    )
    decision_cold = run_cold_start_experiment(
        decision,
        flat,
        seeds=args.seeds,
        test_size=args.test_size,
        Ks=args.ks,
    )

    save_tables({
        "flattened_case_metadata": flat,
        "model_performance": perf,
        "irt_ability_with_performance": ability,
        "paired_comparisons": paired,
        "feature_label_model_metrics": pd.DataFrame([feature_metrics]),
        "difficulty_validity_correlations": corr,
        "difficulty_decile_summary": deciles,
        "damage_label_difficulty": damage_label,
        "vote_fraction_summary": vote_summary,
        "residual_factor_subject_loadings": residual["subject_factors"],
        "residual_factor_item_loadings": residual["item_factors"],
        "residual_factor_variance": residual["variance"],
        "kfactor_correctness_coldstart_summary": correctness_cold["summary"],
        "kfactor_correctness_coldstart_by_split": correctness_cold["metrics_by_split"],
        "kfactor_correctness_coldstart_diagnostics_summary": correctness_cold["diagnostics_summary"],
        "kfactor_correctness_coldstart_diagnostics_by_split": correctness_cold["diagnostics_by_split"],
        "kfactor_decision_coldstart_summary": decision_cold["summary"],
        "kfactor_decision_coldstart_by_split": decision_cold["metrics_by_split"],
        "kfactor_decision_coldstart_diagnostics_summary": decision_cold["diagnostics_summary"],
        "kfactor_decision_coldstart_diagnostics_by_split": decision_cold["diagnostics_by_split"],
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
    plots.plot_coldstart_auc(
        correctness_cold["summary"],
        figs / "kfactor_correctness_coldstart_auc.png",
        "Correctness cold-start: metadata -> item loading",
    )
    plots.plot_coldstart_brier(
        correctness_cold["summary"],
        figs / "kfactor_correctness_coldstart_brier.png",
        "Correctness cold-start: metadata -> item loading",
    )
    plots.plot_coldstart_auc(
        decision_cold["summary"],
        figs / "kfactor_decision_coldstart_auc.png",
        "Raw decision cold-start: metadata -> item loading",
    )
    plots.plot_coldstart_brier(
        decision_cold["summary"],
        figs / "kfactor_decision_coldstart_brier.png",
        "Raw decision cold-start: metadata -> item loading",
    )
    print(f"Done. Outputs saved to {out}")


if __name__ == "__main__":
    main()
