# Disaster Household LLM Evaluation

This folder has two stages:

1. Run household-displacement agents with `run_example6_agents.py`.
2. Analyze the collected model results with IRT and K-factor tools in `irt and k factors/`.

## Folder Layout

The current layout is:

```text
final project/
+-- household_pulse_survey_displaced.csv
+-- run_example6_agents.py
+-- results/
|   +-- <model_name>/
|   |   +-- results_<model_name>_part_0000_0999.json
|   |   +-- results_<model_name>_combined.json
|   |   +-- checkpoints/
|   +-- irt_baseline/
|   +-- irt_baseline_ruleset/
+-- irt and k factors/
    +-- run_disaster_irt_analysis.ipynb
    +-- scripts/
    |   +-- run_all_analysis.py
    +-- src/
        +-- disaster_irt/
```

`irt and k factors` now expects this layout. Its code lives in the subfolder, while input data and outputs live at the project root.

## Stage 1: Run Agents

`run_example6_agents.py` reads `household_pulse_survey_displaced.csv`, filters usable displaced-household records, asks a selected LLM to predict displacement duration, and writes JSON results.

Small test run:

```powershell
python run_example6_agents.py `
  --sample-size 20 `
  --llm-model ollama/qwen2.5:3b `
  --prompting-strategy None `
  --output results/qwen253b_baseline/results_qwen253b_baseline_part_0000_0019.json
```

Typical chunked run:

```powershell
python run_example6_agents.py `
  --sample-size all `
  --start 0 `
  --limit 1000 `
  --llm-model ollama/qwen2.5:3b `
  --prompting-strategy None `
  --output results/qwen253b_baseline/results_qwen253b_baseline_part_0000_0999.json `
  --checkpoint-dir results/qwen253b_baseline/checkpoints `
  --checkpoint-every 200
```

Useful options:

```text
--sample-size all|N
--start N
--limit N
--llm-model MODEL_NAME
--prompting-strategy None|ReadLiterature|ReadRuleset
--temperature FLOAT
--output PATH
--checkpoint-dir PATH
--checkpoint-every N
```

Each JSON contains:

```text
AgentDecisions
DisplacementDurationsMappedToDecisions
DisplacementDurations
HouseholdInfo
Errors
RunMetadata
```

The model decision is correct when `AgentDecisions[i]` equals `DisplacementDurationsMappedToDecisions[i]`.

## Stage 2: Prepare IRT Inputs

The IRT/K-factor analysis does not read every raw JSON file directly by default. It expects a prepared folder:

```text
results/irt_baseline_ruleset/
+-- response_matrix.csv
+-- case_metadata.csv
+-- irt_1pl_item_parameters.csv
+-- irt_1pl_subject_ability.csv
+-- missing_response_summary.csv
```

In this prepared folder:

`response_matrix.csv` is a model-by-case correctness matrix. Rows are models, columns are case/item ids, and values are `1`, `0`, or missing.

`case_metadata.csv` stores the household/case features, true labels, and raw model prediction columns such as `<model_name>_prediction`.

`irt_1pl_item_parameters.csv` stores item difficulty estimates.

`irt_1pl_subject_ability.csv` stores model ability estimates.

`missing_response_summary.csv` records missing or errored responses.

The current `results/irt_baseline_ruleset/` folder already has these files.

## Stage 3: Run IRT and K-factor Analysis

From the project root:

```powershell
python "irt and k factors/scripts/run_all_analysis.py"
```

Or from inside `irt and k factors/`:

```powershell
python scripts/run_all_analysis.py
```

Both commands read:

```text
results/irt_baseline_ruleset/
```

and write:

```text
analysis_outputs/irt_kfactor/
```

## Analysis Outputs

Tables:

```text
analysis_outputs/irt_kfactor/tables/
+-- flattened_case_metadata.csv
+-- model_performance.csv
+-- irt_ability_with_performance.csv
+-- paired_comparisons.csv
+-- feature_label_model_metrics.csv
+-- difficulty_validity_correlations.csv
+-- difficulty_decile_summary.csv
+-- damage_label_difficulty.csv
+-- vote_fraction_summary.csv
+-- residual_factor_subject_loadings.csv
+-- residual_factor_item_loadings.csv
+-- residual_factor_variance.csv
+-- kfactor_correctness_coldstart_summary.csv
+-- kfactor_correctness_coldstart_by_split.csv
+-- kfactor_decision_coldstart_summary.csv
+-- kfactor_decision_coldstart_by_split.csv
```

Figures:

```text
analysis_outputs/irt_kfactor/figures/
+-- irt_ability.png
+-- accuracy_vs_ability.png
+-- coverage.png
+-- difficulty_validity_correlations.png
+-- damage_label_difficulty.png
+-- difficulty_deciles.png
+-- vote_fraction_calibration.png
+-- residual_factor_variance.png
+-- residual_factor1_subject_loadings.png
+-- kfactor_correctness_coldstart_auc.png
+-- kfactor_decision_coldstart_auc.png
```

## Notes

The K-factor analysis is a fast residual-SVD extension on top of a 1PL-style baseline. It is useful for project/report analysis, but it is not a full maximum-likelihood multidimensional IRT model.
