# Disaster Household LLM Evaluation

This repository contains the code and prepared outputs for evaluating LLM
household-displacement predictions and analyzing model behavior with IRT and
K-factor methods.

The reproducible analysis has two stages:

1. `run_example6_agents.py` generates raw LLM decisions from Household Pulse
   Survey records.
2. `irt and k factors/scripts/run_all_analysis.py` regenerates the IRT,
   residual K-factor, cold-start, table, and figure outputs from prepared CSV
   inputs.

## Environment Setup

Recommended environment:

```text
Python 3.11 or 3.12
```

Install dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

The analysis-only pipeline uses `numpy`, `pandas`, `scipy`, `scikit-learn`, and
`matplotlib`. Agent generation also uses the OpenAI Python client against either
OpenAI or an Ollama OpenAI-compatible local server.

For local Ollama runs, install Ollama separately and pull the model you plan to
use, for example:

```powershell
ollama pull qwen2.5:3b
```

The repository includes the minimal local `pyrecodes` subset needed by
`run_example6_agents.py` under `.tmp_pyrecodes_main/`. If that folder is removed,
install the matching `pyrecodes` package/source before running agent generation.

## Data

The survey input expected by the agent runner is:

```text
household_pulse_survey_displaced.csv
```

This CSV is a filtered Household Pulse Survey displacement dataset used by the
project. To reproduce from a fresh copy, place the CSV at the repository root
with the same filename. The script filters rows with `ND_DISPLACE == 1` and keeps
only rows with complete prompting fields.

The IRT/K-factor analysis starts from prepared files in:

```text
results/irt_baseline_ruleset/
+-- response_matrix.csv
+-- case_metadata.csv
+-- irt_1pl_item_parameters.csv
+-- irt_1pl_subject_ability.csv
+-- missing_response_summary.csv
```

These prepared files are committed so the analysis tables and figures can be
regenerated without re-running all LLM calls.

## Repository Structure

```text
final project/
+-- README.md
+-- requirements.txt
+-- household_pulse_survey_displaced.csv
+-- run_example6_agents.py
+-- .tmp_pyrecodes_main/
|   +-- pyrecodes/                         # local subset used by the agent runner
+-- results/
|   +-- <model_name>/
|   |   +-- results_<model_name>_combined.json
|   |   +-- results_<model_name>_part_*.json
|   +-- irt_baseline_ruleset/
|       +-- response_matrix.csv
|       +-- case_metadata.csv
|       +-- irt_1pl_item_parameters.csv
|       +-- irt_1pl_subject_ability.csv
+-- irt and k factors/
|   +-- requirements.txt
|   +-- run_disaster_irt_analysis.ipynb
|   +-- scripts/
|   |   +-- run_all_analysis.py
|   +-- src/disaster_irt/
|       +-- data.py
|       +-- features.py
|       +-- metrics.py
|       +-- irt.py
|       +-- kfactor.py
|       +-- plots.py
|       +-- validity.py
+-- analysis_outputs/
    +-- irt_kfactor/
        +-- tables/
        +-- figures/
```

## Reproduce the Main Results

From the repository root, run:

```powershell
python "irt and k factors/scripts/run_all_analysis.py"
```

This reads:

```text
results/irt_baseline_ruleset/
```

and writes:

```text
analysis_outputs/irt_kfactor/tables/
analysis_outputs/irt_kfactor/figures/
```

The default cold-start experiment uses fixed seeds:

```text
2026, 2027, 2028, 2029, 2030
```

and evaluates:

```text
K = 0, 1, 2, 4, 8
```

To run a faster smoke test without overwriting the main output folder:

```powershell
python "irt and k factors/scripts/run_all_analysis.py" `
  --output-dir analysis_outputs/irt_kfactor_smoke `
  --test-size 50 `
  --seeds 2026 `
  --ks 0,1
```

Useful analysis options:

```text
--irt-dir PATH              prepared IRT input folder
--output-dir PATH           destination for regenerated tables and figures
--test-size N               held-out items per K-factor cold-start split
--seeds 2026,2027           comma-separated cold-start random seeds
--ks 0,1,2,4,8              comma-separated K-factor dimensions
--residual-factors N        number of residual SVD factors to save
```

## Generated Tables and Figures

`run_all_analysis.py` regenerates these main tables:

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
+-- kfactor_correctness_coldstart_diagnostics_summary.csv
+-- kfactor_correctness_coldstart_diagnostics_by_split.csv
+-- kfactor_decision_coldstart_summary.csv
+-- kfactor_decision_coldstart_by_split.csv
+-- kfactor_decision_coldstart_diagnostics_summary.csv
+-- kfactor_decision_coldstart_diagnostics_by_split.csv
```

and these main figures:

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
+-- kfactor_correctness_coldstart_brier.png
+-- kfactor_decision_coldstart_auc.png
+-- kfactor_decision_coldstart_brier.png
```

## Agent Generation

Small smoke test without an LLM call:

```powershell
python run_example6_agents.py `
  --sample-size 20 `
  --mock-decision actual `
  --output results/smoke_mock/results_smoke_mock_part_0000_0019.json
```

Small local Ollama run:

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

Useful agent options:

```text
--sample-size all|N
--start N
--limit N
--seed N
--llm-model GPT|ollama/MODEL_NAME
--prompting-strategy None|ReadLiterature|ReadRuleset
--temperature FLOAT
--mock-decision actual|LeaveHomeForLessThanAWeek|LeaveHomeForMoreThanAWeek
--output PATH
--checkpoint-dir PATH
--checkpoint-every N
```

Each raw JSON contains:

```text
AgentDecisions
DisplacementDurationsMappedToDecisions
DisplacementDurations
HouseholdInfo
Errors
RunMetadata
```

The model decision is correct when `AgentDecisions[i]` equals
`DisplacementDurationsMappedToDecisions[i]`.

## Runtime and Compute

Analysis from prepared CSV files usually runs in a few minutes on a laptop CPU.
The fast smoke-test command above should run much faster because it uses one
seed, two K values, and fewer held-out items.

Agent generation is the expensive step. Runtime depends on the local or remote
LLM backend, model size, and sample size. Chunked runs with 1,000 households can
take from tens of minutes to several hours per model. The committed prepared IRT
inputs let graders reproduce the analysis without repeating those LLM calls.

## Reproducibility Notes

- Python dependencies are pinned in `requirements.txt`.
- Agent sampling uses `--seed` with default `42`.
- Feature-label validity uses `random_state=2026`.
- K-factor cold-start uses fixed seeds by default: `2026,2027,2028,2029,2030`.
- `run_all_analysis.py` is the end-to-end script that regenerates the reported
  analysis tables and figures from the prepared IRT inputs.

## Attribution and Licenses

Most project-specific code is original, including the IRT/K-factor analysis
modules under `irt and k factors/src/disaster_irt/`, the reproducible analysis
runner, and the project README.

The household agent runner builds on a small local subset adapted from the
external `pyrecodes` package under `.tmp_pyrecodes_main/pyrecodes/`. Those files
are marked with module-level attribution comments/docstrings. The adapted
package is licensed as BSD-3-Clause; see `THIRD_PARTY_NOTICES.md` for the source
link, license text, and a list of project-specific modifications.

## Method Note

The K-factor analysis is a fast residual-SVD extension on top of a 1PL-style
baseline. It is useful for project/report analysis, but it is not a full
maximum-likelihood multidimensional IRT model. In the cold-start experiment,
metadata features are one-hot/numeric encoded and a Ridge regression predicts
item difficulty plus residual item loadings for held-out items.
