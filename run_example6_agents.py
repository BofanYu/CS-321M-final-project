import argparse
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_PYRECODES = PROJECT_ROOT / ".tmp_pyrecodes_main"
if LOCAL_PYRECODES.exists():
    sys.path.insert(0, str(LOCAL_PYRECODES))

from pyrecodes.household.convert_survey_to_household_info import convert_survey_to_household_info
from pyrecodes.household.household_survey_gpt import HouseholdSurveyGPT


HOUSEHOLD_INFO_TEMPLATE = {
    "SocioEconomicParameters": {
        "State": None,
        "MSA": None,
        "Tenure": None,
        "Income": None,
        "Occupants": None,
        "Kids5years": None,
        "Kids5-11years": None,
        "Kids12-17years": None,
        "EmploymentStatus": None,
    },
    "BuildingDamage": None,
    "DisasterType": None,
    "DisplacementDuration": None,
    "WaterAccess": None,
    "PowerAccess": None,
    "FoodAccess": None,
    "UnsanitaryConditions": None,
    "InformGPTMethod": None,
}


def conditions_to_include_household_met(household_info):
    if household_info["BuildingDamage"] is None:
        return False
    if household_info["DisplacementDuration"] is None:
        return False
    if household_info["WaterAccess"] is None:
        return False
    if household_info["PowerAccess"] is None:
        return False
    if household_info["FoodAccess"] is None:
        return False
    if household_info["UnsanitaryConditions"] is None:
        return False
    if len(household_info["DisasterType"]) != 1:
        return False
    for value in household_info["SocioEconomicParameters"].values():
        if value is None:
            return False
    return True


def prepare_households(survey_csv):
    household_pulse_survey_df = pd.read_csv(survey_csv)
    households_info = []

    displaced_rows = household_pulse_survey_df[household_pulse_survey_df["ND_DISPLACE"] == 1]
    for _, household_survey_row in displaced_rows.iterrows():
        household_info = convert_survey_to_household_info(
            household_survey_row,
            HOUSEHOLD_INFO_TEMPLATE,
        )
        if conditions_to_include_household_met(household_info):
            household_info["DisasterType"] = household_info["DisasterType"][0]
            households_info.append(household_info)

    return households_info


def select_households(households_info, sample_size, seed, start, limit):
    random.seed(seed)
    if sample_size == "all":
        selected = list(households_info)
    else:
        n = int(sample_size)
        if n > len(households_info):
            raise ValueError(f"sample_size={n} exceeds available households={len(households_info)}")
        selected = random.sample(households_info, n)

    if start:
        selected = selected[start:]
    if limit is not None:
        selected = selected[:limit]
    return selected


def map_actual_duration_to_decision(actual):
    if "Less than a week" in actual:
        return "LeaveHomeForLessThanAWeek"
    return "LeaveHomeForMoreThanAWeek"


def save_results(results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    tmp_path.replace(output_path)


def save_checkpoint(results, output_path, checkpoint_dir, processed_count):
    if checkpoint_dir is None:
        return
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{output_path.stem}_checkpoint_{processed_count:05d}.json"
    save_results(results, checkpoint_path)


def run(args):
    survey_csv = Path(args.survey_csv)
    output_path = Path(args.output)

    households_info = prepare_households(survey_csv)
    selected_households = select_households(
        households_info,
        sample_size=args.sample_size,
        seed=args.seed,
        start=args.start,
        limit=args.limit,
    )

    results = {
        "AgentDecisions": [],
        "DisplacementDurationsMappedToDecisions": [],
        "DisplacementDurations": [],
        "HouseholdInfo": [],
        "Errors": [],
        "RunMetadata": {
            "LLMModel": args.llm_model,
            "PromptingStrategy": args.prompting_strategy,
            "Temperature": args.temperature,
            "Seed": args.seed,
            "SampleSize": args.sample_size,
            "Start": args.start,
            "Limit": args.limit,
            "NumPreparedHouseholds": len(households_info),
            "NumSelectedHouseholds": len(selected_households),
            "HouseholdRuntimeSeconds": [],
        },
    }

    print(f"Prepared {len(households_info)} usable households.")
    print(f"Running {len(selected_households)} households.")
    print(f"Model: {args.llm_model}")
    print(f"Prompting strategy: {args.prompting_strategy}")
    print(f"Temperature: {args.temperature}")
    print(f"Output: {output_path}")

    run_started_at = time.perf_counter()
    for i, household_info in enumerate(selected_households, start=1):
        household_started_at = time.perf_counter()
        print(f"[{i}/{len(selected_households)}] running household...", flush=True)

        household_info = dict(household_info)
        household_info["InformGPTMethod"] = args.prompting_strategy

        try:
            actual = household_info["DisplacementDuration"]
            actual_mapped = map_actual_duration_to_decision(actual)
            if args.mock_decision is None:
                agent = HouseholdSurveyGPT()
                agent.set_parameters(
                    household_info,
                    api_key_filename=args.api_key_filename,
                    temperature=args.temperature,
                    llm_model=args.llm_model,
                )
                agent.create_time_step_narrative(
                    household_info["BuildingDamage"],
                    {
                        "WaterAccess": household_info["WaterAccess"],
                        "PowerAccess": household_info["PowerAccess"],
                        "FoodAccess": household_info["FoodAccess"],
                        "UnsanitaryConditions": household_info["UnsanitaryConditions"],
                    },
                    household_info["DisasterType"],
                )
                agent.decide()
                decision = agent.decisions[-1]
            elif args.mock_decision == "actual":
                decision = actual_mapped
            else:
                decision = args.mock_decision

            results["AgentDecisions"].append(decision)
            results["DisplacementDurationsMappedToDecisions"].append(actual_mapped)
            results["DisplacementDurations"].append(actual)
            results["HouseholdInfo"].append(household_info)
        except Exception as exc:
            error = {
                "Index": i - 1,
                "ErrorType": type(exc).__name__,
                "ErrorMessage": str(exc),
                "HouseholdInfo": household_info,
            }
            results["Errors"].append(error)
            print(f"[{i}/{len(selected_households)}] ERROR {type(exc).__name__}: {exc}", flush=True)
            if args.stop_on_error:
                raise
        finally:
            household_elapsed = time.perf_counter() - household_started_at
            results["RunMetadata"]["HouseholdRuntimeSeconds"].append(household_elapsed)
            print(f"[{i}/{len(selected_households)}] elapsed {household_elapsed:.1f}s", flush=True)

        if args.checkpoint_every and i % args.checkpoint_every == 0:
            elapsed = time.perf_counter() - run_started_at
            results["RunMetadata"]["TotalRuntimeSeconds"] = elapsed
            completed = len(results["AgentDecisions"]) + len(results["Errors"])
            results["RunMetadata"]["AverageRuntimeSecondsPerCompletedHousehold"] = elapsed / completed
            save_results(results, output_path)
            save_checkpoint(results, output_path, args.checkpoint_dir, i)
            print(f"Checkpoint saved after {i} households.", flush=True)

    total_elapsed = time.perf_counter() - run_started_at
    completed = len(results["AgentDecisions"]) + len(results["Errors"])
    results["RunMetadata"]["TotalRuntimeSeconds"] = total_elapsed
    results["RunMetadata"]["AverageRuntimeSecondsPerCompletedHousehold"] = (
        total_elapsed / completed if completed else None
    )
    save_results(results, output_path)

    correct = sum(
        pred == actual
        for pred, actual in zip(
            results["AgentDecisions"],
            results["DisplacementDurationsMappedToDecisions"],
        )
    )
    accuracy = correct / len(results["AgentDecisions"]) if results["AgentDecisions"] else None
    results["RunMetadata"]["Accuracy"] = accuracy
    save_results(results, output_path)

    print(f"Finished in {total_elapsed:.1f}s.")
    print(f"Successful households: {len(results['AgentDecisions'])}")
    print(f"Errors: {len(results['Errors'])}")
    if accuracy is not None:
        print(f"Accuracy: {accuracy:.4f}")
    print(f"Saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run pyrecodes Example 6 household agents without plotting.")
    parser.add_argument("--survey-csv", default="household_pulse_survey_displaced.csv")
    parser.add_argument("--output", default="results_example6_agents.json")
    parser.add_argument("--sample-size", default="20", help="Integer sample size, or 'all'.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=0, help="Start offset after sampling. Useful for job arrays.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum households to run after start offset.")
    parser.add_argument(
        "--prompting-strategy",
        default="None",
        choices=["None", "ReadLiterature", "ReadRuleset"],
    )
    parser.add_argument("--llm-model", default="ollama/qwen2.5:3b")
    parser.add_argument("--api-key-filename", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--mock-decision",
        choices=["actual", "LeaveHomeForLessThanAWeek", "LeaveHomeForMoreThanAWeek"],
        default=None,
        help="Skip the LLM call for smoke tests. Use 'actual' to emit the mapped true label.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=10000)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Optional directory for separate checkpoint JSON snapshots.",
    )
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
