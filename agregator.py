import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime



def find_exp_dirs(experiments_dir: str, model: str, strategy: str) -> list[str]:
    """
    Lists all exp_* folders directly under <experiments_dir>/<model>/<strategy>/
    that contain a pareto_history.json file.

    Returns:
        Sorted list of folder paths.
    """
    search_dir = os.path.join(experiments_dir, model, strategy)

    if not os.path.isdir(search_dir):
        return []

    found = []
    for entry in os.scandir(search_dir):
        if entry.is_dir() and entry.name.startswith("exp_"):
            json_path = os.path.join(entry.path, "pareto_history.json")
            if os.path.isfile(json_path):
                found.append(entry.path)

    return sorted(found)



def load_final_pareto(exp_dir: str) -> tuple[list[dict], list[str], list[str]]:
    """
    Loads pareto_history.json from an exp_* folder and returns only the
    final generation's solutions.

    Returns:
        solutions:       list of {objectives, proportions} dicts
        objective_names: list of objective name strings
        ingredient_names: list of ingredient name strings

    Raises:
        ValueError: if the file is missing required fields
    """
    json_path = os.path.join(exp_dir, "pareto_history.json")
    with open(json_path, "r") as f:
        history = json.load(f)

    if not history:
        raise ValueError("Empty history.")

    final_gen = history[-1]["pareto_front"]

    if not final_gen:
        raise ValueError("Final generation has no solutions.")

    first = final_gen[0]
    if "proportions" not in first:
        raise ValueError(
            'Missing "proportions" field. '
            "Re-run experiments with the updated optimizer.py and logger.py."
        )

    objective_names  = list(first["objectives"].keys())
    ingredient_names = list(first["proportions"].keys())

    solutions = [
        {
            "objectives":  sol["objectives"],
            "proportions": sol["proportions"],
        }
        for sol in final_gen
    ]

    return solutions, objective_names, ingredient_names



def round_proportions(proportions: dict, decimals: int = 2) -> dict:
    """Returns a new dict with all values rounded to `decimals` places."""
    return {k: round(v, decimals) for k, v in proportions.items()}


def proportions_to_key(proportions: dict) -> str:
    """
    Converts a proportions dict into a stable string key.
    """
    return json.dumps(proportions, sort_keys=True)


def compute_solution_frequency(all_solutions: list[dict], decimals: int = 2) -> list[dict]:
    """
    Groups solutions by their rounded proportions and counts how many times
    each unique formulation appeared across all runs.

    Because the same rounded formulation may carry slightly different strength
    values across runs, all observed strength values are collected and the
    average is also stored.

    Args:
        all_solutions: list of solution dicts, each containing 'proportions',
                       'objectives', and 'run' keys.
        decimals:      number of decimal places to consider (default: 2).

    
    """
    frequency: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "runs": [], "strength_values": []}
    )

    for sol in all_solutions:
        rounded = round_proportions(sol["proportions"], decimals)
        key = proportions_to_key(rounded)
        entry = frequency[key]

        entry["proportions"] = rounded
        entry["count"] += 1

        run = sol.get("run", "unknown")
        if run not in entry["runs"]:
            entry["runs"].append(run)

        strength = sol.get("objectives", {}).get("strength")
        if strength is not None:
            entry["strength_values"].append(strength)

    result = []
    for entry in frequency.values():
        values = entry["strength_values"]
        entry["strength_avg"] = round(sum(values) / len(values), 4) if values else None
        result.append(entry)

    result.sort(key=lambda x: x["count"], reverse=True)
    return result


def print_frequency_summary(frequency: list[dict], top_n: int = 10) -> None:
    """Prints a readable summary of the most frequent solutions."""
    print(f"\n{'='*60}")
    print(f"  SOLUTION FREQUENCY ANALYSIS (top {top_n})")
    print(f"{'='*60}")
    print(f"  Unique solutions (2 decimal places): {len(frequency)}\n")

    for i, entry in enumerate(frequency[:top_n], start=1):
        strength_str = (
            f"  strength_avg={entry['strength_avg']}"
            if entry.get("strength_avg") is not None
            else ""
        )
        print(f"  #{i}  count={entry['count']}  runs={len(entry['runs'])}{strength_str}")
        for ingredient, value in entry["proportions"].items():
            print(f"       {ingredient}: {value:.2f}")
        print()



def aggregate(experiments_dir: str, model: str, strategy: str) -> dict:
    """
    Discovers all exp_* folders for the given model/strategy, loads their
    final Pareto fronts, and returns the aggregated data structure.
    """
    exp_dirs = find_exp_dirs(experiments_dir, model, strategy)

    if not exp_dirs:
        search_path = os.path.join(experiments_dir, model, strategy)
        print(f"Error: no exp_* folders found under '{search_path}'.")
        sys.exit(1)

    print(f"Model: {model}  |  Strategy: {strategy}")
    print(f"Found {len(exp_dirs)} exp_* folder(s).\n")

    all_solutions    = []
    objective_names  = None
    ingredient_names = None
    skipped = 0

    for exp_dir in exp_dirs:
        run_name = os.path.basename(exp_dir)
        print(f"  Loading {run_name} ...")
        try:
            sols, obj_names, ing_names = load_final_pareto(exp_dir)
        except Exception as e:
            print(f"    Warning: {e}. Skipping.")
            skipped += 1
            continue

        # validate consistency across runs
        if objective_names is None:
            objective_names  = obj_names
            ingredient_names = ing_names
        elif obj_names != objective_names:
            print(f"    Warning: objective names differ — expected {objective_names}, got {obj_names}. Skipping.")
            skipped += 1
            continue
        elif ing_names != ingredient_names:
            print(f"    Warning: ingredient names differ — expected {ingredient_names}, got {ing_names}. Skipping.")
            skipped += 1
            continue

        # tag each solution with its run name
        for sol in sols:
            sol["run"] = run_name

        all_solutions.extend(sols)
        print(f"    -> {len(sols)} solutions")

    if skipped:
        print(f"\n{skipped} folder(s) skipped.")

    if not all_solutions:
        print("Error: no valid solutions found.")
        sys.exit(1)

    runs_loaded = len(exp_dirs) - skipped
    print(f"\nAggregated {len(all_solutions)} solutions from {runs_loaded} run(s).")

    frequency = compute_solution_frequency(all_solutions, decimals=2)
    print_frequency_summary(frequency, top_n=10)

    return {
        "meta": {
            "model":                    model,
            "strategy":                 strategy,
            "runs":                     runs_loaded,
            "solutions":                len(all_solutions),
            "unique_solutions_2dp":     len(frequency),
            "objectives":               objective_names,
            "ingredients":              ingredient_names,
            "generated_at":             datetime.now().isoformat(timespec="seconds"),
        },
        "solutions": all_solutions,
        "solution_frequency": frequency,   # sorted by count desc
    }



def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate final Pareto fronts from all exp_* folders for a given "
            "model/strategy into a single JSON file for pareto_explorer.html."
        )
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        metavar="MODEL",
        help="Model name (folder name under experiments_dir, e.g. 'rf')."
    )
    parser.add_argument(
        "--strategy", "-s",
        required=True,
        metavar="STRATEGY",
        help="Strategy name (folder name under model, e.g. 'k-fold')."
    )
    parser.add_argument(
        "--experiments-dir",
        default="experiments",
        metavar="DIR",
        help="Root experiments directory (default: experiments/)."
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="FILE",
        help=(
            "Output JSON file path. "
            "Defaults to <experiments_dir>/<model>/<strategy>/aggregated.json."
        )
    )
    parser.add_argument(
        "--frequency-output", "-f",
        default=None,
        metavar="FILE",
        help=(
            "Output JSON file path for the frequency report. "
            "Defaults to <experiments_dir>/<model>/<strategy>/solution_frequency.json."
        )
    )
    return parser.parse_args()


def save_frequency_report(
    frequency: list[dict],
    meta: dict,
    output_path: str,
) -> None:
    """
    Writes the frequency report to a standalone JSON file.

    
    """
    ranked = [
        {
            "rank":            i + 1,
            "count":           entry["count"],
            "runs_count":      len(entry["runs"]),
            "runs":            entry["runs"],
            "strength_avg":    entry.get("strength_avg"),
            "strength_values": entry.get("strength_values", []),
            "proportions":     entry["proportions"],
        }
        for i, entry in enumerate(frequency)
    ]

    report = {
        "meta": {
            "model":                meta["model"],
            "strategy":             meta["strategy"],
            "runs":                 meta["runs"],
            "unique_solutions_2dp": meta["unique_solutions_2dp"],
            "total_solutions":      meta["solutions"],
            "ingredients":          meta["ingredients"],
            "generated_at":         meta["generated_at"],
        },
        "solution_frequency": ranked,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Frequency report saved -> {output_path}")


def main():
    args = parse_args()

    if not os.path.isdir(args.experiments_dir):
        print(f"Error: experiments directory not found — '{args.experiments_dir}'")
        sys.exit(1)

    data = aggregate(args.experiments_dir, args.model, args.strategy)

    # resolve aggregated output path
    if args.output:
        output_path = args.output
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    else:
        output_path = os.path.join(
            args.experiments_dir, args.model, args.strategy, "aggregated.json"
        )

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved -> {output_path}")

    # resolve frequency output path
    if args.frequency_output:
        freq_output_path = args.frequency_output
    else:
        freq_output_path = os.path.join(
            args.experiments_dir, args.model, args.strategy, "solution_frequency.json"
        )

    save_frequency_report(
        frequency=data["solution_frequency"],
        meta=data["meta"],
        output_path=freq_output_path,
    )


if __name__ == "__main__":
    main()