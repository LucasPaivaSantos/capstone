import argparse
import importlib
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import pkgutil
import seaborn as sea

from models.base import MODEL_REGISTRY

def load_modules(package_name):
    package_dir = os.path.join(os.path.dirname(__file__), package_name)
    if not os.path.exists(package_dir):
        return
    for _, module_name, _ in pkgutil.iter_modules([package_dir]):
        if module_name != 'base':
            importlib.import_module(f'{package_name}.{module_name}')


def handle_cli_args():
    parser = argparse.ArgumentParser(description="Plot Boxplot comparing metrics across all models.")
    parser.add_argument(
        "--base_dir",
        default="experiments",
        help="Base directory for experiments (default: experiments)"
    )
    parser.add_argument(
        "--metrics",
        nargs='+',
        choices=['r2', 'rmse'],
        default=['r2'],
        help="Metrics to plot"
    )
    return parser.parse_args()


def load_all_models_data(base_dir="experiments"):
    """Load experiment data for every model folder found in base_dir."""
    if not os.path.exists(base_dir):
        print(f"Base directory '{base_dir}' not found.")
        return pd.DataFrame()

    data = []

    for model_name in os.listdir(base_dir):
        model_dir = os.path.join(base_dir, model_name)

        if not os.path.isdir(model_dir):
            continue

        for strategy in os.listdir(model_dir):
            strategy_dir = os.path.join(model_dir, strategy)

            if not os.path.isdir(strategy_dir):
                continue

            for exp_folder in os.listdir(strategy_dir):
                exp_path = os.path.join(strategy_dir, exp_folder)

                if not os.path.isdir(exp_path):
                    continue

                eval_path = os.path.join(exp_path, "evaluation.json")

                if not os.path.exists(eval_path):
                    continue

                with open(eval_path, 'r') as f:
                    evaluation = json.load(f)

                eval_lower = {k.lower(): v for k, v in evaluation.items()}
                r2   = eval_lower.get('r2')   or eval_lower.get('r2_mean')
                rmse = eval_lower.get('rmse') or eval_lower.get('rmse_mean')

                if r2 is not None:
                    data.append({
                        "Model":      model_name,
                        "Strategy":   strategy,
                        "Experiment": exp_folder,
                        "R2":         r2,
                        "RMSE":       rmse,
                    })
                else:
                    print(f"Metrics not found in {eval_path}")

    return pd.DataFrame(data)


def plot_all_models(df, metric, output_dir="plots"):
    """Boxplot with one box per model+strategy combination on the X-axis."""
    metric_upper = metric.upper()
    clean_df = df.dropna(subset=[metric_upper]).copy()

    if clean_df.empty:
        print(f"No data available to plot for metric '{metric_upper}'.")
        return

    os.makedirs(output_dir, exist_ok=True)

    clean_df["Model_Strategy"] = clean_df["Model"] + "\n" + clean_df["Strategy"]

    sea.set_theme(style="whitegrid")
    plt.figure(figsize=(max(8, len(clean_df["Model_Strategy"].unique()) * 1.5), 8))

    sea.boxplot(data=clean_df,  x="Model_Strategy", y=metric_upper)
    sea.stripplot(data=clean_df, x="Model_Strategy", y=metric_upper,
                  color=".5", alpha=0.5, jitter=True)

    plt.title(f"Model × Strategy Comparison — {metric_upper} Score", fontsize=14)
    plt.ylabel(f"{metric_upper} Score", fontsize=12)
    plt.xlabel("Model / Strategy", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    output_filename = os.path.join(output_dir, f"all_models_{metric.lower()}.png")
    plt.savefig(output_filename, dpi=150)
    plt.close()
    print(f"Plot saved as '{output_filename}'")


def save_metrics_json(df, metric, output_dir="plots"):
    """Save descriptive stats grouped by Model for the given metric."""
    metric_upper = metric.upper()
    stats = df.groupby("Model")[metric_upper].describe()

    metrics_dict = {
        "metric":  metric_upper,
        "models":  {},
        "experiments": (
            df[["Model", "Strategy", "Experiment", metric_upper]]
            .rename(columns={metric_upper: "value"})
            .to_dict(orient="records")
        ),
    }

    for model_name, row in stats.iterrows():
        metrics_dict["models"][model_name] = {
            "count": row.get("count"),
            "mean":  row.get("mean"),
            "std":   row.get("std"),
            "min":   row.get("min"),
            "25%":   row.get("25%"),
            "50%":   row.get("50%"),
            "75%":   row.get("75%"),
            "max":   row.get("max"),
        }

    os.makedirs(output_dir, exist_ok=True)
    json_filename = os.path.join(output_dir, f"metrics_{metric.lower()}.json")
    with open(json_filename, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    print(f"Metrics JSON saved as '{json_filename}'")


def main():
    load_modules('models')
    args = handle_cli_args()

    df = load_all_models_data(args.base_dir)

    if df.empty:
        print("No experiment data found. Exiting.")
        return

    for metric in args.metrics:
        plot_all_models(df, metric)
        save_metrics_json(df, metric)
        print(df.groupby("Model")[metric.upper()].describe())


if __name__ == "__main__":
    main()