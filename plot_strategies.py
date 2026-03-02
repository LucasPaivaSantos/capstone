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
    """
    Dynamically loads all modules in the specified package by joining their names.
    """
    package_dir = os.path.join(os.path.dirname(__file__), package_name)
    # prevents errors if the package does not exist
    if not os.path.exists(package_dir):
        return
    
    # interate through all modules in the package directory
    for _, module_name, _ in pkgutil.iter_modules([package_dir]):
        if module_name != 'base': # skip base module
            importlib.import_module(f'{package_name}.{module_name}')


def handle_cli_args():
    """
    Handles all CLI argument parsing and validation logic.
    Returns the parsed arguments object.
    """

    parser = argparse.ArgumentParser(description="Plot Boxplot comparing R2 Score of validation strategies for a specific model.")
    
    parser.add_argument(
        "--model", "-m",
        required=True,
        choices=MODEL_REGISTRY.keys(),
        help="Machine Learning Model to analyze"
    )
    
    parser.add_argument(
        "--base_dir",
        default="experiments",
        help="Base directory for experiments (default: experiments)"
    )

    return parser.parse_args()


def load_strategy_data(target_model, base_dir="experiments"):
    """
    Reads all strategy folders for a specific model and extracts the R2 metrics.
    """

    model_dir = os.path.join(base_dir, target_model)

    if not os.path.exists(model_dir):
        print(f"Directory '{model_dir}' not found")
        return pd.DataFrame() # empty df
    
    data = []
    
    # loop for strategy folders of a model
    for strategy in os.listdir(model_dir):
        strategy_dir = os.path.join(model_dir, strategy)

        if not os.path.isdir(strategy_dir):
            continue

        # loop for all the experiments folders of a strategy
        for exp_folder in os.listdir(strategy_dir):
            exp_path = os.path.join(strategy_dir, exp_folder)
            
            if not os.path.isdir(exp_path):
                continue

            eval_path = os.path.join(exp_path, "evaluation.json") # get the evaluation file of each folder

            if not os.path.exists(eval_path):
                continue

            
            with open(eval_path, 'r') as f:
                evaluation = json.load(f)

            eval_lower = {k.lower(): v for k, v in evaluation.items()}
            r2 = eval_lower.get('r2') or eval_lower.get('r2_mean') # look for r2 or r2_mean(for k-fold)
            if r2 is not None:
                data.append({
                "Strategy": strategy,
                "Experiment": exp_folder,
                "R2": r2
                })
            else:
                print(f"R2 not found in {eval_path}")

    return pd.DataFrame(data)
        

def plot_strategy_comparison(df, target_model):
    """
    Plots a Boxplot comparing the R2 Score of different strategies for the given model.
    """

    if df.empty or df is None:
        print("No data available to plot.")
        return
    
    sea.set_theme(style="whitegrid")

    # create the figure object, a container for drawing
    plt.figure(figsize=(10, 10))

    # draw the graph (boxes)
    sea.boxplot(data=df, x="Strategy", y="R2")

    # draw each point; jitter adds random horizontal space to the point
    sea.stripplot(data=df, x="Strategy", y="R2", color=".5", alpha=0.5, jitter=True)

    plt.title(f"Validation Strategy Comparison for Model: {target_model}", fontsize=14)
    plt.ylabel("R^2 Score", fontsize=12)
    plt.xlabel("Validation Strategy", fontsize=12)

    plt.tight_layout() # prevents overflow

    output_filename = f"strategy_comparison_r2_{target_model}.png"
    plt.savefig(output_filename)
    print(f"\nPlot saved as '{output_filename}'")



def main():
    # load modules
    load_modules('models')

    # get arguments
    args = handle_cli_args()
    
    target_model = args.model.lower()
    
    df_experiments = load_strategy_data(target_model, args.base_dir)
    plot_strategy_comparison(df_experiments, target_model)

    
    print(df_experiments.groupby('Strategy')['R2'].describe())
    plot_strategy_comparison(df_experiments, target_model)


if __name__ == "__main__":
    main()