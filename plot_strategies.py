import argparse
import importlib
import json
import os
import pkgutil

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

    parser = argparse.ArgumentParser(description="Plot Boxplot comparing R2 Score of training strategies for a specific model.")
    
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
        return 
    
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

