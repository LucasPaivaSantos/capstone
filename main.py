import argparse
import importlib
import numpy as np
import os
import pkgutil
import sys
import traceback

from models.base import MODEL_REGISTRY
from objectives.base import OBJECTIVE_REGISTRY
from strategies.base import STRATEGY_REGISTRY

from optimization.optimizer import run_optimization
from utils.data_loader import load_and_validate_csv
from utils.logger import ExperimentLogger
from utils.plots.plot_convergence import plot_objective_convergence
from utils.plots.plot_pareto import plot_pareto_front

MAX_R2_ATTEMPTS = 100

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

    parser = argparse.ArgumentParser(description="Acid Geopolymer Concrete Optimizer")

    # data arguments
    parser.add_argument(
        "--csv_path",
        default="data/best.csv",
        help="Path to the CSV file containing the dataset (default: data/best.csv)"
    )

    # model arguments
    parser.add_argument(
        "--model", "-m",
        required=True, 
        choices=MODEL_REGISTRY.keys(),
        help="Machine Learning Model to use"
    )
    parser.add_argument(
        "--model_seed",
        type=int,
        default=None,
        help="Seed for ML model (if not provided, a random seed will be generated)"
    )

    # strategy arguments
    parser.add_argument(
        "--strategy", "-s",
        required=True, 
        choices=STRATEGY_REGISTRY.keys(),
        help="Validation Strategy"
    )
    parser.add_argument(
        "--strategy_seed",
        type=int,
        default=None,
        help="Seed for validation strategy (if not provided, a random seed will be generated)"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=None,
        help="Test size for train-test split."
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=None,
        help="Number of folds for K-Fold cross-validation."
    )

    # optimization arguments
    parser.add_argument(
        "--optimize",
        action='store_true',
        help="Enable NSGA-II optimization"
    )
    parser.add_argument(
        "--objectives", "-o",
        nargs='+',
        choices=list(OBJECTIVE_REGISTRY.keys()),
        default=[],
        help="Additional objectives to optimize (strength is always included)."
    )
    parser.add_argument(
        "--pop_size",
        type=int,
        default=10,
        help="Population Size"
    )
    parser.add_argument(
        "--n_gen",
        type=int,
        default=150,
        help="Number of Generations"
    )

    # r2 minimum threshold argument
    parser.add_argument(
        "--min-r2",
        type=float,
        default=None,
        dest="min_r2",
        help=(
            "Minimum R² threshold for the model. Only active when seeds are not manually provided. "
            "The program will retry with new random seeds (up to 100 attempts) until the threshold is met."
        )
    )

    # plot arguments
    parser.add_argument(
        "--plot",
        nargs='+',
        choices=['final_pareto_front', 'strength'] + list(OBJECTIVE_REGISTRY.keys()),
        default=None,
        help=(
            "Plots to generate. 'final_pareto_front' plots the Pareto front. Any objective name plots its best value over generations."
        )
    )

    args = parser.parse_args()

    # error handling for incompatible arguments
    # if test_size is provided, strategy must be 'train-split'
    if args.test_size is not None and args.strategy != 'train-split':
        parser.error(f"The argument '--test_size' is not allowed with strategy '{args.strategy}'. It is only valid for 'train-split'.")

    # if n_splits is provided, strategy must be 'k-fold'
    if args.n_splits is not None and args.strategy != 'k-fold':
        parser.error(f"The argument '--n_splits' is not allowed with strategy '{args.strategy}'. It is only valid for 'k-fold'.")

    # min-r2 is ignored when seeds are manually provided
    if args.min_r2 is not None and (args.model_seed is not None or args.strategy_seed is not None):
        print(
            "Warning: '--min-r2' is ignored when '--model_seed' or '--strategy_seed' are provided. "
            "Proceeding with the supplied seeds."
        )
        args.min_r2 = None

    # validate min-r2 range
    if args.min_r2 is not None and not (0.0 < args.min_r2 <= 1.0):
        parser.error("'--min-r2' must be a value between 0 (exclusive) and 1 (inclusive).")

    # set defaults based on strategy
    if args.strategy == 'train-split' and args.test_size is None:
        args.test_size = 0.2
        print("Note: 'train-split' selected without explicit test_size. Defaulting to 0.2")

    if args.strategy == 'k-fold' and args.n_splits is None:
        args.n_splits = 5
        print("Note: 'k-fold' selected without explicit n_splits. Defaulting to 5")

    return args


def evaluate_model(model_cls, strategy_cls, model_seed, strategy_seed, X, y, args):
    """
    Instantiates and evaluates a model with the given seeds.
    Returns (model_instance, strategy_instance, model_evaluation).
    """
    model_instance = model_cls(seed=model_seed)
    strategy_instance = strategy_cls(seed=strategy_seed)

    eval_kwargs = {}
    if args.strategy == 'train-split':
        eval_kwargs['test_size'] = args.test_size
    elif args.strategy == 'k-fold':
        eval_kwargs['n_splits'] = args.n_splits

    model_evaluation = strategy_instance.evaluate(model_instance, X, y, **eval_kwargs)
    return model_instance, strategy_instance, model_evaluation


def get_r2_from_evaluation(model_evaluation):
    """
    Extracts the R² value from an evaluation dict.
    Looks for keys: 'r2', 'R2', 'r²', 'R²'.
    Returns None if not found.
    """
    for key in ('r2', 'R2', 'r²', 'R²'):
        if key in model_evaluation:
            return model_evaluation[key]
    return None


def main():
    # load modules
    load_modules('models')
    load_modules('strategies')
    load_modules('objectives')

    # get arguments
    args = handle_cli_args()

    try:
        X, y = load_and_validate_csv(args.csv_path)

        model_cls = MODEL_REGISTRY[args.model]
        strategy_cls = STRATEGY_REGISTRY[args.strategy]

        # --- Seed resolution and evaluation ---
        use_min_r2 = args.min_r2 is not None

        if use_min_r2:
            # Retry loop: draw new random seeds until R² threshold is met or limit is reached
            print(f"\nMin-R² mode enabled. Target R² ≥ {args.min_r2} (max {MAX_R2_ATTEMPTS} attempts)")

            model_seed = None
            strategy_seed = None
            model_instance = None
            model_evaluation = None
            achieved = False

            for attempt in range(1, MAX_R2_ATTEMPTS + 1):
                candidate_model_seed = np.random.randint(0, 10_000)
                candidate_strategy_seed = np.random.randint(0, 10_000)

                print(f"\n[Attempt {attempt}/{MAX_R2_ATTEMPTS}] "
                      f"model_seed={candidate_model_seed}, strategy_seed={candidate_strategy_seed}")

                m_instance, _, m_evaluation = evaluate_model(
                    model_cls, strategy_cls,
                    candidate_model_seed, candidate_strategy_seed,
                    X, y, args
                )

                r2 = get_r2_from_evaluation(m_evaluation)

                if r2 is None:
                    print("  Warning: R² metric not found in evaluation results. Skipping min-r2 check.")
                    # treat as achieved so the program continues normally
                    model_seed = candidate_model_seed
                    strategy_seed = candidate_strategy_seed
                    model_instance = m_instance
                    model_evaluation = m_evaluation
                    achieved = True
                    break

                print(f"  R² = {r2:.6f} (target ≥ {args.min_r2})")

                if r2 >= args.min_r2:
                    print(f"  ✓ R² threshold met on attempt {attempt}.")
                    model_seed = candidate_model_seed
                    strategy_seed = candidate_strategy_seed
                    model_instance = m_instance
                    model_evaluation = m_evaluation
                    achieved = True
                    break

            if not achieved:
                print(
                    f"\nWarning: R² threshold of {args.min_r2} was not reached after {MAX_R2_ATTEMPTS} attempts. "
                    "Proceeding with the best available result from the last attempt."
                )
                # use seeds from the last attempt
                model_seed = candidate_model_seed
                strategy_seed = candidate_strategy_seed
                model_instance = m_instance
                model_evaluation = m_evaluation

        else:
            # Original behaviour: use provided seeds or generate once
            model_seed = args.model_seed if args.model_seed is not None else np.random.randint(0, 100)
            strategy_seed = args.strategy_seed if args.strategy_seed is not None else np.random.randint(0, 100)

            print(f"\nInitializing Model: {args.model}")
            if args.model != 'svr':
                print(f" with seed {model_seed}")
            print(f"\nExecuting Strategy: {args.strategy} with seed {strategy_seed}")
            if args.strategy == 'train-split':
                print(f"Configuration: Test Size = {args.test_size}")
            elif args.strategy == 'k-fold':
                print(f"Configuration: Number of Folds = {args.n_splits}")

            model_instance, _, model_evaluation = evaluate_model(
                model_cls, strategy_cls,
                model_seed, strategy_seed,
                X, y, args
            )

        # --- Logging ---
        logger = ExperimentLogger(model_name=args.model, strategy_name=args.strategy, base_dir="experiments")
        logger.log_config(args, model_seed, strategy_seed)
        logger.log_evaluation(model_evaluation)

        # print model evaluation results
        print("\nEvaluation Metrics:")
        for metric, value in model_evaluation.items():
            print(f" - {metric}: {value}")

        print("Retraining model for optimization") # model trained on full dataset for optimization, same seed
        model_instance.fit(X, y)

        # run optimization
        objective_instances = [OBJECTIVE_REGISTRY[obj_name]() for obj_name in args.objectives]
        best_solutions, best_fitness, feat_names, history_pareto_fronts, history_best_strength = run_optimization(
                model=model_instance,
                X_train=X,
                pop_size=args.pop_size,
                n_gen=args.n_gen,
                seed=model_seed,
                objectives=objective_instances
            )
        
        # save Pareto front history
        logger.log_pareto_history(history_pareto_fronts, feat_names, objective_instances)

        # plot and save graphs
        if args.plot:
            # build a list of all objectives: strength is always index 0
            all_objectives = [('strength', 0)] + [
                (obj.get_name(), idx + 1) for idx, obj in enumerate(objective_instances)
            ]

            obj_key_to_plot = {'strength': ('strength', 0)}
            for obj_name, (display_name, idx) in zip(args.objectives, all_objectives[1:]):
                obj_key_to_plot[obj_name] = (display_name, idx)

            for plot_target in args.plot:
                if plot_target in obj_key_to_plot:
                    display_name, obj_idx = obj_key_to_plot[plot_target]
                    plot_objective_convergence(
                        history_pareto_fronts=history_pareto_fronts,
                        objective_name=display_name,
                        objective_index=obj_idx,
                        output_dir=logger.log_dir
                    )    
                elif plot_target == 'final_pareto_front':
                    plot_pareto_front(
                        best_solutions=best_solutions,
                        best_fitness=best_fitness,
                        feature_names=feat_names,
                        objectives=objective_instances,
                        output_dir=logger.log_dir
                    )
    except Exception as e:
        # for erros I didn't anticipate
        print(f"\nUnexpected Error: {e}")
        traceback.print_exc()
        sys.exit(1)



if __name__ == "__main__":
    main()