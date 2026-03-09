import os
import json
import numpy as np
from datetime import datetime

class NpEncoder(json.JSONEncoder):
    """
    Parser to handle numpy data types when saving to JSON.
    Converts numpy integers and floats to their native Python types, and numpy arrays to lists.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class ExperimentLogger:
    def __init__(self, model_name, strategy_name, base_dir="experiments"):
        """
        Create a unique directory with current timestamp nested under model and strategy.
        """
        self.base_dir = base_dir
        # YYYYMMDD_HHMMSS
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # build the nested path
        self.log_dir = os.path.join(self.base_dir, model_name, strategy_name, f"exp_{timestamp}")
        
        # creates all necessary parent directories
        os.makedirs(self.log_dir, exist_ok=True) 

    def log_config(self, args, model_seed, strategy_seed):
        """
        Saves the arguments and the seeds used for the experiment.
        """
        config = vars(args).copy() # get arguments
        
        config['model_seed'] = model_seed # add seeds
        config['strategy_seed'] = strategy_seed

        filepath = os.path.join(self.log_dir, "config.json") # create file
        with open(filepath, 'w') as f: # open to write
            json.dump(config, f, indent=4)

    def log_evaluation(self, model_evaluation):
        """
        Saves the evaluation metrics of the model.
        """
        filepath = os.path.join(self.log_dir, "evaluation.json")
        with open(filepath, 'w') as f:
            json.dump(model_evaluation, f, indent=4, cls=NpEncoder)

    def log_optimization(self, best_solutions, best_fitness, feature_names, objectives):
        """
        Saves the experiment final Pareto front.
        """
        objective_names = ['strength'] + [obj.get_name() for obj in objectives]

        pareto_front = []
        for solution, fitness in zip(best_solutions, best_fitness):
            entry = {}
            entry['mix'] = {name: float(val) for name, val in zip(feature_names, solution)}
            entry['objectives'] = {name: float(val) for name, val in zip(objective_names, fitness)}
            pareto_front.append(entry)

        filepath = os.path.join(self.log_dir, "pareto_front.json")
        with open(filepath, 'w') as f:
            json.dump(pareto_front, f, indent=4, cls=NpEncoder)