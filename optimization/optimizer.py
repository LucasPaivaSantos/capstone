import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

class GeopolymerProblem(ElementwiseProblem):
    def __init__(self, model, feature_names, xl, xu, objectives):
        """
        Initialize the multi-objective problem.
        
        Args:
            model: Trained ML model for strength prediction
            feature_names: List of feature names
            xl: Lower bounds for variables
            xu: Upper bounds for variables
            objectives: List of objective instances
        """

        n_objectives = 1 + len(objectives) # strength + additional objectives

        super().__init__(n_var=len(feature_names), 
                         n_obj=n_objectives, 
                         n_ieq_constr=0, 
                         xl=xl, 
                         xu=xu)
        
        self.model = model
        self.feature_names = feature_names
        self.objectives = objectives


    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate all objectives for a given solution.
        
        Args:
            x: Decision variables (proportions)
            out: Output dictionary
        """

        # for strength objective, always active
        input_data = pd.DataFrame([x], columns=self.feature_names)
        prediction = self.model.predict(input_data)[0] # predict returns an array
        obj_values = [-prediction] # list of objective values, start with strength (minimizing negative), all objectives are minimization for NSGA-II

        # for additional objectives, calculate their values
        for objective in self.objectives:
            obj_value = objective.calculate(x, self.feature_names)
            obj_values.append(obj_value)
        
        out["F"] = obj_values # fill the output dictionary with the objective values for this solution

def run_optimization(model, X_train, pop_size=100, n_gen=50, seed=1, objectives=None):
    """
    Sets up and runs the NSGA-II multi-objective optimization.
    
    Args:
        model: Trained ML model
        X_train: Training data (Full dataframe)
        pop_size: Population size
        n_gen: Number of generations
        seed: Random seed
        objectives: List of objective instances to optimize
    
    Returns: 
        best_solutions: Array of Pareto optimal solutions
        best_fitness: Array of fitness values for Pareto solutions
        feature_names: List of feature names
        history_pareto_fronts: List containing the Pareto front of each generation (List[np.ndarray])
        history_best_strength: List containing the best strength value found in each generation (List[float])
    """

    if objectives is None:
        objectives = []
    
    xl = X_train.min().values
    xu = X_train.max().values
    feature_names = X_train.columns.tolist()

    print(f"\nOptimization Configuration")
    print(f"Objectives: Maximize Strength" + 
          (f" + {', '.join([obj.get_name() for obj in objectives])}" if objectives else ""))
    print(f"Search Space Bounds:")
    for name, low, high in zip(feature_names, xl, xu):
        print(f" - {name}: [{low:.2f}, {high:.2f}]")

    problem = GeopolymerProblem(model, feature_names, xl, xu, objectives)

    algorithm = NSGA2(
        pop_size=pop_size,
        n_offsprings=10,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", n_gen)

    print(f"\nRunning NSGA-II for {n_gen} generations with pop_size={pop_size}")
        
    result = minimize(
            problem,
            algorithm,
            termination,
            seed=seed,
            save_history=True, 
            verbose=True)
    
    # get Pareto front solutions
    # result.X decision variables of final Pareto front
    # result.F objective values of final Pareto front
    if result.X.ndim == 1:
        # single solution
        best_solutions = result.X.reshape(1, -1) # reshape converts a 1D vector into a 2D matrix with 1 row
        best_fitness = result.F.reshape(1, -1) 
    else:
        # multiple solutions (Pareto)
        best_solutions = result.X
        best_fitness = result.F

    # convert strength objective back to positive
    best_fitness[:, 0] = -best_fitness[:, 0] # multiply all rows in the first column by -1

    history_best_strength = []
    history_pareto_fronts = []

    if result.history:
        for gen in result.history:
            target_gen = gen.opt.get("F").copy() # get the objective values for the current generation's Pareto front
            target_gen[:, 0] = -target_gen[:, 0] # convert strength

            # store full Pareto front for this generation
            history_pareto_fronts.append(target_gen)

            # track best strength in this generation
            best_strength = np.max(target_gen[:, 0]) 
            history_best_strength.append(float(best_strength))

    return best_solutions, best_fitness, feature_names, history_pareto_fronts, history_best_strength