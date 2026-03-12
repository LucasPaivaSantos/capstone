import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea


def plot_objective_convergence(history_pareto_fronts, objective_name, objective_index, output_dir):
    """
    Plots best value of a objective vs generations.

    Args:
        history_pareto_fronts: List of per-generation Pareto fronts, shape (n_solutions, n_obj).
        objective_name: Name of the objective for labels
        objective_index: Column index of the objective in the fitness array.
        output_dir: Directory to save the plot.
    """

    if  len(history_pareto_fronts) == 0 or history_pareto_fronts is None:
        print("No data available to plot.")
        return

    history_best = []
    for gen_front in history_pareto_fronts:
        col = gen_front[:, objective_index]
        
        best = np.max(col) if objective_index == 0 else np.min(col) # strength is max, all other objectives are min
        history_best.append(float(best))

    generations = list(range(1, len(history_best) + 1))

    sea.set_theme(style="whitegrid")
    # create the figure object, a container for drawing
    plt.figure(figsize=(10, 10))
    plt.plot(generations, history_best, linewidth=2)

    # labels
    if objective_index == 0:
        ylabel = "Best Strength (MPa)"
        title  = "Best Strength over Generations"
    else:
        ylabel = f"Best {objective_name}"
        title  = f"Best {objective_name} over Generations"

    plt.title(title, fontsize=14)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel("Generation", fontsize=12)

    plt.tight_layout() # prevents overflow

    output_filename = os.path.join(output_dir, f"{objective_name}_convergence.png")
    plt.savefig(output_filename)
    print(f"\nPlot saved as '{output_filename}'")