import os
import matplotlib.pyplot as plt
import seaborn as sea


def plot_pareto_front(best_solutions, best_fitness, feature_names, objectives, output_dir):
    """
    Plots and saves the final Pareto front. Adapts between 2D and 3D plots according to the number of objectives:

    Args:
        best_solutions: Proportions used in the Pareto front, shape (n_solutions, n_vars).
        best_fitness: Objective values reached by the Pareto front, shape (n_solutions, n_obj). Column 0 is compressive strength followed by additional objectives.
        feature_names: Names of the ingredients.
        objectives: Additional objectives, strength always included.
        output_dir: Directory to save the plot.
    """

    if len(best_fitness) == 0 or best_fitness is None:
        print("Nodata available to plot.")
        return

    if len(objectives) == 0:
        print("Pareto front plot requires at least one additional objective besides strength.")
        return

    sea.set_theme(style="whitegrid")

    if len(objectives) == 1:
        _plot_2d(best_fitness, objectives, output_dir)
    else:
        _plot_3d(best_fitness, objectives, output_dir)


def _plot_2d(best_fitness, objectives, output_dir):
    """
    Plots the Pareto front for 2 objectives (strength vs one objective).
    """
    
    if len(best_fitness) == 0 or best_fitness is None:
        print("No data available to plot.")
        return

    strength = best_fitness[:, 0] # all rows, first column (strength)
    additional_obj = best_fitness[:, 1] # all rows, second column (additional objective)

    sea.set_theme(style="whitegrid")

    # create the figure object, a container for drawing
    plt.figure(figsize=(10, 10))

    # sort by strength so the line connects points in order
    order = sorted(range(len(strength)), key=lambda i: strength[i])
    s_sorted = [strength[i] for i in order]
    o_sorted = [additional_obj[i] for i in order]
    
    # plot the line connecting points
    plt.plot(s_sorted, o_sorted, linewidth=2)
    
    # add points
    plt.scatter(strength, additional_obj, linewidths=1)

    plt.title("Final Pareto Front", fontsize=14)
    plt.xlabel("Compressive Strength (MPa)", fontsize=12)
    plt.ylabel(objectives[0].get_name(), fontsize=12)

    plt.tight_layout() # prevents overflow

    output_filename = os.path.join(output_dir, "pareto_front.png")
    plt.savefig(output_filename)
    print(f"\nPlot saved as '{output_filename}'")


def _plot_3d(best_fitness, objectives, output_dir):
    """
    Plots the Pareto front for 3 objectives (strength vs two objectives).
    """

    if len(best_fitness) == 0 or best_fitness is None:
        print("No data available to plot.")
        return

    strength = best_fitness[:, 0]
    obj1 = best_fitness[:, 1]
    obj2 = best_fitness[:, 2]

    sea.set_theme(style="whitegrid")

    # create the figure object, a container for drawing
    plt.figure(figsize=(10, 10))

    plt.axes(projection="3d") 

    plt.gca().scatter(  # gca means get current axes
        strength,
        obj1,
        obj2,
        linewidths=1,
    )

    plt.gca().set_title("Final Pareto Front", fontsize=14)
    plt.gca().set_xlabel("Compressive Strength (MPa)", fontsize=12)
    plt.gca().set_ylabel(objectives[0].get_name(), fontsize=12)
    plt.gca().set_zlabel(objectives[1].get_name(), fontsize=12)

    plt.tight_layout() # prevents overflow

    output_filename = os.path.join(output_dir, "pareto_front.png")
    plt.savefig(output_filename)
    print(f"\nPlot saved as '{output_filename}'")