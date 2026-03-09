import os
import matplotlib.pyplot as plt
import seaborn as sea


def plot_strength_convergence(history_best_strength, output_dir):
    """
    Plots best compressive strength vs generation.
    """

    if  len(history_best_strength) == 0 or history_best_strength is None:
        print("No data available to plot.")
        return
    
    generations = list(range(1, len(history_best_strength) + 1))

    sea.set_theme(style="whitegrid")

    # create the figure object, a container for drawing
    plt.figure(figsize=(10, 10))

    plt.plot(generations, history_best_strength, linewidth=2)

    plt.title("Best strength over Generations", fontsize=14)
    plt.ylabel("Best Strength (MPa)", fontsize=12)
    plt.xlabel("Generation", fontsize=12)

    plt.tight_layout() # prevents overflow

    output_filename = os.path.join(output_dir, "strength_convergence.png")
    plt.savefig(output_filename)
    print(f"\nPlot saved as '{output_filename}'")