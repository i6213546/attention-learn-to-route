import matplotlib.pyplot as plt
import numpy as np

def plot_training_result(cost, legend, title, save_path=None, plus=False):
    if plus:
        epochs = np.arange(1, len(cost[0])+1)
    else:
        epochs = np.arange(len(cost[0]))
    plt.figure(figsize=(10, 6))
    for i in range(len(cost)):
        plt.plot(epochs, cost[i], label=legend[i])

    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.title(title)

    # Add a legend
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        
    # Display the plot
    plt.show()

def plot_tour():
    return
