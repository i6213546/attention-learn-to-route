import matplotlib.pyplot as plt
import numpy as np

# def plot_training_result(train_cost, bl_cost, val_cost, save_path=None):
#     epochs = np.arange(len(train_cost))
#     plt.figure(figsize=(10, 6))
#     plt.plot(epochs, train_cost, label='Training Cost')
#     plt.plot(epochs, bl_cost, label='Baseline Cost')
#     plt.plot(epochs, val_cost, label='Validation Cost')

#     # Add labels and title
#     plt.xlabel('Epochs')
#     plt.ylabel('Cost')
#     plt.title('Training, Baseline and Validation Cost Over Epochs')

#     # Add a legend
#     plt.legend()

#     if save_path:
#         plt.savefig(save_path)
        
#     # Display the plot
#     plt.show()


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
