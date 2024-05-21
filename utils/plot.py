import matplotlib.pyplot as plt
import numpy as np

def plot_training_result(train_cost, bl_cost, val_cost, save_path=None):
    epochs = np.arange(len(train_cost))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_cost, label='Training Cost')
    plt.plot(epochs, bl_cost, label='Baseline Cost')
    plt.plot(epochs, val_cost, label='Validation Cost')

    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.title('Training, Baseline and Validation Cost Over Epochs')

    # Add a legend
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        
    # Display the plot
    plt.show()
