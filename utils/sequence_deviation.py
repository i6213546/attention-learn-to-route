import numpy as np

def sequence_deviation(output_sequences):
    """
    output_sequences should be a list of list (2 dim)
    return: standard deviation of each sequence
    """
    a = [2*np.sum([np.abs(seq[i+1] - seq[i]) -1 for i in range(len(seq)-1)])/(len(seq) * (len(seq)-1)) for seq in output_sequences]
    return a