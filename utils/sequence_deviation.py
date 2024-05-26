import numpy as np
import torch
def sequence_deviation(output_sequences):
    """
    output_sequences should be a list of list (2 dim)
    return: standard deviation of each sequence
    """
    a = [2*np.sum([np.abs(seq[i+1] - seq[i]) -1 for i in range(len(seq)-1)])/(len(seq) * (len(seq)-1)) for seq in output_sequences]
    return a

def rearrange_output_vs_driver(random_indices, pi):
    """"
    random indices and pi must be in tensor
    random_indices: the index used for re-ordering the driver tour to feed into the model
    pi: the output of model
    """
    if not torch.is_tensor(random_indices):
        random_indices = torch.tensor(random_indices)
    
    ret_seq = torch.gather(input=random_indices, dim=1, index=pi)
    return ret_seq