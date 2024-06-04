import numpy as np
import torch
def sequence_deviation(output_sequences):
    """
    output_sequences should be a list of lists (2D list)
    return: standard deviation of each sequence
    """
    # deviations = []
    # for seq in output_sequences:
    #     #seq_tensor = torch.tensor(seq, dtype=torch.float32)
    #     diffs = torch.abs(seq[1:] - seq[:-1]) - 1
    #     deviation = 2 * torch.sum(diffs) / (len(seq) * (len(seq) - 1))
    #     deviations.append(deviation.item())
    length = output_sequences.size(1)
    SD = 2*torch.sum(torch.abs(output_sequences[:,1:] - output_sequences[:,:-1]) - 1, 1)/(length * (length-1))
    return SD

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