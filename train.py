import os
import time
from tqdm import tqdm
import torch
import math
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to

# function used for shifting the tour so that 0 is the starting point (depot)
def shift_row(row):
    zero_index = (row == 0).nonzero(as_tuple=True)[0].item()
    part1 = row[zero_index:]  # From value 0 to the end
    part2 = row[:zero_index]  # From the beginning to value 0
    return torch.cat((part1, part2))

def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts, return_pi=False, sorted_pi=False):
    # Validate
    print('Validating...')
    pi = None
    if return_pi:
        cost, pi = rollout(model, dataset, opts, return_pi=return_pi)
    else:
        cost     = rollout(model, dataset, opts)

    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))
    
    if sorted_pi:
        pi = torch.stack([shift_row(row) for row in pi])

    if return_pi:
        return avg_cost, pi
    else:
        return avg_cost


def rollout(model, dataset, opts, return_pi=False):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")

    def eval_model_bat(bat, bat_id):
        with torch.no_grad():
            if opts.cost_input:
                cost_metric = dataset.cost_data[opts.eval_batch_size*bat_id : 
                                            opts.eval_batch_size*(bat_id+1)]
                if isinstance(cost_metric, list):
                    cost_metric = torch.stack(cost_metric)
                if return_pi:
                    cost, _, pi = model(move_to(bat, opts.device), cost_data=move_to(cost_metric, opts.device), 
                                          return_pi=return_pi)
                    return cost.data.cpu(), pi.data.cpu()
                else:
                    cost, _ = model(move_to(bat, opts.device), cost_data=move_to(cost_metric, opts.device))
                    return cost.data.cpu()
            else:
                if return_pi:
                    cost, _, pi = model(move_to(bat, opts.device), return_pi=return_pi, SD=opts.SD)
                    #print('cost in rollout:', cost[:10])
                    return cost.data.cpu(), pi.data.cpu()
                else:
                    cost, _ = model(move_to(bat, opts.device), SD=opts.SD)
                    return cost.data.cpu()
        
    if return_pi:
        ret_cost = torch.tensor([])
        ret_pi   = torch.tensor([], dtype=torch.int64)
        for bat_id, bat in enumerate(tqdm(DataLoader(dataset, 
                                                     batch_size=opts.eval_batch_size), 
                                                     disable=opts.no_progress_bar)):
            cost_temp, pi_temp = eval_model_bat(bat, bat_id)    
            ret_cost = torch.cat((ret_cost, cost_temp), dim=0)
            ret_pi   = torch.cat((ret_pi, pi_temp), dim=0)                 
        return ret_cost, ret_pi
    
    else:
        return torch.cat([
            eval_model_bat(bat, bat_id)
            for bat_id, bat
            in enumerate(tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar))
        ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    mean_grad_norms = np.mean(grad_norms)
    #grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    grad_norms_clipped = [min(g_norm, mean_grad_norms) for g_norm in grad_norms]
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    training = problem.make_dataset(filename=opts.train_dataset,size=opts.graph_size, 
                                    num_samples=opts.epoch_size, cost_input=opts.cost_input, distribution=opts.data_distribution)
    # shuffling data after every re-train
    training.shuffle_data()

    print('baseline:', baseline)
    print('baseline alpha:', baseline.alpha)
    training_dataset = baseline.wrap_dataset(training)
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)

    if opts.cost_input:
        cost_dataloader = DataLoader(training_dataset.cost_data, batch_size=opts.batch_size, num_workers=1)

        cost_dataloader = [batch for id, batch in enumerate(cost_dataloader)]
    
    # Put model in train mode!
    model.train()
    set_decode_type(model, "greedy")
    training_cost = []

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        if opts.cost_input:
            cost_data = cost_dataloader[batch_id]
        else: 
            cost_data = None

        cost_bat = train_batch(
                    model,
                    optimizer,
                    baseline,
                    epoch,
                    batch_id,
                    step,
                    batch,
                    tb_logger,
                    opts,
                    cost_data=cost_data
                    )
        training_cost.append(cost_bat)
        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))
    #print("Cost in this epoch:", np.sum(training_cost))
    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
    #if (epoch+1) % 10 == 0 or epoch == (opts.n_epochs-1) or epoch==0:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    avg_reward = validate(model, val_dataset, opts)

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)

    candidate_mean = baseline.epoch_callback(model, epoch)
    print('candidate mean:', candidate_mean)
    # lr_scheduler should be called  end of epoch
    lr_scheduler.step()

    return np.sum(training_cost)/training.size, candidate_mean, avg_reward

def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts,
        cost_data=None
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood = model(x, cost_data=cost_data, SD=opts.SD)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)
    
    return cost.sum().item()