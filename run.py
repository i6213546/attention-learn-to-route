#!/usr/bin/env python

import pprint as pp
import pickle, torch, random, os, json
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger

from nets.critic_network import CriticNetwork
from options import get_options
from train import train_epoch, validate, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils import torch_load_cpu, load_problem

from utils.plot import plot_training_result
from utils.sequence_deviation import sequence_deviation

random.seed(None)
def run(opts):

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

    os.makedirs(opts.save_dir, exist_ok=True)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Figure out what's the problem
    problem = load_problem(opts.problem)

    # Load data from load_path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # Initialize model
    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size
    ).to(opts.device)

    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    eval_dataset = problem.make_dataset(size=opts.graph_size, num_samples=opts.eval_size, filename=opts.eval_dataset, 
                                        cost_input=opts.cost_input, distribution=opts.data_distribution)

    # Initialize baseline
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == 'critic' or opts.baseline == 'critic_lstm':
        assert problem.NAME == 'tsp', "Critic only supported for TSP"
        baseline = CriticBaseline(
            (
                CriticNetworkLSTM(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.tanh_clipping
                )
                if opts.baseline == 'critic_lstm'
                else
                CriticNetwork(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.normalization
                )
            ).to(opts.device)
        )
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, problem, opts, dataset=eval_dataset)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    # Start the actual training loop
    val_dataset = problem.make_dataset(size=opts.graph_size, num_samples=opts.val_size, 
                                       filename=opts.val_dataset, cost_input=opts.cost_input, distribution=opts.data_distribution)
    
    # Start the actual training loop
    
    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])
    
        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1
    
    if opts.eval_only:

        if opts.test_dataset:
            test_dataset = problem.make_dataset(size=opts.graph_size, num_samples=opts.test_size, 
                                       filename=opts.test_dataset, cost_input=opts.cost_input, 
                                       distribution=opts.data_distribution)
            #test = test_dataset[:10]
            print(test_dataset[0][:5])
            avg_cost, pi = validate(model, test_dataset, opts, return_pi=True, sorted_pi=True)
            #with open(os.path.join(opts.save_dir, "_test_pi-{}.pkl".format(opts.checkpoint_epoch)), 'wb') as f:
            #    pickle.dump(pi, f, pickle.HIGHEST_PROTOCOL)
            print('avg_cost:', avg_cost)
            SD = sequence_deviation(pi)
            print('sequence_deviation:', SD.max(), SD.min(), SD.mean(), SD.var())
            print(pi[SD.argmin()])

            print()
            [random.shuffle(i) for i in test_dataset]
            #test = test_dataset[:10]
            print(test_dataset[0][:5])
            avg_cost, pi = validate(model, test_dataset, opts, return_pi=True, sorted_pi=True)
            #with open(os.path.join(opts.save_dir, "_test_pi-{}.pkl".format(opts.checkpoint_epoch)), 'wb') as f:
            #    pickle.dump(pi, f, pickle.HIGHEST_PROTOCOL)
            print('avg_cost:', avg_cost)
            SD = sequence_deviation(pi)
            print('sequence_deviation:', SD.max(), SD.min(), SD.mean(), SD.var())
            print(pi[SD.argmin()])

            # print()
            # [random.shuffle(i) for i in test]
            # print(test[0][:5])
            # avg_cost, pi = validate(model, test, opts, return_pi=True, sorted_pi=True)
            # #with open(os.path.join(opts.save_dir, "test_pi{}.pkl".format(opts.checkpoint_epoch)), 'wb') as f:
            # #    pickle.dump(pi, f, pickle.HIGHEST_PROTOCOL)
            # print('avg_cost:', avg_cost)
            # SD = sequence_deviation(pi)
            # print('sequence_deviation:', SD.max(), SD.min(), SD.mean(), SD.var())
            # #print('pi:', pi[0])
        
        #print()
        #avg_cost, pi = validate(model, val_dataset, opts, return_pi=True, sorted_pi=True)
        #with open(os.path.join(opts.save_dir, "_val_pi-{}.pkl".format(opts.checkpoint_epoch)), 'wb') as f:
        #    pickle.dump(pi, f, pickle.HIGHEST_PROTOCOL)
        #print('avg_cost:', avg_cost)
        #SD = sequence_deviation(pi)
        #print('sequence_deviation:', SD.max(), SD.min(), SD.mean(), SD.var())
        #print(pi[SD.argmin()])
    else:
        training_cost   = []
        training_bl_cost = []
        baseline_cost   = []
        freeze_baseline_cost = []
        validation_cost = []
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
        #for epoch in range(5):
            train_cost, train_bl_cost, bl_cost, freeze_bl_cost, val_cost = train_epoch(
                                    model,
                                    optimizer,
                                    baseline,
                                    lr_scheduler,
                                    epoch,
                                    val_dataset,
                                    problem,
                                    tb_logger,
                                    opts
                                    )
            
            training_cost.append(train_cost)
            training_bl_cost.append(train_bl_cost)
            baseline_cost.append(bl_cost)
            freeze_baseline_cost.append(freeze_bl_cost)
            validation_cost.append(val_cost)

            print('avg reward:', val_cost)
        with open(os.path.join(opts.save_dir, 'training_cost.pkl'), 'wb') as f:
            pickle.dump(training_cost, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(opts.save_dir, 'training_bl_cost.pkl'), 'wb') as f:
            pickle.dump(training_bl_cost, f, pickle.HIGHEST_PROTOCOL)
        
        with open(os.path.join(opts.save_dir, 'baseline_cost.pkl'), 'wb') as f:
            pickle.dump(baseline_cost, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(opts.save_dir, 'freeze_baseline_cost.pkl'), 'wb') as f:
            pickle.dump(freeze_baseline_cost, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(opts.save_dir, 'validation_cost.pkl'), 'wb') as f:
            pickle.dump(validation_cost, f, pickle.HIGHEST_PROTOCOL)
        
        if opts.test_dataset:
            test_dataset = problem.make_dataset(size=opts.graph_size, num_samples=opts.test_size, 
                                       filename=opts.test_dataset, cost_input=opts.cost_input, 
                                       distribution=opts.data_distribution)
            avg_cost, pi = validate(model, test_dataset, opts, return_pi=True, sorted_pi=True)
            with open(os.path.join(opts.save_dir, '_test_pi.pkl'), 'wb') as f:
                pickle.dump(pi, f, pickle.HIGHEST_PROTOCOL)
            print('avg_cost:', avg_cost)
            SD = sequence_deviation(pi)
            print('sequence_deviation:', SD.mean())
        
        avg_cost, pi = validate(model, val_dataset, opts, return_pi=True, sorted_pi=True)
        with open(os.path.join(opts.save_dir, '_val_pi.pkl'), 'wb') as f:
            pickle.dump(pi, f, pickle.HIGHEST_PROTOCOL)
        print('avg_cost:', avg_cost)
        SD = sequence_deviation(pi)
        print('sequence_deviation:', SD.mean())

        plot_item = [training_bl_cost[1:], baseline_cost[1:], freeze_baseline_cost[1:], validation_cost[1:]]
        legends   = ['training_cost', 'eval_cost', 'eval_cost_Freezed', 'validation_cost']
        plot_training_result(cost   = plot_item,
                             legend = legends, 
                             title  = 'training-eval-val cost over epochs',
                             save_path=os.path.join(opts.save_dir, '_result.png'),
                             plus     =True)
        # plot_training_result(train_cost =training_cost,
        #                      bl_cost    =baseline_cost,
        #                      val_cost   =validation_cost,
        #                      save_path  =os.path.join(opts.save_dir, 'result.png'))

    return    

if __name__ == "__main__":
    opts = get_options()
    run(opts)

## eval model only
# python run.py --graph_size 100 --baseline rollout --run_name tsp100_rollout 
# --val_dataset data/tsp100/val_location.pkl --eval_dataset data/tsp100/eval_location.pkl  --test_dataset data/tsp100/test_location.pkl 
# --eval_only --use_SD True
# --load_path outputs\tsp_100\tsp100_rollout_20240613T155834_0.01_0.96_100epochs_cost/epoch-99.pt

## use cost input
# python run.py --graph_size 100 --baseline rollout --run_name tsp100_rollout 
# --train_dataset data/tsp100/train_location.pkl  --val_dataset data/tsp100/val_location.pkl 
# --eval_dataset data/tsp100/eval_location.pkl --test_dataset data/tsp100/test_location.pkl 
# --n_epochs 20 --cost_input true --max_grad_norm 0

## use SD
# python run.py --graph_size 100 --baseline rollout --run_name tsp100_rollout 
# --train_dataset data/tsp100/train_location.pkl --val_dataset data/tsp100/val_location.pkl 
# --eval_dataset data/tsp100/eval_location.pkl --test_dataset data/tsp100/test_location.pkl 
# --n_epochs 200 --use_SD True --lr_model 0.0001 --max_grad_norm 0 --lr_decay 0.99
