import argparse
import os
import shutil
import pickle as pickle
import ray
import numpy as np
import random 
from tqdm import tqdm
from tensorboardX import SummaryWriter

from worker import Worker
from environments import Env_config

ENV_CONFIG = Env_config(
    name='rough',
    ground_roughness=0,
    pit_gap=[1,2],
    stump_width=None,
    stump_height=None,
    stump_float=None,
    stair_height=None,
    stair_width=None,
    stair_steps=None,
    )


def initialize_model_simple(args):
    np.random.seed(10)

    h1_size = 100

    if args.saved_model is not None:
        model = pickle.load(open(args.saved_model, 'rb'))

    else:
        model = {}
        model['W0'] = np.random.randn(24, h1_size) / np.sqrt(24)
        model['W1'] = np.random.randn(h1_size, 4) / np.sqrt(h1_size)

        model['morph'] = args.initial_scalar*(1.0 + (np.random.rand(8)*2-1.0)*0.5)

    return model


def get_parallelized_reward_array(workers, perturbed_models, args):
    R = np.zeros(len(perturbed_models))
    num_workers = len(workers)

    # Fully saturate al the workers to start
    assert(len(workers) <= len(perturbed_models))
    batches_to_assign_to_workers = group_jobs_for_workers(jobs=perturbed_models, num_workers=num_workers)
    ongoing_ids = [workers[j].evaluate_model.remote(batches_to_assign_to_workers[j], num_rollouts=3) for j in range(len(workers))]
    returns = ray.get(ongoing_ids)

    for r in returns:
        for model_data in r:
            model_idx, score = model_data
            R[model_idx] = score

    return R

def group_jobs_for_workers(jobs, num_workers):
    import math as m
    batch_size = m.floor(len(jobs)/num_workers)
    tail = len(jobs) - batch_size*num_workers

    batches = []
    idx = 0

    for j in range(tail):
        batch = []
        for i in range(batch_size+1):
            batch.append([idx, jobs.pop(0)])
            idx += 1

        batches.append(batch)
        
    while len(jobs) > 0:
        batch = []
        for i in range(batch_size):
            batch.append([idx, jobs.pop(0)])
            idx += 1

        batches.append(batch)
    
    return batches

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train BipedWalker')

    parser.add_argument('--log_dir', type=str, help='log directory')
    parser.add_argument('--num_cores', type=int, help='num cpu cores')
    parser.add_argument('--npop', type=int, help='num cpu cores')
    parser.add_argument('--num_workers', type=int, help='number of data-collecting workers')
    parser.add_argument('--saved_model', type=str, default=None, help='saved model path')

    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.03)

    parser.add_argument('--scale_limit_lower', type=float, default=1)
    parser.add_argument('--scale_limit_upper', type=float, default=1)
    parser.add_argument('--initial_scalar', type=float, default=1)
    
    parser.add_argument('--debug', type=bool, default=True)


    args = parser.parse_args()
    
    # Logging init
    log_dir = args.log_dir
    if os.path.exists(os.path.join(log_dir, 'tmp-ray-logs')): shutil.rmtree(os.path.join(log_dir, 'tmp-ray-logs'))

    if not os.path.exists(log_dir): os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, 'tmp-ray-logs')): os.makedirs(os.path.join(log_dir, 'tmp-ray-logs'))
    if not os.path.exists(os.path.join(log_dir, 'models')): os.makedirs(os.path.join(log_dir, 'models'))

    # Start
    ray.init(object_store_memory=int(10e8), temp_dir=os.path.join(log_dir, f'tmp-ray-logs'), configure_logging=False, num_cpus=args.num_cores)
    workers = [Worker.remote(ENV_CONFIG) for i in range(args.num_workers)]
    global_evaluator_worker = Worker.remote(ENV_CONFIG)

    model = initialize_model_simple(args)

    npop = args.npop
    aver_reward = None
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir,f'writer'))

    for i in tqdm(range(10000)):

        # npop different perturbations to each weight matrix W1,W2,W2
        N = {}
        for k, v in model.items():
            if k == 'morph': continue
            N[k] = np.random.randn(npop, v.shape[0], v.shape[1])

        N['morph'] = np.vstack([random.uniform(args.scale_limit_lower, args.scale_limit_upper)*(1.0 + (np.random.rand(8)*2-1.0)*0.5) for i in range(npop)])
        
        # npop different scores 
        R = np.zeros(npop)

        # npop different perturbed models
        perturbed_models = []

        for j in range(npop):
            model_try = {}
            for k, v in model.items(): 
                model_try[k] = v + args.sigma*N[k][j]

            perturbed_models.append(model_try)

        # Only for tracking where the overall previous model has reached
        cur_reward_worker_id = global_evaluator_worker.evaluate_model.remote([[-1, model]], num_rollouts=3, debug=args.debug)

        # Launch num_workers workers and keep pushing the npop different perturbed models to them 
        # until they are all done
        R = get_parallelized_reward_array(workers, perturbed_models, args)

        # More confident that it is done here
        cur_reward = ray.get(cur_reward_worker_id)[0][1]
        aver_reward = aver_reward * 0.9 + cur_reward * 0.1 if aver_reward is not None else cur_reward
        print(f'iter {i}, cur_reward {cur_reward}, aver_reward {aver_reward} morphology {model["morph"]}')
        writer.add_scalar('cur_reward', cur_reward, i)
        writer.add_scalar('aver_reward', aver_reward, i)

        if i %10 == 0: pickle.dump(model, open(os.path.join(args.log_dir, f'''models/model-pedal-{cur_reward}-{model['morph'][0]}-{model['morph'][1]}-{model['morph'][2]}-{model['morph'][3]}-{model['morph'][4]}-{model['morph'][5]}-{model['morph'][6]}-{model['morph'][7]}.p'''), 'wb'))

        # New model is a weighted combination (based on resulting rewards) of perturbed models + old model
        A = (R - np.mean(R)) / np.std(R)
        for k in model:
            if k == 'morph': continue
            model[k] = model[k] + args.alpha/(npop*args.sigma) * np.dot(N[k].transpose(1, 2, 0), A)

        model['morph'] = model['morph'] + args.alpha /(npop*args.sigma) * np.dot(N['morph'].transpose(1, 0), A)
