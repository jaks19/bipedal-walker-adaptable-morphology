import gym
gym.logger.set_level(50)
import numpy as np
import ray
from environments import BipedalWalkerCustom

def get_model_keys_in_order(model_keys):
    ordered_keys = [None]*len(model_keys)

    for k in model_keys:
        if k == 'morph': continue

        chars = list(k)
        assert(chars[0]=="W")
        index = int(chars[1])
        ordered_keys[index] = k

    return ordered_keys

@ray.remote(num_cpus=1)
class Worker(object):
    def __init__(self, env_config):
        self.env_config = env_config
        return

    def get_action(self, state, model):
        ordered_keys = get_model_keys_in_order(model.keys())
        h = np.tanh(np.matmul(state, model[ordered_keys.pop(0)]))

        for k in ordered_keys:
            h = np.tanh(np.matmul(h, model[ordered_keys.pop(0)]))

        return h

    def evaluate_model(self, batch, num_rollouts=1, debug=False):
        returned_container = []

        for b in batch:
            model_idx, model = b
            morph_vector = model['morph']

            env = BipedalWalkerCustom(self.env_config)
            env.augment_env(morph_vector)

            total_reward = 0

            for rollout in range(num_rollouts):
                state = env.reset()
                
                steps = 0
                while True:
                    action = self.get_action(state, model)
                    state, reward, done, info = env._step(action)
                    if debug: env.render()
                    total_reward += reward
                    steps += 1

                    if done or steps > 1000: break

            returned_container.append([model_idx, total_reward/num_rollouts])
        return returned_container
