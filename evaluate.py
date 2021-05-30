import argparse
import pickle

import numpy as np

from mcts import EPGame, My_policy, make_policy, save_results

parser = argparse.ArgumentParser()
parser.add_argument('--load_name', type=str, default='tree.pkl', help='Name to save model with')
parser.add_argument('--T_max', type=int, default=100, help='How long to simulate')
parser.add_argument('--seed', type=int, default=200, help='Set random seed')
args = parser.parse_args()

if __name__ == '__main__':
    T_max = args.T_max
    with open(args.load_name, 'rb') as f:
        tree = pickle.load(f)
    data = np.load('data_ps3.npz')
    environment = data['environment']
    env = EPGame(env_map=environment, use_goal=True, seed_num=args.seed)
    env.reset()
    pol = My_policy(tree=tree, env=env)
    print('Path reward: ', pol.plan_reward)
    evader_policy = make_policy(pol)
    pursuer_policy = make_policy(pol)
    env.reset()
    print(len(pol.actions))
    save_results(env=env, evader_policy=evader_policy, pursuer_policy=pursuer_policy, max_iters=T_max, name = args.load_name.split('.')[0])
    print('Success!')
