import pickle

import numpy as np

from mcts import EPGame, My_policy, make_policy, save_results

if __name__ == '__main__':
    T_max = 100
    with open('tree.pkl', 'rb') as f:
        tree = pickle.load(f)
    data = np.load('data_ps3.npz')
    environment = data['environment']
    env = EPGame(env_map=environment, use_goal=True)
    env.reset()
    pol = My_policy(tree=tree, env=env)
    evader_policy = make_policy(pol)
    pursuer_policy = make_policy(pol)
    env.reset()
    print(len(pol.actions))
    save_results(env=env, evader_policy=evader_policy, pursuer_policy=pursuer_policy, max_iters=T_max)
    print('Success!')
