import pickle

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from mcts import simtree, plot_joint_enviroment, State, Tree, EPGame


def UctSearch(env, T_max, num_epochs, UCT=False):
    # env = EPGame(env_map=environment, reward=nogoal_reward)
    state = env.reset()
    x_e = state[:2]
    x_p = state[2:]

    im = plot_joint_enviroment(environment, tuple(x_e), tuple(x_p))
    plt.matshow(im)
    plt.show()

    # Define tree, which at the beginning consists of the current state
    tree = Tree(states=[State(my_id=0, parent_id=-2, e_state=x_e, p_state=x_p, action_applied_p=-1)],
                use_uct=UCT)  # our tree
    # tree.visited_states[make_str_state(x_e=x_e, x_p=x_p, evaders_turn=True)] = 0
    for _ in tqdm(range(num_epochs)):
        env.reset()
        simtree(env=env, tree=tree, T_max=T_max)
    return tree


if __name__ == '__main__':
    T_max = 60
    num_epochs = 100000
    data = np.load('data_ps3.npz')
    environment = data['environment']
    env = EPGame(env_map=environment, use_goal=True)
    tree = UctSearch(env=env, T_max=T_max, num_epochs=num_epochs, UCT=False)
    with open("tree.pkl", "wb") as f:
        pickle.dump(tree, f)
