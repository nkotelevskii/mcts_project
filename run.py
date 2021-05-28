import argparse
import pickle

import numpy as np
from tqdm.auto import tqdm

from mcts import simtree, State, Tree, EPGame


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--save_name', type=str, default='tree.pkl', help='Name to save model with')
parser.add_argument('--use_UCT', type=str2bool, default=True, help='Whether to use UCT')
parser.add_argument('--T_max', type=int, default=100, help='T_max: length of MC path')
parser.add_argument('--seed', type=int, default=200, help='seed')
parser.add_argument('--num_epochs', type=int, default=10000, help='num_epochs: number of epochs')
args = parser.parse_args()


def UctSearch(env, T_max, num_epochs, UCT=False):
    # env = EPGame(env_map=environment, reward=nogoal_reward)
    state = env.reset()
    x_e = state[:2]
    x_p = state[2:]

    # im = plot_joint_enviroment(environment, tuple(x_e), tuple(x_p))
    # plt.matshow(im)
    # plt.show()

    # Define tree, which at the beginning consists of the current state
    tree = Tree(states=[State(my_id=0, parent_id=-2, e_state=x_e, p_state=x_p, action_applied_p=-1)],
                use_uct=UCT)  # our tree
    # tree.visited_states[make_str_state(x_e=x_e, x_p=x_p, evaders_turn=True)] = 0
    for _ in tqdm(range(num_epochs)):
        env.reset()
        simtree(env=env, tree=tree, T_max=T_max)
    return tree


if __name__ == '__main__':
    print(
        f"Save model as {args.save_name} with T_max={args.T_max} and epochs={args.num_epochs} and UCT={args.use_UCT}")
    T_max = args.T_max
    num_epochs = args.num_epochs

    data = np.load('data_ps3.npz')
    environment = data['environment']
    env = EPGame(env_map=environment, use_goal=True, seed_num = args.seed)
    tree = UctSearch(env=env, T_max=T_max, num_epochs=num_epochs, UCT=args.use_UCT)
    with open(args.save_name, "wb") as f:
        pickle.dump(tree, f)

