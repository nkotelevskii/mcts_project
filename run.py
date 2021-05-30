import argparse

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


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
parser.add_argument('--T', type=int, default=100, help='T_max: length of MC path')
parser.add_argument('--T_max_e', type=int, default=50, help='T_max: length of MC path')
parser.add_argument('--T_max_p', type=int, default=10, help='T_max: length of MC path')
parser.add_argument('--n_iteration_e', type=int, default=100, help='num_epochs: number of epochs')
parser.add_argument('--n_iteration_p', type=int, default=10, help='num_epochs: number of epochs')
args = parser.parse_args()

from mcts import EPGame, mcts_two_player_step, plot_joint_enviroment

if __name__ == '__main__':
    print(f"Save model as {args.save_name} and UCT={args.use_UCT}")

    data = np.load('data_ps3.npz')
    environment = data['environment']

    env = EPGame(env_map=environment, use_goal=True)
    env.reset()

    fig = plt.figure()
    imgs = []
    for s in tqdm(range(args.T)):
        im = plot_joint_enviroment(environment, tuple(env.x_e), tuple(env.x_p), tuple(env.goal))
        plot = plt.imshow(im)
        imgs.append([plot])
        u_e = mcts_two_player_step(env=env, n_iterations=args.n_iteration_e, T_max=args.T_max_e, use_uct=args.use_UCT,
                                   is_pursuer=False)
        u_p = mcts_two_player_step(env=env, n_iterations=args.n_iteration_p, T_max=args.T_max_p, use_uct=args.use_UCT,
                                   is_pursuer=True)
        _, _, done, _ = env.evader_step(u_e)
        if done:
            print('game over((')
            break
        state, _, done, _ = env.pursuer_step(u_p)
        if done:
            print('game over((')
            break

    im = plot_joint_enviroment(environment, tuple(env.x_e), tuple(env.x_p), tuple(env.goal))
    plot = plt.imshow(im)
    imgs.append([plot])
    ani = animation.ArtistAnimation(fig, imgs, interval=100, blit=True)

    ani.save('goal.mp4', writer="imagemagick")

    plt.show()
