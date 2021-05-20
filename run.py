import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from mcts import EPGame, mcts_two_player_step, plot_joint_enviroment

data = np.load('data_ps3.npz')
environment = data['environment']

env = EPGame(env_map=environment)
env.reset()

im = plot_joint_enviroment(environment, tuple(env.x_e), tuple(env.x_p))
plt.matshow(im)
plt.show()

fig = plt.figure()
imgs = []
for s in tqdm(range(1000)):
    im = plot_joint_enviroment(environment, tuple(env.x_e), tuple(env.x_p))
    plot = plt.imshow(im)
    imgs.append([plot])
    u_e = mcts_two_player_step(env=env, n_iterations=50, T_max=10, use_uct=False,
                               is_pursuer=False)
    u_p = mcts_two_player_step(env=env, n_iterations=5, T_max=5, use_uct=False,
                               is_pursuer=True)
    _, _, done, _ = env.evader_step(u_e)
    if done:
        print('game over((')
        break
    state, _, done, _ = env.pursuer_step(u_p)
    if done:
        print('game over((')
        break

im = plot_joint_enviroment(environment, tuple(env.x_e), tuple(env.x_p))
plot = plt.imshow(im)
imgs.append([plot])
ani = animation.ArtistAnimation(fig, imgs, interval=100, blit=True)

ani.save('scape_solve.mp4', writer="imagemagick")

plt.show()
