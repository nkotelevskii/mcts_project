import numpy as np


def make_str_state(x_e, x_p):
    return str(x_e) + " " + str(x_p)


def plot_joint_enviroment(env, x_e, x_p):
    """
    env is the grid enviroment
    x_e is the state of the evader
    x_p is the state of the pursuer
    """
    current_env = np.copy(env)
    # plot evader
    current_env[x_e] = 1.0  # yellow
    # plot pursuer
    current_env[x_p] = 0.6  # cyan-ish
    return current_env
