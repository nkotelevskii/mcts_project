import numpy as np


def get_state_reward(name='dist'):
    if name == 'dist':
        return lambda x_e, x_p: np.linalg.norm(x_e - x_p) / 1000  # if np.all(x_e != x_p) else float("inf")
    else:
        raise ValueError('No such state reward name')


def get_reward(name='dist_reward'):
    if name == 'dist_reward':
        def reward(x_e, x_p, sum_distance, t, T_max=100):
            if tuple(x_e) == tuple(x_p):
                return t - T_max
            else:
                return sum_distance

        return reward

    else:
        raise ValueError('No such reward name')
