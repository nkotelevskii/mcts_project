import numpy as np


def get_state_reward(name='dist', is_pursuer=False, goal=None):
    if goal is not None:
        if name == 'dist':
            return lambda x_e, x_p: np.linalg.norm(x_e - x_p)  # if np.all(x_e != x_p) else float("inf")
        else:
            raise ValueError('No such state reward name')
    else:
        if is_pursuer:
            return lambda x_e, x_p: -np.linalg.norm(x_e - x_p)
        else:
            return lambda x_e, x_p: np.linalg.norm(x_e - goal)


def get_reward(name='dist_reward', is_pursuer=False, goal=None):
    if goal is None:
        mult = -1. if is_pursuer else 1.
        if name == 'dist_reward':
            def reward(x_e, x_p, sum_distance, t, T_max=100):
                if tuple(x_e) == tuple(x_p):
                    return mult * (t - T_max)
                else:
                    return mult * sum_distance

            return reward

        else:
            raise ValueError('No such reward name')
    else:
        def reward(x_e, x_p, sum_distance, t, T_max=100):
            if tuple(goal) != (None, None):
                r = -np.linalg.norm(x_e - goal) + np.linalg.norm(x_e - x_p)
            else:
                r = -np.linalg.norm(x_e - goal)
            if tuple(x_e) == tuple(goal):
                r += 100
