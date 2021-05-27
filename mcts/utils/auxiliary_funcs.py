import numpy as np


def make_str_state(x_e, x_p, evaders_turn):
    return str(np.array(x_e)) + " " + str(np.array(x_p)) + " " + str(evaders_turn)


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


class My_policy:
    def __init__(self, env, tree):
        self.actions = []
        self.tree = tree
        self.counter = 0
        self.env = env
        self.is_finihed = False

        self.fit()

    def fit(self):
        state = self.tree.states[0]
        position = -1
        while len(state.children_ids) != 0:
            state = sorted([self.tree.states[i] for i in state.children_ids], key=lambda x: x.value)[position]
            if position == -1:
                position = 0
                self.actions.append(state.action_applied_e)
            else:
                position = -1
                self.actions.append(state.action_applied_p)

    @staticmethod
    def distance(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def policy(self, obs):
        self.counter += 1
        if self.counter - 1 < len(self.actions):
            return self.actions[self.counter - 1]
        else:
            if not self.is_finihed:
                print("Tree has finihed:", self.env.t)
                self.is_finihed = True

            if self.env.evaders_turn:
                action = self.env.allowed_actions[np.argmin(
                    [self.distance(self.env.transition(self.env.x_e, self.env._total_actions[a]), self.env.goal) for a
                     in
                     self.env.allowed_actions])]
            else:
                action = self.env.allowed_actions[np.argmin(
                    [self.distance(self.env.transition(self.env.x_p, self.env._total_actions[a]), self.env.x_e) for a in
                     self.env.allowed_actions])]

            return self.env.allowed_actions[0]


def make_policy(pol):
    return lambda obs: pol.policy(obs)
