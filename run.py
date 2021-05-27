import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from mcts import EPGame, simtree, plot_joint_enviroment, State, Tree, save_results, nogoal_reward


def UctSearch(environment, UCT=False):
    env = EPGame(env_map=environment, reward=nogoal_reward)
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
    for _ in tqdm(range(10000)):
        env.reset()
        simtree(env=env, tree=tree, T_max=100)
    return env, tree


def policy_action(tree, state, is_pursuer):
    sorted_states = sorted([tree.states[i] for i in state.children_ids], key=lambda x: x.value)
    if is_pursuer:
        u = sorted_states[0].action_applied_p
    else:
        u = sorted_states[-1].action_applied_e
    return u


class My_policy:
    def __init__(self, tree):
        self.actions = []
        self.tree = tree
        self.counter = 0

        self.fit()

    def fit(self):
        state = self.tree.states[0]
        position = -1
        while len(state.children_ids) != 0:
            state = sorted([tree.states[i] for i in state.children_ids], key=lambda x: x.value)[position]
            if position == -1:
                position = 0
                self.actions.append(state.action_applied_e)
            else:
                position = -1
                self.actions.append(state.action_applied_p)

    def policy(self, obs):
        self.counter += 1
        if self.counter - 1 < len(self.actions):
            return self.actions[self.counter - 1]
        else:
            print("Random")
            return np.random.randint(low=0, high=4)


def make_policy(pol):
    return lambda obs: pol.policy(obs)


if __name__ == '__main__':
    data = np.load('data_ps3.npz')
    environment = data['environment']
    env, tree = UctSearch(environment=environment, UCT=False)
    pol = My_policy(tree=tree)
    evader_policy = make_policy(pol)
    pursuer_policy = make_policy(pol)
    env.reset()
    print(len(pol.actions))
    save_results(env=env, evader_policy=evader_policy, pursuer_policy=pursuer_policy, max_iters=100)
    print('Success!')
