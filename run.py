import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from mcts import EPGame, simtree, plot_joint_enviroment, State, Tree, get_state_reward, save_results, make_str_state


def UctSearch(environment, UCT=False):
    env = EPGame(env_map=environment)
    env.reset()
    x_e = env.x_e
    x_p = env.x_p
    state_reward = get_state_reward(name='dist')

    im = plot_joint_enviroment(environment, tuple(env.x_e), tuple(env.x_p))
    plt.matshow(im)
    plt.show()

    # Define tree, which at the beginning consists of the current state
    tree = Tree(states=[State(my_id=0, parent_id=-2, e_state=x_e, p_state=x_p,
                              state_reward=state_reward(x_e=x_e, x_p=x_p))],
                use_uct=UCT)  # our tree
    for _ in tqdm(range(1000)):
        env.reset()
        simtree(env=env, tree=tree, T_max=50)
    return env, tree


def policy_action(tree, state, is_pursuer):
    sorted_states = sorted([tree.states[i] for i in state.children_ids], key=lambda x: x.value)
    if is_pursuer:
        u = sorted_states[0].action_applied_p
    else:
        u = sorted_states[-1].action_applied_e
    return u


def policy(obs, tree, is_pursuer):
    node_id = tree.visited_states.get(make_str_state(obs[:2], obs[2:]), None)
    if node_id is not None:
        sorted_states = sorted(
            [tree.states[i] for i in tree.states[node_id].children_ids],
            key=lambda x: x.value)
        if len(sorted_states):
            best_evader = sorted_states[-1]
            if is_pursuer:
                sorted_states = sorted(
                    [tree.states[i] for i in tree.states[best_evader.id].children_ids],
                    key=lambda x: x.value)
                if len(sorted_states):
                    return sorted_states[0].action_applied_p
            else:
                return best_evader.action_applied_e
    return 0


def make_policy(tree, is_pursuer):
    if is_pursuer:
        return lambda obs: policy(obs, tree=tree, is_pursuer=True)
    else:
        return lambda obs: policy(obs, tree=tree, is_pursuer=False)


if __name__ == '__main__':
    data = np.load('data_ps3.npz')
    environment = data['environment']
    env, tree = UctSearch(environment=environment, UCT=False)
    evader_policy = make_policy(tree=tree, is_pursuer=False)
    pursuer_policy = make_policy(tree=tree, is_pursuer=True)
    save_results(env=env, evader_policy=evader_policy, pursuer_policy=pursuer_policy, max_iters=100)
    print('Success!')
