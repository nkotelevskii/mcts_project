import numpy as np

from mcts.mcts_algorithm import get_reward
from mcts.utils import make_str_state

reward = get_reward(name='dist_reward')


def selectstate(env, tree):
    state = tree.states[0]
    position = -1
    while len(state.children_ids) > 0:
        state = sorted([tree.states[i] for i in state.children_ids], key=lambda x: x.value)[position]
        if position == -1:
            position = 0
            _, _, done, _ = env.evader_step(state.action_applied_e)
        else:
            position = -1
            _, _, done, _ = env.pursuer_step(state.action_applied_p)
        if done:
            break

    x_e_best = state.e_state
    x_p_best = state.p_state
    return state, x_e_best, x_p_best


def run_simulation(tree, env, previous_node, x_e, x_p, a_e, a_p, T_max):
    # And we add the node to the tree
    tree.add_node(parent_id=previous_node.id, e_state=x_e, p_state=x_p,
                  action_applied_e=a_e, action_applied_p=a_p)
    tree.visited_states[make_str_state(x_e=x_e, x_p=x_p)] = tree.states[-1].id

    # Third, we launch the simulation
    cum_reward = 0
    for t in range(T_max):
        # Step, according to the default policy
        # TODO: queue
        if env.evaders_turn:
            _, rew, done, _ = env.evader_step(np.random.choice(len(env.actions)))
            cum_reward += rew
        else:
            # Step, according to pursuer policy
            _, rew, done, _ = env.pursuer_step(np.random.choice(len(env.actions)))
            cum_reward += rew

        # If we reached the goal or was eaten, then stop
        if done:
            break

    # rew = reward(x_e=x_e, x_p=x_p, sum_distance=cum_reward, t=last_t, T_max=T_max)
    # Forth, we update all parent's nodes
    tree.update_tree(env=env, node_id=-1, outcome=cum_reward)


def simtree(env, tree, T_max=50):
    """
    Monte-Carlo tree search
    env is the environment
    T_max -- maximal length of each simulation
    """
    a = np.array(env.actions)

    # First, we select the new candidate:
    # Search
    best_node, x_e_best, x_p_best = selectstate(env, tree)
    tree.visited_states[make_str_state(x_e=x_e_best, x_p=x_p_best)] = best_node.id
    # Second, we perform an action.
    for a_e, _ in enumerate(a):  # iterate over all actions of evader
        # TODO: Return not only the new state, but also a flag, if the state is feasible
        env.pursuers_turn = False
        env.evaders_turn = True
        env.state = tuple(np.r_[x_e_best, x_p_best])
        env.x_e = x_e_best
        env.x_p = x_p_best
        new_state, rew, done, _ = env.evader_step(a_e)
        new_x_e = new_state[:2]
        # If the state is not visited, then add it to the tree
        if make_str_state(x_e=new_x_e, x_p=x_p_best) not in tree.visited_states.keys():
            run_simulation(tree=tree, env=env, previous_node=best_node, x_e=new_x_e, x_p=x_p_best, a_e=a_e,
                           a_p=None,
                           T_max=T_max)
        else:
            continue
        last_evader_pos = tree.states[-1]
        for a_p, _ in enumerate(a):  # iterate over all actions of pursuer
            env.pursuers_turn = True
            env.evaders_turn = False
            env.state = tuple(np.r_[new_x_e, x_p_best])
            env.x_e = new_x_e
            env.x_p = x_p_best
            new_state, rew, done, _ = env.pursuer_step(a_p)
            new_x_p = new_state[2:]

            if make_str_state(x_e=new_x_e, x_p=new_x_p) not in tree.visited_states.keys():
                run_simulation(tree=tree, env=env, previous_node=last_evader_pos, x_e=new_x_e, x_p=new_x_p, a_e=None,
                               a_p=a_p, T_max=T_max)
