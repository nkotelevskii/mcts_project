import numpy as np

from mcts.mcts_algorithm import get_reward, get_state_reward
from mcts.utils import make_str_state


def selectstate(tree):
    best_node = tree.queue.pop(0)
    x_e_best = best_node.e_state
    x_p_best = best_node.p_state
    return best_node, x_e_best, x_p_best


def run_simulation(tree, env, previous_node, x_e, x_p, a_e, a_p, T_max):
    # And we add the node to the tree
    a = np.array(env.actions)
    state_reward = get_state_reward(name='dist')
    reward = get_reward(name='dist_reward')
    tree.add_node(parent_id=previous_node.id, e_state=x_e, p_state=x_p,
                  action_applied_e=a_e, action_applied_p=a_p,
                  state_reward=state_reward(x_e=x_e, x_p=x_p))
    tree.visited_states[make_str_state(x_e=x_e, x_p=x_p)] = tree.states[-1].id

    # To compute reward, we have to take distance of previous steps between evader and pursuer into account
    sum_distance = tree.states[-1].state_reward
    node_id = tree.states[-1].id
    while node_id != -2:
        node_id = tree.states[node_id].parent_id
        sum_distance += tree.states[node_id].state_reward

    last_t = T_max
    # Third, we launch the simulation
    for t in range(T_max):
        # Step, according to the default policy
        x_e = env.transition(x=x_e, u=a[np.random.choice(a.shape[0])])
        # Step, according to pursuer policy
        x_p = env.transition(x=x_p, u=a[np.random.choice(a.shape[0])])
        # Accumulate the inverted distance. We need it for reward computation
        sum_distance += state_reward(x_e=x_e, x_p=x_p)
        # If we reached the goal or was eaten, then stop
        if tuple(x_e) == tuple(x_p):
            last_t = t
            break

    rew = reward(x_e=x_e, x_p=x_p, sum_distance=sum_distance, t=last_t, T_max=T_max)
    # Forth, we update all parent's nodes
    tree.update_tree(node_id=-1, outcome=rew)


def simtree(env, tree, T_max=50):
    """
    Monte-Carlo tree search
    env is the environment
    T_max -- maximal length of each simulation
    """
    a = np.array(env.actions)

    # First, we select the new candidate:
    # Search
    best_node, x_e_best, x_p_best = selectstate(tree)
    tree.visited_states[make_str_state(x_e=x_e_best, x_p=x_p_best)] = best_node.id
    # Second, we perform an action.
    for a_e, action_e in enumerate(a):  # iterate over all actions of evader
        for a_p, action_p in enumerate(a):  # iterate over all actions of pursuer
            # TODO: Return not only the new state, but also a flag, if the state is feasible
            new_x_e = env.transition(x=x_e_best, u=action_e)

            # If the state is not visited, then add it to the tree
            if make_str_state(x_e=new_x_e, x_p=x_p_best) not in tree.visited_states.keys():
                run_simulation(tree=tree, env=env, previous_node=best_node, x_e=new_x_e, x_p=x_p_best, a_e=a_e, a_p=a_p,
                               T_max=T_max)
            else:
                continue

            new_x_p = env.transition(x=x_p_best, u=action_p)

            if make_str_state(x_e=new_x_e, x_p=new_x_p) not in tree.visited_states.keys():
                run_simulation(tree=tree, env=env, previous_node=tree.states[-1], x_e=new_x_e, x_p=new_x_p, a_e=a_e,
                               a_p=a_p, T_max=T_max)
            else:
                continue
