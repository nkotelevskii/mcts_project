import numpy as np

from mcts.mcts_algorithm import State, Tree, get_reward, get_state_reward
from mcts.utils import make_str_state


def mcts_two_player_step(env, n_iterations, T_max=50, use_uct=False, is_pursuer=False):
    """
    Monte-Carlo tree search
    env is the environment
    n_iterations - number of iterations (search, expand, simulation, evaluation)
    T_max -- maximal length of each simulation
    is_pursuer -- True if it is pursuer makes a step
    """
    a = np.array(env.actions)
    state_reward = get_state_reward(name='dist')
    reward = get_reward(name='dist_reward', is_pursuer=is_pursuer)

    x_e = env.x_e
    x_p = env.x_p

    # Define tree, which at the beginning consists of the current state
    tree = Tree(states=[State(my_id=0, parent_id=-2, e_state=x_e, p_state=x_p,
                              state_reward=state_reward(x_e=x_e, x_p=x_p))],
                use_uct=use_uct)  # our tree

    # And define a queue for BFS-like search (like in PS3)
    q = [tree.states[0]]
    # Dictionary of visited states
    visited_states = {make_str_state(x_e=x_e, x_p=x_p): True}

    # Cycle over iterations
    for k in range(n_iterations):
        # First, we select the new candidate:
        # Search
        best_node = q.pop(0)  # here we have the best state
        x_e_best = best_node.e_state
        x_p_best = best_node.p_state
        # Define a queue of children, to sort them later
        children_states = []
        # Second, we perform an action.
        for a_e, action_e in enumerate(a):  # iterate over all actions of evader
            for a_p, action_p in enumerate(a):  # iterate over all actions of pursuer
                new_x_e = env.transition(x=x_e_best, u=action_e)
                new_x_p = env.transition(x=x_p_best, u=action_p)

                # If the state is not visited, then add it to the tree
                if make_str_state(x_e=new_x_e, x_p=new_x_p) not in visited_states.keys():
                    # And we add the node to the tree
                    tree.add_node(parent_id=best_node.id, e_state=new_x_e, p_state=new_x_p,
                                  action_applied_e=a_e, action_applied_p=a_p,
                                  state_reward=state_reward(x_e=new_x_e, x_p=new_x_p))
                    children_states.append(tree.states[-1])
                    visited_states[make_str_state(x_e=new_x_e, x_p=new_x_p)] = True
                else:
                    continue

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
                    new_x_e = env.transition(x=new_x_e, u=a[np.random.choice(a.shape[0])])
                    # Step, according to pursuer policy
                    new_x_p = env.transition(x=new_x_p, u=a[np.random.choice(a.shape[0])])
                    # Accumulate the inverted distance. We need it for reward computation
                    sum_distance += state_reward(x_e=new_x_e, x_p=new_x_p)
                    # If we reached the goal or was eaten, then stop
                    if tuple(new_x_e) == tuple(new_x_p):
                        last_t = t
                        break

                rew = reward(x_e=new_x_e, x_p=new_x_p, sum_distance=sum_distance, t=last_t, T_max=T_max)
                # Forth, we update all parent's nodes
                tree.update_tree(node_id=-1, outcome=rew)
        # Here we add children to the whole queue, pushing more promising to the head
        q = q + sorted(children_states, key=lambda x: x.value)[::-1]

    # And here we just extract the most promising action.
    u = sorted([(i, tree.states[i].value) for i in tree.states[0].children_ids], key=lambda x: x[1])
    if is_pursuer:
        u = tree.states[u[-1][0]].action_applied_p
    else:
        u = tree.states[u[-1][0]].action_applied_e
    return u
