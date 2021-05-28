import numpy as np

from mcts.mcts_algorithm import get_reward

reward = get_reward(name='dist_reward')


def UCT(state, position, tree, c=1):
    sign = 1 if position == -1 else 1
    q_sa = state.value
    n_sa = state.n_visits

    parent_id = state.parent_id
    parent_state = tree.states[parent_id]
    n_s = parent_state.n_visits
    if tree.use_uct:
        if n_sa != 0:
            uct = q_sa + sign * c * np.sqrt(np.log(n_s) / n_sa)
        else:
            uct = sign * np.inf
    else:
        uct = q_sa
    return uct


def tree_policy(env, tree, c=1):
    state = tree.states[0]
    position = -1
    while len(state.children_ids) != 0:
        state_upd = sorted([tree.states[i] for i in state.children_ids], key=lambda state: UCT(state, position, tree, c))[
            position]
        if position == -1:
            if state_upd.value < state.value and len(env.allowed_actions) != len(
                    state.children_ids):  # if children are worse, but we can build a sibling
                break
            position = 0
            _, _, done, info = env.evader_step(state_upd.action_applied_e)
        else:
            position = -1
            if state_upd.value > state.value and len(env.allowed_actions) != len(
                    state.children_ids):  # if children are worse, but we can build a sibling
                break
            _, _, done, info = env.pursuer_step(state_upd.action_applied_p)
        state = state_upd
        if done:
            print("")
            print("DONE: ", info)
            break

    x_e_best = state.e_state
    x_p_best = state.p_state
    return state, x_e_best, x_p_best


def run_simulation(tree, env, T_max):
    # Third, we launch the simulation
    cum_reward = 0
    for t in range(T_max):
        # Step, according to the default policy
        if len(env.allowed_actions) == 0:
            break

        if env.evaders_turn:
            _, rew, done, info = env.evader_step(np.random.choice(env.allowed_actions))
            cum_reward += rew
        else:
            # Step, according to pursuer policy
            _, rew, done, info = env.pursuer_step(np.random.choice(env.allowed_actions))
            cum_reward += rew

        # If we reached the goal or was eaten, then stop
        if done:
            print("")
            print("DONE: ", info)
            break

    # rew = reward(x_e=x_e, x_p=x_p, sum_distance=cum_reward, t=last_t, T_max=T_max)
    # Forth, we update all parent's nodes
    tree.update_tree(env=env, node_id=-1, outcome=cum_reward)


def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def simtree(env, tree, T_max=50, c=1):
    """
    Monte-Carlo tree search
    env is the environment
    T_max -- maximal length of each simulation
    """

    # First, we select the new candidate:
    # Search
    best_node, x_e_best, x_p_best = tree_policy(env, tree, c=c)
    # Second, we perform an action.
    if len(env.allowed_actions) > 0:
        action_e, action_p = None, None
        if best_node.action_applied_e is None:
            action = env.allowed_actions[np.argmin(
                [distance(env.transition(env.x_e, env._total_actions[a]), env.goal) for a in env.allowed_actions])]
            env.evader_step(action)
            action_e = action
        else:
            action = env.allowed_actions[np.argmin(
                [distance(env.transition(env.x_p, env._total_actions[a]), env.x_e) for a in env.allowed_actions])]
            env.pursuer_step(action)
            action_p = action
        tree.add_node(parent_id=best_node.id, e_state=env.x_e, p_state=env.x_p, action_applied_e=action_e,
                      action_applied_p=action_p)
    # tree.visited_states[make_str_state(x_e=env.x_e, x_p=env.x_p, evaders_turn=env.evaders_turn)] = tree.states[
    #     -1].id
    run_simulation(tree=tree, env=env, T_max=T_max - env.t)
