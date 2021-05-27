from mcts.mcts_algorithm.state import State


class Tree:
    """
    The class is the Tree implementation
    """

    def __init__(self, states, use_uct=False):
        """

        :param states: list of states
        :param use_uct: if True, we use UCT heuristic.
        """
        self.states = states
        self.use_uct = use_uct
        self.queue = []
        # self.visited_states = {}

    def add_node(self, parent_id, e_state, p_state, action_applied_e, action_applied_p):
        new_node = State(my_id=len(self.states), parent_id=parent_id, e_state=e_state, p_state=p_state,
                         action_applied_e=action_applied_e,
                         action_applied_p=action_applied_p
                         )
        self.states[parent_id].children_ids.append(len(self.states))
        self.states.append(new_node)

    def update_tree(self, env, node_id, outcome):
        cum_reward = outcome
        while node_id != -2:
            self.states[node_id].update_value(outcome=cum_reward)
            node_id = self.states[node_id].parent_id
            cum_reward += env.reward(self.states[node_id].e_state, self.states[node_id].p_state)
