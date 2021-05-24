import heapq

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
        self.visited_states = {}
        heapq.heappush(self.queue, self.states[0])

    def add_node(self, parent_id, e_state, p_state, action_applied_e, action_applied_p, state_reward):
        new_node = State(my_id=len(self.states), parent_id=parent_id, e_state=e_state, p_state=p_state,
                         action_applied_e=action_applied_e,
                         action_applied_p=action_applied_p,
                         state_reward=state_reward)
        self.states[parent_id].children_ids.append(len(self.states))
        self.states.append(new_node)
        heapq.heappush(self.queue, self.states[-1])

    def update_tree(self, node_id, outcome):
        while node_id != -2:
            self.states[node_id].update_value(outcome=outcome)
            node_id = self.states[node_id].parent_id
        self.queue = [self.states[i] for i in range(len(self.states))]
        heapq.heapify(self.queue)
