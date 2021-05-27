class State:
    """
    Class, which represents the state. State is the combination of pursuer and evader positions.
    """

    def __init__(self, my_id, parent_id, e_state, p_state, action_applied_e=None, action_applied_p=None):
        """
        my_id -- id of the node
        parent_id -- id of its parent
        e_state -- evader state
        p_state -- pursuer state
        state_reward -- auxiliary variable, used for the final reward computation
        action_applied_e -- the action applied to node's parent (evader) that leads to this state
        action_applied_p -- the action applied to node's parent (pursuer) that leads to this state
        """
        self.id = my_id
        self.parent_id = parent_id
        self.children_ids = []
        self.e_state = e_state
        self.p_state = p_state
        self.n_visits = 0
        self.sum_reward = 0.
        self.action_applied_e = action_applied_e
        self.action_applied_p = action_applied_p

    @property
    def value(self, ):
        """
        The function computes value of the state. This is the thing we have to change, to receive UCT algorithm
        :return:
        """
        return 0 if self.n_visits == 0 else self.sum_reward / self.n_visits

    def __lt__(self, other):
        """
        This function is needed to define a comparison on State instances.
        We need it in case we implement a heap (for UCT).
        :param other:
        :return:
        """
        return self.value < other.value

    def update_value(self, outcome):
        self.n_visits += 1.
        self.sum_reward += outcome

    def add_children(self, child_id):
        self.children_ids.append(child_id)
