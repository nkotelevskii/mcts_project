#!/usr/bin/python

import heapq

import numpy as np

from mcts.utils.auxiliary_funcs import transition_function, pursuer_transition, action_space


class Node:
    def __init__(self, my_id, parent_id, state, p_state, sum_inv_distance, action_applied=None):
        self.id = my_id
        self.parent_id = parent_id
        self.children_ids = []
        self.children_queue = []
        self.state = state
        self.p_state = p_state
        self.n_visits = 0
        self.sum_reward = 0.
        self.action_applied = action_applied
        self.sum_inv_distance = sum_inv_distance  # aggregated sum of distances, before termination happened

    @property
    def value(self, ):
        return 0 if self.n_visits == 0 else self.sum_reward / self.n_visits

    def __lt__(self, other):
        return self.value < other.value

    def update_value(self, outcome):
        self.n_visits += 1
        self.sum_reward += outcome

    def add_children(self, child_id):
        self.children_ids.append(child_id)

    def heapify_children(self, queue):
        self.children_queue = queue
        heapq.heapify(self.children_queue)


class MyTree:
    def __init__(self, nodes):
        self.nodes = nodes
        self.queue = []
        heapq.heappush(self.queue, self.nodes[0])

    def add_node(self, parent_id, state, p_state, action_applied, sum_inv_distance):
        new_node = Node(my_id=len(self.queue), parent_id=parent_id, state=state, p_state=p_state,
                        action_applied=action_applied, sum_inv_distance=sum_inv_distance)
        self.nodes[parent_id].children_ids.append(len(self.nodes))
        self.nodes.append(new_node)
        heapq.heappush(self.queue, self.nodes[-1])

    def update_tree(self, node_id, outcome):
        while node_id != -2:
            self.nodes[node_id].update_value(outcome=outcome)
            node_id = self.nodes[node_id].parent_id
        self.queue = [self.nodes[i] for i in range(len(self.nodes))]
        heapq.heapify(self.queue)

    def return_best(self):
        # return self.queue[-1]
        return heapq.heappop(self.queue)


def reward(x_e, x_p, sum_inv_distance, goal, t, T_max=100):
    if x_e == x_p:
        return 0
    elif x_e == goal:
        return T_max - t + 0.1 * sum_inv_distance
    else:
        return 0.1 * sum_inv_distance - t


def mcts(env, x_e, x_p, goal, k_budget, default_policy, T_max=100):
    """
    Monte-Carlo tree search
    env is the grid enviroment
    x_e evader
    x_p pursuer
    goal is the goal state
    k_budget number of simulations
    default_policy
    """
    a = np.array(action_space)
    new_x_p = x_p
    new_x_e = x_e
    tree = MyTree([Node(my_id=0, parent_id=-2, state=x_e, p_state=x_p,
                        sum_inv_distance=1. / np.linalg.norm(np.array(x_e) - np.array(x_p)))])  # our tree
    for _ in range(k_budget):
        sum_inv_distance = 0.  # aggregated sum of distances, before termination happened
        # First, we select the new candidate:
        best_node = tree.return_best()  # here we have the best state
        x_e_best = best_node.state

        # Second, we perform action, according to our default policy
        u_e = a[default_policy[x_e_best]]
        new_x_e, _ = transition_function(env=env, x=x_e_best, u=u_e)
        # And perform corresponding step for pursuer
        new_x_p = pursuer_transition(env=env, x_e=new_x_e, x_p=new_x_p)

        # And we add the node to the tree
        tree.add_node(parent_id=best_node.id, state=new_x_e, action_applied=u_e)

        # Third, we launch the simulation
        last_t = T_max
        for t in range(T_max):
            # Step, according to the default policy
            u_e = a[default_policy[new_x_e]]
            new_x_e, _ = transition_function(env=env, x=new_x_e, u=u_e)
            # Step, according to pursuer policy
            new_x_p = pursuer_transition(env=env, x_e=new_x_e, x_p=new_x_p)
            # Accumulate the inversed distance. We need it for reward computation
            if new_x_e != new_x_p:
                sum_inv_distance += 1. / np.linalg.norm(np.array(new_x_e) - np.array(new_x_p))
            # If we reached the goal or was eaten, then stop
            if (new_x_e == new_x_p) or (new_x_e == goal):
                last_t = t
                break

        rew = reward(x_e=new_x_e, x_p=new_x_p, sum_inv_distance=sum_inv_distance, goal=goal, t=last_t, T_max=T_max)

        # Forth, we update all parent's nodes
        tree.update_tree(node_id=-1, outcome=rew)

    u = max([(i, tree.nodes[i].value) for i in tree.nodes[0].children_ids])
    u = tree.nodes[u[0]].action_applied
    return u
