# from epgame_v2 import EPGame,save_results

import gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from gym.envs.classic_control import rendering
from gym.utils import seeding


def compute_rgb_array(env, evader_policy, pursuer_policy, max_iters):
    rgb_array = []
    done = False
    obs = env.reset()
    i = 0
    while (not done) and i < max_iters // 2:
        evader_action_index = evader_policy(obs)  # in [0,1,2,3]
        obs, reward, done, info = env.evader_step(evader_action_index)
        pursuer_action_index = pursuer_policy(obs)  # in [0,1,2,3]
        obs, reward, done, info = env.pursuer_step(pursuer_action_index)
        rgb_array.append(env.render("rgb_array"))
        i += 1
    return rgb_array


def save_results(env, evader_policy, pursuer_policy, max_iters=100, name="policy_example"):
    """
    Usage:
    evader_policy: array of size [x_max,y_max,x_max,y_max] with int values form 0 to 3 (inclusive);
    pursuer_policy: array of size [x_max,y_max,x_max,y_max] with int values form 0 to 3 (inclusive).
    
    """

    im_array = compute_rgb_array(env, evader_policy, pursuer_policy, max_iters)
    fig = plt.figure(figsize=(30, 30))
    imgs = []
    for im in im_array:
        plot = plt.imshow(im)
        imgs.append([plot])

    ani = animation.ArtistAnimation(fig, imgs, interval=50, blit=True)

    ani.save(name + '.mp4')

    # plt.show()


def default_reward(x_e, x_p, goal):
    x_e = np.array(x_e)
    x_p = np.array(x_p)
    goal = np.array(goal)
    if tuple(goal) != (None, None):
        r = -np.linalg.norm(x_e - goal) + np.linalg.norm(x_e - x_p)
    else:
        r = -np.linalg.norm(x_e - goal)
    if tuple(x_e) == tuple(goal):
        r += 100
    return r


class EPGame(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, env_map, reward=None, use_goal=False, seed_num=200):
        self.done = False
        self.use_goal = use_goal

        self.seed_num = seed_num
        self.reward = reward if reward != None else default_reward

        self.env_map = np.array(env_map)
        self._total_actions = [(-1, 0), (0, -1), (1, 0), (0, 1)]

        x_max, y_max = self.env_map.shape

        self.evader_action_space = spaces.MultiDiscrete(len(self._total_actions))
        self.evader_observation_space = spaces.MultiDiscrete([x_max, y_max, x_max, y_max])

        self.pursuer_action_space = spaces.MultiDiscrete(len(self._total_actions))
        self.pursuer_observation_space = spaces.MultiDiscrete([x_max, y_max, x_max, y_max])

        self.seed()

        self.viewer = None
        self.reset()

    def update_allowed_actions(self, evador_actions):
        self.allowed_actions = []
        if evador_actions:
            for a_index, a in enumerate(self._total_actions):
                x_e = tuple(self.transition(self.x_e, a))
                x_p = tuple(self.x_p)
                joint_state = x_e + x_p + (1,)  # 1 = pursuer state
                if self.visited_states[joint_state] == False:
                    self.allowed_actions.append(a_index)
        else:
            for a_index, a in enumerate(self._total_actions):
                x_p = tuple(self.transition(self.x_p, a))
                x_e = tuple(self.x_e)
                joint_state = x_e + x_p + (0,)  # 1 = evador state
                if self.visited_states[joint_state] == False:
                    self.allowed_actions.append(a_index)

    def evader_step(self, action_index):
        assert action_index in self.allowed_actions, "Action is not allowed"
        assert self.evaders_turn == True, "It is the pursuer turn to make step"
        action = self._total_actions[action_index]
        self.x_e = self.transition(self.x_e, action)
        self.state = tuple(np.r_[self.x_e, self.x_p])
        self.visited_states[self.state + (1,)] = True  # 1 = pursuer state
        reward = self.compute_reward()
        done = (tuple(self.x_e) == tuple(self.x_p)) or (tuple(self.x_e) == tuple(self.goal))
        self.evaders_turn = False
        self.pursuers_turn = True
        self.update_allowed_actions(evador_actions=False)
        self.t += 1

        info = None
        if done:
            info = "Evader wins" if tuple(self.x_e) == tuple(self.goal) else "Pursuer wins"

        return self.state, reward, done, info

    def pursuer_step(self, action_index):

        assert action_index in self.allowed_actions, "Action is not allowed"

        assert self.pursuers_turn == True, "It is the evader turn to make step"
        action = self._total_actions[action_index]
        self.x_p = self.transition(self.x_p, action)
        self.state = tuple(np.r_[self.x_e, self.x_p])
        self.visited_states[self.state + (0,)] = True  # 0 = evader state
        reward = self.compute_reward()
        done = (tuple(self.x_e) == tuple(self.x_p)) or (tuple(self.x_e) == tuple(self.goal))
        self.evaders_turn = True
        self.pursuers_turn = False
        self.update_allowed_actions(evador_actions=True)
        self.t += 1

        info = None
        if done:
            info = "Evader wins" if tuple(self.x_e) == tuple(self.goal) else "Pursuer wins"
        return self.state, reward, done, info

    def transition(self, x, u):
        """Transition function for states in this problem
        x: current state, this is a tuple (i,j)
        u: current action, this is a tuple (i,j)
        env: enviroment

        Output:
        new state
        True if correctly propagated
        False if this action can't be executed
        """
        xnew = np.array(x) + np.array(u)
        # print('xnew',xnew)
        if self.state_is_consistent(xnew):
            return xnew
        return x

    def state_is_consistent(self, x):
        x = tuple(x)
        """Checks wether or not the proposed state is a valid state, i.e. is in colision or our of bounds"""
        # check for collision
        if x[0] < 0 or x[1] < 0 or x[0] >= self.env_map.shape[0] or x[1] >= self.env_map.shape[1]:
            # print('out of bonds')
            return False
        if self.env_map[x] >= 1.0 - 1e-4:
            # print('Obstacle')
            return False
        if tuple(self.x_e) == tuple(self.x_p):
            # print('??aught')
            return False
        if tuple(self.x_e) == tuple(self.goal):
            # print('Goal')
            return False
        return True

    def compute_reward(self):
        return self.reward(self.x_e, self.x_p, self.goal)

    def reset(self):
        self.seed(self.seed_num)

        self.evaders_turn = True
        self.pursuers_turn = False
        self.visited_states = np.full(np.r_[self.env_map.shape, self.env_map.shape, 2], False)
        self.t = 0
        x1_max, x2_max = self.env_map.shape
        free_space = np.argwhere(self.env_map == 0)
        x_e_index, x_p_index, goal_index = self.np_random.choice(np.arange(0, len(free_space)), 3, replace=False)
        self.x_e, self.x_p, self.goal = free_space[x_e_index], free_space[x_p_index], free_space[goal_index]
        if not self.use_goal:
            self.goal = (None, None)

        self.state = tuple(np.r_[self.x_e, self.x_p])
        self.visited_states[self.state + (0,)] = True  # 0 = evader state
        self.update_allowed_actions(evador_actions=True)
        return self.state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        map_with_bounds = np.empty((self.env_map.shape[0] + 2, self.env_map.shape[1] + 2))
        map_with_bounds[0, :] = map_with_bounds[-1, :] = map_with_bounds[:, 0] = map_with_bounds[:, -1] = 1
        map_with_bounds[1:-1, 1:-1] = self.env_map

        scale = 20
        screen_width, screen_height = (np.array(map_with_bounds.shape) - 1) * scale
        dx = dy = 1 / 2 * scale
        agent_size = scale

        if self.viewer is None:

            self.viewer = rendering.Viewer(screen_width, screen_height)
            walls = np.argwhere(map_with_bounds > 0) * scale

            evader = rendering.make_circle(agent_size / 2)
            evader.set_color(0, 0, 255)
            self.evadertrans = rendering.Transform()
            evader.add_attr(self.evadertrans)
            self.viewer.add_geom(evader)

            pursuer = rendering.make_circle(agent_size / 2)
            pursuer.set_color(255, 0, 0)
            self.pursuertrans = rendering.Transform()
            pursuer.add_attr(self.pursuertrans)
            self.viewer.add_geom(pursuer)

            goal = rendering.make_circle(agent_size / 2)
            goal.set_color(0, 255, 0)
            self.goaltrans = rendering.Transform()
            goal.add_attr(self.goaltrans)
            self.viewer.add_geom(goal)

            for x, y in walls:
                brick = rendering.FilledPolygon(
                    [(x - dx, y - dy), (x - dx, y + dy), (x + dx, y + dy), (x + dx, y - dy)])
                self.viewer.add_geom(brick)

        x_e = (self.x_e + 1) * scale
        x_p = (self.x_p + 1) * scale
        if self.use_goal:
            goal = (self.goal + 1) * scale
            self.goaltrans.set_translation(*goal)

        self.evadertrans.set_translation(*x_e)
        self.pursuertrans.set_translation(*x_p)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
