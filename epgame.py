import gym
from gym import spaces
import numpy as np
from gym.utils import seeding
from gym.envs.classic_control import rendering


class EPGame(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    
    
    def __init__(self,env_map):
        self.done = False
        
        
        self.env_map = np.array(env_map)
        self.actions = [(-1,0),(0,-1),(1,0),(0,1)]

        x_max,y_max = self.env_map.shape
        
        self.evader_action_space = spaces.MultiDiscrete(self.env_map.shape)
        self.evader_observation_space = spaces.MultiDiscrete([x_max,y_max,x_max,y_max])
        
        self.pursuer_action_space = spaces.MultiDiscrete(self.env_map.shape)
        self.pursuer_observation_space = spaces.MultiDiscrete([x_max,y_max,x_max,y_max])
        
        self.seed()
        
        self.viewer = None
        self.reset()
                
    def evader_step(self,action_index):
        
        assert self.evaders_turn == True, "It is the pursuer turn to make step"
        action = self.actions[action_index]
        self.x_e = self.transition(self.x_e,action)
        self.state = np.r_[self.x_e,self.x_p]
        reward = self.compute_reward()
        done = (self.x_e == self.x_p)
        self.evaders_turn = False
        self.pursuers_turn = True
        return self.state, reward, done,{}
    
    def pursuer_step(self,action_index):
        
        assert self.pursuers_turn == True, "It is the evader turn to make step"
        action = self.actions[action_index]
        self.x_p = self.transition(self.x_p,action)
        self.state = np.r_[self.x_e,self.x_p]
        reward = self.compute_reward()
        done = (self.x_e == self.x_p)
        self.evaders_turn = True
        self.pursuers_turn = False
        return self.state, reward, done,{}

    def transition(self,x,u):
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
        #print('xnew',xnew)
        if self.state_is_consistent(xnew):
            return xnew
        return x

    def state_is_consistent(self,x):
        x = tuple(x)
        """Checks wether or not the proposed state is a valid state, i.e. is in colision or our of bounds"""
        # check for collision
        if x[0] < 0 or x[1] < 0 or x[0] >= self.env_map.shape[0] or x[1] >= self.env_map.shape[1] :
            #print('out of bonds')
            return False
        if self.env_map[x] >= 1.0-1e-4:
            #print('Obstacle')
            return False
        if tuple(self.x_e) == tuple(self.x_p):
            #print('Сaught')
            return False
        return True
        
        
    def compute_reward(self):
        r = 1 if tuple(self.x_e) != tuple(self.x_p) else 0
        return r
        
    def reset(self):
        
        self.evaders_turn = True
        self.pursuers_turn = True
        
        x1_max,x2_max = self.env_map.shape
        while True:
            self.x_e = np.random.randint([0,0],[x1_max,x2_max])
            self.x_p = np.random.randint([0,0],[x1_max,x2_max])
            
            if self.state_is_consistent(self.x_p) and self.state_is_consistent(self.x_p):
                break
        
        self.state = np.r_[self.x_e,self.x_p]
        
        return self.state
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    

    def render(self, mode='human'):
        map_with_bounds = np.empty((self.env_map.shape[0]+2,self.env_map.shape[1]+2))
        map_with_bounds[0,:] = map_with_bounds[-1,:] = map_with_bounds[:,0] = map_with_bounds[:,-1] = 1
        map_with_bounds[1:-1,1:-1] = self.env_map
        
        scale = 20
        screen_width,screen_height = (np.array(map_with_bounds.shape)-1) * scale
        dx=dy= 1/2*scale
        agent_size = scale
        


        if self.viewer is None:
            
            self.viewer = rendering.Viewer(screen_width, screen_height)
            walls = np.argwhere(map_with_bounds >0)*scale
            

            evader = rendering.make_circle(agent_size/2)
            evader.set_color(0, 0, 255)
            self.evadertrans = rendering.Transform()
            evader.add_attr(self.evadertrans)
            self.viewer.add_geom(evader)
            
            pursuer = rendering.make_circle(agent_size/2)
            pursuer.set_color(255, 0, 0)
            self.pursuertrans = rendering.Transform()
            pursuer.add_attr(self.pursuertrans)
            self.viewer.add_geom(pursuer)

            for x,y in walls:
                brick= rendering.FilledPolygon([(x-dx, y-dy),(x-dx, y+dy),(x+dx, y+dy), (x+dx,y-dy)])
                self.viewer.add_geom(brick)

        x_e = (self.x_e+1)*scale
        x_p = (self.x_p+1)*scale
        self.evadertrans.set_translation(*x_e)
        self.pursuertrans.set_translation(*x_p)

        

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

