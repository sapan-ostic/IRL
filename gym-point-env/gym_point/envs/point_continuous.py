#!/usr/bin/python

import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding
import math

import numpy

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

class PointContinuousEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    reward_range = {0, 1}

    low = numpy.array([0, -1, -10000000, -10000000],dtype=numpy.float32)
    high = numpy.array([1.2, 1, 10000000, 10000000],dtype=numpy.float32)
    action_range = [-1, 0, 1]
    observation_range = {-1, 1}
    spec = None

    dimensions = 4
    action_space = spaces.Discrete(9)
    observation_space = spaces.Box(low, high, shape=(dimensions,), dtype=numpy.float32)

    def __init__(self):

        super(PointContinuousEnv, self).__init__()
        self.a_max = 100 # 100
        self.timer = 0
        self.dt = 0.01
        self.goal_margin = 0.2
        self.Tspan = 1

        self.max_position = 1
        self.min_position = -1
        self.viewer = None
        self.state = numpy.array([0.0, 0.0, 0, 0], dtype=numpy.float32)
        self.traj = self.state.reshape(1,4)
        self.done = False

        if self.dimensions <= 0:
            logger.error('The dimensions have to be at least 1')
        elif self.dimensions > 2:
            logger.warn(
                'The dimensions are bigger than 2, only the first 2 dimensions are visualized')

        self.seed()

    def seed(self, given_seed=None):
        self.numpy_random, seed = seeding.np_random(given_seed)

        # TODO: The seed should be passed from seeding
        # gym.spaces.np_random.seed(given_seed)
        return [seed]

    def compute_reward(self):
        x_des = self.timer
        y_des = -0.5*numpy.sin(-6.28*self.timer)
        
        distance = numpy.linalg.norm(numpy.array([x_des, y_des]) - self.state[:2])
        # print(numpy.array([x_des, y_des]))
        # print(numpy.array(self.state[:2]))
        # print(distance)
        out_of_margin = distance > self.goal_margin # If agent is outside the margin 
        # reward = self.timer
        reward = -1 if out_of_margin else 1
        return reward, out_of_margin

    def step(self, action):
        s = self.state
        dt = self.dt
        
        # ax = self.action_range[action[0]]
        # ay = self.action_range[action[1]]

        
        a_max = self.a_max

        if action == 0:
            ax = -1
            ay = -1
        elif action == 1:
            ax = -1
            ay = 0
        elif action == 2:
            ax = -1
            ay = 1
        elif action == 3:
            ax = 0
            ay = -1
        elif action == 4:
            ax = 0
            ay = 0     
        elif action == 5:
            ax = 0
            ay = 1
        elif action == 6:
            ax = 1
            ay = -1
        elif action == 7:
            ax = 1
            ay = 0
        elif action == 8:
            ax = 1
            ay = 1

        self.state[0] = s[0]+s[2]*dt+0.5*ax*dt*dt*a_max
        self.state[1] = s[1]+s[3]*dt+0.5*ay*dt*dt*a_max
        self.state[2] = s[2] + ax*dt*a_max
        self.state[3] = s[3] + ay*dt*a_max
        
        self.state[:2] = numpy.clip(self.state[:2], min(self.observation_range), 
            max(self.observation_range))

        self.timer += dt
        reached_end = (self.timer >= self.Tspan)
        
        reward, out_of_margin = self.compute_reward()
        # done = True if (reached_end or out_of_margin) else False
        done = True if (reached_end) else False
        info = 'all good'
        if reached_end:
            info = 'reached end'
        else:
            info = 'out_of_margin'
            
        self.traj = numpy.concatenate((self.traj, numpy.expand_dims(self.state, axis=0)),axis=0)
        return self.state, reward, done, info

    def reset(self):
        # self.state = self.observation_space.sample()
        # print(self.state[0])
        self.state = numpy.array([0.0, 0.0, 0.0, 0.0], dtype=numpy.float32)
        self.traj = self.state.reshape(1,4)
        self.timer = 0 #self.state[0]
        return self.state

    def render(self, mode='human', close=False):
        screen_width = 400
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width

        if self.viewer is None:
            # Borrowing rendering from classic control
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            self.pointtrans = rendering.Transform()
            point = rendering.make_circle(5)
            point.add_attr(self.pointtrans)
            point.set_color(.5, .5, .5)
            self.viewer.add_geom(point)

            # self.goaltrans = rendering.Transform()
            # goal = rendering.make_circle(self.goal_margin * scale)
            # goal.add_attr(self.goaltrans)
            # goal.set_color(0, 1, 0)
            # self.viewer.add_geom(goal)

            xs = self.traj[:,0]
            ys = self.traj[:,1]
            self.xys = list(zip((xs-self.min_position)*scale, ys*scale))
            self.track = rendering.make_polyline(self.xys)
            self.track.set_linewidth(4)
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.dimensions == 1:
            # If the environment is only one dimensional, add a dimension which
            # is 0 for rendering
            point = [self.state[0], 0]
            # goal = [self.goal[0], 0]
        else:
            point = self.state
            # goal = self.goal

        self.pointtrans.set_translation(
            (point[0] - self.min_position) * scale, (point[1] - self.min_position) * scale)
        # self.trajtrans.set_translation(
        #     (0 - self.min_position) * scale, (0 - self.min_position) * scale)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
