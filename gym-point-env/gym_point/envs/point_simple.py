#!/usr/bin/python

import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding

import numpy

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

class PointSimpleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    reward_range = {0, 1}
    observation_range = {-1, 1}
    action_range = {-1, 1}
    spec = None

    dimensions = 2
    # action_space = spaces.Box(
    #     low=min(action_range), high=max(action_range), shape=(dimensions,), dtype=numpy.float32)
    action_space = spaces.Discrete(4)
    observation_space = spaces.Box(
        low=min(observation_range), high=max(observation_range), shape=(dimensions,), dtype=numpy.float32)

    def __init__(self):

        super(PointSimpleEnv, self).__init__()
        self.max_speed = 0.05
        self.goal_margin = 0.2

        # Fixed goal to 0 in all dimensions. For variable goal use
        # PointGoalEnv.
        self.goal = numpy.array([0.] * self.dimensions)
        self.max_position = 1
        self.min_position = -1
        self.viewer = None
        self.traj = numpy.array([[0.0, 0.0]], dtype=numpy.float32)
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

    def compute_reward(self, achieved_goal, desired_goal=None, info=None):
        if desired_goal is None:
            desired_goal = self.goal
        desired_goal = 0.5*numpy.sin(-6.28*achieved_goal[0])
        distance = numpy.linalg.norm(achieved_goal[1] - desired_goal)

        done = distance < self.goal_margin
        reward = done * 1.0
        # reward = 1 - distance
        # print(distance, reward)
        return reward, distance

    def step(self, action):
        # print("env action", action)
        # self.state += action * self.max_speed
        if action == LEFT:
            self.state[0] -= self.max_speed
        elif action == RIGHT:
            self.state[0] += self.max_speed
        elif action == UP:
            self.state[1] += self.max_speed
        else:
            self.state[1] -= self.max_speed

        self.state = numpy.clip(self.state, min(
            self.observation_range), max(self.observation_range))

        reached_end = numpy.linalg.norm(self.state - numpy.array([1,0])) < self.goal_margin
        reward, distance = self.compute_reward(self.state, self.goal, {})
        done = True if (distance >= self.goal_margin or reached_end) else False

        self.traj = numpy.concatenate((self.traj, numpy.expand_dims(self.state, axis=0)),axis=0)
        return self.state, reward, done, {}

    def reset(self):
        # self.state = self.observation_space.sample()
        self.state = numpy.array([-0.5, 0.0], dtype=numpy.float32)
        self.traj = numpy.array([[0.0, 0.0]], dtype=numpy.float32)
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
            goal = [self.goal[0], 0]
        else:
            point = self.state
            goal = self.goal

        self.pointtrans.set_translation(
            (point[0] - self.min_position) * scale, (point[1] - self.min_position) * scale)
        # self.trajtrans.set_translation(
        #     (0 - self.min_position) * scale, (0 - self.min_position) * scale)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
