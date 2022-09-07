from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from helper import *
from car import Car
from vector import Vector
import pprint

import tensorflow as tf
import numpy as np
import pygame

pygame.init()

import numpy as np

from tf_agents.specs import array_spec
from tf_agents.environments import utils
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.networks import sequential
from tf_agents.specs import tensor_spec
from tf_agents.agents.ppo import ppo_agent
from tf_agents.utils import common


class Environment(py_environment.PyEnvironment):
    def __init__(self):
        # Constants
        self.accFactor = 0.15
        self.dimensions = (800, 600)
        self.laser_amount = 6
        self.discount = 1.0
        # Game variables
        self.car = Car(15, 15, 90, self.laser_amount)
        self.borders = load_boarders("borders.txt")
        # Pygame
        self.win = pygame.display.set_mode(self.dimensions)

        # Environment variables
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(2,), dtype=np.float32, minimum=-1.0, maximum=1.0, name="action"
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1,6), dtype=np.float32, minimum=0, name="observation"
        )
        self._episode_ended = False

    def action_spec(self):
        print("Action spec")
        print(self._action_spec)
        return tensor_spec.from_spec(self._action_spec)
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode_ended = False
        self.car = Car(15, 15, 90, self.laser_amount)
        self.car.set_lasers()
        self.set_laser_length(self.car)

        lasers = [x.length() for x in self.car.lasers]
        # convert lasers to numpy array
        lasers = np.array(lasers)

        return ts.transition(
            observation=lasers, reward=0.0
        )

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        self.car.move(action[0], action[1])
        self.car.update()
        self.car.set_lasers()
        self.set_laser_length(self.car)
            
        # self._episode_ended = True # Reward after every action
        # make true, if the agent crashes or reaches the goal

        # calulate reward
        reward = 0
        if self.car.crashed():
            reward = -10
            self._episode_ended = True
        else:
            reward = 0.1
        
        lasers = [x.length() for x in self.car.lasers]
        # convert lasers to numpy array
        lasers = np.array(lasers)
        return ts.transition(
            observation=lasers, reward=reward
        )

    def set_laser_length(self, car):
        for i, x in enumerate(car.lasers):
            for y in self.borders:
                v = self.get_collision_point(x, y)
                if v is not None:
                    x.nX = v[0]
                    x.nY = v[1]
            d = math.dist((car.xPos, car.yPos), (x.nX, x.nY))
            # car.lengths[i] = min(d, 200)


    def get_collision_point(self, a, b):
        x1 = a.xPos
        x2 = a.nX
        y1 = a.yPos
        y2 = a.nY
        x3 = b[0][0]
        x4 = b[1][0]
        y3 = b[0][1]
        y4 = b[1][1]
        denominator = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        # Line segments are parallel
        if denominator == 0:
            return None

        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator

        if ua >= 0 and ua <= 1 and ub >= 0 and ub <= 1:
            intersectionX = x1 + (ua * (x2 - x1))
            intersectionY = y1 + (ua * (y2 - y1))
            return [intersectionX, intersectionY]
        else:
            return None


def render(self):
    self.win.fill((0, 0, 0))
    self.car.draw_car(self.win)
    for x in self.borders:
        pygame.draw.line(self.win, (255, 255, 255), x[0], x[1])
    pygame.display.update()



env = Environment()
environment = tf_py_environment.TFPyEnvironment(env)

env_name = "CartPole-v0" # @param {type:"string"}
num_iterations = 250 # @param {type:"integer"}
collect_episodes_per_iteration = 2 # @param {type:"integer"}
replay_buffer_capacity = 2000 # @param {type:"integer"}

fc_layer_params = (100,)

learning_rate = 1e-3 # @param {type:"number"}
log_interval = 25 # @param {type:"integer"}
num_eval_episodes = 10 # @param {type:"integer"}
eval_interval = 50 # @param {type:"integer"}

env.reset()

from tf_agents.networks import actor_distribution_network
actor_net = actor_distribution_network.ActorDistributionNetwork(
    env.observation_spec(),
    env.action_spec(),
    fc_layer_params=fc_layer_params)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

from tf_agents.agents.reinforce import reinforce_agent
tf_agent = reinforce_agent.ReinforceAgent(
    env.time_step_spec(),
    env.action_spec(),
    actor_network=actor_net,
    optimizer=optimizer,
    normalize_returns=True,
    train_step_counter=train_step_counter)
tf_agent.initialize()
print("Sucessfully initialized agent")
