from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from helper import *
from car import Car

import numpy as np
import pygame

pygame.init()

import numpy as np

from tf_agents.specs import array_spec
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import tensor_spec


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
            shape=(1, 6), dtype=np.float32, minimum=0, name="observation"
        )

        self._episode_ended = False

    def action_spec(self):
        return tensor_spec.from_spec(self._action_spec)
        # return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        print("reset start")

        self._episode_ended = False
        self.car = Car(15, 15, 90, self.laser_amount)
        self.car.set_lasers()
        self.car.set_laser_length(self.borders)

        # lasers = [x.length() for x in self.car.lasers]
        lasers = [
            math.dist((self.car.xPos, self.car.yPos), (x[1][0], x[1][1]))
            for x in self.car.lasers
        ]
        # convert lasers to numpy array
        lasers = np.array(lasers, dtype=np.float32)
        print("reset last")
        return ts.restart(observation=lasers)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        self.car.move(action[0], action[1])
        self.car.update()
        self.car.set_lasers()
        self.car.set_laser_length(self.borders)

        # self._episode_ended = True # Reward after every action
        # make true, if the agent crashes or reaches the goal

        # calulate reward
        reward = 0
        if self.car.crashed():
            reward = -10
            self._episode_ended = True
        else:
            reward = 0.1

        # lasers = [x.length() for x in self.car.lasers]
        lasers = [
            math.dist((self.car.xPos, self.car.yPos), (x[1][0], x[1][1]))
            for x in self.car.lasers
        ]
        # convert lasers to numpy array
        lasers = np.array(lasers)
        print("step last")
        return ts.transition(observation=lasers, reward=reward)

    def render(self):
        self.win.fill((0, 0, 0))
        self.car.draw_car(self.win)
        for x in self.borders:
            pygame.draw.line(self.win, (255, 255, 255), x[0], x[1])
        pygame.display.update()
