from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from envs.helper import *
from envs.car import Car

import numpy as np
import pygame

pygame.init()
from tf_agents.specs import array_spec
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts


class Environment(py_environment.PyEnvironment):
    def __init__(self):
        # Constants
        self.accFactor = 0.15
        self.dimensions = (800, 600)
        self.laser_amount = 6
        self.discount = 0.98
        # Game variables
        self.car = Car(15, 15, 90, self.laser_amount)
        self.borders = load_boarders("borders.txt")
        # Pygame
        self.win = pygame.display.set_mode(self.dimensions)

        # Environment variables
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(2,), dtype=np.float32, minimum=0, maximum=1.0, name="action"
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.laser_amount,),
            dtype=np.float32,
            minimum=0.0,
            maximum=600.0,
            name="observation",
        )

        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode_ended = False
        self.car = Car(15, 15, 90, self.laser_amount)
        self.car.set_lasers()
        self.car.set_laser_length(self.borders)

        lasers = self.calc_lasers()

        return ts.restart(observation=lasers)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        self.car.move(action[0], action[1])
        self.car.update()
        self.car.set_lasers()
        self.car.set_laser_length(self.borders)
        # self.render()
        # self._episode_ended = True # Reward after every action
        # make true, if the agent crashes or reaches the goal
        lasers = self.calc_lasers()

        if self.car.crashed():
            reward = -200
            self._episode_ended = True
            return ts.termination(observation=lasers, reward=reward)
        else:
            reward = math.sqrt(self.car.driven_distance)
            return ts.transition(
                observation=lasers, reward=reward, discount=self.discount
            )

    def calc_lasers(self):
        lasers = [
            math.dist((self.car.xPos, self.car.yPos), (x[1][0], x[1][1]))
            for x in self.car.lasers
        ]
        return np.array(lasers, dtype=np.float32)

    def render(self, mode="human"):
        self.win.fill((0, 0, 0))
        self.car.draw(self.win)
        for x in self.borders:
            pygame.draw.line(self.win, (255, 255, 255), x[0], x[1])
        pygame.display.update()
        # throw away the events
        pygame.event.get()

    def close(self):
        pygame.quit()
