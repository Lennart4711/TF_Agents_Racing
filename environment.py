import math
import numpy as np
import pygame
import ast

pygame.init()

from car import Car

from tf_agents.specs import array_spec
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts


class Environment(py_environment.PyEnvironment):
    def __init__(self, has_surface=True):
        # Constants
        self.accFactor = 0.15
        self.dimensions = (800, 600)
        self.laser_amount = 8
        self.discount = 0.98
        # Game variables
        self.car = Car(15, 15, 90, self.laser_amount)
        self.borders = load_borders("borders.txt")
        # Pygame
        if has_surface:
            self.win = pygame.display.set_mode(self.dimensions)

        # Environment variables
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(2,), dtype=np.float32, minimum=-1.0, maximum=1.0, name="action"
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.laser_amount,),
            dtype=np.float32,
            minimum=0.0,
            maximum=600.0,
            name="observation",
        )

        self._episode_ended = False
        self.last_action = None

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
        self.last_action = action
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
            reward = self.car.driven_distance
            return ts.transition(
                observation=lasers, reward=reward, discount=self.discount
            )

    def calc_lasers(self):
        lasers = [
            math.dist((self.car.xPos, self.car.yPos), (x[1][0], x[1][1]))
            for x in self.car.lasers
        ]
        return np.array(lasers, dtype=np.float32)

    def render(self, mode="human", telemetry: bool=False):
        self.win.fill((0, 0, 0))
        self.car.draw(self.win)
        for x in self.borders:
            pygame.draw.line(self.win, (255, 255, 255), x[0], x[1])

        if telemetry:
            # Draw a text box with the last action
            font = pygame.font.SysFont("comicsans", 30)
            text = font.render(
                f"Last action: {self.last_action}", 1, (255, 255, 255)
            )
            self.win.blit(text, (10, 10))


        pygame.display.update()
        # throw away the events
        pygame.event.get()

    def close(self):
        pygame.quit()

    
def load_borders(file_name):
    """Load the list of points and convert to numpy array"""
    with open(file_name, "r") as file:
        # Read the file and convert to list of tuples
        out = ast.literal_eval(file.read())
        # Convert to numpy array
        return np.array(out)