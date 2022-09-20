import gym
import pygame
import numpy as np
import math
from gym import spaces
from envs.car import Car
from envs.helper import load_boarders


class RacingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.laser_amount = 6
        self.borders = load_boarders("borders.txt")
        self.car = Car(15, 15, 90, self.laser_amount)
        self.car.set_lasers()
        self.car.set_laser_length(self.borders)

        self.win = pygame.display.set_mode((600, 600))

        # A list of 6 float values between 0 and 600
        self.observation_space = spaces.Box(
            low=0, high=600, shape=(6,), dtype=np.float32
        )
        # self.observation_space = spaces.Dict(
        #     {
        #         "laser": spaces.Box(
        #             low=0, high=600, shape=(self.laser_amount,), dtype=np.float32
        #         ),
        #         "angle": spaces.Box(low=0, high=360, shape=(1,), dtype=np.float32),
        #     }
        # )

        # Allow two actions: steering and speed
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def _get_obs(self):
        lasers = [
            math.dist((self.car.xPos, self.car.yPos), (x[1][0], x[1][1]))
            for x in self.car.lasers
        ]

        # return {"laser": np.array(lasers), "angle": np.asarray(self.car.angle)}

        return np.array(lasers)

    def _get_info(self):
        return {"Carpos": [self.car.xPos, self.car.yPos], "Carangle": self.car.angle}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.car = Car(15, 15, 90, self.laser_amount)
        self.car.set_lasers()
        self.car.set_laser_length(self.borders)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        self.car.move(action[0], action[1])
        self.car.update()
        self.car.set_lasers()
        self.car.set_laser_length(self.borders)

        # An episode is done iff the agent crashed
        terminated = self.car.crashed()
        reward = -10 if terminated else 0.1
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def render(self, mode="human"):
        self.win.fill((80, 100, 100))
        self.car.draw(self.win)
        for border in self.borders:
            pygame.draw.line(self.win, (255, 255, 255), border[0], border[1], 2)
        pygame.display.update()

    def close(self):
        pass
