import gym
from gym import spaces
from car import Car
from helper import load_boarders
import numpy as np
import math


class RacingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.laser_amount = 6
        self.borders = load_boarders("borders.txt")
        self.car = Car(15, 15, 90, self.laser_amount)
        self.car.set_lasers()
        self.car.set_laser_length(self.borders)

        # A list of 6 float values between 0 and 600
        self.observation_space = spaces.Box(
            low=0, high=600, shape=(6,), dtype=np.float32
        )

        # Allow two actions: steering and speed
        self.action_space = spaces.Box(
            low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32
        )

    def _get_obs(self):
        lasers = [
            math.dist((self.car.xPos, self.car.yPos), (x[1][0], x[1][1]))
            for x in self.car.lasers
        ]
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

    def close(self):
        pass


env = RacingEnv()
env.reset()
print(env.observation_space.sample(), type(env.observation_space.sample()))
print(env._get_obs(), type(env._get_obs()))
