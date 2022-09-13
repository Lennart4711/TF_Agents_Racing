import gym
from gym import spaces
from car import Car
from helper import load_boarders
import numpy as np
import math


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.laser_amount = 6
        self.borders = load_boarders("borders.txt")
        self.car = Car(15, 15, 90, self.laser_amount)
        self.car.set_lasers()

        # A list of 6 float values between 0 and 600
        self.observation_space = spaces.Box(
            low=0, high=600, shape=(6,), dtype=np.float32
        )

        # Allow two actions: steering and speed
        self.action_space = spaces.Box(
            low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32
        )


    # def _get_obs(self):
    #     return {"agent": self._agent_location, "target": self._target_location}
    # def _get_info(self):
    #     return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.car = Car(15, 15, 90, self.laser_amount)
        self.car.set_lasers()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def close(self):
        pass


env = GridWorldEnv()
    