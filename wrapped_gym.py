from envs.gym_env import RacingEnv
import gym
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
import numpy as np
from tf_agents.environments import validate_py_environment
from tf_agents.environments import tf_py_environment


# env = RacingEnv()
# print(env._get_obs())
# print(env.observation_space.sample())
# observation, info = env.reset(seed=42)
# env = suite_gym.wrap_env(env)
# env = tf_py_environment.TFPyEnvironment(env)
# validate_py_environment(env, episodes=5)


env = suite_gym.load("Racing-v0")
print("action_spec:", env.action_spec())
print("time_step_spec.observation:", env.time_step_spec().observation)
print("time_step_spec.step_type:", env.time_step_spec().step_type)
print("time_step_spec.discount:", env.time_step_spec().discount)
print("time_step_spec.reward:", env.time_step_spec().reward)

for _ in range(10000):
    action = env.action_space.sample()
    print(action)
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        observation, info = env.reset()
        print("Crashed")
env.close()
