import matplotlib.pyplot as plt
from tf_agents.utils import common

from environment import Environment
from agent import new_agent
from helpers import compute_avg_return, collect_step
from observer import new_replay_buffer
from tf_agents.environments import batched_py_environment, tf_py_environment
from tf_agents.policies import random_tf_policy

env_name = "CartPole-v1"  # @param {type:"string"}
num_iterations = 15000  # @param {type:"integer"}
initial_collect_steps = 1000  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_capacity = 100000  # @param {type:"integer"}
fc_layer_params = (100,)
batch_size = 2  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}
num_atoms = 51  # @param {type:"integer"}
min_q_value = -20  # @param {type:"integer"}
max_q_value = 20  # @param {type:"integer"}
n_step_update = 2  # @param {type:"integer"}
num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

# Create environments
train_env = Environment()
eval_env = Environment()
# Convert to tf environments
train_env = tf_py_environment.TFPyEnvironment(train_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_env)
# reset
train_env.reset()
eval_env.reset()

# Create agent
tf_agent = new_agent(train_env, fc_layer_params, learning_rate)
tf_agent.initialize()


replay_buffer = new_replay_buffer(tf_agent.collect_data_spec, tf_agent, train_env, replay_buffer_capacity)
# print("compute_avg_return")
# compute_avg_return(eval_env, random_policy, num_eval_episodes) # Currently not working

# -- Data Collection --
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

for _ in range(initial_collect_steps):
    collect_step(train_env, random_policy, replay_buffer)
