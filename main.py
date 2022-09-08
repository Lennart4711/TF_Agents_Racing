import matplotlib.pyplot as plt
from tf_agents.utils import common

from environment import Environment
from agent import new_agent
from helpers import compute_avg_return, collect_episode
from observer import new_replay_buffer, new_reverb_observer


env_name = "RaceAI"  # @param {type:"string"}
num_iterations = 250  # @param {type:"integer"}
collect_episodes_per_iteration = 2  # @param {type:"integer"}
replay_buffer_capacity = 2000  # @param {type:"integer"}
fc_layer_params = (100,)
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 25  # @param {type:"integer"}
num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 50  # @param {type:"integer"}

# Train environment
train_env = Environment()
train_env.reset()
# Eval environment
eval_env = Environment()
eval_env.reset()

# Create agent
tf_agent = new_agent(train_env, fc_layer_params, learning_rate)
tf_agent.initialize()

# Replay buffer
replay_buffer = new_replay_buffer(tf_agent.collect_data_spec, replay_buffer_capacity)
rb_observer = new_reverb_observer(
    tf_agent.collect_data_spec, replay_buffer_capacity, replay_buffer
)

# Training the agent
# (Optional) Optimize by wrapping some of the code in a graph using TF function.
tf_agent.train = common.function(tf_agent.train)

# Reset the train step
tf_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):

    # Collect a few episodes using collect_policy and save to the replay buffer.
    collect_episode(train_env, tf_agent.collect_policy, collect_episodes_per_iteration)

    # Use data from the buffer and update the agent's network.
    iterator = iter(replay_buffer.as_dataset(sample_batch_size=1))
    trajectories, _ = next(iterator)
    train_loss = tf_agent.train(experience=trajectories)

    replay_buffer.clear()

    step = tf_agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print("step = {0}: loss = {1}".format(step, train_loss.loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
        print("step = {0}: Average Return = {1}".format(step, avg_return))
        returns.append(avg_return)


# Visualize the results
steps = range(0, num_iterations + 1, eval_interval)
plt.plot(steps, returns)
plt.ylabel("Average Return")
plt.xlabel("Step")
plt.ylim(top=250)
