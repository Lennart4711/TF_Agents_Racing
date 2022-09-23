import matplotlib.pyplot as plt
from environment import Environment
from helpers import compute_avg_return
from observer import new_replay_buffer
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.trajectories import trajectory
from tf_agents.agents.reinforce import reinforce_agent
import tensorflow as tf


num_iterations = 15000  # @param {type:"integer"}
initial_collect_steps = 1000  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_capacity = 100000  # @param {type:"integer"}
fc_layer_params = (200, 100)
batch_size = 2  # @param {type:"integer"}
learning_rate = .2  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}
num_atoms = 51  # @param {type:"integer"}
min_q_value = -20  # @param {type:"integer"}
max_q_value = 20  # @param {type:"integer"}
n_step_update = 2  # @param {type:"integer"}
num_eval_episodes = 5  # @param {type:"integer"}
eval_interval = 50  # @param {type:"integer"}
save_interval = 50

# Create environments
train_env = Environment()
eval_env = Environment()
# Convert to tf environments
train_env = tf_py_environment.TFPyEnvironment(train_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_env)

# # Create agent
# agent = new_agent(train_env, fc_layer_params, learning_rate)
# agent.initialize()
actor_net = actor_distribution_network.ActorDistributionNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params,
)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
train_step_counter = tf.Variable(0)
pre_train_checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=actor_net)
checkpoint_dir = 'tmp/pre_train_checkpoints'
manager = tf.train.CheckpointManager(pre_train_checkpoint, checkpoint_dir, max_to_keep=50, checkpoint_name='save')

tf_agent = reinforce_agent.ReinforceAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    optimizer=optimizer,
    actor_network=actor_net,
    normalize_returns=True,
    train_step_counter=train_step_counter,
)
tf_agent.initialize()

replay_buffer = new_replay_buffer(tf_agent, train_env, replay_buffer_capacity)

# Start Training
eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy

tf_agent.train_step_counter.assign(0)
print("Base policy: ", compute_avg_return(eval_env, eval_policy, num_eval_episodes))
manager.save()

greedy = []
collect = []
def collect_episode(environment, policy, num_episodes=1):
    time_step = environment.reset()
    time_step_counter = 0
    total_reward = 0

    while not time_step.is_last():
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        total_reward += traj.reward

        # Add trajectory to the replay buffer
        replay_buffer.add_batch(traj)

        time_step_counter += 1
        environment.envs[0].render()
    print(f"Episode with with {time_step_counter} steps and return {total_reward}")

if __name__ == '__main__':
    for _ in range(num_iterations):
        collect_episode(train_env, collect_policy)
        experience = replay_buffer.gather_all()
        train_loss = tf_agent.train(experience)
        replay_buffer.clear()
        step = tf_agent.train_step_counter.numpy()
        print("Training episode: ", step)
        if step % save_interval == 0:
            manager.save()
        if step % eval_interval == 0:
            print("step = {0}: loss = {1}".format(step, train_loss.loss))
            avg_greedy = compute_avg_return(eval_env, eval_policy, num_eval_episodes)
            print("step = {0}: Greedy avg Return = {1}".format(step, avg_greedy))
            greedy.append(avg_greedy)
            avg_collect = compute_avg_return(eval_env, collect_policy, num_eval_episodes)
            print("step = {0}: Collect avg Return = {1}".format(step, avg_collect))
            collect.append(avg_collect)

            if avg_greedy > 10000:
                break
        
    print("total training episodes: ", step)
    for i in range(len(greedy)):
        episode = (i+1) * eval_interval
        print("greedy at episode ", episode, " is ", greedy[i])
        print("collect at episode ", episode, " is ", collect[i])