from tf_agents.drivers import py_driver
from tf_agents.policies import py_tf_eager_policy
from tf_agents.trajectories import trajectory


def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    print(environment)
    print()
    print(environment.batch_size)
    for _ in range(num_episodes):

        time_step = environment.reset()
        print(time_step)
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]
    # Please also see the metrics module for standard implementations of different
    # metrics.


def collect_step(environment, policy, replay_buffer):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  replay_buffer.add_batch(traj)
