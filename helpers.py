from tf_agents.drivers import py_driver
from tf_agents.policies import py_tf_eager_policy


def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
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


def collect_episode(environment, policy, num_episodes, observer):

    driver = py_driver.PyDriver(
        environment,
        py_tf_eager_policy.PyTFEagerPolicy(policy, use_tf_function=True),
        [observer],
        max_episodes=num_episodes,
    )
    initial_time_step = environment.reset()
    driver.run(initial_time_step)
