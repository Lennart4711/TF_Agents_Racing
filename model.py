import tensorflow as tf
from tf_agents.environments import tf_py_environment


def new_model(env):
    environment = tf_py_environment.TFPyEnvironment(env)

    env_name = "CartPole-v0"  # @param {type:"string"}
    num_iterations = 250  # @param {type:"integer"}
    collect_episodes_per_iteration = 2  # @param {type:"integer"}
    replay_buffer_capacity = 2000  # @param {type:"integer"}

    fc_layer_params = (100,)

    learning_rate = 1e-3  # @param {type:"number"}
    log_interval = 25  # @param {type:"integer"}
    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 50  # @param {type:"integer"}

    env.reset()

    from tf_agents.networks import actor_distribution_network

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        env.observation_spec(), env.action_spec(), fc_layer_params=fc_layer_params
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    from tf_agents.agents.reinforce import reinforce_agent

    tf_agent = reinforce_agent.ReinforceAgent(
        env.time_step_spec(),
        env.action_spec(),
        actor_network=actor_net,
        optimizer=optimizer,
        normalize_returns=True,
        train_step_counter=train_step_counter,
    )
    tf_agent.initialize()
    print("Sucessfully initialized agent")
    return tf_agent
