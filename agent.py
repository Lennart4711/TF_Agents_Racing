import tensorflow as tf
from tf_agents.environments import tf_py_environment


def new_agent(env, fc_layer_params, learning_rate):

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
    print("Sucessfully created agent")
    return tf_agent
