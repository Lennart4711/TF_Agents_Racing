import tensorflow as tf
from reinforce_agent import eval_env, eval_policy, tf_agent, actor_net, optimizer

WHICH_TO_RESTORE = 6

checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=actor_net)
checkpoint_dir = "tmp/pre_train_checkpoints"
manager = tf.train.CheckpointManager(
    checkpoint, checkpoint_dir, max_to_keep=20, checkpoint_name="save"
)
restore_path = manager.checkpoints[WHICH_TO_RESTORE - 1]
checkpoint.restore(restore_path)
tf_agent.initialize()

while True:
    print("running episode")

    time_step = eval_env.reset()
    episode_return = 0.0
    while not time_step.is_last():
        action_step = eval_policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        episode_return += time_step.reward
        eval_env.envs[0].render()
    print("Episode return: ", episode_return.numpy()[0])
