from gym.envs.registration import register

register(
    id="Racing-v0",
    entry_point="envs.gym_env:RacingEnv",
)
