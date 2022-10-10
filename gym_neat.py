import gym 
from neural_network import NeuralNetwork
from dataclasses import dataclass

@dataclass
class Agent:
    nn: NeuralNetwork
    reward: float = 0
    env: gym.Env = gym.make('InvertedDoublePendulum-v2')


def visualize(nn: NeuralNetwork):
    env = gym.make('InvertedDoublePendulum-v4')
    step = env.reset()
    while True:
        action = nn.forward(step)
        step = env.step(action)
        env.render()

agents = [Agent(NeuralNetwork([8,16, 16, 6,2])) for _ in range(5)]

while True:
    for agent in agents:
        step = agent.env.reset()
        while not step.is_last():
            action = agent.nn.forward(step)
            step = agent.env.step(action)
            agent.reward += step
