from neural_network import NeuralNetwork
from environment import Environment
from dataclasses import dataclass


@dataclass
class Agent:
    nn: NeuralNetwork
    reward: float = 0
    env: Environment = Environment(has_surface=False)

def visualize(nn: NeuralNetwork):
    env = Environment()
    step = env.reset()
    while not step.is_last():
        action = nn.forward(step.observation)
        step = env.step(action)
        env.render(telemetry=True)

def run_agent(agent: Agent):
    step = agent.env.reset()
    while not step.is_last() and agent.reward < 2000:
        action = agent.nn.forward(step.observation)
        action[0] -= 0.5
        step = agent.env.step(action)
        agent.reward += step.reward
    return agent


agent_count = 5
mutation_rate = 0.1

agents = [Agent(NeuralNetwork([8,16, 16, 6,2])) for _ in range(agent_count)]


while True:
    agents = [run_agent(agent) for agent in agents] # Parallelize this
        
    agents.sort(key=lambda agent: agent.reward, reverse=True)
    # print(agents[0].reward)

    last_score = agents[0].reward
    visualize(agents[0].nn)

    new_agents = []
    # Append slightly mutated versions of the best 10% of agents
    for agent in agents[:int(agent_count * 0.1)]:
        new_agents.append(Agent(agent.nn.reproduce(mutation_rate)))
    # Fill 10% with random agents
    for _ in range(int(agent_count * 0.1)):
        new_agents.append(Agent(NeuralNetwork([8,5, 5, 2])))
    # Fill the rest with mutated versions of the best agent
    for _ in range(int(agent_count * 0.8)):
        new_agents.append(Agent(agents[0].nn.reproduce(mutation_rate)))
    agents = new_agents

